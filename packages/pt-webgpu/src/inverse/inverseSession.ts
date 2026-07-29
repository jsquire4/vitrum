/**
 * pt-webgpu inverse-session implementation.
 *
 * `method:'finite-difference'` uses frozen-seed rerenders and a bound-aware
 * one-sided probe for every parameter component. `method:'path-replay'` is a
 * separate, deliberately narrow route: one-bounce RGB camera visibility and
 * the analytic primary-hit material-emissive identity. Its complete scene,
 * parameter, and render regime are validated before initial values are applied;
 * unsupported requests throw rather than downgrade or mix estimators.
 *
 * The host owns cadence. Each `step()` performs one optimizer iteration.
 *
 * Ref: Vicini 2021 (Path Replay Backprop); Kingma & Ba 2015 (Adam).
 */

import type {
  InverseSession,
  InverseSessionOptions,
  InverseStepResult,
  InverseGradientMethod,
  InverseSessionDiagnostic,
  Scene,
  SceneEmitter,
  MaterialSpec,
  BackendSupportMode,
} from '@vitrum/core';
import {
  Adam,
  type AdamSnapshot,
  assertFiniteArray,
  assertFiniteNumber,
  invokeInverseHook,
  l2Loss,
  l1Loss,
  lossValue,
  paramLength,
  parseParamPath,
  normalizeInverseError,
  clampParams,
  validateInverseReadback,
  validateInverseSessionOptions,
} from './optimizer.js';
import {
  type ParamSlot,
  validateParam,
  defaultClampRange,
  validateInitialSceneValue,
  readSceneValue,
  materialPatch,
  emitterPatch,
} from './paramResolution.js';
import {
  type InversePathReplayRenderContext,
  collectPathReplayDiagnostics,
} from './pathReplayDiagnostics.js';

export type { InversePathReplayRenderContext };

/** The engine hooks an InverseSession needs. The engine implements these with
 *  private access to its scene + GPU pipeline; the session stays decoupled from
 *  the engine's internals (and is independently testable with fakes). */
export interface InverseEngineHooks {
  /** The live scene (read-only snapshot used for path resolution + reading the
   *  starting parameter values). */
  getScene(): Scene;
  /** Render `samples` accumulated samples at the target resolution with a FROZEN
   *  RNG seed (so path-replay / FD perturbations differ only in the perturbed
   *  parameter), then read the accum texture back as interleaved RGB float +
   *  its channel count. Async (mapAsync). */
  renderAndReadback(
    width: number,
    height: number,
    samples: number,
  ): Promise<{ rgb: Float32Array; channels: 3 | 4 }>;
  /** Apply a material patch (mirrors Engine.updatePrimitive material fast path). */
  patchMaterial(primitiveId: string, patch: Partial<MaterialSpec>): void;
  /** Apply an emitter patch (mirrors Engine.updateEmitter). */
  patchEmitter(emitterId: string, patch: Partial<SceneEmitter>): void;
  /** Exact forward-regime facts required by fail-closed path-replay preflight. */
  getPathReplayRenderContext?(): InversePathReplayRenderContext;
  /**
   * Material support rows for the active pt-webgpu runtime profile. Full and
   * lite profiles consume different material subsets, so inverse must reject
   * parameters that the active shader path reports as unsupported instead of
   * optimizing a renderer no-op through finite differences.
   */
  getMaterialSupportDetails?(): Readonly<Partial<Record<keyof MaterialSpec, BackendSupportMode>>>;
  /**
   * Emitter support rows for the active pt-webgpu runtime profile. Lite and full
   * profiles differ here (mesh-area is ignored by the lite forward renderer), so
   * inverse must not optimize an emitter target or adjoint direct-light term that
   * the active profile reports as unsupported.
   */
  getEmitterSupportDetails?(): Readonly<Partial<Record<SceneEmitter['kind'], BackendSupportMode>>>;
  /**
   * Certified one-bounce emissive adjoint. It replaces finite-difference probes
   * only after fail-closed preflight establishes the exact production domain.
   */
  computeAdjointGradient?(args: AdjointGradientRequest): Promise<Float32Array>;
}

/** One optimized parameter, located for the engine's adjoint scatter. */
interface AdjointParamSlotDesc {
  readonly domain: 'materials' | 'emitters';
  readonly id: string;
  readonly field: string;
  /** Offset of this slot in the returned flat gradient. */
  readonly offset: number;
  /** Component count (3 for rgb, 2 for vec2, 1 for scalar). */
  readonly length: number;
}

/** Inputs to the engine's path-replay adjoint pass. */
export interface AdjointGradientRequest {
  /** Per-pixel ∂loss/∂rendered at the baseline params (interleaved, `channels`-wide). */
  readonly dLoss_dRendered: Float32Array;
  readonly channels: 3 | 4;
  readonly width: number;
  readonly height: number;
  /** Frozen-seed sample count to replay (matches the baseline render). */
  readonly samples: number;
  /** The parameters to differentiate + where each lands in the flat gradient. */
  readonly params: readonly AdjointParamSlotDesc[];
  /** Total length of the flat gradient to return. */
  readonly gradientLength: number;
}

const F32_MIN_SUBNORMAL = 1.401298464324817e-45;
const f32StepBuffer = new ArrayBuffer(4);
const f32StepValue = new Float32Array(f32StepBuffer);
const f32StepBits = new Uint32Array(f32StepBuffer);

/** Adjacent representable f32 in `direction`. The optimized vector itself is
 * Float32Array-backed, so a nominal epsilon can round back to the baseline for
 * large values. Advancing one ULP prevents a response-changing parameter from
 * silently receiving a zero-length probe. */
function adjacentFloat32(value: number, direction: -1 | 1): number {
  if (value === 0) return direction > 0 ? F32_MIN_SUBNORMAL : -F32_MIN_SUBNORMAL;
  f32StepValue[0] = value;
  const bits = f32StepBits[0]!;
  f32StepBits[0] =
    (value > 0) === (direction > 0)
      ? (bits + 1) >>> 0
      : (bits - 1) >>> 0;
  return f32StepValue[0];
}

interface FiniteDifferenceProbe {
  readonly value: number;
  readonly delta: number;
}

function probeInDirection(
  original: number,
  epsilon: number,
  direction: -1 | 1,
  min: number,
  max: number,
  enforceBounds: boolean,
): FiniteDifferenceProbe | null {
  let desired = original + direction * epsilon;
  if (enforceBounds) {
    desired = Math.min(max, Math.max(min, desired));
  }
  let value = Math.fround(desired);

  // A non-representable authored bound can round outside its own interval.
  // Move one f32 inward before considering the opposite one-sided direction.
  if (enforceBounds && value > max) value = adjacentFloat32(value, -1);
  if (enforceBounds && value < min) value = adjacentFloat32(value, 1);
  if (value === original) value = adjacentFloat32(original, direction);

  const delta = value - original;
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(delta) ||
    delta === 0 ||
    Math.sign(delta) !== direction ||
    (enforceBounds && (value < min || value > max))
  ) {
    return null;
  }
  return { value, delta };
}

function probeTowardInterval(
  original: number,
  epsilon: number,
  min: number,
  max: number,
): FiniteDifferenceProbe | null {
  const direction: -1 | 1 = original > max ? -1 : 1;
  const nearestBound = direction < 0 ? max : min;
  const distanceToInterval = Math.abs(original - nearestBound);
  const desired = original + direction * Math.min(epsilon, distanceToInterval);
  let value = Math.fround(desired);

  // When the capped step lands on a non-representable bound, round one ULP
  // inward rather than crossing the complete legal interval.
  if (direction < 0 && desired <= max && value > max) {
    value = adjacentFloat32(value, -1);
  } else if (direction > 0 && desired >= min && value < min) {
    value = adjacentFloat32(value, 1);
  }
  if (value === original) value = adjacentFloat32(original, direction);

  const delta = value - original;
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(delta) ||
    delta === 0 ||
    Math.sign(delta) !== direction ||
    // A capped probe may land at/just inside the nearest boundary, but must
    // never traverse through the complete legal interval.
    (direction < 0 && value < min) ||
    (direction > 0 && value > max)
  ) {
    return null;
  }
  return { value, delta };
}

/** Resolve a legal finite-difference sample against the same effective bounds
 * used by the post-Adam clamp. Legal boundary values use the inward one-sided
 * derivative. Legacy initial values outside an explicit interval retain their
 * documented first-step semantics and probe by epsilon toward that interval. */
function finiteDifferenceProbe(
  slot: ParamSlot,
  original: number,
  epsilon: number,
): FiniteDifferenceProbe | null {
  const [defaultMin, defaultMax] = defaultClampRange(slot.target.field);
  const min = slot.param.min ?? defaultMin;
  const max = slot.param.max ?? defaultMax;

  // A zero-width interval is an explicitly fixed parameter.
  if (min === max) return null;

  // An authored decimal bound may round to the adjacent f32 when stored in the
  // optimizer vector. Treat that exact stored representation as the boundary,
  // not as a legacy out-of-range initial value with a one-ULP probe.
  const atStoredMin = Number.isFinite(min) && original === Math.fround(min);
  const atStoredMax = Number.isFinite(max) && original === Math.fround(max);
  const inside = (original >= min && original <= max) || atStoredMin || atStoredMax;
  if (!inside) {
    return probeTowardInterval(original, epsilon, min, max);
  }

  let preferred: -1 | 1;
  if (original >= max || atStoredMax) {
    preferred = -1;
  } else if (original <= min || atStoredMin) {
    preferred = 1;
  } else if (original + epsilon <= max) {
    preferred = 1;
  } else if (original - epsilon >= min) {
    preferred = -1;
  } else {
    preferred = max - original >= original - min ? 1 : -1;
  }
  return (
    probeInDirection(original, epsilon, preferred, min, max, true) ??
    probeInDirection(original, epsilon, preferred === 1 ? -1 : 1, min, max, true)
  );
}

export class PtWebgpuInverseSession implements InverseSession {
  readonly #hooks: InverseEngineHooks;
  readonly #target: InverseSessionOptions['target'];
  readonly #lossFn: typeof l2Loss;
  readonly #lossKind: 'l2' | 'l1';
  readonly #method: InverseGradientMethod;
  readonly #parameterMethods: readonly InverseGradientMethod[];
  readonly #diagnostics: readonly InverseSessionDiagnostic[];
  readonly #samplesPerStep: number;
  readonly #fdEpsilon: number;
  readonly #slots: ParamSlot[];
  readonly #flat: Float32Array;
  readonly #adam: Adam;
  #stepIndex = 0;
  #disposed = false;
  #generation = 0;
  #stepInFlight = false;
  #poisoned: AggregateError | null = null;

  constructor(hooks: InverseEngineHooks, opts: InverseSessionOptions) {
    const config = validateInverseSessionOptions(opts, 'pt-webgpu');
    this.#hooks = hooks;
    this.#target = config.target;
    const loss = config.loss;
    this.#lossKind = loss;
    this.#lossFn = loss === 'l1' ? l1Loss : l2Loss;

    const requestedMethod: InverseGradientMethod = config.method;

    this.#samplesPerStep = config.samplesPerStep;
    this.#fdEpsilon = config.optimizer.fdEpsilon;

    const scene = invokeInverseHook(
      'createInverseSession getScene',
      () => hooks.getScene(),
    );
    const materialSupportDetails = hooks.getMaterialSupportDetails == null
      ? undefined
      : invokeInverseHook(
          'createInverseSession getMaterialSupportDetails',
          () => hooks.getMaterialSupportDetails!(),
        );
    const emitterSupportDetails = hooks.getEmitterSupportDetails == null
      ? undefined
      : invokeInverseHook(
          'createInverseSession getEmitterSupportDetails',
          () => hooks.getEmitterSupportDetails!(),
        );
    // Resolve every parameter path against the live scene, validate the field
    // matches the declared kind, and lay out the flat parameter vector.
    this.#slots = [];
    let offset = 0;
    for (const param of config.parameters) {
      const target = parseParamPath(param.path);
      validateParam(
        scene,
        param,
        target,
        materialSupportDetails,
        emitterSupportDetails,
      );
      const length = paramLength(param);
      this.#slots.push({ param, target, offset, length });
      offset += length;
    }
    this.#flat = new Float32Array(offset);

    // Path replay is all-or-nothing. Its diagnostics run before reading or
    // applying any parameter `initial` override.
    const pathReplayRenderContext = hooks.getPathReplayRenderContext == null
      ? {}
      : invokeInverseHook(
          'createInverseSession getPathReplayRenderContext',
          () => hooks.getPathReplayRenderContext!(),
        );
    const pathReplayDiagnostics = requestedMethod === 'path-replay'
      ? collectPathReplayDiagnostics(scene, this.#slots, {
          hasHook: hooks.computeAdjointGradient != null,
          renderContext: pathReplayRenderContext,
        })
      : [];
    this.#diagnostics = pathReplayDiagnostics;
    for (const diagnostic of pathReplayDiagnostics) {
      try {
        config.onDiagnostic?.(diagnostic);
      } catch {
        // Host diagnostic callbacks must not abort inverse-session creation.
      }
    }
    if (requestedMethod === 'path-replay' && pathReplayDiagnostics.length > 0) {
      const failures = pathReplayDiagnostics.map((diagnostic) => new Error(
        `${diagnostic.code}${diagnostic.path == null ? '' : ` (${diagnostic.path})`}: ` +
          diagnostic.message,
      ));
      throw new AggregateError(
        failures,
        'createInverseSession: requested path-replay is outside the certified ' +
          'pt-webgpu domain. No session was created and no scene values were changed. ' +
          "Select method:'finite-difference' explicitly or adjust the scene, parameter, " +
          'and render regime to match capabilities.inverseRendering.pathReplay.',
      );
    }
    this.#method = requestedMethod;
    this.#parameterMethods = this.#slots.map(() => requestedMethod);

    // Seed the flat vector from the parameter `initial` override or the current
    // scene value.
    const originalSceneValues = this.#captureSceneValues(scene);
    for (const slot of this.#slots) {
      const initial = slot.param.initial ?? readSceneValue(scene, slot.target, slot.length);
      if (initial.length !== slot.length) {
        throw new Error(
          `createInverseSession: parameter "${slot.param.path}" initial value has ` +
            `length ${initial.length}, expected ${slot.length}.`,
        );
      }
      validateInitialSceneValue(slot, initial, slot.param.initial != null);
      this.#flat.set(initial, slot.offset);
    }
    assertFiniteArray(this.#flat, 'createInverseSession parameter values');

    this.#adam = new Adam(this.#flat.length, config.optimizer);

    // Push the initial values so the first render reflects the (possibly
    // overridden) starting point.
    try {
      this.#applyFlatToScene();
    } catch (error) {
      const cause = normalizeInverseError(
        error,
        'createInverseSession initial scene update',
      );
      const rollbackErrors = this.#restoreSceneValues(originalSceneValues);
      if (rollbackErrors.length > 0) {
        throw new AggregateError(
          [cause, ...rollbackErrors],
          'createInverseSession: initial scene update and rollback both failed.',
        );
      }
      throw cause;
    }
  }

  get parameterCount(): number {
    return this.#slots.length;
  }

  get method(): InverseGradientMethod {
    return this.#method;
  }

  get parameterMethods(): readonly InverseGradientMethod[] {
    return [...this.#parameterMethods];
  }

  get diagnostics(): readonly InverseSessionDiagnostic[] {
    return this.#diagnostics;
  }

  currentValues(): readonly (readonly number[])[] {
    return this.#slots.map((s) => Array.from(this.#flat.subarray(s.offset, s.offset + s.length)));
  }

  async step(): Promise<InverseStepResult> {
    if (this.#disposed) {
      throw new Error('InverseSession.step: session is disposed.');
    }
    if (this.#poisoned != null) {
      throw this.#poisoned;
    }
    if (this.#stepInFlight) {
      throw new Error('InverseSession.step: a step is already in progress.');
    }
    this.#stepInFlight = true;
    try {
      const generation = this.#generation;
      const flatSnapshot = this.#flat.slice();
      const adamSnapshot = this.#adam.snapshot();
      const sceneSnapshot = this.#captureSceneValues(invokeInverseHook(
        'InverseSession.step getScene',
        () => this.#hooks.getScene(),
      ));
      try {
        const { width, height } = this.#target;
        this.#applyFlatToScene();
        const base = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
        this.#assertGeneration(generation);
        validateInverseReadback(base.rgb, base.channels, width, height, 'baseline readback');
        const { loss, dLoss_dRendered } =
          this.#lossFn(base.rgb, base.channels, this.#target);
        assertFiniteNumber(loss, 'baseline loss');
        assertFiniteArray(dLoss_dRendered, 'InverseSession.step loss gradient');

        let grad: Float32Array;
        if (this.#method === 'path-replay') {
          const adjointGrad = await this.#hooks.computeAdjointGradient!({
            dLoss_dRendered,
            channels: base.channels,
            width,
            height,
            samples: this.#samplesPerStep,
            params: this.#slots.map((slot) => ({
              domain: slot.target.domain,
              id: slot.target.id,
              field: slot.target.field,
              offset: slot.offset,
              length: slot.length,
            })),
            gradientLength: this.#flat.length,
          });
          this.#assertGeneration(generation);
          if (!(adjointGrad instanceof Float32Array)) {
            throw new Error(
              'InverseSession.step: adjoint gradient must be a Float32Array.',
            );
          }
          if (adjointGrad.length !== this.#flat.length) {
            throw new Error(
              `InverseSession.step: adjoint gradient length ${adjointGrad.length} ≠ ` +
                `parameter length ${this.#flat.length}.`,
            );
          }
          assertFiniteArray(adjointGrad, 'InverseSession.step adjoint gradient');
          grad = adjointGrad;
        } else {
          grad = new Float32Array(this.#flat.length);
          await this.#fillFiniteDifferenceGradient(
            grad,
            loss,
            width,
            height,
            generation,
            () => true,
          );
        }
        assertFiniteArray(grad, 'InverseSession.step gradient');

        const gradByParam = this.#slots.map((s) =>
          Array.from(grad.subarray(s.offset, s.offset + s.length)),
        );
        this.#adam.step(this.#flat, grad);
        clampParams(
          this.#flat,
          this.#slots.map((s) => s.param),
          this.#slots.map((s) => {
            const [defaultMin, defaultMax] = defaultClampRange(s.target.field);
            return { offset: s.offset, length: s.length, defaultMin, defaultMax };
          }),
        );
        assertFiniteArray(this.#flat, 'InverseSession.step parameter values');
        this.#applyFlatToScene();
        this.#assertGeneration(generation);

        const result: InverseStepResult = {
          step: this.#stepIndex,
          loss,
          values: this.currentValues(),
          gradient: gradByParam,
        };
        this.#stepIndex += 1;
        return result;
      } catch (error) {
        this.#rollbackStep(error, flatSnapshot, adamSnapshot, sceneSnapshot);
      }
    } finally {
      this.#stepInFlight = false;
    }
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#generation += 1;
    // No session-owned GPU buffers in the Phase-0 path (readback buffers are
    // created + destroyed per readback inside readOidnInputsFromTextures). The
    // optimized values stay applied to the scene by contract.
  }

  // ── internals ──────────────────────────────────────────────────────────────

  /** Fill selected gradient slots with bound-aware one-sided finite differences. */
  async #fillFiniteDifferenceGradient(
    grad: Float32Array,
    baselineLoss: number,
    width: number,
    height: number,
    generation: number,
    shouldProbeSlot: (slotIndex: number) => boolean,
  ): Promise<void> {
    for (let slotIndex = 0; slotIndex < this.#slots.length; slotIndex++) {
      if (!shouldProbeSlot(slotIndex)) continue;
      const slot = this.#slots[slotIndex]!;
      for (let local = 0; local < slot.length; local++) {
        const flatIndex = slot.offset + local;
        const original = this.#flat[flatIndex]!;
        const probeSpec = finiteDifferenceProbe(slot, original, this.#fdEpsilon);
        if (probeSpec == null) {
          grad[flatIndex] = 0;
          continue;
        }
        this.#flat[flatIndex] = probeSpec.value;
        let failed = false;
        let failure: unknown;
        let componentGradient = 0;
        try {
          assertFiniteNumber(this.#flat[flatIndex], 'finite-difference perturbation');
          this.#applyFlatToScene();
          const probe = await this.#hooks.renderAndReadback(
            width,
            height,
            this.#samplesPerStep,
          );
          this.#assertGeneration(generation);
          validateInverseReadback(probe.rgb, probe.channels, width, height, 'probe readback');
          const probeLoss = lossValue(
            probe.rgb,
            probe.channels,
            this.#target,
            this.#lossKind,
          );
          assertFiniteNumber(probeLoss, 'probe loss');
          componentGradient = (probeLoss - baselineLoss) / probeSpec.delta;
          assertFiniteNumber(componentGradient, 'finite-difference gradient');
        } catch (error) {
          failed = true;
          failure = normalizeInverseError(
            error,
            'InverseSession finite-difference probe',
          );
        }
        this.#flat[flatIndex] = original;
        try {
          this.#applyFlatToScene();
        } catch (restoreError) {
          const restoreCause = normalizeInverseError(
            restoreError,
            'InverseSession finite-difference probe restoration',
          );
          if (failed) {
            throw new AggregateError(
              [failure, restoreCause],
              'InverseSession.step: finite-difference probe and restoration both failed.',
            );
          }
          throw restoreCause;
        }
        if (failed) throw failure;
        grad[flatIndex] = componentGradient;
      }
    }
  }

  #applyFlatToScene(): void {
    for (const slot of this.#slots) {
      const value = Array.from(this.#flat.subarray(slot.offset, slot.offset + slot.length));
      this.#patchSlot(slot, value);
    }
  }

  #assertGeneration(generation: number): void {
    if (this.#disposed || generation !== this.#generation) {
      throw new Error('InverseSession.step: session was disposed while the step was in progress.');
    }
  }

  #captureSceneValues(scene: Scene): number[][] {
    return this.#slots.map((slot) =>
      readSceneValue(scene, slot.target, slot.length).slice(),
    );
  }

  #restoreSceneValues(values: readonly (readonly number[])[]): unknown[] {
    const errors: unknown[] = [];
    for (let i = 0; i < this.#slots.length; i++) {
      try {
        this.#patchSlot(this.#slots[i]!, values[i]!);
      } catch (error) {
        errors.push(normalizeInverseError(error, 'InverseSession scene rollback'));
      }
    }
    return errors;
  }

  #rollbackStep(
    error: unknown,
    flatSnapshot: Float32Array,
    adamSnapshot: AdamSnapshot,
    sceneSnapshot: readonly (readonly number[])[],
  ): never {
    const cause = normalizeInverseError(error, 'InverseSession.step');
    this.#flat.set(flatSnapshot);
    const rollbackErrors: unknown[] = [];
    try {
      this.#adam.restore(adamSnapshot);
    } catch (rollbackError) {
      rollbackErrors.push(normalizeInverseError(
        rollbackError,
        'InverseSession Adam rollback',
      ));
    }
    rollbackErrors.push(...this.#restoreSceneValues(sceneSnapshot));
    if (rollbackErrors.length > 0) {
      const aggregate = new AggregateError(
        [cause, ...rollbackErrors],
        'InverseSession.step: rollback failed; the session is poisoned.',
      );
      this.#poisoned = aggregate;
      throw aggregate;
    }
    throw cause;
  }

  #patchSlot(slot: ParamSlot, value: readonly number[]): void {
    if (slot.target.domain === 'materials') {
      this.#hooks.patchMaterial(slot.target.id, materialPatch(slot.target.field, value));
    } else {
      this.#hooks.patchEmitter(slot.target.id, emitterPatch(slot.target.field, value));
    }
  }
}
