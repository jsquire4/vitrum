/**
 * inverseSession.ts — the pt-webgpu InverseSession implementation (WS5).
 *
 * Phase 0 (finite-difference) is fully wired here and backend-portable in shape:
 *   1. render N samples with a FROZEN frame seed/index,
 *   2. read the accum texture back to CPU (reuses `readOidnInputsFromTextures`),
 *   3. compute the image-space loss vs the target,
 *   4. estimate the gradient by +epsilon perturbing each parameter and re-rendering,
 *   5. Adam-step the tiny flat parameter vector,
 *   6. push the updated params into the scene via updatePrimitive/updateEmitter.
 *
 * Phase 1 (path-replay analytic adjoint) plugs into the SAME step skeleton via
 * the OPTIONAL `InverseEngineHooks.computeAdjointGradient`: instead of the
 * N-render FD probe loop, the per-pixel loss gradient `dLoss/dRendered` is handed
 * to the engine's adjoint pass, which re-traces the frozen-seed path + NEE and
 * feeds the analytic BSDF partials (`../wgsl/pathTrace/pathTraceAdjoint.wgsl.ts`,
 * oracle in `brdfAdjoint.ts`). The two adjoint stages are GPU-validated on
 * lavapipe (V24): the partials match the FD oracle to f32 precision, and the
 * chain rule + fixed-point accumulation match an on-device finite-difference.
 * The session requests 'path-replay' only when the engine provides the hook,
 * every parameter appears in its end-to-end GPU-fit proof manifest, and every target
 * material stays within the scoped direct-light domain this pass mirrors
 * (delta/soft directional, point, spot, stochastic area-measure rect/disc/mesh
 * area lights, and direct HDRI/procedural-sky environment NEE), or
 * is a baseColorMap / COLOR_0-aware `shadingModel:'unlit'` baseColor primary-hit fit.
 * The built-in release proof currently covers material emissive only. Other
 * implemented local partials are not part of the production path-replay
 * contract until their complete GPU inverse fit agrees with the reference
 * gradient gate. The production engine uses `failurePolicy:'error'`: preflight
 * rejects the whole path-replay request before scene mutation if any slot,
 * geometry, lighting regime, or required hook is outside that proof. There is
 * no mixed analytic/finite-difference production session. This unexported
 * implementation class also supports an explicit finite-difference fallback
 * policy solely for isolated kernel and diagnostic tests. The hook and proof
 * manifest together form the hardware-validation claim.
 *
 * The host owns the cadence: each `step()` is one optimizer iteration. The
 * session never schedules itself.
 *
 * Ref: Vicini 2021 (Path Replay Backprop); Nimier-David 2020 (Radiative
 *      Backprop); Kingma & Ba 2015 (Adam).
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
  type InversePathReplayGeometryCapabilities,
  collectPathReplayDiagnostics,
  pathReplayRenderRegimeIssue,
  diagnosePathReplaySlot,
  isPathReplayZeroGradientSlot,
} from './pathReplayDiagnostics.js';

export type { InversePathReplayRenderContext, InversePathReplayGeometryCapabilities };

/** End-to-end inverse-fit evidence exposed by an engine implementation.
 * Implementation-capable local derivatives are not enough: a field appears
 * here only after its GPU session fit converges against finite differences. */
export interface InversePathReplayProofManifest {
  readonly materialFields: readonly string[];
  readonly emitterFields: readonly string[];
}

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
  /**
   * Optional render-regime facts for deciding whether the scoped path-replay
   * adjoint matches the most recent forward baseline. When omitted, the session
   * keeps legacy permissive behavior for non-engine fakes; the real pt-webgpu
   * engine supplies this so multi-bounce/spectral baselines downgrade honestly.
   */
  getPathReplayRenderContext?(): InversePathReplayRenderContext;
  /**
   * Optional geometry facts for deciding whether the scoped path-replay adjoint
   * can build an exact temporary triangle replay stream. Omitted means analytic
   * primitives remain outside path replay. The production pt-webgpu engine
   * intentionally reports no replayable analytic shapes: deterministic
   * tessellation is an approximation and therefore cannot certify an exact
   * path-replay gradient. Private implementation tests may supply capabilities
   * for isolated diagnostic coverage.
   */
  getPathReplayGeometryCapabilities?(): InversePathReplayGeometryCapabilities;
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
  /** Exact end-to-end proof manifest. The production engine supplies its
   *  certified public domain explicitly; isolated implementation tests may
   *  supply wider manifests to exercise private adjoint kernels. */
  getPathReplayProofManifest?(): InversePathReplayProofManifest;
  /**
   * Public-engine policy for an out-of-domain `method:'path-replay'` request.
   * The production pt-webgpu engine sets `error`, matching its published
   * capability contract. Omission preserves the finite-difference downgrade
   * used by isolated implementation tests; this session class is not exported
   * from the package public facade.
   */
  readonly pathReplayFailurePolicy?: 'error' | 'finite-difference';
  /**
   * OPTIONAL Phase-1 path-replay adjoint. When present, the engine dispatches a
   * single-bounce adjoint compute pass over its scene buffers — re-tracing the
   * frozen-seed primary path + NEE, evaluating the analytic BSDF partials
   * (`pathTraceAdjoint.wgsl`, GPU-validated vs the FD oracle to f32 precision),
   * and accumulating `∂loss/∂θ` (the chain rule GPU-validated vs on-device FD) —
   * and returns the flat gradient. Replaces the N-render FD probe loop with one
   * baseline render + one adjoint pass. The session only requests this when the
   * hook exists, every parameter is in the engine's proof manifest, and every target material
   * stays inside the adjoint-compatible direct-light domain (delta/soft
   * directional, point, spot, stochastic area-measure rect/disc/mesh area
   * lights, and direct HDRI/procedural-sky environment NEE). A field only
   * graduates to public path replay once its end-to-end inverse fit converges
   * and the engine includes it in `getPathReplayProofManifest`. The production
   * engine uses an error policy, so an unsupported slot rejects the complete
   * request before scene mutation. The optional fallback policy exists only for
   * this unexported class's isolated implementation tests.
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

const DEFAULT_PATH_REPLAY_PROOF: InversePathReplayProofManifest = {
  materialFields: ['emissive'],
  // Emitter derivatives have implementation/oracle coverage, but no completed
  // full-forward replay-vs-render proof. Keep them off the certified default.
  emitterFields: [],
};

function hasPathReplayProof(
  _scene: Scene,
  slot: ParamSlot,
  materialFields: ReadonlySet<string>,
  emitterFields: ReadonlySet<string>,
): boolean {
  return slot.target.domain === 'materials'
    ? materialFields.has(slot.target.field)
    : emitterFields.has(slot.target.field);
}

function pathReplayProofDiagnostic(
  scene: Scene,
  slot: ParamSlot,
  materialFields: ReadonlySet<string>,
  emitterFields: ReadonlySet<string>,
  failurePolicy: 'error' | 'finite-difference',
): InverseSessionDiagnostic | null {
  if (hasPathReplayProof(scene, slot, materialFields, emitterFields)) return null;
  const domainLabel = slot.target.domain === 'materials' ? 'material' : 'emitter';
  return {
    severity: 'info',
    code: 'path-replay-unsupported-field',
    path: slot.param.path,
    message:
      `[vitrum/pt-webgpu] InverseSession path "${slot.param.path}" targets ` +
      `${domainLabel} field "${slot.target.field}", which has local adjoint coverage ` +
      'but no registered end-to-end GPU inverse-fit proof; ' +
      (failurePolicy === 'error'
        ? 'the path-replay request will be rejected. Select finite-difference explicitly.'
        : 'using finite-difference.'),
    details: { field: slot.target.field, proof: 'missing-end-to-end-gpu-fit' },
  };
}

function pathReplayDiagnosticForPolicy(
  diagnostic: InverseSessionDiagnostic,
  failurePolicy: 'error' | 'finite-difference',
): InverseSessionDiagnostic {
  if (failurePolicy === 'finite-difference') return diagnostic;
  return {
    ...diagnostic,
    message: diagnostic.message.replace(
      /using finite-difference\./g,
      'the path-replay request will be rejected. Select finite-difference explicitly.',
    ),
  };
}

export class PtWebgpuInverseSession implements InverseSession {
  readonly #hooks: InverseEngineHooks;
  readonly #target: InverseSessionOptions['target'];
  readonly #lossFn: typeof l2Loss;
  readonly #lossKind: 'l2' | 'l1';
  readonly #method: InverseGradientMethod;
  readonly #parameterMethods: readonly InverseGradientMethod[];
  readonly #diagnostics: readonly InverseSessionDiagnostic[];
  readonly #pathReplaySlotEligible: readonly boolean[];
  readonly #pathReplayShaderSlotEligible: readonly boolean[];
  readonly #usesPartialPathReplay: boolean;
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

    // Resolve the EFFECTIVE gradient method from the request + backend capability
    // (InverseSessionOptions.method contract). Public 'path-replay' requires the
    // engine to provide the adjoint hook, every parameter to be in the
    // adjoint-differentiable set, and every target material to stay in the
    // compatible direct-light domain. The built-in proof manifest currently
    // advertises only material emissive. Additional local adjoint partials
    // remain finite-difference until the native GPU gate agrees with finite
    // differences. The production engine's error policy rejects any shortfall
    // before applying initial values, so its public facade never creates a
    // downgraded or mixed session. This unexported class retains an opt-in
    // finite-difference policy solely for isolated kernel/diagnostic tests. The
    // two adjoint stages the hook relies on (partials;
    // chain rule + accumulation) are GPU-validated on lavapipe (V24); the proof
    // manifest controls which complete fits may consume that machinery. Public
    // production hooks use `error`, so an uncertified slot cannot silently
    // downgrade; this internal class's fallback mode is test-only.
    const iridescenceOptimizedPrimitiveIds = new Set(
      this.#slots
        .filter(
          (s) =>
            s.target.domain === 'materials' &&
            (
              s.target.field === 'iridescence' ||
              s.target.field === 'iridescenceIor' ||
              s.target.field === 'iridescenceThicknessRange'
            ),
        )
        .map((s) => s.target.id),
    );
    const pathReplayRenderContext = hooks.getPathReplayRenderContext == null
      ? {}
      : invokeInverseHook(
          'createInverseSession getPathReplayRenderContext',
          () => hooks.getPathReplayRenderContext!(),
        );
    const pathReplayGeometryCapabilities =
      hooks.getPathReplayGeometryCapabilities == null
        ? {}
        : invokeInverseHook(
            'createInverseSession getPathReplayGeometryCapabilities',
            () => hooks.getPathReplayGeometryCapabilities!(),
          );
    const proofManifest = hooks.getPathReplayProofManifest == null
      ? DEFAULT_PATH_REPLAY_PROOF
      : invokeInverseHook(
          'createInverseSession getPathReplayProofManifest',
          () => hooks.getPathReplayProofManifest!(),
        );
    const provenMaterialFields = new Set(proofManifest.materialFields);
    const provenEmitterFields = new Set(proofManifest.emitterFields);
    const pathReplayFailurePolicy = hooks.pathReplayFailurePolicy ?? 'finite-difference';
    const implementationDiagnostics = requestedMethod === 'path-replay'
      ? collectPathReplayDiagnostics(scene, this.#slots, {
          hasHook: hooks.computeAdjointGradient != null,
          iridescenceOptimizedPrimitiveIds,
          renderContext: pathReplayRenderContext,
          geometryCapabilities: pathReplayGeometryCapabilities,
          emitterSupportDetails,
        }).map((diagnostic) =>
          pathReplayDiagnosticForPolicy(diagnostic, pathReplayFailurePolicy),
        )
      : [];
    const pathsWithImplementationIssue = new Set(
      implementationDiagnostics
        .map((diagnostic) => diagnostic.path)
        .filter((path): path is string => path != null),
    );
    const proofDiagnostics =
      requestedMethod === 'path-replay' && hooks.computeAdjointGradient != null
        ? this.#slots
            .filter((slot) => !pathsWithImplementationIssue.has(slot.param.path))
            .map((slot) =>
              pathReplayProofDiagnostic(
                scene,
                slot,
                provenMaterialFields,
                provenEmitterFields,
                pathReplayFailurePolicy,
              ),
            )
            .filter((diagnostic): diagnostic is InverseSessionDiagnostic =>
              diagnostic != null,
            )
        : [];
    const pathReplayDiagnostics = [...implementationDiagnostics, ...proofDiagnostics];
    this.#diagnostics = pathReplayDiagnostics;
    for (const diagnostic of pathReplayDiagnostics) {
      try {
        config.onDiagnostic?.(diagnostic);
      } catch {
        // Host diagnostic callbacks must not abort inverse-session creation.
      }
    }
    if (
      requestedMethod === 'path-replay' &&
      pathReplayDiagnostics.length > 0 &&
      hooks.pathReplayFailurePolicy === 'error'
    ) {
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
    const allEligible = pathReplayDiagnostics.length === 0;
    this.#method =
      requestedMethod === 'path-replay' && hooks.computeAdjointGradient != null && allEligible
        ? 'path-replay'
        : 'finite-difference';
    const canUseScopedAdjoint =
      requestedMethod === 'path-replay' &&
      hooks.computeAdjointGradient != null &&
      pathReplayRenderRegimeIssue(pathReplayRenderContext) == null;
    this.#pathReplaySlotEligible = canUseScopedAdjoint
      ? this.#slots.map((slot) =>
          hasPathReplayProof(
            scene,
            slot,
            provenMaterialFields,
            provenEmitterFields,
          ) &&
          diagnosePathReplaySlot(
              scene,
              slot,
              iridescenceOptimizedPrimitiveIds,
              pathReplayGeometryCapabilities,
              pathReplayRenderContext,
              emitterSupportDetails,
            ).length === 0,
        )
      : this.#slots.map(() => false);
    this.#pathReplayShaderSlotEligible = canUseScopedAdjoint
      ? this.#slots.map((slot, index) =>
          this.#pathReplaySlotEligible[index] === true &&
          !isPathReplayZeroGradientSlot(scene, slot),
        )
      : this.#slots.map(() => false);
    this.#usesPartialPathReplay =
      this.#method === 'finite-difference' &&
      canUseScopedAdjoint &&
      this.#pathReplaySlotEligible.some((eligible) => eligible);
    this.#parameterMethods = this.#pathReplaySlotEligible.map((eligible) =>
      eligible ? 'path-replay' : 'finite-difference',
    );

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
        if (this.#method === 'path-replay' || this.#usesPartialPathReplay) {
          grad = new Float32Array(this.#flat.length);
          const replaySlots = this.#slots.filter(
            (_slot, index) => this.#pathReplayShaderSlotEligible[index] === true,
          );
          if (replaySlots.length > 0) {
            const adjointGrad = await this.#hooks.computeAdjointGradient!({
              dLoss_dRendered,
              channels: base.channels,
              width,
              height,
              samples: this.#samplesPerStep,
              params: replaySlots.map((s) => ({
                domain: s.target.domain,
                id: s.target.id,
                field: s.target.field,
                offset: s.offset,
                length: s.length,
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
            for (const slot of replaySlots) {
              grad.set(
                adjointGrad.subarray(slot.offset, slot.offset + slot.length),
                slot.offset,
              );
            }
          }
          if (this.#usesPartialPathReplay) {
            await this.#fillFiniteDifferenceGradient(
              grad,
              loss,
              width,
              height,
              generation,
              (slotIndex) => this.#pathReplaySlotEligible[slotIndex] !== true,
            );
          }
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

  /** Fill selected gradient slots with forward finite differences. */
  async #fillFiniteDifferenceGradient(
    grad: Float32Array,
    baselineLoss: number,
    width: number,
    height: number,
    generation: number,
    shouldProbeSlot: (slotIndex: number) => boolean,
  ): Promise<void> {
    const eps = this.#fdEpsilon;
    for (let slotIndex = 0; slotIndex < this.#slots.length; slotIndex++) {
      if (!shouldProbeSlot(slotIndex)) continue;
      const slot = this.#slots[slotIndex]!;
      for (let local = 0; local < slot.length; local++) {
        const flatIndex = slot.offset + local;
        const original = this.#flat[flatIndex]!;
        this.#flat[flatIndex] = original + eps;
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
          componentGradient = (probeLoss - baselineLoss) / eps;
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
