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
 * every parameter is in the adjoint-differentiable set, and every target
 * material stays within the scoped direct-light domain this pass mirrors
 * (delta/soft directional, point, spot, stochastic area-measure rect/disc/mesh
 * area lights, and direct HDRI/procedural-sky environment NEE), or
 * is a baseColorMap / COLOR_0-aware `shadingModel:'unlit'` baseColor primary-hit fit;
 * `ADJOINT_ELIGIBLE_FIELDS`: material baseColor / roughness / metallic /
 * AO intensity / light-map intensity / emissive / env-map intensity /
 * specularColor / specularIntensity / clearcoat / sheen / iridescence /
 * iridescenceThicknessRange / anisotropy controls plus normalScale, bumpScale,
 * and clearcoatNormalScale with their readable local map factors); any
 * shortfall (no hook, an unsupported emitter/material target, an `ior` param, etc.) resolves the
 * effective method to 'finite-difference', reported via `session.method` — no
 * silent wrong-gradient path. Mixed sessions may still route individually
 * eligible slots through the adjoint hook and finite-difference only the
 * holdouts; the public method stays conservative until every slot is replayed.
 * An engine providing the hook vouches that its re-trace dispatch is
 * hardware-validated.
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
  DEFAULT_ADAM,
  l2Loss,
  l1Loss,
  lossValue,
  paramLength,
  parseParamPath,
  clampParams,
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
   * can build a temporary triangle replay stream. Omitted means analytic
   * primitives remain finite-difference. The real pt-webgpu engine supplies this
   * so full-tier supported analytics can replay via deterministic tessellation
   * while lite-tier unsupported analytics stay fail-closed.
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
  /**
   * OPTIONAL Phase-1 path-replay adjoint. When present, the engine dispatches a
   * single-bounce adjoint compute pass over its scene buffers — re-tracing the
   * frozen-seed primary path + NEE, evaluating the analytic BSDF partials
   * (`pathTraceAdjoint.wgsl`, GPU-validated vs the FD oracle to f32 precision),
   * and accumulating `∂loss/∂θ` (the chain rule GPU-validated vs on-device FD) —
   * and returns the flat gradient. Replaces the N-render FD probe loop with one
   * baseline render + one adjoint pass. The session only requests this when the
   * hook exists, every parameter is adjoint-eligible, and every target material
   * stays inside the adjoint-compatible direct-light domain (delta/soft
   * directional, point, spot, stochastic area-measure rect/disc/mesh area
   * lights, and direct HDRI/procedural-sky environment NEE;
   * `ADJOINT_ELIGIBLE_FIELDS`: material baseColor / roughness / metallic /
   * AO intensity / light-map intensity / emissive / env-map intensity /
   * specularColor / specularIntensity / clearcoat / sheen / iridescence /
   * iridescenceThicknessRange / anisotropy controls plus normalScale,
   * bumpScale, and clearcoatNormalScale with their readable local map factors);
   * otherwise it reports + uses
   * 'finite-difference' (no silently-wrong gradient). An engine that provides
   * this hook is vouching that its adjoint pass is hardware-validated — a field
   * only graduates to path-replay once its end-to-end inverse fit converges.
   * Mixed sessions keep `session.method === 'finite-difference'`, but the
   * session may call this hook for eligible slots and use finite-difference for
   * unsupported holdouts.
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

export class PtWebgpuInverseSession implements InverseSession {
  readonly #hooks: InverseEngineHooks;
  readonly #target: InverseSessionOptions['target'];
  readonly #lossFn: typeof l2Loss;
  readonly #lossKind: 'l2' | 'l1';
  readonly #method: InverseGradientMethod;
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

  constructor(hooks: InverseEngineHooks, opts: InverseSessionOptions) {
    this.#hooks = hooks;
    this.#target = opts.target;
    if (opts.parameters.length === 0) {
      throw new Error('createInverseSession: at least one parameter is required.');
    }
    if (opts.target.width <= 0 || opts.target.height <= 0) {
      throw new Error('createInverseSession: target image must have positive dimensions.');
    }

    const loss = opts.loss ?? 'l2';
    if (loss === 'ssim' || loss === 'lpips') {
      throw new Error(
        `createInverseSession: loss '${loss}' is a reserved perceptual loss and ` +
          "is not yet implemented in pt-webgpu (use 'l2' or 'l1').",
      );
    }
    this.#lossKind = loss === 'l1' ? 'l1' : 'l2';
    this.#lossFn = loss === 'l1' ? l1Loss : l2Loss;

    const requestedMethod: InverseGradientMethod = opts.method ?? 'finite-difference';

    this.#samplesPerStep = Math.max(1, Math.floor(opts.samplesPerStep ?? 8));
    const optimizerCfg = opts.optimizer ?? {};
    this.#fdEpsilon = optimizerCfg.fdEpsilon ?? 1e-3;

    const scene = hooks.getScene();
    // Resolve every parameter path against the live scene, validate the field
    // matches the declared kind, and lay out the flat parameter vector.
    this.#slots = [];
    let offset = 0;
    for (const param of opts.parameters) {
      const target = parseParamPath(param.path);
      validateParam(
        scene,
        param,
        target,
        hooks.getMaterialSupportDetails?.(),
        hooks.getEmitterSupportDetails?.(),
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
    // compatible direct-light domain, or a mapped/unmapped unlit baseColor primary-hit
    // fit (`ADJOINT_ELIGIBLE_FIELDS`: material
    // baseColor / roughness / metallic / aoMapIntensity / lightMapIntensity / emissive /
    // emissiveIntensity / specularColor / specularIntensity / clearcoat / sheen /
    // iridescence / iridescenceThicknessRange / anisotropy slices — see its doc
    // for scoped map coverage and exclusions). Any shortfall degrades the public
    // `method` to finite-difference, so the host never receives a silently-wrong
    // all-adjoint promise. When the hook/render regime are valid but only some
    // slots are supported, the step still replays the eligible slots and FD-probes
    // only the holdouts. The two adjoint stages the hook relies on (partials;
    // chain rule + accumulation) are GPU-validated on lavapipe (V24); an engine
    // exposing the hook vouches for the re-trace.
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
    const pathReplayRenderContext = hooks.getPathReplayRenderContext?.() ?? {};
    const pathReplayGeometryCapabilities = hooks.getPathReplayGeometryCapabilities?.() ?? {};
    const emitterSupportDetails = hooks.getEmitterSupportDetails?.();
    const pathReplayDiagnostics = requestedMethod === 'path-replay'
      ? collectPathReplayDiagnostics(scene, this.#slots, {
          hasHook: hooks.computeAdjointGradient != null,
          iridescenceOptimizedPrimitiveIds,
          renderContext: pathReplayRenderContext,
          geometryCapabilities: pathReplayGeometryCapabilities,
          emitterSupportDetails,
        })
      : [];
    this.#diagnostics = pathReplayDiagnostics;
    for (const diagnostic of pathReplayDiagnostics) {
      try {
        opts.onDiagnostic?.(diagnostic);
      } catch {
        // Host diagnostic callbacks must not abort inverse-session creation.
      }
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

    // Seed the flat vector from the parameter `initial` override or the current
    // scene value.
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

    this.#adam = new Adam(this.#flat.length, {
      learningRate: optimizerCfg.learningRate ?? DEFAULT_ADAM.learningRate,
      beta1: optimizerCfg.beta1 ?? DEFAULT_ADAM.beta1,
      beta2: optimizerCfg.beta2 ?? DEFAULT_ADAM.beta2,
      epsilon: optimizerCfg.epsilon ?? DEFAULT_ADAM.epsilon,
    });

    // Push the initial values so the first render reflects the (possibly
    // overridden) starting point.
    this.#applyFlatToScene();
  }

  get parameterCount(): number {
    return this.#slots.length;
  }

  get method(): InverseGradientMethod {
    return this.#method;
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
    const { width, height } = this.#target;

    // 1. Baseline render at the current params + its loss.
    this.#applyFlatToScene();
    const base = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
    const { loss, dLoss_dRendered } = this.#lossFn(base.rgb, base.channels, this.#target);

    // 2. Gradient.
    let grad: Float32Array;
    if (this.#method === 'path-replay' || this.#usesPartialPathReplay) {
      // Phase 1: ONE adjoint pass replaces the N-render FD probe loop. The engine
      // re-traces the frozen-seed primary path + NEE and accumulates ∂loss/∂θ from
      // the per-pixel `dLoss_dRendered` through the GPU-validated BSDF partials.
      // (Method resolution already guaranteed the hook + adjoint-eligibility for
      // every slot included in `replaySlots`.)
      grad = new Float32Array(this.#flat.length);
      const replaySlots = this.#slots.filter((_slot, index) => this.#pathReplayShaderSlotEligible[index] === true);
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
        if (adjointGrad.length !== this.#flat.length) {
          throw new Error(
            `InverseSession.step: adjoint gradient length ${adjointGrad.length} ≠ ` +
              `parameter length ${this.#flat.length}.`,
          );
        }
        for (const slot of replaySlots) {
          grad.set(adjointGrad.subarray(slot.offset, slot.offset + slot.length), slot.offset);
        }
      }
      if (this.#usesPartialPathReplay) {
        await this.#fillFiniteDifferenceGradient(
          grad,
          loss,
          width,
          height,
          (slotIndex) => this.#pathReplaySlotEligible[slotIndex] !== true,
        );
      }
    } else {
      // Phase 0: forward-difference each scalar component by re-rendering with the
      // component perturbed by +ε. dLoss/dθ_i ≈ (loss(θ + ε·e_i) − loss(θ)) / ε.
      // The FD path needs only the scalar loss delta, so it uses the
      // allocation-free `lossValue` (dLoss_dRendered is the Phase-1 input).
      void dLoss_dRendered;
      grad = new Float32Array(this.#flat.length);
      await this.#fillFiniteDifferenceGradient(grad, loss, width, height, () => true);
    }

    // 3. Adam step + clamp, then push the updated params back to the scene.
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
    this.#applyFlatToScene();

    const result: InverseStepResult = {
      step: this.#stepIndex,
      loss,
      values: this.currentValues(),
      gradient: gradByParam,
    };
    this.#stepIndex += 1;
    return result;
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    // No session-owned GPU buffers in the Phase-0 path (readback buffers are
    // created + destroyed per readback inside readOidnInputsFromTextures). The
    // optimized values stay applied to the scene by contract.
  }

  // ── internals ──────────────────────────────────────────────────────────────

  /** Push every parameter slot's current flat value into the scene via the
   *  engine's incremental-update hooks. */
  async #fillFiniteDifferenceGradient(
    grad: Float32Array,
    baselineLoss: number,
    width: number,
    height: number,
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
        this.#applyFlatToScene();
        const probe = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
        const probeLoss = lossValue(probe.rgb, probe.channels, this.#target, this.#lossKind);
        grad[flatIndex] = (probeLoss - baselineLoss) / eps;
        this.#flat[flatIndex] = original; // restore before the next probe
      }
    }
  }

  #applyFlatToScene(): void {
    for (const slot of this.#slots) {
      const value = Array.from(this.#flat.subarray(slot.offset, slot.offset + slot.length));
      if (slot.target.domain === 'materials') {
        this.#hooks.patchMaterial(slot.target.id, materialPatch(slot.target.field, value));
      } else {
        this.#hooks.patchEmitter(slot.target.id, emitterPatch(slot.target.field, value));
      }
    }
  }
}
