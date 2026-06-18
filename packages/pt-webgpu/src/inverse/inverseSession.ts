/**
 * inverseSession.ts — the pt-webgpu InverseSession implementation (WS5).
 *
 * Phase 0 (finite-difference) is fully wired here and backend-portable in shape:
 *   1. render N samples with a FROZEN frame seed/index,
 *   2. read the accum texture back to CPU (reuses `readOidnInputsFromTextures`),
 *   3. compute the image-space loss vs the target,
 *   4. estimate the gradient by ±ε perturbing each parameter and re-rendering,
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
 * material stays within the direct-light base/specular/extension-lobe domain this pass mirrors
 * (delta directional, point, spot, and stochastic area-measure rect/disc/mesh
 * area lights), or
 * is a baseColorMap / COLOR_0-aware `shadingModel:'unlit'` baseColor primary-hit fit;
 * `ADJOINT_ELIGIBLE_FIELDS`: material baseColor / roughness / metallic /
 * AO intensity / emissive / specularColor / specularIntensity / clearcoat / sheen /
 * iridescence / iridescenceThicknessRange / anisotropy controls with their
 * readable local map factors); any
 * shortfall (no hook, an unsupported emitter/material target, an `ior` param, etc.) resolves the
 * effective method to 'finite-difference', reported via `session.method` — no
 * silent wrong-gradient path. An engine providing the hook vouches that its
 * re-trace dispatch is hardware-validated.
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
  InverseParam,
  InverseGradientMethod,
  InverseSessionDiagnostic,
  Scene,
  ScenePrimitive,
  SceneEmitter,
  MaterialSpec,
} from '@vitrum/core';
import type { Vec2, Vec3 } from '@vitrum/core';
import {
  Adam,
  DEFAULT_ADAM,
  l2Loss,
  l1Loss,
  lossValue,
  paramLength,
  parseParamPath,
  clampParams,
  type ResolvedParamTarget,
} from './optimizer.js';
import { meshAreaEmitterAdjointRangeForScene } from '../scene/emitterPacking.js';

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
   * OPTIONAL Phase-1 path-replay adjoint. When present, the engine dispatches a
   * single-bounce adjoint compute pass over its scene buffers — re-tracing the
   * frozen-seed primary path + NEE, evaluating the analytic BSDF partials
   * (`pathTraceAdjoint.wgsl`, GPU-validated vs the FD oracle to f32 precision),
   * and accumulating `∂loss/∂θ` (the chain rule GPU-validated vs on-device FD) —
   * and returns the flat gradient. Replaces the N-render FD probe loop with one
   * baseline render + one adjoint pass. The session only requests this when the
   * hook exists, every parameter is adjoint-eligible, and every target material
   * stays inside the adjoint-compatible direct-light domain (delta directional,
   * point, spot, and stochastic area-measure rect/disc/mesh area lights;
   * `ADJOINT_ELIGIBLE_FIELDS`: material baseColor / roughness / metallic /
   * AO intensity / emissive / specularColor / specularIntensity / clearcoat / sheen /
   * iridescence / iridescenceThicknessRange / anisotropy controls with their
   * readable local map factors); otherwise it reports + uses
   * 'finite-difference' (no silently-wrong gradient). An engine that provides
   * this hook is vouching that its adjoint pass is hardware-validated — a field
   * only graduates to path-replay once its end-to-end inverse fit converges.
   */
  computeAdjointGradient?(args: AdjointGradientRequest): Promise<Float32Array>;
}

export interface InversePathReplayRenderContext {
  readonly bounces?: number;
  readonly spectral?: boolean;
  readonly bdpt?: boolean;
  readonly restirPtReuse?: boolean;
  readonly causticStrategy?: 'none' | 'manifold-nee' | 'photon-map';
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

/**
 * Material fields the path-replay adjoint differentiates (resolves the method to
 * 'path-replay' instead of FD).
 *
 *  - `baseColor`, `roughness` — the original Phase-1 BSDF partials
 *    (`dBrdf_dBaseColor` / `dBrdf_dRoughness`), GPU-validated end-to-end (V24).
 *    `baseColor` also covers baseColorMap / COLOR_0-aware
 *    `shadingModel:'unlit'` primary hits:
 *    forward contributes `throughput · baseColor` and terminates, so the
 *    derivative is the direct contribution-level identity rather than a BRDF
 *    partial or light-domain term.
 *  - `metallic` — the opaque base-BRDF partial through the diffuse fade-out
 *    and F0 blend in the same direct-light replay domain.
 *  - `emissive` / `emissiveIntensity` — the camera-direct emission partials
 *    (`∂rendered_c/∂emissive_c = throughput · emissiveIntensity` and
 *    `∂rendered_c/∂emissiveIntensity = throughput · emissive_c`, scattered at
 *    the PRIMARY hit where the camera sees the emissive surface directly — NOT
 *    a NEE term, so they need no light). `emissive` is GPU-validated end-to-end
 *    on lavapipe
 *    (`wsl-gpu tests/v24-emissive-fit.mjs`): the path-replay engine adjoint
 *    gradient SIGN-MATCHES the full-render FD on the decisive channels and the fit
 *    converges (param error 3→~0.4). The earlier divergent trial scattered emissive
 *    inside the NEE loop / without folding the live emissiveIntensity through the
 *    descriptor; the fix scatters it at the primary hit gated by the matId match
 *    and hands the fixed emissiveIntensity in the descriptor `.w` (bitcast f32).
 *  - `specularColor`, `specularIntensity` — KHR_materials_specular dielectric F0
 *    controls through the same frozen direct-light BRDF derivative. These affect
 *    the diffuse/specular partition and the specular Fresnel colour for
 *    non-metallic and partially-metallic surfaces; fully metallic surfaces still
 *    source F0 from baseColor and therefore have zero derivative here.
 *  - `clearcoat`, `clearcoatRoughness` — KHR_materials_clearcoat direct lobe
 *    controls. Clearcoat/clearcoatRoughness maps replay as local scalar
 *    chain-rule factors; clearcoatNormalMap stays on finite difference until
 *    the adjoint pass mirrors normal-map sampling and path/visibility effects.
 *  - `sheen`, `sheenColor`, `sheenRoughness` — KHR_materials_sheen direct lobe
 *    controls through the additive Charlie lobe. Sheen color/roughness maps
 *    replay as local chain-rule factors.
 *  - `iridescence` — KHR_materials_iridescence scalar through the
 *    thin-film-modified base F0 in the same scoped opaque direct-light domain.
 *    Iridescence maps replay as local scalar chain-rule factors.
 *  - `iridescenceIor` and `iridescenceThicknessRange` — scalar thin-film IOR
 *    and authored min/max thickness ranges. Iridescence thickness maps replay by
 *    collapsing the range to the sampled forward value for the BRDF derivative;
 *    thickness-range gradients chain the sampled texel (or map-free `V·H`) back
 *    to the min/max endpoints.
 *  - `aoMapIntensity` — local derivative of glTF AO's
 *    `mix(1, sampledR, intensity)` baseColor multiplier.
 *  - `lightMapIntensity` — additive primary-hit baked-radiance partial:
 *    `∂rendered/∂lightMapIntensity = lightMapRadianceTexel` in the same
 *    camera-direct emission slot as `emissive`, with no direct-light requirement.
 *  - baseColorMap/COLOR_0, roughnessMap/metallicMap, AO, specular
 *    color/intensity maps, clearcoat/sheen maps, iridescence/thickness maps,
 *    anisotropy maps, and light maps are replayed as local chain-rule factors
 *    for the lit BRDF / primary-hit emission domains above. Additive primary-hit
 *    emissiveMap/lightMap terms are allowed on BRDF/unlit targets because they
 *    do not change the derivative of those optimized fields; dLossDRendered
 *    already contains their forward contribution. Alpha, transmission,
 *    normal/bump, displacement, and
 *    clearcoat-normal maps remain finite-difference fallbacks until their
 *    visibility/transport/normal terms are mirrored.
 *  - `anisotropy` / `anisotropyRotation` — map-free scalar anisotropic-GGX
 *    controls through a local symmetric derivative of the direct-light specular
 *    lobe. Anisotropy maps replay the B-channel strength multiplier and RG
 *    rotation offset used by the forward shader.
 *
 * `ior` is deliberately NOT here — it optimizes via finite difference (correct,
 * just slower) and has a GPU-validated analytic partial (`dFrDielectric_dIor` —
 * analytic == FD in isolation, `ADJOINT_EMISSIVE_IOR_FD_WGSL`) ready for a future
 * path-replay wire: `∂evaluateBrdf/∂ior ≡ 0` in the current forward
 * (opaque F0 is controlled by KHR_materials_specular/baseColor, not ior), and the single-bounce
 * adjoint doesn't trace the transmissive Fresnel partition where ior IS
 * differentiable.
 *
 * NOTE: adding a field here makes `inverseSession` REQUEST path-replay; the
 * engine's `computeAdjointGradient` hook must actually accumulate that field's
 * gradient and the field needs proof appropriate to its risk. baseColor,
 * roughness, and emissive have GPU inverse-fit captures; specular, metallic,
 * scalar clearcoat, sheen controls, scalar iridescence,
 * scalar iridescenceIor, anisotropy controls, and lightMapIntensity are
 * CPU-FD-oracle + shader-gate covered and remain on the recapture tail.
 */
const ADJOINT_ELIGIBLE_FIELDS = new Set([
  'baseColor',
  'roughness',
  'metallic',
  'aoMapIntensity',
  'lightMapIntensity',
  'emissive',
  'emissiveIntensity',
  'specularColor',
  'specularIntensity',
  'clearcoat',
  'clearcoatRoughness',
  'sheen',
  'sheenColor',
  'sheenRoughness',
  'iridescence',
  'iridescenceIor',
  'iridescenceThicknessRange',
  'anisotropy',
  'anisotropyRotation',
  'envMapIntensity',
]);
const ADJOINT_ELIGIBLE_EMITTER_FIELDS = new Set(['color', 'intensity']);
const ADJOINT_MAPPED_EMISSION_EPS = 1e-8;
const PATH_REPLAY_TRANSPORT_ONLY_FIELDS = new Set([
  'ior',
  'transmission',
  'thickness',
  'attenuationColor',
  'attenuationDistance',
  'dispersionAbbeNumber',
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'scatteringCoefficientRGB',
]);
const PATH_REPLAY_VISIBILITY_ONLY_FIELDS = new Set(['opacity', 'alphaCutoff']);
const PATH_REPLAY_NORMAL_ONLY_FIELDS = new Set(['normalScale', 'bumpScale', 'clearcoatNormalScale']);

interface ParamSlot {
  readonly param: InverseParam;
  readonly target: ResolvedParamTarget;
  readonly offset: number;
  readonly length: number;
}

const MATERIAL_RGB_FIELDS = new Set([
  'baseColor',
  'emissive',
  'attenuationColor',
  'specularColor',
  'sheenColor',
  'scatteringCoefficientRGB',
]);
const MATERIAL_VEC2_FIELDS = new Set([
  'iridescenceThicknessRange',
]);
const MATERIAL_SCALAR_FIELDS = new Set([
  'roughness',
  'metallic',
  'emissiveIntensity',
  'opacity',
  'alphaCutoff',
  'ior',
  'transmission',
  'thickness',
  'attenuationDistance',
  'specularIntensity',
  'clearcoat',
  'clearcoatRoughness',
  'sheen',
  'sheenRoughness',
  'iridescence',
  'iridescenceIor',
  'anisotropy',
  'anisotropyRotation',
  'normalScale',
  'bumpScale',
  'clearcoatNormalScale',
  'aoMapIntensity',
  'lightMapIntensity',
  'envMapIntensity',
  'dispersionAbbeNumber',
  'scatteringCoefficient',
  'scatteringAnisotropy',
]);
const EMITTER_RGB_FIELDS = new Set(['color']);
const EMITTER_SCALAR_FIELDS = new Set(['intensity']);

type PathReplayUnsupportedCode = InverseSessionDiagnostic['code'];
type PathReplayMaterialIssue = {
  readonly code?: PathReplayUnsupportedCode;
  readonly message: string;
  readonly details: Record<string, string | number | boolean | readonly string[]>;
};

export class PtWebgpuInverseSession implements InverseSession {
  readonly #hooks: InverseEngineHooks;
  readonly #target: InverseSessionOptions['target'];
  readonly #lossFn: typeof l2Loss;
  readonly #lossKind: 'l2' | 'l1';
  readonly #method: InverseGradientMethod;
  readonly #diagnostics: readonly InverseSessionDiagnostic[];
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
      validateParam(scene, param, target);
      const length = paramLength(param);
      this.#slots.push({ param, target, offset, length });
      offset += length;
    }
    this.#flat = new Float32Array(offset);

    // Resolve the EFFECTIVE gradient method from the request + backend capability
    // (InverseSessionOptions.method contract). 'path-replay' requires the engine
    // to provide the adjoint hook, every parameter to be in the
    // adjoint-differentiable set, and every target material to stay in the
    // compatible direct-light domain, or a mapped/unmapped unlit baseColor primary-hit
    // fit (`ADJOINT_ELIGIBLE_FIELDS`: material
    // baseColor / roughness / metallic / aoMapIntensity / lightMapIntensity / emissive /
    // emissiveIntensity / specularColor / specularIntensity / clearcoat / sheen /
    // iridescence / iridescenceThicknessRange / anisotropy slices — see its doc
    // for scoped map coverage and exclusions). Any shortfall
    // degrades to finite-difference and is reported via `method`, so the host
    // never receives a silently-wrong gradient. The two adjoint stages
    // the hook relies on (partials; chain rule + accumulation) are GPU-validated
    // on lavapipe (V24); an engine exposing the hook vouches for the re-trace.
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
    const pathReplayDiagnostics = requestedMethod === 'path-replay'
      ? collectPathReplayDiagnostics(scene, this.#slots, {
          hasHook: hooks.computeAdjointGradient != null,
          iridescenceOptimizedPrimitiveIds,
          renderContext: hooks.getPathReplayRenderContext?.() ?? {},
        })
      : [];
    this.#diagnostics = pathReplayDiagnostics;
    for (const diagnostic of pathReplayDiagnostics) {
      opts.onDiagnostic?.(diagnostic);
    }
    const allEligible = pathReplayDiagnostics.length === 0;
    this.#method =
      requestedMethod === 'path-replay' && hooks.computeAdjointGradient != null && allEligible
        ? 'path-replay'
        : 'finite-difference';

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
    if (this.#method === 'path-replay') {
      // Phase 1: ONE adjoint pass replaces the N-render FD probe loop. The engine
      // re-traces the frozen-seed primary path + NEE and accumulates ∂loss/∂θ from
      // the per-pixel `dLoss_dRendered` through the GPU-validated BSDF partials.
      // (Method resolution already guaranteed the hook + adjoint-eligibility.)
      grad = await this.#hooks.computeAdjointGradient!({
        dLoss_dRendered,
        channels: base.channels,
        width,
        height,
        samples: this.#samplesPerStep,
        params: this.#slots.map((s) => ({
          domain: s.target.domain,
          id: s.target.id,
          field: s.target.field,
          offset: s.offset,
          length: s.length,
        })),
        gradientLength: this.#flat.length,
      });
      if (grad.length !== this.#flat.length) {
        throw new Error(
          `InverseSession.step: adjoint gradient length ${grad.length} ≠ ` +
            `parameter length ${this.#flat.length}.`,
        );
      }
    } else {
      // Phase 0: forward-difference each scalar component by re-rendering with the
      // component perturbed by +ε. dLoss/dθ_i ≈ (loss(θ + ε·e_i) − loss(θ)) / ε.
      // The FD path needs only the scalar loss delta, so it uses the
      // allocation-free `lossValue` (dLoss_dRendered is the Phase-1 input).
      void dLoss_dRendered;
      grad = new Float32Array(this.#flat.length);
      const eps = this.#fdEpsilon;
      for (let i = 0; i < this.#flat.length; i++) {
        const original = this.#flat[i]!;
        this.#flat[i] = original + eps;
        this.#applyFlatToScene();
        const probe = await this.#hooks.renderAndReadback(width, height, this.#samplesPerStep);
        const probeLoss = lossValue(probe.rgb, probe.channels, this.#target, this.#lossKind);
        grad[i] = (probeLoss - loss) / eps;
        this.#flat[i] = original; // restore before the next probe
      }
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

function collectPathReplayDiagnostics(
  scene: Scene,
  slots: readonly ParamSlot[],
  options: {
    readonly hasHook: boolean;
    readonly iridescenceOptimizedPrimitiveIds: ReadonlySet<string>;
    readonly renderContext: InversePathReplayRenderContext;
  },
): InverseSessionDiagnostic[] {
  const diagnostics: InverseSessionDiagnostic[] = [];
  if (!options.hasHook) {
    diagnostics.push({
      severity: 'info',
      code: 'path-replay-hook-missing',
      message:
        '[vitrum/pt-webgpu] InverseSession requested path-replay, but this engine instance ' +
        'does not expose the adjoint gradient hook; using finite-difference.',
    });
  }

  const renderRegimeIssue = pathReplayRenderRegimeIssue(options.renderContext);
  if (renderRegimeIssue != null) {
    diagnostics.push({
      severity: 'info',
      code: 'path-replay-unsupported-render-regime',
      message:
        '[vitrum/pt-webgpu] InverseSession requested path-replay, but the most recent ' +
        `${renderRegimeIssue.message}; using finite-difference.`,
      details: renderRegimeIssue.details,
    });
  }

  for (const slot of slots) {
    diagnostics.push(...diagnosePathReplaySlot(scene, slot, options.iridescenceOptimizedPrimitiveIds));
  }
  return diagnostics;
}

function pathReplayRenderRegimeIssue(
  context: InversePathReplayRenderContext,
): { readonly message: string; readonly details: Record<string, string | number | boolean> } | null {
  if (context.spectral === true) {
    return {
      message: 'forward baseline used spectral transport that the scoped RGB direct-light adjoint does not mirror',
      details: { spectral: true },
    };
  }
  if (typeof context.bounces === 'number' && Number.isFinite(context.bounces) && context.bounces > 1) {
    return {
      message: `forward baseline used ${context.bounces} bounces while the adjoint pass is single-bounce direct-light only`,
      details: { bounces: context.bounces, supportedBounces: 1 },
    };
  }
  if (context.bdpt === true) {
    return {
      message: 'forward baseline used BDPT contributions that the scoped path-replay adjoint does not mirror',
      details: { bdpt: true, unsupportedFeature: 'bdpt' },
    };
  }
  if (context.restirPtReuse === true) {
    return {
      message: 'forward baseline used ReSTIR-PT reuse contributions that the scoped path-replay adjoint does not mirror',
      details: { restirPtReuse: true, unsupportedFeature: 'restir-pt-reuse' },
    };
  }
  if (context.causticStrategy != null && context.causticStrategy !== 'none') {
    return {
      message: `forward baseline used caustic strategy "${context.causticStrategy}" that the scoped path-replay adjoint does not mirror`,
      details: { causticStrategy: context.causticStrategy, unsupportedFeature: 'caustic-strategy' },
    };
  }
  return null;
}

function diagnosePathReplaySlot(
  scene: Scene,
  slot: ParamSlot,
  iridescenceOptimizedPrimitiveIds: ReadonlySet<string>,
): InverseSessionDiagnostic[] {
  const path = slot.param.path;
  const target = slot.target;
  if (target.domain !== 'materials') {
    return diagnosePathReplayEmitterSlot(scene, slot);
  }
  if (!ADJOINT_ELIGIBLE_FIELDS.has(target.field)) {
    const finiteDifferenceOnlyIssue = pathReplayFiniteDifferenceOnlyFieldIssue(target.field);
    if (finiteDifferenceOnlyIssue != null) {
      return [{
        severity: 'info',
        code: finiteDifferenceOnlyIssue.code,
        path,
        message:
          `[vitrum/pt-webgpu] InverseSession path "${path}" targets material field ` +
          `"${target.field}", ${finiteDifferenceOnlyIssue.message}; using finite-difference.`,
        details: finiteDifferenceOnlyIssue.details,
      }];
    }
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-field',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets material field ` +
        `"${target.field}", which is not in the path-replay differentiable field set; ` +
        'using finite-difference.',
      details: { field: target.field },
    }];
  }

  const prim = findPrimitive(scene, target.id);
  if (prim == null) return [];
  const primitiveIssue = pathReplayPrimitiveIssue(prim);
  if (primitiveIssue != null) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-primitive',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets primitive "${target.id}", ` +
        `${primitiveIssue.message}; using finite-difference.`,
      details: primitiveIssue.details,
    }];
  }

  const materialIssue = pathReplayMaterialIssue(prim.material, target.field, iridescenceOptimizedPrimitiveIds.has(target.id));
  if (materialIssue != null) {
    return [{
      severity: 'info',
      code: materialIssue.code ?? 'path-replay-unsupported-material',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" is outside the scoped path-replay ` +
        `material domain (${materialIssue.message}); using finite-difference.`,
      details: materialIssue.details,
    }];
  }

  if (pathReplayTargetRequiresLighting(target.field, prim.material)) {
    const lightingIssue = pathReplayLightingIssue(scene);
    if (lightingIssue != null) {
      return [{
        severity: 'info',
        code: 'path-replay-unsupported-lighting',
        path,
        message:
          `[vitrum/pt-webgpu] InverseSession path "${path}" needs the direct-light replay domain, ` +
          `${lightingIssue.message}; using finite-difference.`,
        details: lightingIssue.details,
      }];
    }
  }
  return [];
}

function pathReplayFiniteDifferenceOnlyFieldIssue(
  field: string,
): {
  readonly code:
    | 'path-replay-unsupported-transport'
    | 'path-replay-unsupported-visibility'
    | 'path-replay-unsupported-normal';
  readonly message: string;
  readonly details: Record<string, string | readonly string[]>;
} | null {
  if (PATH_REPLAY_TRANSPORT_ONLY_FIELDS.has(field)) {
    return {
      code: 'path-replay-unsupported-transport',
      message:
        'which changes transmissive/medium transport that the scoped path-replay adjoint does not mirror yet',
      details: {
        field,
        finiteDifferenceReason: 'transport',
        affectedTerms: ['fresnel-partition', 'refraction-direction', 'medium-attenuation'],
      },
    };
  }
  if (PATH_REPLAY_VISIBILITY_ONLY_FIELDS.has(field)) {
    return {
      code: 'path-replay-unsupported-visibility',
      message:
        'which changes alpha coverage and visibility discontinuities that the scoped path-replay adjoint does not mirror yet',
      details: {
        field,
        finiteDifferenceReason: 'visibility',
        affectedTerms: ['alpha-coverage', 'ray-visibility', 'shadow-visibility'],
      },
    };
  }
  if (PATH_REPLAY_NORMAL_ONLY_FIELDS.has(field)) {
    return {
      code: 'path-replay-unsupported-normal',
      message:
        'which changes shading normals that the scoped path-replay adjoint does not mirror yet',
      details: {
        field,
        finiteDifferenceReason: 'normal',
        affectedTerms: ['normal-map-frame', 'bump-gradient', 'clearcoat-normal-frame'],
      },
    };
  }
  return null;
}

function diagnosePathReplayEmitterSlot(
  scene: Scene,
  slot: ParamSlot,
): InverseSessionDiagnostic[] {
  const path = slot.param.path;
  const target = slot.target;
  if (!ADJOINT_ELIGIBLE_EMITTER_FIELDS.has(target.field)) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-field',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets emitter field ` +
        `"${target.field}", which is not in the path-replay differentiable emitter field set; ` +
        'using finite-difference.',
      details: { field: target.field },
    }];
  }

  const emitter = scene.emitters.find((e) => e.id === target.id);
  if (emitter == null) return [];
  const emitterIssue = pathReplayEmitterTargetIssue(scene, emitter, target.field);
  if (emitterIssue != null) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-emitter',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" targets emitter "${target.id}", ` +
        `${emitterIssue.message}; using finite-difference.`,
      details: emitterIssue.details,
    }];
  }

  const lightingIssue = pathReplayLightingIssue(scene);
  if (lightingIssue != null) {
    return [{
      severity: 'info',
      code: 'path-replay-unsupported-lighting',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" needs the direct-light replay domain, ` +
        `${lightingIssue.message}; using finite-difference.`,
      details: lightingIssue.details,
    }];
  }

  const receiverIssue = pathReplayEmitterReceiverSceneIssue(scene);
  if (receiverIssue != null) {
    return [{
      severity: 'info',
      code: receiverIssue.code ?? 'path-replay-unsupported-receiver',
      path,
      message:
        `[vitrum/pt-webgpu] InverseSession path "${path}" differentiates an emitter through ` +
        `scene receivers, but ${receiverIssue.message}; using finite-difference.`,
      details: receiverIssue.details,
    }];
  }
  return [];
}

function pathReplayPrimitiveIssue(
  primitive: ScenePrimitive,
): { message: string; details: Record<string, string | boolean> } | null {
  if (primitive.kind !== 'mesh' && primitive.kind !== 'skinned-mesh') {
    return {
      message: `primitive kind "${primitive.kind}" is not triangle-backed for path-replay`,
      details: { primitiveKind: primitive.kind },
    };
  }
  if (primitive.transform != null && !isIdentityMat4(primitive.transform)) {
    return {
      message: 'non-identity primitive transforms are not mirrored by the adjoint replay pass',
      details: { primitiveKind: primitive.kind, nonIdentityTransform: true },
    };
  }
  return null;
}

function pathReplayEmitterTargetIssue(
  scene: Scene,
  emitter: SceneEmitter,
  field: string,
): { message: string; details: Record<string, string | number | readonly string[]> } | null {
  switch (emitter.kind) {
    case 'directional': {
      return null;
    }
    case 'point':
    case 'spot':
    case 'rect-area':
    case 'disc-area':
      return null;
    case 'mesh-area': {
      const range = meshAreaEmitterAdjointRangeForScene(scene, emitter.id);
      if (range == null) {
        return {
          message: 'mesh-area emitter target produces no contiguous packed triangle range',
          details: { emitterKind: emitter.kind },
        };
      }
      if (range.capped) {
        return {
          message:
            'mesh-area emitter target is in a globally capped/reordered triangle stream and needs source-triangle PDF mapping first',
          details: {
            emitterKind: emitter.kind,
            meshAreaTriangleCount: range.totalMeshAreaTriangles,
          },
        };
      }
      const mappedEmissionIssue = meshAreaEmitterMappedEmissionIssue(scene, emitter, field);
      if (mappedEmissionIssue != null) return mappedEmissionIssue;
      return null;
    }
    default: {
      const emitterKind = (emitter as { readonly kind: string }).kind;
      return {
        message: `emitter kind "${emitterKind}" is outside the path-replay target domain`,
        details: { emitterKind },
      };
    }
  }
}

function meshAreaEmitterMappedEmissionIssue(
  scene: Scene,
  emitter: Extract<SceneEmitter, { readonly kind: 'mesh-area' }>,
  field: string,
): { message: string; details: Record<string, string | number | readonly string[]> } | null {
  const primitive = scene.primitives.find((p) => p.id === emitter.meshId);
  if (primitive == null || primitive.kind === 'analytic') return null;
  if (primitive.material.emissiveMap == null) return null;
  if (field === 'intensity') {
    if (Number.isFinite(emitter.intensity) && Math.abs(emitter.intensity) > ADJOINT_MAPPED_EMISSION_EPS) {
      return null;
    }
    return {
      message:
        'mesh-area emitter target uses material emissiveMap radiance with zero/near-zero intensity; packed-radiance chain-rule ratio is undefined',
      details: {
        emitterKind: emitter.kind,
        meshId: emitter.meshId,
        unsupportedMaterialFields: ['emissiveMap'],
        finiteDifferenceReason: 'zero-intensity-mapped-emission',
      },
    };
  }
  if (field === 'color') {
    const hasZeroChannel = emitter.color.some((v) =>
      !Number.isFinite(v) || Math.abs(v) <= ADJOINT_MAPPED_EMISSION_EPS,
    );
    if (!hasZeroChannel) return null;
    return {
      message:
        'mesh-area emitter target uses material emissiveMap radiance with a zero/near-zero color channel; packed-radiance chain-rule ratio is undefined',
      details: {
        emitterKind: emitter.kind,
        meshId: emitter.meshId,
        unsupportedMaterialFields: ['emissiveMap'],
        finiteDifferenceReason: 'zero-color-channel-mapped-emission',
      },
    };
  }
  return {
    message:
      'mesh-area emitter target uses material emissiveMap radiance outside the scoped emitter color/intensity replay fields',
    details: {
      emitterKind: emitter.kind,
      meshId: emitter.meshId,
      unsupportedMaterialFields: ['emissiveMap'],
    },
  };
}

function pathReplayEmitterReceiverSceneIssue(
  scene: Scene,
): PathReplayMaterialIssue | null {
  for (const primitive of scene.primitives) {
    const primitiveIssue = pathReplayPrimitiveIssue(primitive);
    if (primitiveIssue != null) {
      return {
        message: `receiver primitive "${primitive.id}" ${primitiveIssue.message}`,
        details: { primitiveId: primitive.id, ...primitiveIssue.details },
      };
    }
    const materialIssue = pathReplayEmitterReceiverMaterialIssue(primitive.material);
    if (materialIssue != null) {
      return {
        ...(materialIssue.code != null ? { code: materialIssue.code } : {}),
        message: `receiver primitive "${primitive.id}" has material outside the scoped direct-light replay domain (${materialIssue.message})`,
        details: { primitiveId: primitive.id, ...materialIssue.details },
      };
    }
  }
  return null;
}

function pathReplayEmitterReceiverMaterialIssue(
  material: MaterialSpec,
): PathReplayMaterialIssue | null {
  const common = materialIssueCommon(material, { allowIridescence: false, allowAnisotropy: true });
  if (common != null) return common;
  const maps = listPathReplayTransportOrGeometryMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function pathReplayMaterialIssue(
  material: MaterialSpec,
  field: string,
  iridescenceCoupled: boolean,
): PathReplayMaterialIssue | null {
  if (field === 'baseColor' && isPathReplayCompatibleUnlitBaseColorMaterial(material)) return null;
  if (field === 'emissive' || field === 'emissiveIntensity') {
    return materialIssueForEmissive(material);
  }
  if (field === 'aoMapIntensity') {
    return materialIssueForAoMapIntensity(material);
  }
  if (field === 'lightMapIntensity') {
    return materialIssueForLightMapIntensity(material);
  }
  if (field === 'iridescence' || field === 'iridescenceIor' || field === 'iridescenceThicknessRange') {
    return materialIssueForIridescence(material);
  }
  if (field === 'anisotropy' || field === 'anisotropyRotation') {
    return materialIssueForAnisotropy(material);
  }
  if (iridescenceCoupled) {
    return {
      message: 'another optimized parameter on this material targets iridescence, which is coupled to this BRDF field',
      details: { reason: 'coupled-iridescence-parameter' },
    };
  }
  return materialIssueForBrdf(material);
}

function materialIssueForEmissive(
  material: MaterialSpec,
): PathReplayMaterialIssue | null {
  const common = materialIssueCommon(material, { allowIridescence: true, allowAnisotropy: true });
  if (common != null) return common;
  const maps = listPathReplayPrimaryEmissionUnsupportedMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function materialIssueForBrdf(
  material: MaterialSpec,
): PathReplayMaterialIssue | null {
  if (material.shadingModel === 'unlit') {
    return { message: 'unlit materials only support path-replay for baseColor primary-hit fitting', details: { reason: 'unlit' } };
  }
  const common = materialIssueCommon(material, { allowIridescence: false, allowAnisotropy: false });
  if (common != null) return common;
  const maps = listPathReplayTransportOrGeometryMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function materialIssueForAoMapIntensity(
  material: MaterialSpec,
): PathReplayMaterialIssue | null {
  const common = materialIssueCommon(material, {
    allowIridescence: material.shadingModel === 'unlit',
    allowAnisotropy: material.shadingModel === 'unlit',
  });
  if (common != null) return common;
  const maps = listPathReplayTransportOrGeometryMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function materialIssueForLightMapIntensity(
  material: MaterialSpec,
): PathReplayMaterialIssue | null {
  const common = materialIssueCommon(material, { allowIridescence: true, allowAnisotropy: true });
  if (common != null) return common;
  const maps = listPathReplayPrimaryEmissionUnsupportedMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function materialIssueForIridescence(
  material: MaterialSpec,
): PathReplayMaterialIssue | null {
  if (material.shadingModel === 'unlit') {
    return { message: 'unlit materials do not evaluate the iridescence direct-light lobe', details: { reason: 'unlit' } };
  }
  const common = materialIssueCommon(material, { allowIridescence: true, allowAnisotropy: false });
  if (common != null) return common;
  const maps = listPathReplayTransportOrGeometryMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function materialIssueForAnisotropy(
  material: MaterialSpec,
): PathReplayMaterialIssue | null {
  if (material.shadingModel === 'unlit') {
    return { message: 'unlit materials do not evaluate the anisotropic direct-light lobe', details: { reason: 'unlit' } };
  }
  const common = materialIssueCommon(material, { allowIridescence: false, allowAnisotropy: true });
  if (common != null) return common;
  const maps = listPathReplayTransportOrGeometryMaps(material);
  if (maps.length > 0) {
    return materialMapIssue(maps);
  }
  return null;
}

function materialIssueCommon(
  material: MaterialSpec,
  options: { readonly allowIridescence: boolean; readonly allowAnisotropy: boolean },
): PathReplayMaterialIssue | null {
  if (material.alphaMode != null && material.alphaMode !== 'opaque') {
    return {
      code: 'path-replay-unsupported-visibility',
      message: `alphaMode "${material.alphaMode}" changes visibility/coverage`,
      details: {
        field: 'alphaMode',
        value: material.alphaMode,
        finiteDifferenceReason: 'visibility',
        affectedTerms: ['alpha-coverage', 'ray-visibility', 'shadow-visibility'],
      },
    };
  }
  if (material.opacity != null && material.opacity < 1) {
    return {
      code: 'path-replay-unsupported-visibility',
      message: 'opacity below 1 changes visibility/coverage',
      details: {
        field: 'opacity',
        value: material.opacity,
        finiteDifferenceReason: 'visibility',
        affectedTerms: ['alpha-coverage', 'ray-visibility', 'shadow-visibility'],
      },
    };
  }
  if ((material.transmission ?? 0) > 1e-6) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'transmission transport is not replayed',
      details: {
        field: 'transmission',
        value: material.transmission ?? 0,
        finiteDifferenceReason: 'transport',
        affectedTerms: ['fresnel-partition', 'refraction-direction', 'medium-attenuation'],
      },
    };
  }
  if (!options.allowIridescence && (material.iridescence ?? 0) > 1e-6) {
    return { message: 'iridescence is coupled to the optimized BRDF field', details: { field: 'iridescence', value: material.iridescence ?? 0 } };
  }
  if (!options.allowAnisotropy && (material.anisotropy ?? 0) > 1e-6) {
    return { message: 'anisotropy is coupled to the optimized BRDF field', details: { field: 'anisotropy', value: material.anisotropy ?? 0 } };
  }
  if (material.frontLayer != null || material.backLayer != null || material.thinFilmStack != null) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'layered/thin-film material stacks are not replayed',
      details: {
        field: 'layeredMaterial',
        finiteDifferenceReason: 'transport',
        affectedTerms: ['layer-selection', 'thin-film-phase', 'transmission-mis'],
      },
    };
  }
  if (material.spectralAttenuation != null || material.dispersionAbbeNumber != null) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'spectral/dispersion material transport is not replayed',
      details: {
        field: 'spectralOrDispersion',
        finiteDifferenceReason: 'transport',
        affectedTerms: ['hero-wavelength', 'spectral-attenuation', 'dispersion-ior'],
      },
    };
  }
  if ((material.scatteringCoefficient ?? 0) > 0 || material.scatteringCoefficientRGB != null) {
    return {
      code: 'path-replay-unsupported-transport',
      message: 'volume/scattering material transport is not replayed',
      details: {
        field: 'scattering',
        finiteDifferenceReason: 'transport',
        affectedTerms: ['volume-walk', 'medium-scattering', 'phase-function'],
      },
    };
  }
  if (material.extensions != null && Object.keys(material.extensions).length > 0) {
    return { message: 'opaque MaterialSpec.extensions are not replayed by the adjoint pass', details: { field: 'extensions' } };
  }
  return null;
}

function materialMapIssue(maps: readonly string[]): PathReplayMaterialIssue {
  const categories = new Set(maps.map(pathReplayMaterialMapCategory));
  const details: Record<string, string | readonly string[]> = { unsupportedMaterialFields: maps };
  if (categories.size === 1) {
    const category = categories.values().next().value as PathReplayMaterialMapCategory;
    switch (category) {
      case 'transport':
        details.finiteDifferenceReason = 'transport';
        details.affectedTerms = ['fresnel-partition', 'refraction-direction', 'medium-attenuation'];
        return {
          code: 'path-replay-unsupported-transport',
          message: `transport maps are not replayed: ${maps.join(', ')}`,
          details,
        };
      case 'visibility':
        details.finiteDifferenceReason = 'visibility';
        details.affectedTerms = ['alpha-coverage', 'ray-visibility', 'shadow-visibility'];
        return {
          code: 'path-replay-unsupported-visibility',
          message: `visibility maps are not replayed: ${maps.join(', ')}`,
          details,
        };
      case 'normal':
        details.finiteDifferenceReason = 'normal';
        details.affectedTerms = ['normal-map-frame', 'bump-gradient', 'clearcoat-normal-frame'];
        return {
          code: 'path-replay-unsupported-normal',
          message: `normal maps are not replayed: ${maps.join(', ')}`,
          details,
        };
      case 'geometry':
        details.finiteDifferenceReason = 'geometry';
        details.affectedTerms = ['micro-displacement', 'bvh-geometry', 'visibility'];
        return {
          code: 'path-replay-unsupported-material',
          message: `geometry maps are not replayed: ${maps.join(', ')}`,
          details,
        };
    }
  }
  details.finiteDifferenceReason = 'mixed-material-domain';
  return {
    code: 'path-replay-unsupported-material',
    message: `mixed transport/visibility/normal/geometry maps are not replayed: ${maps.join(', ')}`,
    details,
  };
}

type PathReplayMaterialMapCategory = 'transport' | 'visibility' | 'normal' | 'geometry';

function pathReplayMaterialMapCategory(field: string): PathReplayMaterialMapCategory {
  switch (field) {
    case 'transmissionMap':
    case 'thicknessMap':
      return 'transport';
    case 'alphaMap':
      return 'visibility';
    case 'normalMap':
    case 'bumpMap':
    case 'clearcoatNormalMap':
      return 'normal';
    case 'displacementMap':
    default:
      return 'geometry';
  }
}

function pathReplayTargetRequiresLighting(field: string, material: MaterialSpec): boolean {
  if (field === 'emissive' || field === 'emissiveIntensity') return false;
  if (field === 'lightMapIntensity') return false;
  if (field === 'baseColor' && isPathReplayCompatibleUnlitBaseColorMaterial(material)) return false;
  if (field === 'aoMapIntensity' && material.shadingModel === 'unlit') return false;
  return true;
}

function pathReplayLightingIssue(
  scene: Scene,
): { message: string; details: Record<string, string | number | readonly string[]> } | null {
  const unsupported = (scene.emitters as unknown as ReadonlyArray<{
    readonly id: string;
    readonly kind: string;
    readonly angularDiameter?: number;
  }>).find((e) =>
    e.kind !== 'directional' &&
    e.kind !== 'point' &&
    e.kind !== 'spot' &&
    e.kind !== 'rect-area' &&
    e.kind !== 'disc-area' &&
    e.kind !== 'mesh-area'
  );
  if (unsupported == null) return null;
  return {
    message: `emitter "${unsupported.id}" (${unsupported.kind}) is outside the deterministic direct-light replay domain`,
    details: {
      emitterId: unsupported.id,
      emitterKind: unsupported.kind,
      ...(unsupported.kind === 'directional' && unsupported.angularDiameter != null
        ? { angularDiameter: unsupported.angularDiameter }
        : {}),
    },
  };
}

function isIdentityMat4(transform: Float32Array): boolean {
  const expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  if (transform.length < 16) return false;
  for (let i = 0; i < 16; i += 1) {
    if (Math.abs((transform[i] ?? 0) - expected[i]!) > 1e-6) return false;
  }
  return true;
}

function isPathReplayCompatibleUnlitBaseColorMaterial(m: MaterialSpec): boolean {
  if (m.shadingModel !== 'unlit') return false;
  if (m.alphaMode != null && m.alphaMode !== 'opaque') return false;
  if (m.opacity != null && m.opacity < 1) return false;
  if ((m.transmission ?? 0) > 1e-6) return false;
  if (m.frontLayer != null || m.backLayer != null || m.thinFilmStack != null) return false;
  if (m.spectralAttenuation != null || m.dispersionAbbeNumber != null) return false;
  if ((m.scatteringCoefficient ?? 0) > 0 || (m.scatteringCoefficientRGB != null)) return false;
  if (m.extensions != null && Object.keys(m.extensions).length > 0) return false;
  return !hasPathReplayTransportOrGeometryMap(m);
}

function hasPathReplayTransportOrGeometryMap(m: MaterialSpec): boolean {
  return listPathReplayTransportOrGeometryMaps(m).length > 0;
}

function listPathReplayTransportOrGeometryMaps(m: MaterialSpec): readonly string[] {
  const out: string[] = [];
  if (m.normalMap != null) out.push('normalMap');
  if (m.transmissionMap != null) out.push('transmissionMap');
  if (m.thicknessMap != null) out.push('thicknessMap');
  if (m.alphaMap != null) out.push('alphaMap');
  if (m.clearcoatNormalMap != null) out.push('clearcoatNormalMap');
  if (m.bumpMap != null) out.push('bumpMap');
  if (m.displacementMap != null) out.push('displacementMap');
  return out;
}

function listPathReplayPrimaryEmissionUnsupportedMaps(m: MaterialSpec): readonly string[] {
  const out: string[] = [];
  if (m.alphaMap != null) out.push('alphaMap');
  if (m.displacementMap != null) out.push('displacementMap');
  if (m.transmissionMap != null) out.push('transmissionMap');
  if (m.thicknessMap != null) out.push('thicknessMap');
  return out;
}

// ── path resolution / field validation ────────────────────────────────────────

function validateParam(scene: Scene, param: InverseParam, target: ResolvedParamTarget): void {
  if (param.kind === 'texture') {
    throw new Error(
      `createInverseSession: parameter kind 'texture' (path "${param.path}") is reserved ` +
        'for Phase 2 (texture optimization) and is not yet differentiable in pt-webgpu.',
    );
  }
  if (target.domain === 'materials') {
    const prim = findPrimitive(scene, target.id);
    if (prim == null) {
      throw new Error(
        `createInverseSession: no primitive with id "${target.id}" for path "${param.path}".`,
      );
    }
    const isRgb = MATERIAL_RGB_FIELDS.has(target.field);
    const isVec2 = MATERIAL_VEC2_FIELDS.has(target.field);
    const isScalar = MATERIAL_SCALAR_FIELDS.has(target.field);
    if (!isRgb && !isVec2 && !isScalar) {
      throw new Error(
        `createInverseSession: material field "${target.field}" (path "${param.path}") is not ` +
          `optimizable. Supported: ${[
            ...MATERIAL_RGB_FIELDS,
            ...MATERIAL_VEC2_FIELDS,
            ...MATERIAL_SCALAR_FIELDS,
          ].join(', ')}.`,
      );
    }
    assertKind(param, isRgb ? 'rgb' : isVec2 ? 'vec2' : 'scalar');
  } else {
    const emitter = scene.emitters.find((e) => e.id === target.id);
    if (emitter == null) {
      throw new Error(
        `createInverseSession: no emitter with id "${target.id}" for path "${param.path}".`,
      );
    }
    const isRgb = EMITTER_RGB_FIELDS.has(target.field);
    const isScalar = EMITTER_SCALAR_FIELDS.has(target.field);
    if (!isRgb && !isScalar) {
      throw new Error(
        `createInverseSession: emitter field "${target.field}" (path "${param.path}") is not ` +
          `optimizable. Supported: ${[...EMITTER_RGB_FIELDS, ...EMITTER_SCALAR_FIELDS].join(', ')}.`,
      );
    }
    assertKind(param, isRgb ? 'rgb' : 'scalar');
  }
}

/** Field-aware default [min, max] clamp range, used when a parameter doesn't
 *  supply its own `min`/`max`. baseColor / roughness / metallic / emissive
 *  saturate at [0, 1] (physical reflectance / microfacet range);
 *  attenuationColor uses the finite material-packer clamp [1e-4, 1]; emissive
 *  intensity and emitter intensity / color are non-negative but unbounded above
 *  (an explicit `max` from the host narrows them); `ior` is bounded to the
 *  dielectric range [1, 2.5] the material decoder clamps to (material.wgsl.ts:615
 *  `clamp(m2.y, 1.0, 2.5)`) — optimizing outside it would hit a flat clamp and
 *  stall. */
function defaultClampRange(field: string): [number, number] {
  switch (field) {
    case 'baseColor':
    case 'specularColor':
    case 'sheenColor':
    case 'roughness':
    case 'metallic':
    case 'opacity':
    case 'alphaCutoff':
    case 'transmission':
    case 'specularIntensity':
    case 'clearcoat':
    case 'clearcoatRoughness':
    case 'sheen':
    case 'sheenRoughness':
    case 'iridescence':
    case 'anisotropy':
    case 'aoMapIntensity':
      return [0, 1];
    case 'attenuationColor':
      return [1e-4, 1];
    case 'ior':
      return [1, 2.5];
    case 'attenuationDistance':
      return [1e-6, Infinity];
    case 'dispersionAbbeNumber':
      return [0, Infinity];
    case 'scatteringCoefficient':
      return [0, Infinity];
    case 'scatteringAnisotropy':
      return [-0.95, 0.95];
    case 'iridescenceIor':
      return [1, 3];
    case 'iridescenceThicknessRange':
      return [0, Infinity];
    case 'anisotropyRotation':
      return [-Infinity, Infinity];
    case 'emissive':
    case 'emissiveIntensity':
    case 'thickness':
    case 'normalScale':
    case 'bumpScale':
    case 'clearcoatNormalScale':
    case 'lightMapIntensity':
    case 'envMapIntensity':
    case 'color':
    case 'intensity':
      return [0, Infinity];
    default:
      return [0, Infinity];
  }
}

function assertKind(param: InverseParam, expected: 'rgb' | 'vec2' | 'scalar'): void {
  if (param.kind !== expected) {
    throw new Error(
      `createInverseSession: parameter "${param.path}" is declared kind '${param.kind}' ` +
        `but the resolved field is '${expected}'.`,
    );
  }
}

function findPrimitive(scene: Scene, id: string): ScenePrimitive | null {
  return scene.primitives.find((p) => p.id === id) ?? null;
}

function readSceneValue(scene: Scene, target: ResolvedParamTarget, length: number): number[] {
  if (target.domain === 'materials') {
    const prim = findPrimitive(scene, target.id)!;
    const m = prim.material;
    switch (target.field) {
      case 'baseColor': return [...m.baseColor];
      case 'roughness': return [m.roughness];
      case 'metallic': return [m.metallic];
      case 'emissive': return [...(m.emissive ?? [0, 0, 0])];
      case 'emissiveIntensity': return [m.emissiveIntensity ?? 1];
      case 'opacity': return [m.opacity ?? 1];
      case 'alphaCutoff': return [m.alphaCutoff ?? 0.5];
      case 'ior': return [m.ior ?? 1.5];
      case 'transmission': return [m.transmission ?? 0];
      case 'thickness': return [m.thickness ?? 0];
      case 'attenuationColor': return [...(m.attenuationColor ?? [1, 1, 1])];
      case 'attenuationDistance': return [m.attenuationDistance ?? 1];
      case 'dispersionAbbeNumber': return [m.dispersionAbbeNumber ?? 0];
      case 'scatteringCoefficient': return [m.scatteringCoefficient ?? 0];
      case 'scatteringAnisotropy': return [m.scatteringAnisotropy ?? 0];
      case 'scatteringCoefficientRGB': return [...(m.scatteringCoefficientRGB ?? [0, 0, 0])];
      case 'specularColor': return [...(m.specularColor ?? [1, 1, 1])];
      case 'specularIntensity': return [m.specularIntensity ?? 1];
      case 'clearcoat': return [m.clearcoat ?? 0];
      case 'clearcoatRoughness': return [m.clearcoatRoughness ?? 0];
      case 'sheen': return [m.sheen ?? 0];
      case 'sheenColor': return [...(m.sheenColor ?? [1, 1, 1])];
      case 'sheenRoughness': return [m.sheenRoughness ?? 0];
      case 'iridescence': return [m.iridescence ?? 0];
      case 'iridescenceIor': return [m.iridescenceIor ?? 1.3];
      case 'iridescenceThicknessRange': return [...(m.iridescenceThicknessRange ?? [100, 400])];
      case 'anisotropy': return [m.anisotropy ?? 0];
      case 'anisotropyRotation': return [m.anisotropyRotation ?? 0];
      case 'normalScale': return [m.normalScale ?? 1];
      case 'bumpScale': return [m.bumpScale ?? 1];
      case 'clearcoatNormalScale': return [m.clearcoatNormalScale ?? 1];
      case 'aoMapIntensity': return [m.aoMapIntensity ?? 1];
      case 'lightMapIntensity': return [m.lightMapIntensity ?? 1];
      case 'envMapIntensity': return [m.envMapIntensity ?? 1];
      default: break;
    }
  } else {
    const e = scene.emitters.find((em) => em.id === target.id)!;
    switch (target.field) {
      case 'color': return [...e.color];
      case 'intensity': return [e.intensity];
      default: break;
    }
  }
  // unreachable — validateParam already rejected unknown fields
  return new Array<number>(length).fill(0);
}

function materialPatch(field: string, value: number[]): Partial<MaterialSpec> {
  switch (field) {
    case 'baseColor': return { baseColor: value as unknown as Vec3 };
    case 'roughness': return { roughness: value[0]! };
    case 'metallic': return { metallic: value[0]! };
    case 'emissive': return { emissive: value as unknown as Vec3 };
    case 'emissiveIntensity': return { emissiveIntensity: value[0]! };
    case 'opacity': return { opacity: value[0]! };
    case 'alphaCutoff': return { alphaCutoff: value[0]! };
    case 'ior': return { ior: value[0]! };
    case 'transmission': return { transmission: value[0]! };
    case 'thickness': return { thickness: value[0]! };
    case 'attenuationColor': return { attenuationColor: value as unknown as Vec3 };
    case 'attenuationDistance': return { attenuationDistance: value[0]! };
    case 'dispersionAbbeNumber': return { dispersionAbbeNumber: value[0]! };
    case 'scatteringCoefficient': return { scatteringCoefficient: value[0]! };
    case 'scatteringAnisotropy': return { scatteringAnisotropy: value[0]! };
    case 'scatteringCoefficientRGB': return { scatteringCoefficientRGB: value as unknown as Vec3 };
    case 'specularColor': return { specularColor: value as unknown as Vec3 };
    case 'specularIntensity': return { specularIntensity: value[0]! };
    case 'clearcoat': return { clearcoat: value[0]! };
    case 'clearcoatRoughness': return { clearcoatRoughness: value[0]! };
    case 'sheen': return { sheen: value[0]! };
    case 'sheenColor': return { sheenColor: value as unknown as Vec3 };
    case 'sheenRoughness': return { sheenRoughness: value[0]! };
    case 'iridescence': return { iridescence: value[0]! };
    case 'iridescenceIor': return { iridescenceIor: value[0]! };
    case 'iridescenceThicknessRange':
      return {
        iridescenceThicknessRange: [
          Math.max(value[0] ?? 100, 0),
          Math.max(value[1] ?? 400, 0),
        ] as unknown as Vec2,
      };
    case 'anisotropy': return { anisotropy: value[0]! };
    case 'anisotropyRotation': return { anisotropyRotation: value[0]! };
    case 'normalScale': return { normalScale: value[0]! };
    case 'bumpScale': return { bumpScale: value[0]! };
    case 'clearcoatNormalScale': return { clearcoatNormalScale: value[0]! };
    case 'aoMapIntensity': return { aoMapIntensity: value[0]! };
    case 'lightMapIntensity': return { lightMapIntensity: value[0]! };
    case 'envMapIntensity': return { envMapIntensity: value[0]! };
    default: throw new Error(`inverse: unsupported material field "${field}".`);
  }
}

function emitterPatch(field: string, value: number[]): Partial<SceneEmitter> {
  switch (field) {
    case 'color': return { color: value as unknown as Vec3 };
    case 'intensity': return { intensity: value[0]! };
    default: throw new Error(`inverse: unsupported emitter field "${field}".`);
  }
}
