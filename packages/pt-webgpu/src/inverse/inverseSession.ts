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
 * material stays within the direct-light base/specular domain this pass mirrors
 * (delta directional, point, spot, and center-sampled rect/disc area lights), or
 * is a map-free `shadingModel:'unlit'` baseColor primary-hit fit;
 * `ADJOINT_ELIGIBLE_FIELDS`: material baseColor / roughness / metallic /
 * emissive / specularColor / specularIntensity / clearcoat / map-free sheen /
 * map-free iridescence / map-free anisotropy controls); any
 * shortfall (no hook, an emitter param, an `ior` param, etc.) resolves the
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
  Scene,
  ScenePrimitive,
  SceneEmitter,
  MaterialSpec,
} from '@vitrum/core';
import type { Vec3 } from '@vitrum/core';
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
   * OPTIONAL Phase-1 path-replay adjoint. When present, the engine dispatches a
   * single-bounce adjoint compute pass over its scene buffers — re-tracing the
   * frozen-seed primary path + NEE, evaluating the analytic BSDF partials
   * (`pathTraceAdjoint.wgsl`, GPU-validated vs the FD oracle to f32 precision),
   * and accumulating `∂loss/∂θ` (the chain rule GPU-validated vs on-device FD) —
   * and returns the flat gradient. Replaces the N-render FD probe loop with one
   * baseline render + one adjoint pass. The session only requests this when the
   * hook exists, every parameter is adjoint-eligible, and every target material
   * stays inside the adjoint-compatible direct-light domain (delta directional,
   * point, spot, and center-sampled rect/disc/mesh area lights;
   * `ADJOINT_ELIGIBLE_FIELDS`: material baseColor / roughness / metallic /
   * emissive / specularColor / specularIntensity / clearcoat / map-free sheen /
   * map-free iridescence / map-free anisotropy controls); otherwise it reports + uses
   * 'finite-difference' (no silently-wrong gradient). An engine that provides
   * this hook is vouching that its adjoint pass is hardware-validated — a field
   * only graduates to path-replay once its end-to-end inverse fit converges.
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
  /** Component count (3 for rgb, 1 for scalar). */
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
 *    `baseColor` also covers map-free `shadingModel:'unlit'` primary hits:
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
 *  - `clearcoat`, `clearcoatRoughness` — map-free KHR_materials_clearcoat direct
 *    lobe controls. Clearcoat maps and clearcoatNormalMap stay on finite
 *    difference until the adjoint pass mirrors texture/normal-map sampling.
 *  - `sheen`, `sheenColor`, `sheenRoughness` — map-free KHR_materials_sheen
 *    direct lobe controls through the additive Charlie lobe. Sheen maps and
 *    sheen-roughness maps stay on finite difference until the adjoint pass
 *    mirrors those texture derivatives.
 *  - `iridescence` — map-free KHR_materials_iridescence scalar through the
 *    thin-film-modified base F0 in the same scoped opaque direct-light domain.
 *  - `iridescenceIor` — map-free scalar thin-film IOR through a local symmetric
 *    derivative of that F0 term. Thickness range, iridescence maps, and
 *    thickness maps stay on finite difference until their derivatives are
 *    mirrored and validated.
 *  - `anisotropy` / `anisotropyRotation` — map-free scalar anisotropic-GGX
 *    controls through a local symmetric derivative of the direct-light specular
 *    lobe. Anisotropy maps stay on finite difference.
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
 * scalar clearcoat, map-free sheen controls, scalar iridescence,
 * scalar iridescenceIor, and map-free anisotropy controls are
 * CPU-FD-oracle + shader-gate covered and remain on the recapture tail.
 */
const ADJOINT_ELIGIBLE_FIELDS = new Set([
  'baseColor',
  'roughness',
  'metallic',
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
  'anisotropy',
  'anisotropyRotation',
]);

interface ParamSlot {
  readonly param: InverseParam;
  readonly target: ResolvedParamTarget;
  readonly offset: number;
  readonly length: number;
}

const MATERIAL_RGB_FIELDS = new Set([
  'baseColor',
  'emissive',
  'specularColor',
  'sheenColor',
]);
const MATERIAL_SCALAR_FIELDS = new Set([
  'roughness',
  'metallic',
  'emissiveIntensity',
  'ior',
  'specularIntensity',
  'clearcoat',
  'clearcoatRoughness',
  'sheen',
  'sheenRoughness',
  'iridescence',
  'iridescenceIor',
  'anisotropy',
  'anisotropyRotation',
]);
const EMITTER_RGB_FIELDS = new Set(['color']);
const EMITTER_SCALAR_FIELDS = new Set(['intensity']);

export class PtWebgpuInverseSession implements InverseSession {
  readonly #hooks: InverseEngineHooks;
  readonly #target: InverseSessionOptions['target'];
  readonly #lossFn: typeof l2Loss;
  readonly #lossKind: 'l2' | 'l1';
  readonly #method: InverseGradientMethod;
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
    // compatible direct-light domain, or a map-free unlit baseColor primary-hit
    // fit (`ADJOINT_ELIGIBLE_FIELDS`: material
    // baseColor / roughness / metallic / emissive / emissiveIntensity /
    // specularColor / specularIntensity — see its doc for the ior exclusion). Any shortfall
    // degrades to finite-difference and is reported via `method`, so the host
    // never receives a silently-wrong gradient. The two adjoint stages
    // the hook relies on (partials; chain rule + accumulation) are GPU-validated
    // on lavapipe (V24); an engine exposing the hook vouches for the re-trace.
    const iridescenceOptimizedPrimitiveIds = new Set(
      this.#slots
        .filter(
          (s) =>
            s.target.domain === 'materials' &&
            (s.target.field === 'iridescence' || s.target.field === 'iridescenceIor'),
        )
        .map((s) => s.target.id),
    );
    const allEligible = this.#slots.every(
      (s) =>
        s.target.domain === 'materials' &&
        ADJOINT_ELIGIBLE_FIELDS.has(s.target.field) &&
        isPathReplayCompatibleTarget(scene, s.target, iridescenceOptimizedPrimitiveIds),
    );
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

function isPathReplayCompatibleTarget(
  scene: Scene,
  target: ResolvedParamTarget,
  iridescenceOptimizedPrimitiveIds: ReadonlySet<string>,
): boolean {
  const prim = findPrimitive(scene, target.id);
  if (prim == null) return false;
  if (!isPathReplayTriangleBackedPrimitive(prim)) return false;
  const m = prim.material;
  if (target.field === 'baseColor' && isPathReplayCompatibleUnlitBaseColorMaterial(m)) {
    return true;
  }
  if (target.field === 'emissive' || target.field === 'emissiveIntensity') {
    return isPathReplayCompatibleEmissiveMaterial(m);
  }
  if (target.field === 'iridescence' || target.field === 'iridescenceIor') {
    return isPathReplayCompatibleLighting(scene) && isPathReplayCompatibleIridescenceMaterial(m);
  }
  if (target.field === 'anisotropy' || target.field === 'anisotropyRotation') {
    return isPathReplayCompatibleLighting(scene) && isPathReplayCompatibleAnisotropyMaterial(m);
  }
  if (iridescenceOptimizedPrimitiveIds.has(target.id)) return false;
  return isPathReplayCompatibleLighting(scene) && isPathReplayCompatibleBrdfMaterial(m);
}

function isIdentityMat4(transform: Float32Array): boolean {
  const expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  if (transform.length < 16) return false;
  for (let i = 0; i < 16; i += 1) {
    if (Math.abs((transform[i] ?? 0) - expected[i]!) > 1e-6) return false;
  }
  return true;
}

function isPathReplayTriangleBackedPrimitive(primitive: ScenePrimitive): boolean {
  if (primitive.kind !== 'mesh' && primitive.kind !== 'skinned-mesh') return false;
  return primitive.transform == null || isIdentityMat4(primitive.transform);
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
  return !hasPathReplayUnsupportedMap(m);
}

function isPathReplayCompatibleEmissiveMaterial(m: MaterialSpec): boolean {
  if (m.shadingModel === 'unlit') return false;
  if (m.alphaMode != null && m.alphaMode !== 'opaque') return false;
  if (m.opacity != null && m.opacity < 1) return false;
  if ((m.transmission ?? 0) > 1e-6) return false;
  return m.alphaMap == null;
}

function isPathReplayCompatibleBrdfMaterial(m: MaterialSpec): boolean {
  if (m.shadingModel === 'unlit') return false;
  if (m.alphaMode != null && m.alphaMode !== 'opaque') return false;
  if (m.opacity != null && m.opacity < 1) return false;
  if ((m.transmission ?? 0) > 1e-6) return false;
  if ((m.iridescence ?? 0) > 1e-6) return false;
  if ((m.anisotropy ?? 0) > 1e-6) return false;
  if (m.frontLayer != null || m.backLayer != null || m.thinFilmStack != null) return false;
  if (m.spectralAttenuation != null || m.dispersionAbbeNumber != null) return false;
  if ((m.scatteringCoefficient ?? 0) > 0 || (m.scatteringCoefficientRGB != null)) return false;
  if (m.extensions != null && Object.keys(m.extensions).length > 0) return false;
  return !hasPathReplayUnsupportedMap(m);
}

function isPathReplayCompatibleIridescenceMaterial(m: MaterialSpec): boolean {
  if (m.shadingModel === 'unlit') return false;
  if (m.alphaMode != null && m.alphaMode !== 'opaque') return false;
  if (m.opacity != null && m.opacity < 1) return false;
  if ((m.transmission ?? 0) > 1e-6) return false;
  if ((m.anisotropy ?? 0) > 1e-6) return false;
  if (m.frontLayer != null || m.backLayer != null || m.thinFilmStack != null) return false;
  if (m.spectralAttenuation != null || m.dispersionAbbeNumber != null) return false;
  if ((m.scatteringCoefficient ?? 0) > 0 || (m.scatteringCoefficientRGB != null)) return false;
  if (m.extensions != null && Object.keys(m.extensions).length > 0) return false;
  return !hasPathReplayUnsupportedMap(m);
}

function isPathReplayCompatibleAnisotropyMaterial(m: MaterialSpec): boolean {
  if (m.shadingModel === 'unlit') return false;
  if (m.alphaMode != null && m.alphaMode !== 'opaque') return false;
  if (m.opacity != null && m.opacity < 1) return false;
  if ((m.transmission ?? 0) > 1e-6) return false;
  if ((m.iridescence ?? 0) > 1e-6) return false;
  if (m.frontLayer != null || m.backLayer != null || m.thinFilmStack != null) return false;
  if (m.spectralAttenuation != null || m.dispersionAbbeNumber != null) return false;
  if ((m.scatteringCoefficient ?? 0) > 0 || (m.scatteringCoefficientRGB != null)) return false;
  if (m.extensions != null && Object.keys(m.extensions).length > 0) return false;
  return !hasPathReplayUnsupportedMap(m);
}

function hasPathReplayUnsupportedMap(m: MaterialSpec): boolean {
  return (
    m.baseColorMap != null ||
    m.normalMap != null ||
    m.roughnessMap != null ||
    m.metallicMap != null ||
    m.transmissionMap != null ||
    m.thicknessMap != null ||
    m.emissiveMap != null ||
    m.alphaMap != null ||
    m.aoMap != null ||
    m.clearcoatMap != null ||
    m.clearcoatRoughnessMap != null ||
    m.clearcoatNormalMap != null ||
    m.sheenColorMap != null ||
    m.sheenRoughnessMap != null ||
    m.iridescenceMap != null ||
    m.iridescenceThicknessMap != null ||
    m.anisotropyMap != null ||
    m.specularColorMap != null ||
    m.specularIntensityMap != null ||
    m.bumpMap != null ||
    m.displacementMap != null ||
    m.lightMap != null
  );
}

function isPathReplayCompatibleLighting(scene: Scene): boolean {
  if ((scene.environment?.kind ?? 'none') !== 'none') return false;
  // The path-replay pass differentiates deterministic direct-light terms for
  // delta directional, point, spot, and center-sampled rect/disc/mesh-area lights.
  // Soft-sun angular-diameter directionals and environment lighting
  // stay on finite difference until their stochastic/source terms are mirrored
  // and validated end-to-end.
  return scene.emitters.every((e) => {
    if (
      e.kind === 'point' ||
      e.kind === 'spot' ||
      e.kind === 'rect-area' ||
      e.kind === 'disc-area' ||
      e.kind === 'mesh-area'
    ) {
      return true;
    }
    if (e.kind === 'directional') {
      const angularDiameter = e.angularDiameter;
      return angularDiameter == null || !Number.isFinite(angularDiameter) || angularDiameter <= 1e-6;
    }
    return false;
  });
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
    const isScalar = MATERIAL_SCALAR_FIELDS.has(target.field);
    if (!isRgb && !isScalar) {
      throw new Error(
        `createInverseSession: material field "${target.field}" (path "${param.path}") is not ` +
          `optimizable. Supported: ${[...MATERIAL_RGB_FIELDS, ...MATERIAL_SCALAR_FIELDS].join(', ')}.`,
      );
    }
    assertKind(param, isRgb ? 'rgb' : 'scalar');
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
 *  saturate at [0, 1] (physical reflectance / microfacet range); emissive
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
    case 'specularIntensity':
    case 'clearcoat':
    case 'clearcoatRoughness':
    case 'sheen':
    case 'sheenRoughness':
    case 'iridescence':
    case 'anisotropy':
      return [0, 1];
    case 'ior':
      return [1, 2.5];
    case 'iridescenceIor':
      return [1, 3];
    case 'anisotropyRotation':
      return [0, Math.PI];
    case 'emissive':
    case 'emissiveIntensity':
    case 'color':
    case 'intensity':
      return [0, Infinity];
    default:
      return [0, Infinity];
  }
}

function assertKind(param: InverseParam, expected: 'rgb' | 'scalar'): void {
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
      case 'ior': return [m.ior ?? 1.5];
      case 'specularColor': return [...(m.specularColor ?? [1, 1, 1])];
      case 'specularIntensity': return [m.specularIntensity ?? 1];
      case 'clearcoat': return [m.clearcoat ?? 0];
      case 'clearcoatRoughness': return [m.clearcoatRoughness ?? 0];
      case 'sheen': return [m.sheen ?? 0];
      case 'sheenColor': return [...(m.sheenColor ?? [1, 1, 1])];
      case 'sheenRoughness': return [m.sheenRoughness ?? 0];
      case 'iridescence': return [m.iridescence ?? 0];
      case 'iridescenceIor': return [m.iridescenceIor ?? 1.3];
      case 'anisotropy': return [m.anisotropy ?? 0];
      case 'anisotropyRotation': return [m.anisotropyRotation ?? 0];
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
    case 'ior': return { ior: value[0]! };
    case 'specularColor': return { specularColor: value as unknown as Vec3 };
    case 'specularIntensity': return { specularIntensity: value[0]! };
    case 'clearcoat': return { clearcoat: value[0]! };
    case 'clearcoatRoughness': return { clearcoatRoughness: value[0]! };
    case 'sheen': return { sheen: value[0]! };
    case 'sheenColor': return { sheenColor: value as unknown as Vec3 };
    case 'sheenRoughness': return { sheenRoughness: value[0]! };
    case 'iridescence': return { iridescence: value[0]! };
    case 'iridescenceIor': return { iridescenceIor: value[0]! };
    case 'anisotropy': return { anisotropy: value[0]! };
    case 'anisotropyRotation': return { anisotropyRotation: value[0]! };
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
