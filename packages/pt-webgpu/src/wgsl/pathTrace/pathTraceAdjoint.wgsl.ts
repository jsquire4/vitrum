/**
 * pathTraceAdjoint.wgsl.ts — path-replay BSDF adjoint (WS5 Phase 1).
 *
 * STATUS (2026-06): NOT WIRED. This WGSL is an unwired GPU twin of the CPU
 * adjoint oracle (`../../inverse/brdfAdjoint.ts`). It is NOT composed into any
 * pipeline and NOT dispatched — it is only string-shape-pinned by
 * `__tests__/brdfAdjoint.test.ts` to stay arithmetically identical to the
 * oracle, against future wiring. The live inverse path is finite-difference
 * (`inverse/inverseSession.ts` hardcodes `method = 'finite-difference'`; the GPU
 * adjoint dispatch is gated pending real-GPU validation — V24). The present-
 * tense description below is the INTENDED Phase-1 design, not current runtime.
 *
 * Emits the WGSL functions that compute the analytic partials of the
 * Cook-Torrance BRDF (`evaluateBrdf`) w.r.t. the two Phase-1 optimizable
 * parameters — `baseColor` (rgb) and `roughness` (scalar) — for a FROZEN
 * sampled direction `wi`. This is the GPU twin of the CPU oracle in
 * `../../inverse/brdfAdjoint.ts`; the two are hand-verified line-for-line and
 * the codegen-shape tests pin that they keep emitting the same arithmetic.
 *
 * Path-replay (Vicini 2021): the adjoint re-traces the forward path with the
 * SAME RNG seed (`params.frameSeed ^ params.frameIndex`) so the hit point, the
 * frozen light/BSDF sample direction, and the visibility term are bit-identical
 * to the forward render. Only the continuous shading is differentiated — the
 * sampled direction is held constant, so there is NO differentiation through
 * sampling and the visibility / lobe-choice discontinuities never enter the
 * gradient. With Phase-1's single-bounce scope the per-pixel replay state is a
 * single hit record, far under the `GpuResources.BDPT_EYE_STACK_MAX_BYTES`
 * (384 MiB) ceiling that forced path-replay over a stored adjoint graph.
 *
 * The gradient of the image-space L2 loss w.r.t. a parameter θ is
 *   dLoss/dθ = Σ_pixels  2·(rendered_p − target_p) · dRendered_p/dθ,
 * and dRendered_p/dθ = dBrdf/dθ · NdotL · Li (the cosine and incident radiance
 * are frozen in replay). The per-pixel contribution is reduced into a tiny
 * per-parameter gradient buffer via i32 fixed-point atomics — the SAME
 * NRC_GRAD_FP = 2^20 fixed-point discipline `fusedMlp.wgsl.ts` uses, so a
 * downstream Adam step reads an integer-accumulated gradient with no float
 * atomic-add (which core WebGPU lacks).
 *
 * Ref: Vicini, Speierer, Jakob, "Path Replay Backpropagation," ACM TOG 40(4),
 *      SIGGRAPH 2021; Nimier-David et al., "Radiative Backpropagation," ACM TOG
 *      39(4), SIGGRAPH 2020. BRDF: PBR 4th ed. §9.6–9.8.
 */

/** Fixed-point scale for gradient atomics — matches fusedMlp.wgsl.ts NRC_GRAD_FP. */
export const ADJOINT_GRAD_FP = 1048576.0; // 2^20

export const PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL = /* wgsl */ `
// Fixed-point scale for the i32 gradient atomics (2^20). Mirrors NRC_GRAD_FP so
// the adjoint and the NRC trainer share one fixed-point convention.
const ADJOINT_GRAD_FP = ${ADJOINT_GRAD_FP};

// ── analytic ∂(evaluateBrdf)_c / ∂baseColor_c (diagonal Jacobian) ───────────
// Mirror of inverse/brdfAdjoint.ts:dBrdf_dBaseColor. f0 mixes baseColor with
// metallic, so baseColor perturbs BOTH the diffuse term and (through f0) the
// per-channel specular Fresnel weight. No cross-channel coupling → diagonal.
fn dBrdf_dBaseColor(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  let specScale = (d * g) / max(4.0 * nDotV * nDotL, 1e-6);
  let kd0 = 1.0 - metallic;
  let m = clamp(1.0 - vDotH, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  var outv = vec3f(0.0);
  for (var c: u32 = 0u; c < 3u; c = c + 1u) {
    let bc = baseColor[c];
    let f0c = 0.04 + (bc - 0.04) * metallic;
    let fc = f0c + (1.0 - f0c) * m5;
    let dfc = (1.0 - m5) * metallic;               // df_c/dbaseColor_c
    let dDiff = kd0 * INV_PI * ((1.0 - fc) + bc * (-dfc));
    let dSpec = specScale * dfc;
    outv[c] = dDiff + dSpec;
  }
  return outv;
}

// ── analytic ∂(evaluateBrdf)_c / ∂roughness (per channel) ───────────────────
// Mirror of inverse/brdfAdjoint.ts:dBrdf_dRoughness. Diffuse term is
// roughness-independent; only the specular D·G product carries the derivative.
fn dBrdf_dRoughness(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let f = fresnelSchlick(vDotH, f0);

  let alpha = max(roughness * roughness, 1e-3);
  let alphaClamped = (roughness * roughness) < 1e-3;
  let dAlpha_dRough = select(2.0 * roughness, 0.0, alphaClamped);

  // dD/da²  (den = nDotH²(a²-1)+1) ; da²/droughness = 2·alpha·dAlpha_dRough.
  let a2 = alpha * alpha;
  let den = nDotH * nDotH * (a2 - 1.0) + 1.0;
  let dD_da2 = (den - 2.0 * a2 * (nDotH * nDotH)) / max(PI * den * den * den, 1e-12);
  let da2_dRough = 2.0 * alpha * dAlpha_dRough;
  let dD_dRough = dD_da2 * da2_dRough;
  let d = ggxD(nDotH, alpha);

  // dG1/droughness. k = (roughness+1)²/8 ; dk/droughness = (roughness+1)/4.
  let k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
  let dk_dRough = (roughness + 1.0) * 0.25;
  let g1V = smithG1(nDotV, roughness);
  let g1L = smithG1(nDotL, roughness);
  let denV = nDotV * (1.0 - k) + k;
  let denL = nDotL * (1.0 - k) + k;
  let dG1V = select((-nDotV * (1.0 - nDotV) / (denV * denV)) * dk_dRough, 0.0, denV <= 1e-6);
  let dG1L = select((-nDotL * (1.0 - nDotL) / (denL * denL)) * dk_dRough, 0.0, denL <= 1e-6);
  let g = g1V * g1L;
  let dG_dRough = dG1V * g1L + g1V * dG1L;

  let invDenom = 1.0 / max(4.0 * nDotV * nDotL, 1e-6);
  let dSpecScale = (dD_dRough * g + d * dG_dRough) * invDenom;
  return f * dSpecScale;
}

// Scatter a scalar gradient component into the i32 fixed-point accumulator.
// Atomic add of round(g · ADJOINT_GRAD_FP); a host-side pass divides back out.
fn adjointScatter(slot: u32, g: f32) {
  let q = i32(round(g * ADJOINT_GRAD_FP));
  atomicAdd(&gradAccum[slot], q);
}
`;
