/**
 * pathTraceAdjoint.wgsl.ts — path-replay BSDF adjoint (WS5 Phase 1).
 *
 * STATUS (2026-06): WIRED + GPU-VALIDATED (V24, commits 3d022f9/5a79307). These
 * partials are composed into the engine adjoint compute pass
 * (`adjointPass.wgsl.ts` → PT_WEBGPU_ADJOINT_PASS_WGSL), which
 * `index.ts:#computeAdjointGradient` builds into a focused pipeline and
 * dispatches; the same string is also composed into the GPU validation harness
 * (`../../inverse/adjointHarness.wgsl.ts`) and string-shape-pinned against the
 * CPU oracle by `__tests__/brdfAdjoint.test.ts`. `inverse/inverseSession.ts`
 * resolves the effective method to 'path-replay' (NOT finite-difference)
 * whenever the engine supplies the `computeAdjointGradient` hook AND every
 * optimized parameter is in the Phase-1 differentiable set (material baseColor /
 * roughness). GPU-validated on lavapipe: the partials match the FD oracle to f32
 * precision, and the chain rule + fixed-point accumulation match an on-device
 * finite-difference.
 *
 * Emits the WGSL functions that compute the analytic partials of:
 *  - the Cook-Torrance BRDF (`evaluateBrdf`) w.r.t. `baseColor` (rgb) and
 *    `roughness` (scalar), for a FROZEN sampled direction `wi`;
 *  - the additive emission term w.r.t. `emissive` (rgb) — a CONTRIBUTION-level
 *    identity (×emissiveIntensity), NOT a BRDF partial (`dContribution_dEmissive`);
 *  - the dielectric Fresnel reflectance `frDielectric` w.r.t. `ior` (scalar)
 *    (`dFrDielectric_dIor`) — the only differentiable `ior` dependence in the
 *    forward kernel (the opaque-reflective F0 is a fixed 0.04, so
 *    `∂evaluateBrdf/∂ior ≡ 0`; see the CPU oracle doc for the consumption caveat).
 * These are the GPU twins of the CPU oracle in `../../inverse/brdfAdjoint.ts`;
 * the two are hand-verified line-for-line and the codegen-shape tests pin that
 * they keep emitting the same arithmetic.
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

// ── analytic ∂(contribution)_c / ∂emissive_c (diagonal identity) ────────────
// Mirror of inverse/brdfAdjoint.ts:dContribution_dEmissive. Emission is an
// additive Le, NOT a BSDF term: ∂(throughput·emissive_packed)/∂emissive_param =
// throughput · emissiveIntensity (the packing folds intensity in). Diagonal;
// for a primary hit throughput = 1 so this is the identity × emissiveIntensity.
fn dContribution_dEmissive(throughput: vec3f, emissiveIntensity: f32) -> vec3f {
  return throughput * emissiveIntensity;
}

// ── analytic ∂(frDielectric)/∂ior (scalar) ──────────────────────────────────
// Mirror of inverse/brdfAdjoint.ts:dFrDielectric_dIor. The ONLY differentiable
// ior dependence in the forward kernel (opaque-reflective F0 is a fixed 0.04, so
// ∂evaluateBrdf/∂ior ≡ 0). eta = ior (front) or 1/ior (back); TIR / grazing
// return 0 (frozen-discontinuity convention). NOT yet wired into an end-to-end
// gradient — the Phase-1 pass does not trace the transmissive partition.
fn dFrDielectric_dIor(cosThetaI_in: f32, ior: f32) -> f32 {
  var cosThetaI = clamp(cosThetaI_in, -1.0, 1.0);
  var eta: f32;
  var dEta_dIor: f32;
  if (cosThetaI < 0.0) {
    eta = 1.0 / ior;
    dEta_dIor = -1.0 / (ior * ior);
    cosThetaI = -cosThetaI;
  } else {
    eta = ior;
    dEta_dIor = 1.0;
  }
  let s = max(0.0, 1.0 - cosThetaI * cosThetaI);
  let sin2ThetaT = s / (eta * eta);
  if (sin2ThetaT >= 1.0) { return 0.0; }            // TIR — Fr pinned to 1.
  let cosThetaT = sqrt(max(0.0, 1.0 - sin2ThetaT));
  if (cosThetaT <= 1e-6) { return 0.0; }            // grazing guard.

  let dCosT_dEta = s / (cosThetaT * eta * eta * eta);

  let a = eta * cosThetaI - cosThetaT;
  let b = eta * cosThetaI + cosThetaT;
  let da = cosThetaI - dCosT_dEta;
  let db = cosThetaI + dCosT_dEta;
  let rPar = a / b;
  let dRPar_dEta = (da * b - a * db) / (b * b);

  let c = cosThetaI - eta * cosThetaT;
  let dd = cosThetaT + eta * dCosT_dEta;            // d(eta·cosT)/deta
  let cc = cosThetaI + eta * cosThetaT;
  let dcNum = -dd;                                   // dc/deta
  let rPerp = c / cc;
  let dRPerp_dEta = (dcNum * cc - c * dd) / (cc * cc);

  let dFr_dEta = rPar * dRPar_dEta + rPerp * dRPerp_dEta;
  return dFr_dEta * dEta_dIor;
}

// Scatter a scalar gradient component into the i32 fixed-point accumulator.
// Atomic add of round(g · ADJOINT_GRAD_FP); a host-side pass divides back out.
fn adjointScatter(slot: u32, g: f32) {
  let q = i32(round(g * ADJOINT_GRAD_FP));
  atomicAdd(&gradAccum[slot], q);
}
`;
