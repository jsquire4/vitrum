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
 * optimized parameter is in the currently differentiable set (baseColor,
 * roughness, metallic, aoMapIntensity, emissive, specularColor,
 * specularIntensity, clearcoat, clearcoatRoughness, sheen, sheenColor,
 * sheenRoughness, iridescence, iridescenceIor, iridescenceThicknessRange,
 * anisotropy, anisotropyRotation).
 * GPU-validated on lavapipe for the original V24 path: the baseColor/roughness
 * partials match the FD oracle to f32 precision, the chain rule + fixed-point
 * accumulation match an on-device finite-difference, and
 * baseColor/roughness/emissive end-to-end inverse fits converge + sign-match
 * the full-render FD (`v24-inverse-fit.mjs`, `v24-emissive-fit.mjs`). Later
 * specular/metallic/AO/clearcoat/sheen/iridescence/anisotropy partials are
 * CPU-FD-oracle + WGSL-shape + shader-gate covered until their GPU inverse-fit
 * recaptures land.
 *
 * Emits the WGSL functions that compute the analytic partials of:
 *  - the Cook-Torrance BRDF (`evaluateBrdf`) w.r.t. `baseColor` (rgb) and
 *    `roughness` (scalar), for a FROZEN sampled direction `wi`;
 *  - the opaque base-BRDF `metallic` scalar through the diffuse/specular
 *    partition and F0 blend;
 *  - the additive, map-free KHR_materials_clearcoat lobe w.r.t. `clearcoat`
 *    and `clearcoatRoughness`;
 *  - the additive, map-free KHR_materials_sheen lobe w.r.t. `sheen`,
 *    `sheenColor`, and `sheenRoughness`;
 *  - map-free KHR_materials_iridescence scalar through the thin-film-modified
 *    base F0 in the opaque direct-light domain;
 *  - map-free KHR_materials_iridescence IOR through a local symmetric
 *    derivative of the thin-film F0 term (single replay pass; not a full-render
 *    finite-difference probe);
 *  - map-free KHR_materials_anisotropy scalar controls through a local
 *    symmetric derivative of the anisotropic GGX specular lobe;
 *  - the additive emission term w.r.t. `emissive` (rgb) — a CONTRIBUTION-level
 *    identity (×emissiveIntensity), NOT a BRDF partial (`dContribution_dEmissive`);
 *  - the dielectric Fresnel reflectance `frDielectric` w.r.t. `ior` (scalar)
 *    (`dFrDielectric_dIor`) — the only differentiable `ior` dependence in the
 *    forward kernel (opaque F0 is controlled by KHR_materials_specular/baseColor,
 *    so `∂evaluateBrdf/∂ior ≡ 0`; see the CPU oracle doc for the consumption caveat).
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

fn adjointMaterialSpecularF0(
  baseColor: vec3f,
  metallic: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let dielectricF0 = clamp(
    vec3f(0.04) * clamp(specularColor, vec3f(0.0), vec3f(1.0)) * clamp(specularIntensity, 0.0, 1.0),
    vec3f(0.0),
    vec3f(1.0),
  );
  return mix(dielectricF0, baseColor, metallic);
}

// ── analytic ∂(evaluateBrdf)_c / ∂baseColor_c (diagonal Jacobian) ───────────
// Mirror of inverse/brdfAdjoint.ts:dBrdf_dBaseColor. f0 mixes baseColor with
// metallic, so baseColor perturbs BOTH the diffuse term and (through f0) the
// per-channel specular Fresnel weight. No cross-channel coupling → diagonal.
fn dBrdf_dBaseColor(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
) -> vec3f {
  return dBrdf_dBaseColorWithSpecular(
    baseColor, roughness, metallic, normal, wo, wi, vec3f(1.0), 1.0,
  );
}
fn dBrdf_dBaseColorWithSpecular(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f, specularIntensity: f32,
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
    let dielectricF0 = clamp(0.04 * clamp(specularColor[c], 0.0, 1.0) * clamp(specularIntensity, 0.0, 1.0), 0.0, 1.0);
    let f0c = dielectricF0 + (bc - dielectricF0) * metallic;
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
  return dBrdf_dRoughnessWithSpecular(
    baseColor, roughness, metallic, normal, wo, wi, vec3f(1.0), 1.0,
  );
}
fn dBrdf_dRoughnessWithSpecular(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f, specularIntensity: f32,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = adjointMaterialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
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

// ── analytic ∂(evaluateBrdf)_c / ∂metallic (per channel) ────────────────────
fn dBrdf_dMetallic(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f, specularIntensity: f32,
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
    let dielectricF0 = clamp(0.04 * clamp(specularColor[c], 0.0, 1.0) * clamp(specularIntensity, 0.0, 1.0), 0.0, 1.0);
    let f0c = dielectricF0 + (bc - dielectricF0) * metallic;
    let fc = f0c + (1.0 - f0c) * m5;
    let dfc = (1.0 - m5) * (bc - dielectricF0);
    let dDiff = bc * INV_PI * (-kd0 * dfc - (1.0 - fc));
    let dSpec = specScale * dfc;
    outv[c] = dDiff + dSpec;
  }
  return outv;
}

// ── analytic KHR_materials_specular partials ───────────────────────────────
fn dBrdf_dSpecularF0(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  dF0: vec3f,
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
  return dF0 * (1.0 - m5) * (vec3f(specScale) - kd0 * baseColor * INV_PI);
}
fn dBrdf_dSpecularColor(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f, specularIntensity: f32,
) -> vec3f {
  let dF0 = vec3f(0.04 * clamp(specularIntensity, 0.0, 1.0) * (1.0 - metallic));
  return dBrdf_dSpecularF0(baseColor, roughness, metallic, normal, wo, wi, dF0);
}
fn dBrdf_dSpecularIntensity(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f,
) -> vec3f {
  let dF0 = 0.04 * clamp(specularColor, vec3f(0.0), vec3f(1.0)) * (1.0 - metallic);
  return dBrdf_dSpecularF0(baseColor, roughness, metallic, normal, wo, wi, dF0);
}

// ── KHR_materials_iridescence scalar partials ───────────────────────────────
// Mirrors inverse/brdfAdjoint.ts:dBrdf_dIridescence*. This is map-free scalar
// iridescence only: no iridescence maps, no thickness maps, no anisotropy.
const IRIDESCENCE_IOR_DERIV_STEP = 1e-3;
const IRIDESCENCE_THICKNESS_DERIV_STEP = 1e-2;
const ANISOTROPY_DERIV_STEP = 1e-3;
const ANISOTROPY_ROTATION_DERIV_STEP = 1e-3;
fn adjointIridXyzToRec709(xyz: vec3f) -> vec3f {
  return vec3f(
     3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
    -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
     0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z,
  );
}
fn adjointIridFresnel0ToIor(f0: vec3f) -> vec3f {
  let sqrtF0 = sqrt(clamp(f0, vec3f(0.0), vec3f(0.9999)));
  return (vec3f(1.0) + sqrtF0) / (vec3f(1.0) - sqrtF0);
}
fn adjointIridIorToFresnel0Scalar(transmittedIor: f32, incidentIor: f32) -> f32 {
  let r = (transmittedIor - incidentIor) / (transmittedIor + incidentIor);
  return r * r;
}
fn adjointIridIorToFresnel0Vec(transmittedIor: vec3f, incidentIor: f32) -> vec3f {
  let r = (transmittedIor - vec3f(incidentIor)) / (transmittedIor + vec3f(incidentIor));
  return r * r;
}
fn adjointIridSchlickScalar(cosTheta: f32, f0: f32) -> f32 {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (1.0 - f0) * m2 * m2 * m;
}
fn adjointIridSchlickVec(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3f(1.0) - f0) * m2 * m2 * m;
}
fn adjointIridEvalSensitivity(OPD: f32, shift: vec3f) -> vec3f {
  let phase = 2.0 * PI * OPD * 1.0e-9;
  let val = vec3f(5.4856e-13, 4.4201e-13, 5.2481e-13);
  let pos = vec3f(1.6810e+06, 1.7953e+06, 2.2084e+06);
  let vari = vec3f(4.3278e+09, 9.3046e+09, 6.6121e+09);
  var xyz = val * sqrt(2.0 * PI * vari) * cos(pos * phase + shift) * exp(-phase * phase * vari);
  xyz.x = xyz.x + 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09)
      * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
  xyz = xyz / 1.0685e-7;
  return adjointIridXyzToRec709(xyz);
}
fn adjointEvalIridescence(
  outsideIOR: f32,
  eta2: f32,
  cosTheta1: f32,
  thicknessNm: f32,
  baseF0: vec3f,
) -> vec3f {
  let iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thicknessNm));
  let sinTheta2Sq = (outsideIOR / iridescenceIor) * (outsideIOR / iridescenceIor)
      * max(0.0, 1.0 - cosTheta1 * cosTheta1);
  let cosTheta2Sq = 1.0 - sinTheta2Sq;
  if (cosTheta2Sq < 0.0) { return vec3f(1.0); }
  let cosTheta2 = sqrt(cosTheta2Sq);
  let R0_scalar = adjointIridIorToFresnel0Scalar(iridescenceIor, outsideIOR);
  let R12 = adjointIridSchlickScalar(cosTheta1, R0_scalar);
  let T121 = 1.0 - R12;
  let phi12 = select(0.0, PI, iridescenceIor < outsideIOR);
  let phi21 = PI - phi12;
  let baseIOR = adjointIridFresnel0ToIor(clamp(baseF0, vec3f(0.0), vec3f(0.9999)));
  let R1_vec = adjointIridIorToFresnel0Vec(baseIOR, iridescenceIor);
  let R23 = adjointIridSchlickVec(cosTheta2, R1_vec);
  var phi23 = vec3f(0.0);
  phi23.x = select(0.0, PI, baseIOR.x < iridescenceIor);
  phi23.y = select(0.0, PI, baseIOR.y < iridescenceIor);
  phi23.z = select(0.0, PI, baseIOR.z < iridescenceIor);
  let OPD = 2.0 * iridescenceIor * thicknessNm * cosTheta2;
  let phi = vec3f(phi21) + phi23;
  let R123 = clamp(R12 * R23, vec3f(1e-5), vec3f(0.9999));
  let r123 = sqrt(R123);
  let Rs = (T121 * T121) * R23 / (vec3f(1.0) - R123);
  var I = vec3f(R12) + Rs;
  var Cm = Rs - vec3f(T121);
  for (var m = 1; m <= 2; m = m + 1) {
    Cm = Cm * r123;
    let Sm = 2.0 * adjointIridEvalSensitivity(f32(m) * OPD, f32(m) * phi);
    I = I + Cm * Sm;
  }
  return max(I, vec3f(0.0));
}
fn dBrdf_dIridescence(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f, specularIntensity: f32,
  iridescenceIor: f32, iridescenceThicknessMin: f32, iridescenceThicknessMax: f32,
) -> vec3f {
  let h = safe_normalize(wi + wo);
  let vDotH = max(dot(wo, h), 0.0);
  let baseF0 = adjointMaterialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  let thicknessNm = mix(iridescenceThicknessMin, iridescenceThicknessMax, clamp(vDotH, 0.0, 1.0));
  let iridF = adjointEvalIridescence(1.0, iridescenceIor, vDotH, thicknessNm, baseF0);
  return dBrdf_dSpecularF0(baseColor, roughness, metallic, normal, wo, wi, iridF - baseF0);
}
fn dBrdf_dIridescenceIor(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f, specularIntensity: f32,
  iridescence: f32, iridescenceIor: f32,
  iridescenceThicknessMin: f32, iridescenceThicknessMax: f32,
) -> vec3f {
  if (iridescence < 1e-4) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let vDotH = max(dot(wo, h), 0.0);
  let baseF0 = adjointMaterialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  let thicknessNm = mix(iridescenceThicknessMin, iridescenceThicknessMax, clamp(vDotH, 0.0, 1.0));
  let iorP = max(1.0, iridescenceIor + IRIDESCENCE_IOR_DERIV_STEP);
  let iorM = max(1.0, iridescenceIor - IRIDESCENCE_IOR_DERIV_STEP);
  let denom = iorP - iorM;
  if (denom <= 1e-6) { return vec3f(0.0); }
  let fp = adjointEvalIridescence(1.0, iorP, vDotH, thicknessNm, baseF0);
  let fm = adjointEvalIridescence(1.0, iorM, vDotH, thicknessNm, baseF0);
  return dBrdf_dSpecularF0(
    baseColor, roughness, metallic, normal, wo, wi,
    iridescence * (fp - fm) / denom,
  );
}

struct IridescenceThicknessRangePartial {
  min: vec3f,
  max: vec3f,
}
fn dBrdf_dIridescenceThicknessRange(
  baseColor: vec3f, roughness: f32, metallic: f32,
  normal: vec3f, wo: vec3f, wi: vec3f,
  specularColor: vec3f, specularIntensity: f32,
  iridescence: f32, iridescenceIor: f32,
  iridescenceThicknessMin: f32, iridescenceThicknessMax: f32,
  iridescenceThicknessTexel: f32,
) -> IridescenceThicknessRangePartial {
  if (iridescence < 1e-4) {
    return IridescenceThicknessRangePartial(vec3f(0.0), vec3f(0.0));
  }
  let h = safe_normalize(wi + wo);
  let vDotH = clamp(max(dot(wo, h), 0.0), 0.0, 1.0);
  let rangeT = select(vDotH, clamp(iridescenceThicknessTexel, 0.0, 1.0), iridescenceThicknessTexel >= 0.0);
  let baseF0 = adjointMaterialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  let thicknessNm = mix(iridescenceThicknessMin, iridescenceThicknessMax, rangeT);
  let tp = max(0.0, thicknessNm + IRIDESCENCE_THICKNESS_DERIV_STEP);
  let tm = max(0.0, thicknessNm - IRIDESCENCE_THICKNESS_DERIV_STEP);
  let denom = tp - tm;
  if (denom <= 1e-6) {
    return IridescenceThicknessRangePartial(vec3f(0.0), vec3f(0.0));
  }
  let fp = adjointEvalIridescence(1.0, iridescenceIor, vDotH, tp, baseF0);
  let fm = adjointEvalIridescence(1.0, iridescenceIor, vDotH, tm, baseF0);
  let dBrdf_dThickness = dBrdf_dSpecularF0(
    baseColor, roughness, metallic, normal, wo, wi,
    iridescence * (fp - fm) / denom,
  );
  return IridescenceThicknessRangePartial(
    dBrdf_dThickness * (1.0 - rangeT),
    dBrdf_dThickness * rangeT,
  );
}

fn adjointBuildTangent(n: vec3f) -> vec3f {
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
  return safe_normalize(cross(up, n));
}

fn adjointGgxDAnis(hT: f32, hB: f32, hN: f32, ax: f32, ay: f32) -> f32 {
  let d = (hT / ax) * (hT / ax) + (hB / ay) * (hB / ay) + hN * hN;
  return 1.0 / max(PI * ax * ay * d * d, 1e-10);
}

fn adjointSmithG1Anis(vT: f32, vB: f32, vN: f32, ax: f32, ay: f32) -> f32 {
  let vN2 = max(vN * vN, 1e-10);
  let numer = 2.0 * vN;
  let denom = vN + sqrt(vN2 + (vT * ax) * (vT * ax) + (vB * ay) * (vB * ay));
  return numer / max(denom, 1e-6);
}

fn adjointEvalBrdfSpecAnisotropic(
  f0: vec3f,
  roughness: f32,
  anisotropy: f32,
  normal: vec3f,
  tangent: vec3f,
  bitangent: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let nDotV = max(dot(normal, wo), 1e-6);
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let vDotH = max(dot(wo, h), 1e-6);
  let f = fresnelSchlick(vDotH, f0);
  let alpha = max(roughness * roughness, 1e-3);
  let aspect = sqrt(max(1.0 - 0.9 * anisotropy, 1e-4));
  let ax = max(alpha / aspect, 1e-4);
  let ay = max(alpha * aspect, 1e-4);
  let hT = dot(h, tangent);
  let hB = dot(h, bitangent);
  let hN = max(dot(h, normal), 0.0);
  let woT = dot(wo, tangent);
  let woB = dot(wo, bitangent);
  let woN = max(dot(wo, normal), 1e-6);
  let wiT = dot(wi, tangent);
  let wiB = dot(wi, bitangent);
  let wiN = max(dot(wi, normal), 1e-6);
  let d = adjointGgxDAnis(hT, hB, hN, ax, ay);
  let g = adjointSmithG1Anis(woT, woB, woN, ax, ay) *
    adjointSmithG1Anis(wiT, wiB, wiN, ax, ay);
  return (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
}

fn adjointEvaluateBrdfWithAnisotropy(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = adjointMaterialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  let f = fresnelSchlick(vDotH, f0);
  let diff = (vec3f(1.0) - f) * (1.0 - metallic) * baseColor * INV_PI;
  if (anisotropy <= 1e-4) {
    let alpha = max(roughness * roughness, 1e-3);
    let d = ggxD(nDotH, alpha);
    let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
    let spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
    return diff + spec;
  }
  let tanT = adjointBuildTangent(normal);
  let tanB = cross(normal, tanT);
  let c = cos(anisotropyRotation);
  let s = sin(anisotropyRotation);
  let anisoT = c * tanT + s * tanB;
  let anisoB = -s * tanT + c * tanB;
  let spec = adjointEvalBrdfSpecAnisotropic(
    f0, roughness, anisotropy, normal, anisoT, anisoB, wo, wi,
  );
  return diff + spec;
}

fn dBrdf_dAnisotropy(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let ap = clamp(anisotropy + ANISOTROPY_DERIV_STEP, 0.0, 1.0);
  let am = clamp(anisotropy - ANISOTROPY_DERIV_STEP, 0.0, 1.0);
  let denom = ap - am;
  if (denom <= 1e-6) { return vec3f(0.0); }
  let fp = adjointEvaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    ap, anisotropyRotation, specularColor, specularIntensity,
  );
  let fm = adjointEvaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    am, anisotropyRotation, specularColor, specularIntensity,
  );
  return (fp - fm) / denom;
}

fn dBrdf_dAnisotropyRotation(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  if (anisotropy <= 1e-4) { return vec3f(0.0); }
  let fp = adjointEvaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation + ANISOTROPY_ROTATION_DERIV_STEP,
    specularColor, specularIntensity,
  );
  let fm = adjointEvaluateBrdfWithAnisotropy(
    baseColor, roughness, metallic, normal, wo, wi,
    anisotropy, anisotropyRotation - ANISOTROPY_ROTATION_DERIV_STEP,
    specularColor, specularIntensity,
  );
  return (fp - fm) / (2.0 * ANISOTROPY_ROTATION_DERIV_STEP);
}

// ── analytic KHR_materials_clearcoat partials ───────────────────────────────
// Mirrors inverse/brdfAdjoint.ts:dBrdf_dClearcoat*. This is the additive
// clearcoat lobe only: no clearcoat normal map, no clearcoat maps, frozen wi.
fn adjointClearcoatLobe(
  clearcoat: f32,
  clearcoatRoughness: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  if (clearcoat < 1e-4) { return vec3f(0.0); }
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f = fresnelSchlick(vDotH, vec3f(0.04));
  let alpha = max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, clearcoatRoughness) * smithG1(nDotL, clearcoatRoughness);
  let specScale = (d * g) / max(4.0 * nDotV * nDotL, 1e-6);
  return clearcoat * f * specScale;
}
fn dBrdf_dClearcoat(
  clearcoatRoughness: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  return adjointClearcoatLobe(1.0, clearcoatRoughness, normal, wo, wi);
}
fn dBrdf_dClearcoatRoughness(
  clearcoat: f32,
  clearcoatRoughness: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  if (clearcoat < 1e-4) { return vec3f(0.0); }
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f = fresnelSchlick(vDotH, vec3f(0.04));

  let alpha = max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  let alphaClamped = (clearcoatRoughness * clearcoatRoughness) < 1e-3;
  let dAlpha_dRough = select(2.0 * clearcoatRoughness, 0.0, alphaClamped);

  let a2 = alpha * alpha;
  let den = nDotH * nDotH * (a2 - 1.0) + 1.0;
  let dD_da2 = (den - 2.0 * a2 * (nDotH * nDotH)) / max(PI * den * den * den, 1e-12);
  let da2_dRough = 2.0 * alpha * dAlpha_dRough;
  let dD_dRough = dD_da2 * da2_dRough;
  let d = ggxD(nDotH, alpha);

  let k = (clearcoatRoughness + 1.0) * (clearcoatRoughness + 1.0) * 0.125;
  let dk_dRough = (clearcoatRoughness + 1.0) * 0.25;
  let g1V = smithG1(nDotV, clearcoatRoughness);
  let g1L = smithG1(nDotL, clearcoatRoughness);
  let denV = nDotV * (1.0 - k) + k;
  let denL = nDotL * (1.0 - k) + k;
  let dG1V = select((-nDotV * (1.0 - nDotV) / (denV * denV)) * dk_dRough, 0.0, denV <= 1e-6);
  let dG1L = select((-nDotL * (1.0 - nDotL) / (denL * denL)) * dk_dRough, 0.0, denL <= 1e-6);
  let g = g1V * g1L;
  let dG_dRough = dG1V * g1L + g1V * dG1L;

  let invDenom = 1.0 / max(4.0 * nDotV * nDotL, 1e-6);
  let dSpecScale = (dD_dRough * g + d * dG_dRough) * invDenom;
  return clearcoat * f * dSpecScale;
}

// ── analytic KHR_materials_sheen partials ───────────────────────────────────
// Mirrors inverse/brdfAdjoint.ts:dBrdf_dSheen*. This is the additive
// map-free Charlie sheen lobe only: no sheen maps, frozen sheenColor, frozen wi.
fn adjointCharlieD(nDotH: f32, alpha: f32) -> f32 {
  let invAlpha = 1.0 / max(alpha, 1e-4);
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  return (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / (2.0 * PI);
}
fn adjointSheenVisibility(nDotL: f32, nDotV: f32) -> f32 {
  return 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}
fn adjointSheenLobe(
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  if (sheen < 1e-4) { return vec3f(0.0); }
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let alpha = max(sheenRoughness * sheenRoughness, 1e-3);
  let d = adjointCharlieD(nDotH, alpha);
  let vis = adjointSheenVisibility(nDotL, nDotV);
  return sheen * sheenColor * d * vis;
}
fn dBrdf_dSheen(
  sheenRoughness: f32,
  sheenColor: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  return adjointSheenLobe(1.0, sheenRoughness, sheenColor, normal, wo, wi);
}
fn dBrdf_dSheenColor(
  sheen: f32,
  sheenRoughness: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  return adjointSheenLobe(sheen, sheenRoughness, vec3f(1.0), normal, wo, wi);
}
fn dBrdf_dSheenRoughness(
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  if (sheen < 1e-4) { return vec3f(0.0); }
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  let alphaRaw = sheenRoughness * sheenRoughness;
  let dAlpha_dRough = select(2.0 * sheenRoughness, 0.0, alphaRaw < 1e-3);
  if (sinThetaH <= 1e-6 || dAlpha_dRough == 0.0) { return vec3f(0.0); }

  let alpha = max(alphaRaw, 1e-3);
  let q = 1.0 / max(alpha, 1e-4);
  let powTerm = pow(sinThetaH, q);
  let logSin = log(sinThetaH);
  let dD_dQ = powTerm * (1.0 + (2.0 + q) * logSin) / (2.0 * PI);
  let dQ_dAlpha = -1.0 / (alpha * alpha);
  let dD_dRough = dD_dQ * dQ_dAlpha * dAlpha_dRough;
  let vis = adjointSheenVisibility(nDotL, nDotV);
  return sheen * sheenColor * dD_dRough * vis;
}

// ── analytic ∂(contribution)_c / ∂emissive_c (diagonal identity) ────────────
// Mirror of inverse/brdfAdjoint.ts:dContribution_dEmissive. Emission is an
// additive Le, NOT a BSDF term: ∂(throughput·emissive_packed)/∂emissive_param =
// throughput · emissiveIntensity (the packing folds intensity in). Diagonal;
// for a primary hit throughput = 1 so this is the identity × emissiveIntensity.
fn dContribution_dEmissive(throughput: vec3f, emissiveIntensity: f32) -> vec3f {
  return throughput * emissiveIntensity;
}

// Mirror of inverse/brdfAdjoint.ts:dContribution_dEmissiveIntensity. This is
// the scalar intensity partial's per-channel factor; the pass dots it with
// dLoss/dRendered. The descriptor carries UNFACTORED emissive RGB so intensity=0
// remains differentiable.
fn dContribution_dEmissiveIntensity(throughput: vec3f, emissive: vec3f) -> vec3f {
  return throughput * emissive;
}

// ── analytic ∂(frDielectric)/∂ior (scalar) ──────────────────────────────────
// Mirror of inverse/brdfAdjoint.ts:dFrDielectric_dIor. The ONLY differentiable
// ior dependence in the forward kernel (opaque F0 is controlled by
// KHR_materials_specular/baseColor, so ∂evaluateBrdf/∂ior ≡ 0). eta = ior
// (front) or 1/ior (back); TIR / grazing
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
