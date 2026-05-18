/**
 * Shared BSDF primitives — Fresnel + microfacet helpers used by every BSDF
 * sample/PDF/eval triple.
 *
 *  - `luminance`, `powerHeuristic` — Veach MIS weighting (1997 Ch. 9)
 *  - `fresnelSchlick`, `frDielectric` — PBR4e §9.3
 *  - `ggxD`, `smithG1` — Walter et al. 2007 GGX distribution + Smith G
 *
 * Future (W4-A4 follow-up): introduce a shared `BsdfSample {wi, pdf, value}`
 * return struct so glossy/dielectric/diffuse sample functions all speak the
 * same shape. Today's per-BSDF return types are heterogeneous; unifying them
 * is a behavior-equivalent but non-trivial signature change so it's deferred.
 */
export const PT_WEBGPU_BSDF_MIS_WGSL = /* wgsl */ `
fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return f0 + (vec3f(1.0) - f0) * m5;
}

/**
 * Unpolarised Fresnel reflectance for a smooth dielectric interface.
 * Handles TIR (returns 1.0) and entering-from-inside (cosTheta_i < 0).
 * Ref: Pharr, Jakob, Humphreys. Physically Based Rendering 4th ed. §9.3
 *      "Specular Reflection and Transmission" — FrDielectric().
 *      https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
 */
fn frDielectric(cosTheta_i_in: f32, eta_in: f32) -> f32 {
  var cosTheta_i = clamp(cosTheta_i_in, -1.0, 1.0);
  var eta = eta_in;
  // Entering from the inside — flip so cosTheta_i is positive and invert eta.
  if (cosTheta_i < 0.0) {
    eta = 1.0 / eta;
    cosTheta_i = -cosTheta_i;
  }
  let sin2Theta_i = max(0.0, 1.0 - cosTheta_i * cosTheta_i);
  let sin2Theta_t = sin2Theta_i / (eta * eta);
  if (sin2Theta_t >= 1.0) { return 1.0; } // Total Internal Reflection.
  let cosTheta_t = sqrt(max(0.0, 1.0 - sin2Theta_t));
  let r_par  = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
  let r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
  return 0.5 * (r_par * r_par + r_perp * r_perp);
}

fn ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}

fn smithG1(nDotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}

fn powerHeuristic(pdfA: f32, pdfB: f32) -> f32 {
  let a2 = pdfA * pdfA;
  let b2 = pdfB * pdfB;
  return a2 / max(a2 + b2, 1e-6);
}
`;
