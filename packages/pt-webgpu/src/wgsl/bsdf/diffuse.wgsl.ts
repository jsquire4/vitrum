/**
 * Layered Cook-Torrance BRDF evaluation + directional PDF.
 *
 * `evaluateBrdf` returns the unified specular+diffuse value for a given
 * (wo, wi, normal, material) tuple; `brdfDirectionalPdf` returns the matching
 * PDF used by Veach 1997 power-heuristic MIS weighting.
 *
 * Despite the file name (`diffuse.wgsl`), both functions cover the full
 * layered model — they're grouped because eval and PDF must stay in lock-step.
 * The naming follows the W4-A4 plan; a future split can introduce per-lobe
 * modules (pure-diffuse, pure-glossy, pure-conductor) once the BsdfSample
 * unification lands.
 */
export const PT_WEBGPU_BSDF_DIFFUSE_WGSL = /* wgsl */ `
fn evaluateBrdf(baseColor: vec3f, roughness: f32, metallic: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) {
    return vec3f(0.0);
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let f = fresnelSchlick(vDotH, f0);
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  let spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
  let kd = (vec3f(1.0) - f) * (1.0 - metallic);
  let diff = kd * baseColor * INV_PI;
  return diff + spec;
}

fn brdfDirectionalPdf(baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> f32 {
  let wiDotN = dot(normal, wi);
  let woDotN = dot(normal, wo);
  let nDotV = max(woDotN, 0.0);
  if (nDotV <= 1e-5) {
    return 0.0;
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 1e-6);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let fresnel = fresnelSchlick(vDotH, f0);
  let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
  let baseTransProb = clamp(transmission * (1.0 - metallic), 0.0, 0.95);
  let baseDiffProb = max(0.0, (1.0 - metallic) * (1.0 - transmission));
  let sumProb = max(baseSpecProb + baseTransProb + baseDiffProb, 1e-4);
  let specProb = baseSpecProb / sumProb;
  let transProb = baseTransProb / sumProb;
  let diffProb = baseDiffProb / sumProb;
  let sameHemisphere = wiDotN * woDotN > 0.0;
  if (!sameHemisphere) {
    let nDotT = max(abs(wiDotN), 1e-5);
    let pdfTransApprox = nDotT * INV_PI;
    return max(transProb * pdfTransApprox, 1e-8);
  }
  let nDotL = max(wiDotN, 0.0);
  if (nDotL <= 1e-5) {
    return 0.0;
  }
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let pdfSpec = d * nDotH / max(4.0 * vDotH, 1e-6);
  let pdfDiff = nDotL * INV_PI;
  return diffProb * pdfDiff + specProb * pdfSpec;
}
`;
