/**
 * BSDF -> emissive connection contributions (the BSDF-direction half of MIS).
 *
 *  - `bsdfAreaLightConnectionContribution` — BSDF-sampled ray intersected
 *    against every rect/mesh area light; nearest unoccluded contributes.
 *  - `bsdfEnvironmentConnectionContribution` — BSDF-sampled ray against the
 *    environment radiance + PDF.
 *
 * Both use Veach 1997 power-heuristic MIS weights vs. the light-sampling
 * branches in `main.wgsl.ts`. Sum-MIS over all lights is unbiased (D9
 * decision); the per-light loops are O(N_lights) acceptable for ≤ 8 lights.
 */
export const PT_WEBGPU_LIGHT_EMISSIVE_WGSL = /* wgsl */ `
fn bsdfAreaLightConnectionContribution(
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughputAtVertex: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) {
    return vec3f(0.0);
  }
  let bsdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
  if (bsdfPdf <= 1e-6) {
    return vec3f(0.0);
  }
  // Sum MIS over all area lights: iterate every rect and mesh light, keep the
  // closest unoccluded hit. Cost is O(N_lights) intersection tests — acceptable
  // for prototype scenes with ≤ 8 lights (D9 decision).
  // Ref: Veach 1997 Ch. 9 — sum-MIS is unbiased; choosing the closest hit along
  //      the BSDF-sampled direction is correct because the sample is a direction,
  //      not a point, so only the nearest light along that direction contributes.
  let offsetOrigin = hitPos + normal * 1e-3;
  var bestDist = INFINITY;
  var bestLightPdf = 0.0;
  var bestEmission = vec3f(0.0);
  for (var li = 0u; li < params.rectAreaLightCount; li = li + 1u) {
    var rectDist = INFINITY;
    var rectPdf = 0.0;
    if (intersectRectAreaLightRay(li, offsetOrigin, wi, &rectDist, &rectPdf)) {
      let shadowRay = Ray(offsetOrigin, wi);
      if (!traceAny(shadowRay, 1e-4, max(rectDist - 2e-3, 1e-3)) && rectDist < bestDist) {
        bestDist = rectDist;
        bestLightPdf = rectPdf;
        bestEmission = rectAreaLights[li * 4u + 3u].rgb;
      }
    }
  }
  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
    var meshDist = INFINITY;
    var meshPdf = 0.0;
    if (intersectMeshAreaLightRay(mi, offsetOrigin, wi, &meshDist, &meshPdf)) {
      let shadowRay = Ray(offsetOrigin, wi);
      if (!traceAny(shadowRay, 1e-4, max(meshDist - 2e-3, 1e-3)) && meshDist < bestDist) {
        bestDist = meshDist;
        bestLightPdf = meshPdf;
        bestEmission = meshAreaLights[mi * 4u + 3u].rgb;
      }
    }
  }
  if (bestDist >= INFINITY || bestLightPdf <= 1e-6) {
    return vec3f(0.0);
  }
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  let misWeight = powerHeuristic(bsdfPdf, bestLightPdf);
  return throughputAtVertex * brdf * nDotL * bestEmission * misWeight / max(bsdfPdf, 1e-6);
}

fn bsdfEnvironmentConnectionContribution(
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughputAtVertex: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return vec3f(0.0); }
  let bsdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
  if (bsdfPdf <= 1e-6) { return vec3f(0.0); }
  let shadowRay = Ray(hitPos + normal * 1e-3, wi);
  if (traceAny(shadowRay, 1e-4, INFINITY)) { return vec3f(0.0); }
  let envPdf = environmentPdf(wi);
  let envColor = sampleEnvironmentColor(wi);
  let misWeight = powerHeuristic(bsdfPdf, envPdf);
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  return throughputAtVertex * brdf * nDotL * envColor * misWeight / max(bsdfPdf, 1e-6);
}
`;
