/**
 * Manifold Next-Event Estimation (MNEE) — caustic strategy mode 1.
 *
 *  - `perturbAroundDirection` — cone-perturbed direction sampler used by MNEE
 *    (also reused by other cone-based importance schemes).
 *  - `traceSpecularTransmissiveChain` — walks the path through specular
 *    transmissive surfaces (collected attenuation, exit position/direction).
 *  - `manifoldNeeContribution` — direct-light contribution that integrates
 *    over a perturbation cone, applies cone-PDF / BSDF-PDF MIS, and emits
 *    contributions only for paths whose exit direction aligns with the
 *    primary light (caustic visibility).
 *
 * Refs:
 *  - Hanika, Droske, Weidlich. "Manifold Next Event Estimation."
 *    Computer Graphics Forum 34(4) (EGSR 2015).
 *  - Veach 1997 Ch. 9 power-heuristic MIS for the cone/BSDF blend.
 */
export const PT_WEBGPU_MNEE_WGSL = /* wgsl */ `
fn perturbAroundDirection(baseDir: vec3f, xi: vec2f, coneAngle: f32) -> vec3f {
  var t: vec3f;
  var b: vec3f;
  buildOnb(baseDir, &t, &b);
  let cosThetaMin = cos(coneAngle);
  let cosTheta = mix(cosThetaMin, 1.0, xi.x);
  let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
  let phi = 2.0 * PI * xi.y;
  let local = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  return safe_normalize(local.x * t + local.y * b + local.z * baseDir);
}

fn traceSpecularTransmissiveChain(
  startPos: vec3f,
  startNormal: vec3f,
  startDir: vec3f,
  maxChain: u32,
  exitPos: ptr<function, vec3f>,
  exitDir: ptr<function, vec3f>,
  chainAttenuation: ptr<function, vec3f>,
) -> bool {
  var ray = Ray(startPos + startNormal * 1e-3, safe_normalize(startDir));
  var att = vec3f(1.0);
  for (var step = 0u; step < 8u; step = step + 1u) {
    if (step >= maxChain) {
      *exitPos = ray.origin;
      *exitDir = ray.direction;
      *chainAttenuation = att;
      return true;
    }
    let hit = traceClosest(ray, 1e-4, INFINITY);
    if (!hit.didHit) {
      *exitPos = ray.origin;
      *exitDir = ray.direction;
      *chainAttenuation = att;
      return true;
    }
    let matId = hitMaterialId(hit);
    let m0Index = matId * MATERIAL_VEC4_STRIDE;
    let m2Index = m0Index + 2u;
    let m0 = select(vec4f(1.0, 1.0, 1.0, 0.5), materials[m0Index], m0Index < arrayLength(&materials));
    let m2 = select(vec4f(0.0, 1.5, 0.0, 0.0), materials[m2Index], m2Index < arrayLength(&materials));
    let transmission = clamp(m2.x, 0.0, 1.0);
    if (transmission <= 1e-4) {
      return false;
    }
    let ior = clamp(m2.y, 1.0, 2.5);
    let hitPos = ray.origin + ray.direction * hit.dist;
    let frontFace = dot(ray.direction, hit.normal) < 0.0;
    let surfaceNormal = select(-hit.normal, hit.normal, frontFace);
    let eta = select(ior, 1.0 / ior, frontFace);
    let refr = refract(ray.direction, surfaceNormal, eta);
    let hasRefr = dot(refr, refr) > 1e-8;
    let nextDir = select(reflect(ray.direction, surfaceNormal), safe_normalize(refr), hasRefr);
    att = att * mix(vec3f(1.0), clamp(m0.rgb, vec3f(0.0), vec3f(1.0)), 0.2) * max(transmission, 0.05);
    if (max(att.r, max(att.g, att.b)) < 1e-4) {
      return false;
    }
    ray.origin = hitPos + nextDir * 1e-3;
    ray.direction = nextDir;
  }
  *exitPos = ray.origin;
  *exitDir = ray.direction;
  *chainAttenuation = att;
  return true;
}

fn manifoldNeeContribution(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughput: vec3f,
) -> vec3f {
  if (transmission <= 1e-4 || params.lightDir.w <= 1e-6) {
    return vec3f(0.0);
  }
  let mneeSteps = clamp(params.mneeMaxIterations, 1u, 8u);
  let maxChain = clamp(params.mneeMaxChainLength, 1u, 8u);
  let baseLightDir = safe_normalize(params.lightDir.xyz);
  let coneAngle = mix(0.01, 0.12, clamp(roughness, 0.0, 1.0));
  var contribution = vec3f(0.0);
  for (var step = 0u; step < 8u; step = step + 1u) {
    if (step >= mneeSteps) {
      break;
    }
    let jitter = vec2f(rand_f32(rng), rand_f32(rng));
    let candidateDir = perturbAroundDirection(baseLightDir, jitter, coneAngle);
    let nDotL = max(dot(normal, candidateDir), 0.0);
    if (nDotL <= 1e-5) {
      continue;
    }
    var exitPos = vec3f(0.0);
    var exitDir = vec3f(0.0, 1.0, 0.0);
    var chainAtt = vec3f(1.0);
    if (!traceSpecularTransmissiveChain(hitPos, normal, candidateDir, maxChain, &exitPos, &exitDir, &chainAtt)) {
      continue;
    }
    let align = max(dot(exitDir, baseLightDir), 0.0);
    if (align <= 0.75) {
      continue;
    }
    let visibilityRay = Ray(exitPos + exitDir * 1e-3, baseLightDir);
    if (traceAny(visibilityRay, 1e-4, INFINITY)) {
      continue;
    }
    let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, candidateDir);
    let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, candidateDir);
    let conePdf = 1.0 / max(2.0 * PI * (1.0 - cos(coneAngle)), 1e-6);
    let samplePdf = conePdf / f32(mneeSteps);
    let misWeight = powerHeuristic(samplePdf, brdfPdf);
    let lightRadiance = vec3f(params.lightDir.w) * align;
    contribution = contribution +
      throughput * chainAtt * brdf * nDotL * lightRadiance * misWeight / max(samplePdf, 1e-6);
  }
  return contribution;
}
`;
