import { PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL } from './connectCore.wgsl.js';

/**
 * Lite connect module — procedural sky only (no HDRI / area-light storage buffers).
 * Used when the adapter cannot bind the full trace pass layout.
 */
export const PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL = /* wgsl */ `
${PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL}

fn hasEnvironmentMap() -> bool {
  return false;
}

fn environmentDimensions() -> vec2u {
  return vec2u(0u, 0u);
}

fn sampleEnvironmentColor(dir: vec3f) -> vec3f {
  return sampleSky(dir);
}

fn environmentPdf(dir: vec3f) -> f32 {
  _ = dir;
  return 1.0 / (4.0 * PI);
}

fn sampleEnvironmentImportance(rng: ptr<function, u32>) -> BsdfSample {
  var result: BsdfSample;
  result.wi = vec3f(0.0, 1.0, 0.0);
  result.value = vec3f(0.0);
  result.pdf = 0.0;
  let xi = vec2f(rand_f32(rng), rand_f32(rng));
  let dir = uniformSphere(xi);
  result.wi = dir;
  result.value = sampleSky(dir);
  result.pdf = 1.0 / (4.0 * PI);
  return result;
}

fn bsdfAreaLightConnectionContribution(
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  throughputAtVertex: vec3f,
) -> vec3f {
  _ = hitPos;
  _ = normal;
  _ = wo;
  _ = wi;
  _ = baseColor;
  _ = roughness;
  _ = metallic;
  _ = transmission;
  _ = ior;
  _ = throughputAtVertex;
  return vec3f(0.0);
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
  ior: f32,
  throughputAtVertex: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return vec3f(0.0); }
  let bsdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
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
