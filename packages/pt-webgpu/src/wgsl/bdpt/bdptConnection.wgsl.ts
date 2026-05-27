/**
 * BDPT eye↔light connection (Veach 1997 §10.3) — WebGPU port of fork GLSL.
 * @see packages/three-gpu-pathtracer/.../bdpt_connection.glsl.js
 */
export const PT_WEBGPU_BDPT_CONNECTION_WGSL = /* wgsl */ `
const BDPT_KIND_INVALID: f32 = 3.0;
const BDPT_CONTRIBUTION_CLAMP: f32 = 100.0;

fn bdptGeometricTerm(posX: vec3f, nX: vec3f, posY: vec3f, nY: vec3f) -> f32 {
  let d = posY - posX;
  let dist2 = dot(d, d);
  if (dist2 <= 1e-12) {
    return 0.0;
  }
  let w = d * inverseSqrt(dist2);
  let cosX = abs(dot(nX, w));
  let cosY = abs(dot(nY, -w));
  return (cosX * cosY) / dist2;
}

fn bdptMISWeight2(p_s: f32, p_alt: f32) -> f32 {
  if (p_s <= 0.0) {
    return 0.0;
  }
  let p2s = p_s * p_s;
  let p2alt = p_alt * p_alt;
  let denom = p2s + p2alt;
  return select(0.0, p2s / denom, denom > 0.0);
}

fn evaluateBdptConnection(
  eyePos: vec3f,
  eyeNormal: vec3f,
  eyeWo: vec3f,
  eyeThroughput: vec3f,
  eyePdfFwd: f32,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  lightVtxIdx: i32,
) -> vec3f {
  let lv0 = textureLoad(bdptLightPath, vec2i(lightVtxIdx, 0), 0);
  let lv1 = textureLoad(bdptLightPath, vec2i(lightVtxIdx, 1), 0);
  let lv2 = textureLoad(bdptLightPath, vec2i(lightVtxIdx, 2), 0);
  if (lv0.w == BDPT_KIND_INVALID) {
    return vec3f(0.0);
  }
  let eyeIsSpecular = transmission > 0.5 && roughness < 0.05;
  if (eyeIsSpecular) {
    return vec3f(0.0);
  }
  let lightPos = lv0.xyz;
  let lightNormal = lv1.xyz;
  let lightPdfFwd = lv1.w;
  let lightThroughput = lv2.xyz;
  let toLight = lightPos - eyePos;
  let dist = length(toLight);
  if (dist < 1e-4) {
    return vec3f(0.0);
  }
  let connDir = toLight / dist;
  let gTerm = bdptGeometricTerm(eyePos, eyeNormal, lightPos, lightNormal);
  if (gTerm <= 0.0) {
    return vec3f(0.0);
  }
  let shadowRay = Ray(eyePos + eyeNormal * 1e-3, connDir);
  if (traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
    return vec3f(0.0);
  }
  let eyeBrdf = evaluateBrdf(baseColor, roughness, metallic, eyeNormal, eyeWo, connDir);
  let cosEye = max(dot(eyeNormal, connDir), 0.0);
  if (cosEye <= 0.0) {
    return vec3f(0.0);
  }
  let eyeBsdfCosTheta = eyeBrdf * cosEye;
  let eyePdfFwdUse = max(eyePdfFwd, 1e-8);
  let cosLight = max(dot(lightNormal, -connDir), 0.0);
  let lightBsdfCosTheta = vec3f(cosLight / PI);
  let p_s = lightPdfFwd * gTerm * eyePdfFwdUse;
  let p_alt = eyePdfFwdUse * gTerm;
  let misW = bdptMISWeight2(p_s, p_alt);
  if (misW <= 0.0) {
    return vec3f(0.0);
  }
  var contribution = lightThroughput * lightBsdfCosTheta * gTerm * eyeBsdfCosTheta * misW;
  contribution = contribution * eyeThroughput;
  if (any(isNan(contribution)) || any(isInf(contribution))) {
    return vec3f(0.0);
  }
  return clamp(contribution, vec3f(0.0), vec3f(BDPT_CONTRIBUTION_CLAMP));
}
`;
