/**
 * Environment / sky lighting — procedural sky fallback + HDRI lookup +
 * importance sampling against the CDF uploaded by the host.
 *
 *  - `sampleSky` — Hosek-Lilley-flavoured procedural sky (legacy fallback)
 *  - `hasEnvironmentMap` / `environmentDimensions` — UBO probes
 *  - `sampleEnvironmentColor` / `environmentPdf` — radiance + PDF lookup
 *  - `sampleEnvironmentImportance` — CDF-binary-search importance sample
 *
 * The HDRI presence + dimensions live in dedicated u32 FrameParams fields
 * (post-D9 cleanup); legacy "flag-in-.w" packing has been removed.
 */
export const PT_WEBGPU_LIGHT_ENVIRONMENT_WGSL = /* wgsl */ `
fn sampleSky(dir: vec3f) -> vec3f {
  let t = 0.5 * (dir.y + 1.0);
  var sky = mix(vec3f(0.06, 0.08, 0.12), vec3f(0.45, 0.62, 0.95), clamp(t, 0.0, 1.0));
  let sunDir = safe_normalize(params.environmentSun.xyz);
  let sunGlow = pow(max(0.0, dot(dir, sunDir)), 512.0) * params.environmentSun.w;
  sky = sky + vec3f(1.0, 0.95, 0.85) * sunGlow;
  return sky * params.environmentTint.rgb;
}

// HDRI environment presence + dimensions are now dedicated u32 fields in
// FrameParams (hasEnvironmentMap / environmentMapWidth / environmentMapHeight).
// Previously these lived in the .w lanes of meshAreaTri{B,C} / environmentTint —
// a space-saving hack that has been removed.
// The second clause below guards the legacy "flag set but dims=0" edge case:
// if the host writes hasEnvironmentMap=1 but never uploads a non-zero map,
// we still fall back to the procedural sky.
fn hasEnvironmentMap() -> bool {
  return params.hasEnvironmentMap > 0u && params.environmentMapWidth > 0u;
}

fn environmentDimensions() -> vec2u {
  return vec2u(params.environmentMapWidth, params.environmentMapHeight);
}

fn sampleEnvironmentColor(dir: vec3f) -> vec3f {
  if (!hasEnvironmentMap()) {
    return sampleSky(dir);
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return sampleSky(dir);
  }
  let phi = atan2(dir.z, dir.x);
  let theta = acos(clamp(dir.y, -1.0, 1.0));
  let u = fract(phi * INV_2PI + 0.5);
  let v = clamp(theta * INV_PI, 0.0, 0.999999);
  let x = min(u32(floor(u * f32(dims.x))), dims.x - 1u);
  let y = min(u32(floor(v * f32(dims.y))), dims.y - 1u);
  let idx = y * dims.x + x;
  if (idx >= arrayLength(&environmentMapTexels)) {
    return sampleSky(dir);
  }
  let texel = environmentMapTexels[idx];
  return texel.rgb * max(params.environmentSun.w, 0.0);
}

fn environmentPdf(dir: vec3f) -> f32 {
  if (!hasEnvironmentMap()) {
    return 1.0 / (4.0 * PI);
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return 1.0 / (4.0 * PI);
  }
  let phi = atan2(dir.z, dir.x);
  let theta = acos(clamp(dir.y, -1.0, 1.0));
  let u = fract(phi * INV_2PI + 0.5);
  let v = clamp(theta * INV_PI, 0.0, 0.999999);
  let x = min(u32(floor(u * f32(dims.x))), dims.x - 1u);
  let y = min(u32(floor(v * f32(dims.y))), dims.y - 1u);
  let idx = y * dims.x + x;
  if (idx >= arrayLength(&environmentMapTexels)) {
    return 1.0 / (4.0 * PI);
  }
  return max(environmentMapTexels[idx].w, 1e-8);
}

fn sampleEnvironmentImportance(rng: ptr<function, u32>, outDir: ptr<function, vec3f>, outColor: ptr<function, vec3f>, outPdf: ptr<function, f32>) -> bool {
  if (!hasEnvironmentMap()) {
    return false;
  }
  let dims = environmentDimensions();
  let count = dims.x * dims.y;
  if (count == 0u || arrayLength(&environmentMapCdf) < count + 1u) {
    return false;
  }
  let xi = rand_f32(rng);
  var lo = 0u;
  var hi = count;
  loop {
    if (lo + 1u >= hi) { break; }
    let mid = (lo + hi) >> 1u;
    if (environmentMapCdf[mid] <= xi) { lo = mid; } else { hi = mid; }
  }
  let idx = min(lo, count - 1u);
  let x = idx % dims.x;
  let y = idx / dims.x;
  let u = (f32(x) + 0.5) / f32(dims.x);
  let v = (f32(y) + 0.5) / f32(dims.y);
  let phi = (u - 0.5) * (2.0 * PI);
  let theta = v * PI;
  let sinTheta = sin(theta);
  let dir = vec3f(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
  let texel = environmentMapTexels[idx];
  *outDir = safe_normalize(dir);
  *outColor = texel.rgb * max(params.environmentSun.w, 0.0);
  *outPdf = max(texel.w, 1e-8);
  return true;
}
`;
