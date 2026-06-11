import { PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL } from './connectCore.wgsl.js';

/**
 * Lite connect module — B12: texture-based HDRI env sampling + BSDF env
 * connection.  Uses the `liteEnvTex` (binding 12) and `liteEnvCdfTex` (binding 13)
 * texture_2d<f32> slots added to the lite group-0 layout.  `textureLoad` is used
 * for all access (no sampler needed).
 *
 * When `params.hasEnvironmentMap == 0` (no HDRI / procedural-sky with CDF) the
 * textures are 1×1 black placeholders — `hasEnvironmentMap()` returns false and
 * the code falls back to the procedural sky.
 *
 * Area-light BSDF→light MIS (`bsdfAreaLightConnectionContribution`) is a zero stub:
 * rect-area lights fire via the NEE loop in kernelLite only (analytic contribution),
 * not via the BSDF→light reconnection path (which would require iterating all rect
 * lights on a direction-sampled bounce — that would add a loop here that the lite
 * kernel isn't structured for, and the full-tier uses its own connect.wgsl.ts path
 * for that).  Area-light energy on the lite tier therefore comes from NEE only (same
 * estimator as point/spot lights).
 */
export const PT_WEBGPU_PATH_TRACE_CONNECT_LITE_WGSL = /* wgsl */ `
${PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL}

// B12 — lite-tier env presence: driven by the UBO hasEnvironmentMap flag + dims.
// The liteEnvTex / liteEnvCdfTex are 1×1 black placeholders when no HDRI is loaded.
fn hasEnvironmentMap() -> bool {
  return params.hasEnvironmentMap > 0u && params.environmentMapWidth > 0u;
}

fn environmentDimensions() -> vec2u {
  return vec2u(params.environmentMapWidth, params.environmentMapHeight);
}

// D9.13 — rotateYNeg / rotateYPos are now in connectCore.wgsl.ts (shared).

// B12 — look up a texel from the lite env radiance texture via textureLoad.
// Returns (rgb=radiance, a=pdf_per_sr).  Falls back to (sampleSky, 0) when
// the map is absent (hasEnvironmentMap()=false).
fn liteEnvLookup(dir: vec3f) -> vec4f {
  if (!hasEnvironmentMap()) {
    return vec4f(sampleSky(dir), 0.0);
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return vec4f(sampleSky(dir), 0.0);
  }
  let rotY = params.environmentTint.w;
  let lookupDir = rotateYNeg(dir, rotY);
  let phi = atan2(lookupDir.z, lookupDir.x);
  let theta = acos(clamp(lookupDir.y, -1.0, 1.0));
  let u = fract(phi * INV_2PI + 0.5);
  let v = clamp(theta * INV_PI, 0.0, 0.999999);
  let x = min(u32(floor(u * f32(dims.x))), dims.x - 1u);
  let y = min(u32(floor(v * f32(dims.y))), dims.y - 1u);
  let texel = textureLoad(liteEnvTex, vec2i(i32(x), i32(y)), 0);
  return vec4f(texel.rgb * max(params.environmentHdriIntensity, 0.0), max(texel.a, 1e-8));
}

fn sampleEnvironmentColor(dir: vec3f) -> vec3f {
  let lk = liteEnvLookup(dir);
  if (lk.a <= 0.0) { return sampleSky(dir); }
  return lk.rgb;
}

fn environmentPdf(dir: vec3f) -> f32 {
  let lk = liteEnvLookup(dir);
  if (lk.a <= 0.0) { return 1.0 / (4.0 * PI); }
  return max(lk.a, 1e-8);
}

// B12 — importance sample the lite env CDF via binary search over liteEnvCdfTex.
// The CDF is stored as a W×H texture: texel at (x, y).r = cdf[(y*W + x) + 1].
// cdf[0] = 0 is implicit.  Returns pdf ≤ 0 on failure (no env or empty CDF).
fn sampleEnvironmentImportance(rng: ptr<function, u32>) -> BsdfSample {
  var result: BsdfSample;
  result.wi = vec3f(0.0, 1.0, 0.0);
  result.value = vec3f(0.0);
  result.pdf = 0.0;

  if (!hasEnvironmentMap()) {
    // Procedural-sky fallback: uniform-sphere sample + sky eval.
    let xi = vec2f(rand_f32(rng), rand_f32(rng));
    let dir = uniformSphere(xi);
    result.wi = dir;
    result.value = sampleSky(dir);
    result.pdf = 1.0 / (4.0 * PI);
    return result;
  }

  let dims = environmentDimensions();
  let count = dims.x * dims.y;
  if (count == 0u) { return result; }

  // Binary search in the CDF texture.  cdf[0] = 0 (implicit), cdf[i] in texel ((i-1)%W, (i-1)/W).
  let xi = rand_f32(rng);
  var lo = 0u;
  var hi = count;
  for (var _i = 0u; _i < 32u; _i = _i + 1u) {
    if (lo + 1u >= hi) { break; }
    let mid = (lo + hi) >> 1u;
    // cdf[mid] → texel (mid-1 % W, mid-1 / W) for mid ≥ 1; cdf[0]=0 (guard).
    var cdfMid = 0.0;
    if (mid > 0u) {
      let ti = mid - 1u;
      let cx = i32(ti % dims.x);
      let cy = i32(ti / dims.x);
      cdfMid = textureLoad(liteEnvCdfTex, vec2i(cx, cy), 0).r;
    }
    if (cdfMid <= xi) { lo = mid; } else { hi = mid; }
  }
  let idx = min(lo, count - 1u);
  let x = idx % dims.x;
  let y = idx / dims.x;
  let u = (f32(x) + 0.5) / f32(dims.x);
  let v = (f32(y) + 0.5) / f32(dims.y);
  let phi = (u - 0.5) * (2.0 * PI);
  let theta = v * PI;
  let sinTheta = sin(theta);
  let mapDir = vec3f(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
  let texel = textureLoad(liteEnvTex, vec2i(i32(x), i32(y)), 0);
  let rotY = params.environmentTint.w;
  result.wi = safe_normalize(rotateYPos(mapDir, rotY));
  result.value = texel.rgb * max(params.environmentHdriIntensity, 0.0);
  result.pdf = max(texel.a, 1e-8);
  return result;
}

// Area-light BSDF→light MIS: zero stub on the lite tier.
// Rect-area lights are sampled via the NEE loop in kernelLite only (analytic NEE
// from the liteLightTex).  The BSDF-sampled direction that happens to hit a rect
// light is not reconnected here; kernelLite therefore uses a one-sided area-NEE
// estimator for rect/disc records instead of applying an unmatched MIS weight.
// Implementing the rect intersect here would require iterating all rect lights
// for every bounce on the lite tier and is deferred.
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
