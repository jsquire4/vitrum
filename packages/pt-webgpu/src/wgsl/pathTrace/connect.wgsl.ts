import { PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL } from './connectCore.wgsl.js';

/**
 * Connect module — light-direction sampling, MIS connections, and the
 * environment-map / HDRI helpers consumed by both the main kernel and the
 * BSDF→light MIS contributions.
 *
 * Bundled here:
 *  - Procedural sky + HDRI bookkeeping:
 *      - `sampleSky` — analytic sky fallback (sun glow + zenith tint)
 *      - `hasEnvironmentMap`, `environmentDimensions` — UBO/binding guards
 *      - `sampleEnvironmentColor` — equirect lookup with sky fallback
 *      - `environmentPdf` — equirect importance PDF
 *      - `sampleEnvironmentImportance` — RNG-driven HDRI importance sample
 *  - Area-light directional intersectors used by BSDF→light MIS:
 *      - `intersectRectAreaLightRay`
 *      - `intersectMeshAreaLightRay`
 *  - BSDF-direction MIS connection contributions:
 *      - `bsdfAreaLightConnectionContribution`
 *      - `bsdfEnvironmentConnectionContribution`
 *
 * Depends on FrameParams bindings (rectAreaLights, meshAreaLights,
 * environmentMap*) from `material.wgsl.ts`, evaluateBrdf + brdfDirectionalPdf
 * from `bsdf.wgsl.ts`, and traceAny from `intersection.wgsl.ts`.
 */
export const PT_WEBGPU_PATH_TRACE_CONNECT_WGSL = /* wgsl */ `
${PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL}

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

struct EnvironmentLookup {
  color: vec3f,
  pdf: f32,
};

// H6 (2026-06-09) — Y-axis rotation helpers for the HDRI environment dome.
//
// Convention (matches HdriEnvironment.rotationY JSDoc and the pt-webgl2 mat4):
//   A CCW rotationY of the environment dome means a world-space direction d
//   looks up the UNROTATED map at rotateY(d, -rotationY).
//   The CDF-sampled direction (in unrotated-map space) is rotated by +rotationY
//   to yield the world-space light direction.
//
// rotationY is stored in params.environmentTint.w (the previously-zero .w lane).
// rotationY = 0 => cos=1, sin=0 => both helpers return dir unchanged (zero-rotation
// invariant: output is byte-identical to the pre-H6 code).
//
// rotateYNeg(dir, rotY) = RY(-rotY) * dir:
//   x' =  cos(rotY)*x - sin(rotY)*z
//   y' =  y
//   z' =  sin(rotY)*x + cos(rotY)*z
fn rotateYNeg(dir: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY);
  let s = sin(rotY);
  return vec3f(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);
}

// rotateYPos(dir, rotY) = RY(+rotY) * dir:
//   x' =  cos(rotY)*x + sin(rotY)*z
//   y' =  y
//   z' = -sin(rotY)*x + cos(rotY)*z
fn rotateYPos(dir: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY);
  let s = sin(rotY);
  return vec3f(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
}

fn environmentLookup(dir: vec3f) -> EnvironmentLookup {
  if (!hasEnvironmentMap()) {
    return EnvironmentLookup(sampleSky(dir), 1.0 / (4.0 * PI));
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return EnvironmentLookup(sampleSky(dir), 1.0 / (4.0 * PI));
  }
  // H6: rotate the lookup direction by -rotationY so the unrotated map is
  // sampled at the correct texel for a CCW-rotated environment dome.
  // rotationY = 0 → rotateYNeg is identity → zero-rotation invariant.
  let rotY = params.environmentTint.w;
  let lookupDir = rotateYNeg(dir, rotY);
  let phi = atan2(lookupDir.z, lookupDir.x);
  let theta = acos(clamp(lookupDir.y, -1.0, 1.0));
  let u = fract(phi * INV_2PI + 0.5);
  let v = clamp(theta * INV_PI, 0.0, 0.999999);
  let x = min(u32(floor(u * f32(dims.x))), dims.x - 1u);
  let y = min(u32(floor(v * f32(dims.y))), dims.y - 1u);
  let idx = y * dims.x + x;
  if (idx >= arrayLength(&environmentMapTexels)) {
    return EnvironmentLookup(sampleSky(dir), 1.0 / (4.0 * PI));
  }
  let texel = environmentMapTexels[idx];
  return EnvironmentLookup(
    texel.rgb * max(params.environmentHdriIntensity, 0.0),
    max(texel.w, 1e-8),
  );
}

fn sampleEnvironmentColor(dir: vec3f) -> vec3f {
  return environmentLookup(dir).color;
}

fn environmentPdf(dir: vec3f) -> f32 {
  return environmentLookup(dir).pdf;
}

// Environment-map importance sampler. Returns a BsdfSample where
// .value is the emitted radiance along .wi and .pdf <= 0 signals failure
// (no environment map, or empty CDF). Same RNG consumption (one rand_f32 call)
// and identical sampled direction / radiance / pdf as the prior pointer-out
// signature it replaces.
fn sampleEnvironmentImportance(rng: ptr<function, u32>) -> BsdfSample {
  var result: BsdfSample;
  result.wi = vec3f(0.0, 1.0, 0.0);
  result.value = vec3f(0.0);
  result.pdf = 0.0;
  if (!hasEnvironmentMap()) {
    return result;
  }
  let dims = environmentDimensions();
  let count = dims.x * dims.y;
  if (count == 0u || arrayLength(&environmentMapCdf) < count + 1u) {
    return result;
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
  // dir is in unrotated-map space (the CDF is built from the unrotated map).
  let mapDir = vec3f(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
  let texel = environmentMapTexels[idx];
  // H6: rotate the map-space sample direction by +rotationY to get the world-space
  // light direction for a CCW-rotated environment dome.
  // rotationY = 0 → rotateYPos is identity → zero-rotation invariant.
  let rotY = params.environmentTint.w;
  result.wi = safe_normalize(rotateYPos(mapDir, rotY));
  result.value = texel.rgb * max(params.environmentHdriIntensity, 0.0);
  result.pdf = max(texel.w, 1e-8);
  return result;
}

// Intersect the BSDF sample ray against rect area light index li.
// Accepts a light index so BSDF->light MIS can check all lights.
// Ref: Veach, E. PhD thesis, Stanford 1997, Ch. 9 -- power-heuristic MIS;
//      sum-MIS over all lights is unbiased (D9 decision).
fn intersectRectAreaLightRay(li: u32, rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let rb = li * 4u;
  let rectPos = rectAreaLights[rb].xyz;
  let uAxis = rectAreaLights[rb + 1u].xyz;
  let vAxis = rectAreaLights[rb + 2u].xyz;
  let lightNormal = safe_normalize(cross(uAxis, vAxis));
  let denom = dot(lightNormal, rayDir);
  if (abs(denom) < 1e-6) {
    return false;
  }
  let t = dot(lightNormal, rectPos - rayOrigin) / denom;
  if (t <= 1e-4) {
    return false;
  }
  let p = rayOrigin + rayDir * t;
  let rel = p - rectPos;
  let uLen2 = max(dot(uAxis, uAxis), 1e-6);
  let vLen2 = max(dot(vAxis, vAxis), 1e-6);
  let uCoord = dot(rel, uAxis) / uLen2;
  let vCoord = dot(rel, vAxis) / vLen2;
  if (abs(uCoord) > 1.0 || abs(vCoord) > 1.0) {
    return false;
  }
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = max(4.0 * length(cross(uAxis, vAxis)), 1e-6);
  *distOut = t;
  *lightPdfOut = (t * t) / max(cosLight * area, 1e-6);
  return true;
}

// Intersect the BSDF sample ray against mesh area light index li.
// Accepts a light index so BSDF->light MIS can check all lights.
// Ref: Veach 1997 Ch. 9 -- sum-MIS over all lights (D9 decision).
fn intersectMeshAreaLightRay(li: u32, rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let mb = li * 4u;
  let a = meshAreaLights[mb].xyz;
  let b = meshAreaLights[mb + 1u].xyz;
  let c = meshAreaLights[mb + 2u].xyz;
  let t = intersectTriangle(rayOrigin, rayDir, a, b, c);
  if (t <= 1e-4 || t >= INFINITY) {
    return false;
  }
  let lightNormal = safe_normalize(cross(b - a, c - a));
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
  *distOut = t;
  *lightPdfOut = (t * t) / max(cosLight * area, 1e-6);
  return true;
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
  heroLambda: f32,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) {
    return vec3f(0.0);
  }
  let bsdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
  if (bsdfPdf <= 1e-6) {
    return vec3f(0.0);
  }
  // Sum MIS over all area lights: iterate every rect and mesh light, keep the
  // closest unoccluded hit. Cost is O(N_lights) intersection tests — acceptable
  // for experimental scenes with ≤ 8 lights (D9 decision).
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
  // A3 — spectralize the BSDF-connection emission at the hero λ in spectral mode,
  // matching the NEE half (kernel.wgsl.ts §631/676) so both halves of the MIS pair
  // use the same emission model for chromatic emitters. RGB mode: byte-identical.
  let emitOut = select(bestEmission, spectralEmissionAtHero(bestEmission, heroLambda), params.spectralEnabled != 0u);
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  let misWeight = powerHeuristic(bsdfPdf, bestLightPdf);
  return throughputAtVertex * brdf * nDotL * emitOut * misWeight / max(bsdfPdf, 1e-6);
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
  heroLambda: f32,
  matId: u32,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return vec3f(0.0); }
  let bsdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
  if (bsdfPdf <= 1e-6) { return vec3f(0.0); }
  let shadowRay = Ray(hitPos + normal * 1e-3, wi);
  if (traceAny(shadowRay, 1e-4, INFINITY)) { return vec3f(0.0); }
  let env = environmentLookup(wi);
  let misWeight = powerHeuristic(bsdfPdf, env.pdf);
  let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
  // A3 — spectralize the env connection at the hero λ in spectral mode, matching the
  // NEE miss-shader path (kernel.wgsl.ts §431) and the NEE env branch (§724).
  // D3 — apply per-material envMapIntensity to the BSDF-connection env term, matching
  // the NEE env branch (kernel.wgsl.ts §723-724). envMapIntensity == 1.0 (default) →
  // envScale == 1.0 → byte-identical. Non-unit values scale BOTH halves identically so
  // the converged env contribution is consistent across the two MIS strategies.
  let envScale = materialEnvMapIntensity(matId);
  let envColorOut = select(env.color, spectralEmissionAtHero(env.color, heroLambda), params.spectralEnabled != 0u) * envScale;
  return throughputAtVertex * brdf * nDotL * envColorOut * misWeight / max(bsdfPdf, 1e-6);
}
`;
