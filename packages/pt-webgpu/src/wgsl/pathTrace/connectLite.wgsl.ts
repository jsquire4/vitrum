import { PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL } from './connectCore.wgsl.js';

/**
 * Lite connect module — B12: texture-based HDRI env sampling + BSDF env
 * connection.  Uses the `liteEnvTex` (binding 12) and `liteEnvCdfTex` (binding 13)
 * texture_2d<f32> slots added to the lite group-0 layout.  `textureLoad` is used
 * for all access (no sampler needed).
 *
 * When `params.hasEnvironmentMap == 0` (no HDRI / procedural-sky with CDF) the
 * textures are 1×1 black placeholders — `hasEnvironmentMap()` returns false and
 * the code returns black no-environment radiance. Procedural sky is CPU-baked to
 * the HDRI/CDF path before this shader runs.
 *
 * Area-light BSDF→light MIS is paired with the NEE loop in kernelLite: the NEE
 * half samples a point on the packed rect/disc light, and this module intersects
 * the BSDF-sampled direction against the same `liteLightTex` records.  That keeps
 * lite rect/disc records in the same matched Veach MIS family as the full tier
 * without binding the full rectAreaLights storage buffer.
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
// Returns (rgb=radiance, a=pdf_per_sr).  Falls back to black when the map is
// absent (hasEnvironmentMap()=false).
fn liteEnvLookup(dir: vec3f) -> vec4f {
  if (!hasEnvironmentMap()) {
    return vec4f(0.0);
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return vec4f(0.0);
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
  return lk.rgb;
}

fn environmentPdf(dir: vec3f) -> f32 {
  let lk = liteEnvLookup(dir);
  if (lk.a <= 0.0) { return 0.0; }
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

fn liteRectLightBase() -> u32 {
  return params.directionalLightCount * 2u + params.pointLightCount * 3u + params.spotLightCount * 4u;
}

// Intersect the BSDF-sampled ray against a packed lite rect/disc record.
// Layout mirrors kernelLite.wgsl.ts: rpos, u axis, v axis, emission+shape.
fn intersectLiteRectAreaLightRay(li: u32, rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let rb = liteRectLightBase() + li * 4u;
  let rectPos = textureLoad(liteLightTex, vec2i(i32(rb), 0), 0).xyz;
  let uAxis = textureLoad(liteLightTex, vec2i(i32(rb + 1u), 0), 0).xyz;
  let vAxis = textureLoad(liteLightTex, vec2i(i32(rb + 2u), 0), 0).xyz;
  let rshape = textureLoad(liteLightTex, vec2i(i32(rb + 3u), 0), 0);
  let isDisc = abs(rshape.w - 1.0) < 0.5;
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
  let inside = select(
    abs(uCoord) <= 1.0 && abs(vCoord) <= 1.0,
    uCoord * uCoord + vCoord * vCoord <= 1.0,
    isDisc,
  );
  if (!inside) {
    return false;
  }
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = select(
    max(4.0 * length(cross(uAxis, vAxis)), 1e-6),
    max(PI * dot(uAxis, uAxis), 1e-6),
    isDisc,
  );
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
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  throughputAtVertex: vec3f,
  heroLambda: f32,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) {
    return vec3f(0.0);
  }
  let bsdfPdf = brdfDirectionalPdfFullSampled(
    baseColor, roughness, metallic, transmission, ior, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    0.0, 0.0,
  );
  if (bsdfPdf <= 1e-6) {
    return vec3f(0.0);
  }

  let offsetOrigin = hitPos + normal * 1e-3;
  var bestDist = INFINITY;
  var bestLightPdf = 0.0;
  var bestEmission = vec3f(0.0);
  for (var li = 0u; li < params.rectAreaLightCount; li = li + 1u) {
    var rectDist = INFINITY;
    var rectPdf = 0.0;
    if (intersectLiteRectAreaLightRay(li, offsetOrigin, wi, &rectDist, &rectPdf)) {
      let rb = liteRectLightBase() + li * 4u;
      let rectShadowDisabled = textureLoad(liteLightTex, vec2i(i32(rb), 0), 0).w > 0.5;
      let shadowRay = Ray(offsetOrigin, wi);
      if ((rectShadowDisabled || !traceAny(shadowRay, 1e-4, max(rectDist - 2e-3, 1e-3))) && rectDist < bestDist) {
        bestDist = rectDist;
        bestLightPdf = rectPdf;
        bestEmission = textureLoad(liteLightTex, vec2i(i32(rb + 3u), 0), 0).rgb;
      }
    }
  }
  if (bestDist >= INFINITY || bestLightPdf <= 1e-6) {
    return vec3f(0.0);
  }

  let brdf = evaluateBrdfFull(
    baseColor, roughness, metallic, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    0.0, 0.0,
  );
  let emitOut = select(bestEmission, spectralEmissionAtHero(bestEmission, heroLambda), params.spectralEnabled != 0u);
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
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  throughputAtVertex: vec3f,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return vec3f(0.0); }
  let bsdfPdf = brdfDirectionalPdfFullSampled(
    baseColor, roughness, metallic, transmission, ior, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    0.0, 0.0,
  );
  if (bsdfPdf <= 1e-6) { return vec3f(0.0); }
  let shadowRay = Ray(hitPos + normal * 1e-3, wi);
  if (traceAny(shadowRay, 1e-4, INFINITY)) { return vec3f(0.0); }
  let envPdf = environmentPdf(wi);
  let envColor = sampleEnvironmentColor(wi);
  let misWeight = powerHeuristic(bsdfPdf, envPdf);
  let brdf = evaluateBrdfFull(
    baseColor, roughness, metallic, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    0.0, 0.0,
  );
  return throughputAtVertex * brdf * nDotL * envColor * misWeight / max(bsdfPdf, 1e-6);
}
`;
