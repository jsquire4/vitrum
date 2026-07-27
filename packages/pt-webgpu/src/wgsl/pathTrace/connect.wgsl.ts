import { PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL } from './connectCore.wgsl.js';

/**
 * Connect module — light-direction sampling, MIS connections, and the
 * environment-map / HDRI helpers consumed by both the main kernel and the
 * BSDF→light MIS contributions.
 *
 * Bundled here:
 *  - Procedural sky + HDRI bookkeeping:
 *      - `sampleSky` — legacy analytic sky helper retained in the shared core
 *      - `hasEnvironmentMap`, `environmentDimensions` — UBO/binding guards
 *      - `sampleEnvironmentColor` — equirect lookup with black no-env fallback
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
// The second clause below guards the legacy "flag set but dims=0" edge case.
// Procedural sky is CPU-baked into the HDRI path before this shader runs; if no
// map is present here, the authored environment is none or invalid and must
// contribute no radiance.
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

// D9.13 — rotateYNeg / rotateYPos moved to connectCore.wgsl.ts (shared by both tiers).
// Convention (matches HdriEnvironment.rotationY JSDoc and the pt-webgl2 mat4):
//   A CCW rotationY of the environment dome means a world-space direction d
//   looks up the UNROTATED map at rotateYNeg(d, rotationY).
//   The CDF-sampled direction (in unrotated-map space) is rotated by rotateYPos
//   to yield the world-space light direction.
// rotationY is stored in params.environmentTint.w.  rotationY=0 → identity.

fn environmentLookup(dir: vec3f) -> EnvironmentLookup {
  if (!hasEnvironmentMap()) {
    if (params.environmentSun.w > 0.0) {
      return EnvironmentLookup(sampleSky(dir), 0.25 * INV_PI);
    }
    return EnvironmentLookup(vec3f(0.0), 0.0);
  }
  let dims = environmentDimensions();
  if (dims.x == 0u || dims.y == 0u) {
    return EnvironmentLookup(vec3f(0.0), 0.0);
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
    return EnvironmentLookup(vec3f(0.0), 0.0);
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
fn sampleEnvironmentImportance(rng: ptr<function, PtRngState>) -> BsdfSample {
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

fn environmentImportanceSamplerReady() -> bool {
  if (!hasEnvironmentMap()) { return false; }
  let dims = environmentDimensions();
  let count = dims.x * dims.y;
  return count > 0u && arrayLength(&environmentMapCdf) >= count + 1u;
}

// Density of the proposal used by environment NEE. When the importance table
// is unavailable every transport family falls back to the same uniform-sphere
// proposal. This keeps p0/p1/p2+ densities comparable without a receiver-local
// normal hidden in the infinite-root light subpath.
fn environmentNeeProposalPdf(dir: vec3f, normal: vec3f) -> f32 {
  if (environmentImportanceSamplerReady()) {
    return max(environmentPdf(dir), 1e-8);
  }
  return 0.25 * INV_PI;
}

const MESH_AREA_LIGHT_VEC4_STRIDE: u32 = 7u;

fn meshAreaLightBase(index: u32) -> u32 {
  return index * MESH_AREA_LIGHT_VEC4_STRIDE;
}

fn meshAreaLightWeightsAtPoint(index: u32, point: vec3f) -> vec3f {
  let base = meshAreaLightBase(index);
  let a = meshAreaLights[base].xyz;
  let ab = meshAreaLights[base + 1u].xyz - a;
  let ac = meshAreaLights[base + 2u].xyz - a;
  let ap = point - a;
  let d00 = dot(ab, ab);
  let d01 = dot(ab, ac);
  let d11 = dot(ac, ac);
  let d20 = dot(ap, ab);
  let d21 = dot(ap, ac);
  let denom = d00 * d11 - d01 * d01;
  if (denom <= 0.0) { return vec3f(1.0, 0.0, 0.0); }
  let wb = (d11 * d20 - d01 * d21) / denom;
  let wc = (d00 * d21 - d01 * d20) / denom;
  return vec3f(1.0 - wb - wc, wb, wc);
}

// Evaluate the exact authored emissive texture at a sampled packed-light point.
// Rows 4..6 retain the source triangle's raw selected UVs, material descriptor,
// original world area, and unmodulated emitter Le. This reproduces the forward
// material sampler's transform, wrap, min/mag/mipmap policy and footprint LOD.
fn sampleMeshAreaLightRadiance(
  index: u32,
  weights: vec3f,
  worldPosition: vec3f,
) -> vec3f {
  let base = meshAreaLightBase(index);
  let averageAndShadow = meshAreaLights[base + 3u];
  let uvAB = meshAreaLights[base + 4u];
  let uvCAndMaterial = meshAreaLights[base + 5u];
  let materialIdPlusOne = uvCAndMaterial.z;
  if (materialIdPlusOne < 0.5) { return averageAndShadow.rgb; }
  let matId = u32(materialIdPlusOne - 1.0);
  let descriptorBase = matId * MATERIAL_TEX_VEC4_STRIDE;
  if (descriptorBase + 13u >= arrayLength(&materialTexDescriptors)) {
    return vec3f(0.0);
  }
  let layerIdx = i32(materialTexDescriptors[descriptorBase].w);
  if (layerIdx < 0) { return vec3f(0.0); }
  let rawA = uvAB.xy;
  let rawB = uvAB.zw;
  let rawC = uvCAndMaterial.xy;
  let rawUv = rawA * weights.x + rawB * weights.y + rawC * weights.z;
  let uvMeta = materialTexDescriptors[
    descriptorBase + MATERIAL_TEX_UV_EMISSIVE
  ];
  let uvScale = materialTexDescriptors[
    descriptorBase + MATERIAL_TEX_UV_EMISSIVE + 1u
  ];
  let c = cos(uvMeta.w);
  let s = sin(uvMeta.w);
  let transformUv = mat2x2f(
    uvScale.x * c, -uvScale.y * s,
    uvScale.x * s, uvScale.y * c,
  );
  let offset = uvMeta.yz;
  let uvA = transformUv * rawA + offset;
  let uvB = transformUv * rawB + offset;
  let uvC = transformUv * rawC + offset;
  let uv = transformUv * rawUv + offset;
  let wrapMode = materialTexDescriptors[descriptorBase + 13u].zw;
  let uvFitScale = materialTexDescriptors[descriptorBase + 7u].zw;
  let sourceBaseSize = materialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTexturesEmissive, 0)), uvFitScale,
  );
  let sourceMipCount = f32(materialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelArea = max(
    abs((uvB.x - uvA.x) * (uvC.y - uvA.y) -
        (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y,
    1.0,
  );
  let worldArea = uvCAndMaterial.w;
  if (worldArea <= 0.0) { return vec3f(0.0); }
  let cameraDistance = max(
    length(worldPosition - params.cameraPos.xyz), 1e-3,
  );
  let pixelsPerMeter =
    0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(
    log2(sqrt(texelArea) / projectedPixels),
    0.0,
    max(sourceMipCount - 1.0, 0.0),
  );
  let mipPolicy = materialTextureMipPolicy(
    descriptorBase, MATERIAL_TEX_MIP_EMISSIVE,
  );
  let policyLod = materialTexturePolicyLod(lod, sourceMipCount, mipPolicy);
  let filterPolicy = materialTextureFilterPolicy(
    descriptorBase, MATERIAL_TEX_MIP_EMISSIVE,
  );
  let filterMode = select(filterPolicy.x, filterPolicy.y, lod > 0.0);
  let texel = sampleMaterialEmissiveSourceRect(
    layerIdx, uv, sourceBaseSize, wrapMode,
    policyLod, filterMode, mipPolicy,
  );
  return meshAreaLights[base + 6u].rgb * texel.rgb;
}

// Intersect the BSDF sample ray against rect/disc area light index li.
// Reads the shape discriminator from emission.w: ≈ 0 → rect, ≈ 1 → analytic disc.
// Rect:  uCoord/vCoord ∈ [-1,1] box test; area = 4·|u×v|.
// Disc:  uCoord² + vCoord² ≤ 1 circle test; area = π·|u|² (|u| = radius).
// Ref: Veach, E. PhD thesis, Stanford 1997, Ch. 9 -- power-heuristic MIS;
//      sum-MIS over all lights is unbiased (D9 decision).
// Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
// RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
fn intersectRectAreaLightRay(li: u32, rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let rb = li * 4u;
  let rectPos = rectAreaLights[rb].xyz;
  let uAxis = rectAreaLights[rb + 1u].xyz;
  let vAxis = rectAreaLights[rb + 2u].xyz;
  let rshape = rectAreaLights[rb + 3u];
  let isDisc = abs(rshape.w - 1.0) < 0.5;
  let axisCross = cross(uAxis, vAxis);
  let axisCrossLen2 = dot(axisCross, axisCross);
  if (axisCrossLen2 <= 0.0) { return false; }
  let lightNormal = axisCross * inverseSqrt(axisCrossLen2);
  let denom = dot(lightNormal, rayDir);
  if (denom == 0.0) {
    return false;
  }
  let t = dot(lightNormal, rectPos - rayOrigin) / denom;
  if (t <= 1e-4) {
    return false;
  }
  let p = rayOrigin + rayDir * t;
  let rel = p - rectPos;
  let uLen2 = dot(uAxis, uAxis);
  let vLen2 = dot(vAxis, vAxis);
  if (uLen2 <= 0.0 || vLen2 <= 0.0) { return false; }
  let uCoord = dot(rel, uAxis) / uLen2;
  let vCoord = dot(rel, vAxis) / vLen2;
  // Containment test: disc uses circle (u²+v²≤1), rect uses square (|u|,|v|≤1).
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
  // Area formula: disc → π·r² (r = |uAxis|); rect → 4·|u×v|.
  let area = select(
    4.0 * sqrt(axisCrossLen2),
    PI * uLen2,
    isDisc,
  );
  if (area <= 0.0) { return false; }
  *distOut = t;
  *lightPdfOut = (t * t) / (cosLight * area);
  return true;
}

// Intersect the BSDF sample ray against mesh area light index li.
// Accepts a light index so BSDF->light MIS can check all lights.
// Ref: Veach 1997 Ch. 9 -- sum-MIS over all lights (D9 decision).
fn intersectMeshAreaLightRay(li: u32, rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let mb = meshAreaLightBase(li);
  let a = meshAreaLights[mb].xyz;
  let b = meshAreaLights[mb + 1u].xyz;
  let c = meshAreaLights[mb + 2u].xyz;
  let t = intersectTriangle(rayOrigin, rayDir, a, b, c);
  if (t <= 1e-4 || t >= INFINITY) {
    return false;
  }
  let triangleCross = cross(b - a, c - a);
  let triangleCrossLength = length(triangleCross);
  if (triangleCrossLength <= 0.0) { return false; }
  let lightNormal = triangleCross / triangleCrossLength;
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = 0.5 * triangleCrossLength;
  *distOut = t;
  *lightPdfOut = (t * t) / (cosLight * area);
  return true;
}

fn bsdfAreaLightConnectionContribution(
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  anisotropy: f32,
  anisotropyRotation: f32,
  throughputAtVertex: vec3f,
  heroLambda: f32,
  includeMeshAreaLights: bool,
) -> vec3f {
  let nDotL = abs(dot(normal, wi));
  if (nDotL <= 0.0) {
    return vec3f(0.0);
  }
  let bsdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation,
  );
  if (bsdfPdf <= 0.0) {
    return vec3f(0.0);
  }
  // Sum MIS over all area lights: iterate every rect/disc and, when allowed,
  // mesh light, keep the closest unoccluded hit. Cost is O(N_lights)
  // intersection tests — intended for scenes with no more than eight area lights
  // (D9 decision). ReSTIR-PT composite mode disables the mesh branch for
  // contributed pixels because the resolve already carries xs-on-emissive-mesh
  // radiance, while rect/disc analytic emitters cannot be reconnection vertices.
  // Ref: Veach 1997 Ch. 9 — sum-MIS is unbiased; choosing the closest hit along
  //      the BSDF-sampled direction is correct because the sample is a direction,
  //      not a point, so only the nearest light along that direction contributes.
  let offsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
  let offsetOrigin = hitPos + offsetNormal * 1e-3;
  var bestDist = INFINITY;
  var bestLightPdf = 0.0;
  var bestEmission = vec3f(0.0);
  for (var li = 0u; li < params.rectAreaLightCount; li = li + 1u) {
    var rectDist = INFINITY;
    var rectPdf = 0.0;
    if (intersectRectAreaLightRay(li, offsetOrigin, wi, &rectDist, &rectPdf)) {
      let shadowRay = Ray(offsetOrigin, wi);
      // SHADOW-01 — rectAreaLights[li*4].w carries the emitter castShadowDisabled
      // flag; skip the visibility test for that light (matches the NEE half so
      // both MIS strategies see the same lighting).
      let rectShadowDisabled = rectAreaLights[li * 4u].w > 0.5;
      if ((rectShadowDisabled || !traceAny(shadowRay, 1e-4, max(rectDist - 2e-3, 1e-3))) && rectDist < bestDist) {
        bestDist = rectDist;
        bestLightPdf = rectPdf;
        bestEmission = rectAreaLights[li * 4u + 3u].rgb;
      }
    }
  }
  if (includeMeshAreaLights) {
    for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
      var meshDist = INFINITY;
      var meshPdf = 0.0;
      if (intersectMeshAreaLightRay(mi, offsetOrigin, wi, &meshDist, &meshPdf)) {
        let shadowRay = Ray(offsetOrigin, wi);
        // SHADOW-01 — row 3.w carries castShadowDisabled (NEE parity).
        let meshBase = meshAreaLightBase(mi);
        let meshShadowDisabled = meshAreaLights[meshBase + 3u].w > 0.5;
        if ((meshShadowDisabled || !traceAny(shadowRay, 1e-4, max(meshDist - 2e-3, 1e-3))) && meshDist < bestDist) {
          bestDist = meshDist;
          bestLightPdf = meshPdf;
          let lightPoint = offsetOrigin + wi * meshDist;
          bestEmission = sampleMeshAreaLightRadiance(
            mi, meshAreaLightWeightsAtPoint(mi, lightPoint), lightPoint,
          );
        }
      }
    }
  }
  if (bestDist >= INFINITY || bestLightPdf <= 0.0) {
    return vec3f(0.0);
  }
  // A3 — spectralize the BSDF-connection emission at the hero λ in spectral mode,
  // matching the NEE half (kernel.wgsl.ts §631/676) so both halves of the MIS pair
  // use the same emission model for chromatic emitters. RGB mode: byte-identical.
  let emitOut = select(bestEmission, spectralEmissionAtHero(bestEmission, heroLambda), params.spectralEnabled != 0u);
  let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, false,
  );
  let misWeight = powerHeuristic(bsdfPdf, bestLightPdf);
  return throughputAtVertex * brdf * nDotL * emitOut * misWeight / bsdfPdf;
}

fn bsdfEnvironmentConnectionContribution(
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  anisotropy: f32,
  anisotropyRotation: f32,
  throughputAtVertex: vec3f,
  heroLambda: f32,
  matId: u32,
  misWeightOverride: f32,
) -> vec3f {
  let nDotL = abs(dot(normal, wi));
  if (nDotL <= 0.0) { return vec3f(0.0); }
  let bsdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation,
  );
  if (bsdfPdf <= 0.0) { return vec3f(0.0); }
  let offsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
  let shadowRay = Ray(hitPos + offsetNormal * 1e-3, wi);
  if (traceAny(shadowRay, 1e-4, INFINITY)) { return vec3f(0.0); }
  let env = environmentLookup(wi);
  let envLightPdf = environmentNeeProposalPdf(wi, normal);
  let ordinaryMisWeight = powerHeuristic(bsdfPdf, envLightPdf);
  let misWeight = select(
    ordinaryMisWeight, misWeightOverride, misWeightOverride >= 0.0,
  );
  let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, false,
  );
  // A3 — spectralize the env connection at the hero λ in spectral mode, matching the
  // NEE miss-shader path (kernel.wgsl.ts §431) and the NEE env branch (§724).
  // D3 — apply per-material envMapIntensity to the BSDF-connection env term, matching
  // the NEE env branch (kernel.wgsl.ts §723-724). envMapIntensity == 1.0 (default) →
  // envScale == 1.0 → byte-identical. Non-unit values scale BOTH halves identically so
  // the converged env contribution is consistent across the two MIS strategies.
  let envScale = materialEnvMapIntensity(matId);
  let envColorOut = select(env.color, spectralEmissionAtHero(env.color, heroLambda), params.spectralEnabled != 0u) * envScale;
  return throughputAtVertex * brdf * nDotL * envColorOut * misWeight / bsdfPdf;
}
`;
