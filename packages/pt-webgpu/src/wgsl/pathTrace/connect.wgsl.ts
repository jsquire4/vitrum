import { PT_WEBGPU_PATH_TRACE_CONNECT_CORE_WGSL } from './connectCore.wgsl.js';

/**
 * Connect module — light-direction sampling, MIS connections, and the
 * environment-map / HDRI helpers consumed by both the main kernel and the
 * BSDF→light MIS contributions.
 *
 * Bundled here:
 *  - CPU-baked procedural sky + HDRI bookkeeping:
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

struct SurfaceVisibility {
  transmittance: vec3f,
  visible: bool,
};

// Alpha/sidedness-aware straight visibility with exact smooth thin-sheet
// transport. Rough compound sheets have no arbitrary-direction connection
// measure and therefore remain blockers. The fixed-origin exact cursor keeps
// adjacent f32 surfaces distinct without an epsilon hop.
fn traceSurfaceVisibility(
  ray: Ray,
  tMin: f32,
  tMax: f32,
  heroLambda: f32,
  incidentIor: f32,
  rng: ptr<function, PtRngState>,
) -> SurfaceVisibility {
  var result = SurfaceVisibility(vec3f(1.0), false);
  var cursor = select(tMin, 0.0, opticalContinuationSourceIsActive());
  let surfaceHitLimit = sceneSurfaceHitLimit();
  var surfaceHitCount = 0u;
  loop {
    let hit = traceClosestRaw(ray, cursor, tMax);
    if (!hit.didHit) {
      opticalClearContinuationSource();
      result.visible = true;
      return result;
    }
    if (surfaceHitCount >= surfaceHitLimit) {
      opticalClearContinuationSource();
      return result;
    }
    surfaceHitCount = surfaceHitCount + 1u;
    let matId = hitMaterialId(hit);
    if (
      !materialAcceptsSidedHit(matId, hit.frontFace) ||
      materialShadowCastDisabled(matId) ||
      alphaTestPassThrough(
        matId, hit.triIndex, hit.baryVW, hit.instanceIndex, rng,
      )
    ) {
      if (!(hit.dist > cursor)) {
        opticalClearContinuationSource();
        return result;
      }
      cursor = hit.dist;
      continue;
    }
    let sheetAttenuation = thinSheetExactVisibilityTransmission(
      hit, ray.direction, heroLambda, incidentIor,
    );
    if (max(
      sheetAttenuation.x,
      max(sheetAttenuation.y, sheetAttenuation.z),
    ) > 0.0) {
      result.transmittance = result.transmittance * sheetAttenuation;
      if (!(hit.dist > cursor)) {
        opticalClearContinuationSource();
        return SurfaceVisibility(vec3f(0.0), false);
      }
      cursor = hit.dist;
      continue;
    }
    opticalClearContinuationSource();
    return SurfaceVisibility(vec3f(0.0), false);
  }
  return result;
}

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
    ptScaleEnvironmentRadiance(texel.rgb, params.environmentHdriIntensity),
    texel.w,
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
// (no environment map, or empty CDF). One variate selects the discrete cell
// and two residual variates sample uniformly over that cell's solid angle.
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
  let cellXi = vec2f(rand_f32(rng), rand_f32(rng));
  // mapDir is in unrotated-map space (the CDF is built from the unrotated map).
  let mapDir = sampleEnvironmentCellDirection(x, y, dims, cellXi);
  let texel = environmentMapTexels[idx];
  // H6: rotate the map-space sample direction by +rotationY to get the world-space
  // light direction for a CCW-rotated environment dome.
  // rotationY = 0 → rotateYPos is identity → zero-rotation invariant.
  let rotY = params.environmentTint.w;
  result.wi = safe_normalize(rotateYPos(mapDir, rotY));
  result.value = ptScaleEnvironmentRadiance(
    texel.rgb,
    params.environmentHdriIntensity,
  );
  result.pdf = texel.w;
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
    return environmentPdf(dir);
  }
  return 0.25 * INV_PI;
}

const MESH_AREA_LIGHT_VEC4_STRIDE: u32 = 7u;

fn meshAreaLightBase(index: u32) -> u32 {
  return index * MESH_AREA_LIGHT_VEC4_STRIDE;
}

fn meshAreaLightIsTwoSided(index: u32) -> bool {
  return meshAreaLights[meshAreaLightBase(index) + 6u].w > 0.5;
}

// towardReceiver points away from the sampled light surface. One-sided
// emitters retain their authored winding; double-sided materials expose the
// same Le over both hemispheres.
fn meshAreaLightCosineTowardReceiver(
  index: u32,
  geometricNormal: vec3f,
  towardReceiver: vec3f,
) -> f32 {
  let signedCosine = dot(geometricNormal, towardReceiver);
  return select(
    max(signedCosine, 0.0),
    abs(signedCosine),
    meshAreaLightIsTwoSided(index),
  );
}

fn meshAreaLightWeightsAtPoint(index: u32, point: vec3f) -> vec3f {
  let base = meshAreaLightBase(index);
  let a = meshAreaLights[base].xyz;
  let ab = meshAreaLights[base + 1u].xyz - a;
  let ac = meshAreaLights[base + 2u].xyz - a;
  let ap = point - a;
  let areaMeasure = measureAreaVector(ab, ac, 0.5);
  let areaCoordinates = solveAreaVectorCoordinates(
    ab, ac, ap, areaMeasure,
  );
  if (areaCoordinates.z == 0.0) {
    return vec3f(1.0, 0.0, 0.0);
  }
  let wb = areaCoordinates.x;
  let wc = areaCoordinates.y;
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
  if (index > 0xffffffffu / MESH_AREA_LIGHT_VEC4_STRIDE) {
    return vec3f(0.0);
  }
  let base = meshAreaLightBase(index);
  let meshLightCount = arrayLength(&meshAreaLights);
  if (base > meshLightCount || 7u > meshLightCount - base) {
    return vec3f(0.0);
  }
  let averageAndShadow = meshAreaLights[base + 3u];
  let uvAB = meshAreaLights[base + 4u];
  let uvCAndMaterial = meshAreaLights[base + 5u];
  let materialIdPlusOne = uvCAndMaterial.z;
  if (!materialTextureFiniteF32(materialIdPlusOne)) { return vec3f(0.0); }
  if (!materialTextureFiniteVec4(averageAndShadow)) { return vec3f(0.0); }
  if (materialIdPlusOne < 0.5) { return averageAndShadow.rgb; }
  let materialCount = arrayLength(&materialTexDescriptors) / MATERIAL_TEX_VEC4_STRIDE;
  let matId = materialTextureExactU32(materialIdPlusOne - 1.0, materialCount);
  if (matId == 0xffffffffu) { return vec3f(0.0); }
  let descriptorBase = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(descriptorBase, 0u, MATERIAL_TEX_VEC4_STRIDE)) {
    return vec3f(0.0);
  }
  let layerIdx = materialTextureLayerIndex(
    materialTexDescriptors[descriptorBase].w,
    textureNumLayers(materialTexturesEmissive),
  );
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
  let wrapMode = materialTexDescriptors[descriptorBase + 13u].zw;
  let uvFitScale = materialTexDescriptors[descriptorBase + 7u].zw;
  if (
    !materialTextureFiniteVec4(averageAndShadow) ||
    !materialTextureFiniteVec4(uvAB) ||
    !materialTextureFiniteVec4(uvCAndMaterial) ||
    !materialTextureFiniteVec3(weights) ||
    !materialTextureFiniteVec3(worldPosition) ||
    !materialTextureFiniteVec4(uvMeta) ||
    !materialTextureFiniteVec4(uvScale) ||
    !materialTextureWrapModesValid(wrapMode) ||
    !materialTextureFiniteVec2(uvFitScale) ||
    any(uvFitScale <= vec2f(0.0)) ||
    any(uvFitScale > vec2f(1.0))
  ) { return vec3f(0.0); }
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
  let sourceBaseSize = materialTextureSourceBaseSize(
    vec2u(textureDimensions(materialTexturesEmissive, i32(0))), uvFitScale,
  );
  let sourceMipCount = f32(materialTextureSourceMipCount(sourceBaseSize));
  let texDim = vec2f(sourceBaseSize);
  let texelArea = max(
    abs((uvB.x - uvA.x) * (uvC.y - uvA.y) -
        (uvB.y - uvA.y) * (uvC.x - uvA.x)) * texDim.x * texDim.y,
    1.0,
  );
  let worldArea = uvCAndMaterial.w;
  if (
    !materialTextureFiniteF32(texelArea) ||
    !materialTextureFiniteF32(worldArea) ||
    worldArea <= 0.0
  ) { return vec3f(0.0); }
  let cameraDistance = max(
    safe_length(worldPosition - params.cameraPos.xyz), ptRayTMin(),
  );
  let pixelsPerMeter =
    0.5 * f32(max(params.width, params.height)) / cameraDistance;
  let projectedPixels = max(sqrt(worldArea) * pixelsPerMeter, 1.0);
  let lod = clamp(
    log2(sqrt(texelArea) / projectedPixels),
    0.0,
    max(sourceMipCount - 1.0, 0.0),
  );
  if (!materialTextureFiniteF32(lod)) { return vec3f(0.0); }
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
  if (!texel.valid) { return vec3f(0.0); }
  return ptFiniteNonNegativeRadianceProduct(
    meshAreaLights[base + 6u].rgb,
    texel.value.rgb,
  );
}

// Intersect the BSDF sample ray against rect/disc area light index li.
// Reads the shape discriminator from emission.w: ≈ 0 → rect, ≈ 1 → analytic disc.
// Rect:  uCoord/vCoord ∈ [-1,1] box test; area = 4·|u×v|.
// Disc:  uCoord² + vCoord² ≤ 1 circle test; area = π·|u×v|.
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
  let areaMeasure = measureAreaVector(
    uAxis, vAxis, select(4.0, PI, isDisc),
  );
  if (areaMeasure.valid == 0u) { return false; }
  let lightNormal = areaMeasure.normal;
  let denom = dot(lightNormal, rayDir);
  if (denom == 0.0) {
    return false;
  }
  let t = dot(lightNormal, rectPos - rayOrigin) / denom;
  if (t <= ptRayTMin()) {
    return false;
  }
  let p = rayOrigin + rayDir * t;
  let rel = p - rectPos;
  // Solve through the dominant 2D projection. This preserves the exact affine
  // coordinates without squaring the axes into an overflow-prone Gram matrix.
  let areaCoordinates = solveAreaVectorCoordinates(
    uAxis, vAxis, rel, areaMeasure,
  );
  if (areaCoordinates.z == 0.0) { return false; }
  let uCoord = areaCoordinates.x;
  let vCoord = areaCoordinates.y;
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
  // Area formula: the unit-disc parameterisation has Jacobian |u×v|;
  // disc → π·|u×v|, rect → 4·|u×v|.
  let lightPdf = ptAreaToSolidAnglePdf(t, cosLight, areaMeasure);
  if (!(lightPdf > 0.0)) { return false; }
  *distOut = t;
  *lightPdfOut = lightPdf;
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
  let t = intersectTriangle(rayOrigin, rayDir, a, b, c, ptRayTMin());
  if (t <= ptRayTMin() || t >= INFINITY) {
    return false;
  }
  let areaMeasure = measureAreaVector(b - a, c - a, 0.5);
  if (areaMeasure.valid == 0u) { return false; }
  let lightNormal = areaMeasure.normal;
  let cosLight = meshAreaLightCosineTowardReceiver(
    li, lightNormal, -rayDir,
  );
  if (cosLight <= 0.0) {
    return false;
  }
  let lightPdf = ptAreaToSolidAnglePdf(t, cosLight, areaMeasure);
  if (!(lightPdf > 0.0)) { return false; }
  *distOut = t;
  *lightPdfOut = lightPdf;
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
  thinFilm: ThinFilmInterface,
  throughputAtVertex: vec3f,
  heroLambda: f32,
  includeMeshAreaLights: bool,
  rng: ptr<function, PtRngState>,
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
    anisotropy, anisotropyRotation, thinFilm,
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
  let offsetOrigin = hitPos + offsetNormal * ptRayOriginBias();
  var bestDist = INFINITY;
  var bestLightPdf = 0.0;
  var bestEmission = vec3f(0.0);
  var bestVisibility = vec3f(1.0);
  for (var li = 0u; li < params.rectAreaLightCount; li = li + 1u) {
    var rectDist = INFINITY;
    var rectPdf = 0.0;
    if (intersectRectAreaLightRay(li, offsetOrigin, wi, &rectDist, &rectPdf)) {
      let shadowRay = Ray(offsetOrigin, wi);
      // SHADOW-01 — rectAreaLights[li*4].w carries the emitter castShadowDisabled
      // flag; skip the visibility test for that light (matches the NEE half so
      // both MIS strategies see the same lighting).
      let rectShadowDisabled = rectAreaLights[li * 4u].w > 0.5;
      var visibility = SurfaceVisibility(vec3f(1.0), true);
      if (!rectShadowDisabled) {
        visibility = traceSurfaceVisibility(
          shadowRay, ptRayTMin(), ptFiniteSegmentTMax(rectDist),
          heroLambda, 1.0, rng,
        );
      }
      if (visibility.visible && rectDist < bestDist) {
        bestDist = rectDist;
        bestLightPdf = rectPdf;
        bestEmission = rectAreaLights[li * 4u + 3u].rgb;
        bestVisibility = visibility.transmittance;
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
        var visibility = SurfaceVisibility(vec3f(1.0), true);
        if (!meshShadowDisabled) {
          visibility = traceSurfaceVisibility(
            shadowRay, ptRayTMin(), ptFiniteSegmentTMax(meshDist),
            heroLambda, 1.0, rng,
          );
        }
        if (visibility.visible && meshDist < bestDist) {
          bestDist = meshDist;
          bestLightPdf = meshPdf;
          let lightPoint = offsetOrigin + wi * meshDist;
          bestEmission = sampleMeshAreaLightRadiance(
            mi, meshAreaLightWeightsAtPoint(mi, lightPoint), lightPoint,
          );
          bestVisibility = visibility.transmittance;
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
    anisotropy, anisotropyRotation, thinFilm, false,
  );
  let misWeight = powerHeuristic(bsdfPdf, bestLightPdf);
  return throughputAtVertex * brdf * nDotL * emitOut * bestVisibility *
    misWeight / bsdfPdf;
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
  thinFilm: ThinFilmInterface,
  throughputAtVertex: vec3f,
  heroLambda: f32,
  matId: u32,
  misWeightOverride: f32,
  rng: ptr<function, PtRngState>,
) -> vec3f {
  let nDotL = abs(dot(normal, wi));
  if (nDotL <= 0.0) { return vec3f(0.0); }
  let bsdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, thinFilm,
  );
  if (bsdfPdf <= 0.0) { return vec3f(0.0); }
  let offsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
  let shadowRay = Ray(
    hitPos + offsetNormal * ptRayOriginBias(),
    wi,
  );
  let visibility = traceSurfaceVisibility(
    shadowRay, ptRayTMin(), INFINITY, heroLambda, 1.0, rng,
  );
  if (!visibility.visible) {
    return vec3f(0.0);
  }
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
    anisotropy, anisotropyRotation, thinFilm, false,
  );
  // A3 — spectralize the env connection at the hero λ in spectral mode, matching the
  // NEE miss-shader path (kernel.wgsl.ts §431) and the NEE env branch (§724).
  // D3 — apply per-material envMapIntensity to the BSDF-connection env term, matching
  // the NEE env branch (kernel.wgsl.ts §723-724). envMapIntensity == 1.0 (default) →
  // envScale == 1.0 → byte-identical. Non-unit values scale BOTH halves identically so
  // the converged env contribution is consistent across the two MIS strategies.
  let envScale = materialEnvMapIntensity(matId);
  let envColorOut = ptScaleEnvironmentRadiance(
    select(
      env.color,
      spectralEmissionAtHero(env.color, heroLambda),
      params.spectralEnabled != 0u,
    ),
    envScale,
  );
  return throughputAtVertex * brdf * nDotL * envColorOut *
    visibility.transmittance * misWeight / bsdfPdf;
}
`;
