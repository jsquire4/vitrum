/**
 * Camera-visible transparent composition for walkaround-hybrid.
 *
 * Fractional `alphaMode:'blend'` coverage is resolved here, deterministically,
 * front-to-back over the opaque/background result. Camera-primary reservoir and
 * shade passes intentionally skip fractional blend layers so coverage is owned
 * by exactly one pass. Secondary transport uses separately seeded stochastic
 * coverage; shadow transport integrates analytic alpha transmittance.
 *
 * Lighting policy: transparent layer radiance is evaluated at each admitted
 * layer. The sky/environment, direct sun, and analytic
 * point/spot/finite-emitter terms use the same atlas-backed material-lobe BRDF
 * as opaque shade/ReSTIR material scoring; their shadow rays use the canonical
 * material-atlas alpha transmittance helper, including the post-budget opaque
 * blocker fallback used by shade/ReSTIR.
 * Emissive, light-map, finite-emitter, DDGI, and RC indirect receiver terms are
 * evaluated at each visible blend layer. DDGI/RC are combined as two estimates
 * of the same diffuse irradiance measure; ReSTIR-GI reservoirs remain tied to
 * the opaque primary surface and are not sampled at unrelated layer geometry.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { analyticLightFalloffWgsl } from './analyticLightFalloff.wgsl.js';

export const TRANSPARENT_OIT_WGSL = /* wgsl */ `

@group(1) @binding(5) var bvh_beer: texture_2d<u32>;
@group(1) @binding(12) var bvh_emissive: texture_2d<f32>;
@group(1) @binding(13) var analytic_lights: texture_2d<f32>;
@group(1) @binding(14) var bvh_material: texture_2d<u32>;

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

@group(3) @binding(0) var ddgiIrradiance: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibility: texture_2d<f32>;
@group(3) @binding(2) var ddgiSampler: sampler;
@group(3) @binding(6) var oit_background: texture_2d<f32>;
@group(3) @binding(7) var oit_transparentOut: texture_storage_2d<rgba16float, write>;

fn oitMaterialWord(triIndex: u32) -> u32 {
  return textureLoad(
    bvh_material,
    vec2i(i32(triIndex % BVH_MATERIAL_TEX_WIDTH), i32(triIndex / BVH_MATERIAL_TEX_WIDTH)),
    0,
  ).r;
}

fn oitHitIsMaskDiscarded(hit: IntersectionResult, alpha: MaterialAlphaCoverage) -> bool {
  if (alpha.scalarDiscarded != 0u) {
    return true;
  }
  if (alpha.mode == 1u) {
    return alpha.coverage < alpha.cutoff;
  }
  return false;
}

fn oitShadowTransmittance(origin: vec3f, dir: vec3f, tMax: f32, triEps: f32) -> vec3f {
  return traceSceneAlphaTintTransmittanceTextured(
    ubo.bvhMode,
    ubo.tlasNodeCount,

    origin,
    dir,
    max(tMax, 0.0),
    triEps,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );
}

${analyticLightFalloffWgsl('oit')}

struct OitAnalyticAliasDraw {
  index: u32,
  pmf: f32,
}

fn oitSamplingHashU32(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn oitSamplingHashToF32(seed: u32) -> f32 {
  return f32(oitSamplingHashU32(seed) >> 8u) * (1.0 / 16777216.0);
}

fn oitAnalyticAliasColumn(seed: u32, count: u32) -> u32 {
  let threshold = ((0xffffffffu % count) + 1u) % count;
  var word = oitSamplingHashU32(seed);
  loop {
    if (word >= threshold) { return word % count; }
    word = oitSamplingHashU32(word ^ 0x27d4eb2du);
  }
  return 0u;
}

fn oitAnalyticAliasDraw(
  count: u32,
  aliasOffset: u32,
  dims: vec2u,
  seed: u32,
) -> OitAnalyticAliasDraw {
  let column = oitAnalyticAliasColumn(seed, count);
  let coord = aliasOffset + column;
  let entry = textureLoad(analytic_lights, vec2i(i32(coord % dims.x), i32(coord / dims.x)), 0);
  let aliasIndex = bitcast<u32>(entry.y);
  let selected = select(aliasIndex, column, oitSamplingHashToF32(seed ^ 0x85ebca6bu) < entry.x);
  let selectedCoord = aliasOffset + selected;
  let selectedEntry = textureLoad(
    analytic_lights,
    vec2i(i32(selectedCoord % dims.x), i32(selectedCoord / dims.x)),
    0,
  );
  var draw: OitAnalyticAliasDraw;
  draw.index = selected;
  draw.pmf = selectedEntry.z;
  return draw;
}

struct OitLayerNormals {
  smoothNormal: vec3f,
  shadingNormal: vec3f,
};

fn oitLayerNormals(hit: IntersectionResult) -> OitLayerNormals {
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = hit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < tlasWorldToLocalColumnCount();
  let n_i = select(0u, n_base, n_ok);
  let n0 = sceneLoadBvhNormal(hit.indices.x);
  let n1 = sceneLoadBvhNormal(hit.indices.y);
  let n2 = sceneLoadBvhNormal(hit.indices.z);
  let smoothNormal = smoothShadingNormal(
    hit,
    hit.normal,
    n0.xyz,
    n1.xyz,
    n2.xyz,
    n_ok,
    tlasLoadWorldToLocalColumn(n_i),
    tlasLoadWorldToLocalColumn(n_i + 1u),
    tlasLoadWorldToLocalColumn(n_i + 2u),
  );
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  var normals: OitLayerNormals;
  normals.smoothNormal = smoothNormal;
  normals.shadingNormal = applyBumpMapForHit(hit, normalMapped);
  return normals;
}

// Ray Tracing Gems, Ch. 6: scale-independent ULP origin offset.  Unlike an
// absolute world-space epsilon this remains on the correct side of thin
// geometry at every supported scene scale.
fn oitOffsetRayOrigin(p: vec3f, geometricNormal: vec3f, dir: vec3f) -> vec3f {
  let n = select(-geometricNormal, geometricNormal, dot(geometricNormal, dir) >= 0.0);
  let ofi = vec3i(n * 256.0);
  let pi = vec3f(
    bitcast<f32>(bitcast<i32>(p.x) + select(ofi.x, -ofi.x, p.x < 0.0)),
    bitcast<f32>(bitcast<i32>(p.y) + select(ofi.y, -ofi.y, p.y < 0.0)),
    bitcast<f32>(bitcast<i32>(p.z) + select(ofi.z, -ofi.z, p.z < 0.0)),
  );
  let fallback = p + n * (1.0 / 65536.0);
  return vec3f(
    select(pi.x, fallback.x, abs(p.x) < (1.0 / 32.0)),
    select(pi.y, fallback.y, abs(p.y) < (1.0 / 32.0)),
    select(pi.z, fallback.z, abs(p.z) < (1.0 / 32.0)),
  );
}

fn oitCosineHemisphereDir(n: vec3f, xi: vec2f) -> vec3f {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.95) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  let tangent = safe_normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  let r = sqrt(xi.x);
  let phi = 2.0 * PI * xi.y;
  let local = vec3f(r * cos(phi), r * sin(phi), sqrt(1.0 - xi.x));
  return safe_normalize(tangent * local.x + bitangent * local.y + n * local.z);
}

fn oitLayerEnvSampleRadiance(
  payload: RestirDIMaterialPayload,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let brdf = evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
    payload.albedo,
    payload.rough,
    payload.metal,
    payload.specular.rgb,
    payload.specular.a,
    payload.anisotropy.x,
    payload.anisotropy.y,
    payload.iridescence,
    payload.clearcoat.x,
    payload.clearcoat.y,
    payload.sheen.a,
    payload.sheenRoughness,
    payload.sheen.rgb,
    payload.anisotropyTangent,
    payload.anisotropyBitangent,
    normal,
    payload.clearcoatNormal,
    wo,
    wi,
  );
  return envRadiance(wi) * max(payload.envMapIntensity, 0.0) * brdf;
}

fn oitFiniteVec3(value: vec3f) -> bool {
  return all(value == value) &&
    all(abs(value) <= vec3f(3.402823466e38));
}

// The 24-bit hash maps xi.x to [0, 1 - 2^-24], so cosine sampling has
// cos(theta) >= 2^-12 and pdf >= INV_PI/4096. Keep that proven lower bound
// despite basis-roundoff, reject non-finite inputs, and cap to the largest
// finite rgba16float value before the storage write. Clamping the numerator
// before division prevents the division itself from overflowing.
fn oitBoundedCosineImportanceDivide(numerator: vec3f, pdf: f32) -> vec3f {
  if (
    !oitFiniteVec3(numerator) ||
    !(pdf > 0.0) ||
    !(pdf <= 3.402823466e38)
  ) {
    return vec3f(0.0);
  }
  let safePdf = max(pdf, INV_PI / 4096.0);
  let maxOutput = 65504.0;
  let boundedNumerator = min(
    max(numerator, vec3f(0.0)),
    vec3f(maxOutput * safePdf),
  );
  return boundedNumerator / safePdf;
}

fn oitLayerSkyRadiance(
  payload: RestirDIMaterialPayload,
  normal: vec3f,
  wo: vec3f,
  seed: u32,
) -> vec3f {
  // One cosine-density sample per temporal sample.  The frame seed rotates the
  // proposal and temporal accumulation performs the integration; dividing by
  // the exact proposal PDF makes the estimator unbiased for arbitrary HDRIs.
  let xi = vec2f(
    oitSamplingHashToF32(seed ^ 0x9e3779b9u),
    oitSamplingHashToF32(seed ^ 0x85ebca6bu),
  );
  let wi = oitCosineHemisphereDir(normal, xi);
  let pdf = max(dot(normal, wi), 0.0) * INV_PI;
  return oitBoundedCosineImportanceDivide(
    oitLayerEnvSampleRadiance(payload, normal, wo, wi),
    pdf,
  );
}

fn oitLayerAnalyticNEE(
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  payload: RestirDIMaterialPayload,
  wo: vec3f,
  seed0: u32,
) -> vec3f {
  let analyticDims = textureDimensions(analytic_lights);
  let analyticHeader = textureLoad(analytic_lights, vec2i(0, 0), 0);
  let count = u32(max(analyticHeader.x, 0.0));
  let aliasOffset = u32(max(analyticHeader.y, 0.0));
  let texelCount = analyticDims.x * analyticDims.y;
  if (
    count == 0u ||
    aliasOffset != 1u + count * 4u ||
    aliasOffset + count > texelCount
  ) { return vec3f(0.0); }

  var Lo = vec3f(0.0);
  let sampleCount = min(count, 4u);
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
    var li = sampleIndex;
    var estimatorWeight = 1.0;
    if (count > 4u) {
      let draw = oitAnalyticAliasDraw(
        count,
        aliasOffset,
        analyticDims,
        seed0 ^ (sampleIndex * 0xc2b2ae35u),
      );
      if (draw.pmf <= 0.0) { continue; }
      li = draw.index;
      estimatorWeight = 1.0 / (f32(sampleCount) * draw.pmf);
    }
    let base = 1u + li * 4u;
    let light0 = textureLoad(analytic_lights, vec2i(i32(base % analyticDims.x), i32(base / analyticDims.x)), 0);
    let light1 = textureLoad(analytic_lights, vec2i(i32((base + 1u) % analyticDims.x), i32((base + 1u) / analyticDims.x)), 0);
    let light2 = textureLoad(analytic_lights, vec2i(i32((base + 2u) % analyticDims.x), i32((base + 2u) / analyticDims.x)), 0);
    let light3 = textureLoad(analytic_lights, vec2i(i32((base + 3u) % analyticDims.x), i32((base + 3u) / analyticDims.x)), 0);
    let lightPos = light0.xyz;
    let lightLe = light1.xyz;
    let lightDir = light2.xyz;
    let cosInner = light2.w;
    let cosOuter = light3.x;
    let castShadowDisabled = light3.y > 0.5;
    let cutoffDistance = light3.z;
    let decay = light3.w;

    let toL = lightPos - hitPos;
    let dist = length(toL);
    if (dist <= 0.0) { continue; }
    let wi = toL / dist;
    let nDotL = dot(normal, wi);
    if (nDotL <= 0.0) { continue; }

    let cone = oitSpotConeFalloff(lightDir, wi, cosInner, cosOuter);
    if (cone <= 0.0) { continue; }

    var shadowT = vec3f(1.0);
    if (!castShadowDisabled) {
      let shadowOrigin = oitOffsetRayOrigin(hitPos, geoNormal, wi);
      shadowT = oitShadowTransmittance(
        shadowOrigin,
        wi,
        length(lightPos - shadowOrigin),
        ubo.triIntersectEpsilon,
      );
      if (max(max(shadowT.x, shadowT.y), shadowT.z) <= 0.0) { continue; }
    }

    let attenuation = oitPointSpotAttenuation(dist, cutoffDistance, decay, ubo.emitterDist2Floor);
    let brdf = evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
      payload.albedo,
      payload.rough,
      payload.metal,
      payload.specular.rgb,
      payload.specular.a,
      payload.anisotropy.x,
      payload.anisotropy.y,
      payload.iridescence,
      payload.clearcoat.x,
      payload.clearcoat.y,
      payload.sheen.a,
      payload.sheenRoughness,
      payload.sheen.rgb,
      payload.anisotropyTangent,
      payload.anisotropyBitangent,
      normal,
      clearcoatNormal,
      wo,
      wi,
    );
    // evalGGX* already includes the receiver cosine; nDotL is only a gate here.
    Lo += lightLe * shadowT * brdf * cone * attenuation * estimatorWeight;
  }
  return Lo;
}

const OIT_AREA_EMITTER_SAMPLE_COUNT = 4u;

fn oitAreaEmitterXi(sampleIndex: u32) -> vec2f {
  if (sampleIndex == 0u) { return vec2f(0.125, 0.375); }
  if (sampleIndex == 1u) { return vec2f(0.375, 0.875); }
  if (sampleIndex == 2u) { return vec2f(0.625, 0.125); }
  return vec2f(0.875, 0.625);
}

fn oitLayerAreaEmitterNEE(
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  payload: RestirDIMaterialPayload,
  wo: vec3f,
  seed0: u32,
) -> vec3f {
  let count = min(ubo.emitterCount, sceneEmitterCount());
  if (count == 0u) { return vec3f(0.0); }

  var Lo = vec3f(0.0);
  // Four stratified emitter-CDF draws make cost independent of emitter count.
  // A Cranley-Patterson rotation keeps the four established within-triangle
  // strata while decorrelating them per pixel/layer/frame.
  for (var si = 0u; si < OIT_AREA_EMITTER_SAMPLE_COUNT; si = si + 1u) {
    let emitterXi = (
      f32(si) + oitSamplingHashToF32(seed0 ^ (si * 0x9e3779b9u) ^ 0x68bc21ebu)
    ) / f32(OIT_AREA_EMITTER_SAMPLE_COUNT);
    let lid = sampleEmitterIdx(count, emitterXi);
    let emitterPmf = emitterCdfPmf(count, lid);
    if (emitterPmf <= 0.0) { continue; }
    let e = sceneLoadEmitter(lid);
      let rotation = vec2f(
        oitSamplingHashToF32(seed0 ^ (si * 0x85ebca6bu) ^ 0x243f6a88u),
        oitSamplingHashToF32(seed0 ^ (si * 0xc2b2ae35u) ^ 0xb7e15162u),
      );
      let xi = fract(oitAreaEmitterXi(si) + rotation);
      let ls = sampleEmitterPoint(e, xi);
      let toL = ls.pos - hitPos;
      let dist2 = dot(toL, toL);
      if (dist2 <= 0.0 || ls.area <= 0.0) { continue; }

      let dist = sqrt(dist2);
      let wi = toL / dist;
      let nDotL = max(0.0, dot(normal, wi));
      let nlDotL = max(0.0, dot(-ls.normal, wi));
      if (nDotL <= 0.0 || nlDotL <= 0.0) { continue; }

      var shadowT = vec3f(1.0);
      if (e.castShadowDisabled < 0.5) {
        let shadowOrigin = oitOffsetRayOrigin(hitPos, geoNormal, wi);
        shadowT = oitShadowTransmittance(
          shadowOrigin,
          wi,
          length(ls.pos - shadowOrigin),
          ubo.triIntersectEpsilon,
        );
        if (max(max(shadowT.x, shadowT.y), shadowT.z) <= 0.0) { continue; }
      }

      let G = nlDotL / dist2;
      let brdf = evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
        payload.albedo,
        payload.rough,
        payload.metal,
        payload.specular.rgb,
        payload.specular.a,
        payload.anisotropy.x,
        payload.anisotropy.y,
        payload.iridescence,
        payload.clearcoat.x,
        payload.clearcoat.y,
        payload.sheen.a,
        payload.sheenRoughness,
        payload.sheen.rgb,
        payload.anisotropyTangent,
        payload.anisotropyBitangent,
        normal,
        clearcoatNormal,
        wo,
        wi,
      );
      let Le = sampleEmitterLeAtXi(e, xi);
      let estimatorWeight = 1.0 /
        (f32(OIT_AREA_EMITTER_SAMPLE_COUNT) * emitterPmf);
      Lo += Le * shadowT * brdf * G * ls.area * estimatorWeight;
  }
  return Lo;
}

fn oitLayerRadiance(
  hit: IntersectionResult,
  hitPos: vec3f,
  rayDir: vec3f,
  materialWord: u32,
  seed0: u32,
) -> vec3f {
  let scalarBase = decodeMaterialColor(hit.matColorPacked).rgb;
  let uv1 = materialAtlasUv1ForHit(hit);
  let normals = oitLayerNormals(hit);
  let normal = normals.shadingNormal;
  let wo = safe_normalize(-rayDir);
  let payload = sampleRestirDIMaterialPayloadForHit(
    hit, normals.smoothNormal, normal, scalarBase, materialWord, wo,
  );

  let emitCoord = vec2u(hit.indices.w % BVH_MATERIAL_TEX_WIDTH, hit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let emissive = sampleEmissiveMap(
    hit.indices.w,
    hit.uv,
    uv1,
    textureLoad(bvh_emissive, vec2i(emitCoord), 0).rgb,
  );
  let bakedIrradiance = sampleLightMap(hit);
  let baked = payload.albedo * INV_PI * bakedIrradiance;

  // Both caches return the same diffuse receiver measure. DDGI exposes
  // irradiance E, so apply albedo/pi here. sampleCascadeC0 already applies
  // 1/pi but deliberately omits albedo for the opaque denoiser path.
  let ddgiIndirect = payload.albedo * INV_PI * sampleDDGIAtPoint(hitPos, normal);
  let rcIndirect = payload.albedo * sampleCascadeC0(hitPos, normal);
  let indirect = mix(ddgiIndirect, rcIndirect, clamp(rcParams.rcWeight, 0.0, 1.0));

  let skyAmbient = oitLayerSkyRadiance(payload, normal, wo, seed0 ^ 0xd1b54a35u);
  let analyticDirect = oitLayerAnalyticNEE(
    hitPos, normal, payload.clearcoatNormal, hit.normal, payload, wo,
    seed0 ^ 0xa511e9b3u,
  );
  let areaDirect = oitLayerAreaEmitterNEE(
    hitPos, normal, payload.clearcoatNormal, hit.normal, payload, wo,
    seed0 ^ 0x63d83595u,
  );
  let sunBase = safe_normalize(ubo.sunDirection);
  // Match the opaque soft-sun estimator's temporal integration policy: keep
  // the world/material identity while changing the disk sample each frame.
  let sunXi = worldHash2(hitPos, hit.indices.w ^ ubo.frameSeed ^ 0x4f495431u);
  let sunUpRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let sunTan = safe_normalize(cross(sunUpRef, sunBase));
  let sunBit = cross(sunBase, sunTan);
  let sunAngularRadius = max(ubo.sunAngular.x, 0.0);
  let sunR = sunAngularRadius * sqrt(sunXi.x);
  let sunPhi = 6.2831853 * sunXi.y;
  let toSun = safe_normalize(sunBase + sunTan * (sunR * cos(sunPhi)) + sunBit * (sunR * sin(sunPhi)));
  let sunBrdf = evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
    payload.albedo,
    payload.rough,
    payload.metal,
    payload.specular.rgb,
    payload.specular.a,
    payload.anisotropy.x,
    payload.anisotropy.y,
    payload.iridescence,
    payload.clearcoat.x,
    payload.clearcoat.y,
    payload.sheen.a,
    payload.sheenRoughness,
    payload.sheen.rgb,
    payload.anisotropyTangent,
    payload.anisotropyBitangent,
    normal,
    payload.clearcoatNormal,
    wo,
    toSun,
  );
  var sunVisibility = vec3f(1.0);
  if ((ubo.stainedGlassFlags & SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) == 0u) {
    sunVisibility = oitShadowTransmittance(
      oitOffsetRayOrigin(hitPos, hit.normal, toSun),
      toSun,
      1e30,
      ubo.triIntersectEpsilon,
    );
  }
  let sunDirect = vec3f(ubo.sunIntensity) * sunBrdf * sunVisibility;
  return applyHomogeneousVolumeSingleScatter(
    (skyAmbient + sunDirect + analyticDirect + areaDirect + indirect + emissive + baked) *
      payload.layerTransmission,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    normal,
    wo,
  );
}

@compute @workgroup_size(8, 8, 1)
fn transparentOitMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(oit_transparentOut);
  let pix = gid.xy;
  if (any(pix >= dims)) { return; }

  let background = textureLoad(oit_background, vec2i(pix), 0).rgb;
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let primaryRay = generatePrimaryRay_common(
    pix.x, pix.y, dims.x, dims.y, ubo.cameraPos, invVP,
  );

  var walkRay = primaryRay;
  var transmittance = 1.0;
  var accum = vec3f(0.0);
  let instanceMultiplier = select(1u, max(tlasBlasRootCount(), 1u), ubo.bvhMode == 1u);
  let triangleCount = bvhIndexCount();
  // A straight ray can cross each world-space triangle at most once.  For TLAS
  // scenes every instance may contribute every BLAS triangle, giving this
  // finite scene-derived upper bound without discarding a fixed black tail.
  let layerCapacity = select(
    triangleCount * instanceMultiplier,
    0xffffffffu,
    triangleCount > 0xffffffffu / instanceMultiplier,
  );
  let layerBudget = max(layerCapacity, 1u);

  for (var layer = 0u; layer < layerBudget; layer = layer + 1u) {
    let hit = traceSceneFirstHit(
      ubo.bvhMode,
      ubo.tlasNodeCount,
      walkRay,
      ubo.triIntersectEpsilon,
    );
    if (!hit.didHit || transmittance <= 0.0) {
      break;
    }

    let word = oitMaterialWord(hit.indices.w);
    if (!materialSideAdmittedForHit(hit)) {
      let hitPos = walkRay.origin + walkRay.direction * hit.dist;
      walkRay.origin = oitOffsetRayOrigin(hitPos, hit.normal, walkRay.direction);
      continue;
    }
    let alpha = materialAlphaCoverageForHit(hit, word);
    if (oitHitIsMaskDiscarded(hit, alpha)) {
      let hitPos = walkRay.origin + walkRay.direction * hit.dist;
      walkRay.origin = oitOffsetRayOrigin(hitPos, hit.normal, walkRay.direction);
      continue;
    }

    if (alpha.mode == 2u && alpha.coverage <= 0.0) {
      let hitPos = walkRay.origin + walkRay.direction * hit.dist;
      walkRay.origin = oitOffsetRayOrigin(hitPos, hit.normal, walkRay.direction);
      continue;
    }

    if (alpha.mode == 2u) {
      let coverage = clamp(alpha.coverage, 0.0, 1.0);
      let hitPos = walkRay.origin + walkRay.direction * hit.dist;
      let layerRadiance = oitLayerRadiance(
        hit,
        hitPos,
        primaryRay.direction,
        word,
        ubo.frameSeed ^ (pix.x * 0x9e3779b9u) ^
          (pix.y * 0x85ebca6bu) ^ (layer * 0xc2b2ae35u),
      );
      // Always pay only the current layer's actual coverage. Any residual after
      // the scene-derived bound remains assigned to the already-rendered
      // background; assigning it to layerRadiance would invent uncovered light.
      accum = accum + layerRadiance * coverage * transmittance;
      transmittance = transmittance * (1.0 - coverage);
      if (coverage >= 1.0) {
        break;
      }
      walkRay.origin = oitOffsetRayOrigin(hitPos, hit.normal, walkRay.direction);
      continue;
    }

    // The background pass starts at this first accepted opaque/fully-covered
    // surface. Stop before evaluating it again.
    break;
  }

  textureStore(
    oit_transparentOut,
    pix,
    vec4f(accum + background * transmittance, 1.0),
  );
}
`;

export const TRANSPARENT_OIT_MODULE: WgslModule = {
  name: 'transparentOit',
  source: TRANSPARENT_OIT_WGSL,
  requires: ['common', 'materialAtlas', 'surfaceTextures', 'environmentSample', 'ggxBrdf', 'emitterLeAtXi', 'ddgiGridUbo', 'sampleCascadeC0'],
};
