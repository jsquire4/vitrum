/**
 * Camera-visible transparent composition for walkaround-hybrid.
 *
 * The opaque shade/denoise chain now skips fractional `alphaMode:'blend'`
 * surfaces. This pass walks the same primary ray front-to-back, collects those
 * transparent layers using the atlas-backed alpha coverage helper, and
 * composites them over the already-denoised opaque/background radiance.
 *
 * Lighting policy: transparent layer radiance is an intentionally cheap
 * camera-visible approximation. The sky/environment, direct sun, and analytic
 * point/spot/finite-emitter terms use the same atlas-backed material-lobe BRDF
 * as opaque shade/ReSTIR material scoring; their shadow rays use the canonical
 * material-atlas alpha transmittance helper, including the post-budget opaque
 * blocker fallback used by shade/ReSTIR.
 * Emissive, light-map, and finite-emitter terms remain camera-visible
 * approximations rather than transparent ReSTIR/GI reservoir participation.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const TRANSPARENT_OIT_WGSL = /* wgsl */ `

@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(1) @binding(12) var bvh_emissive: texture_2d<f32>;
@group(1) @binding(13) var analytic_lights: texture_2d<f32>;
@group(1) @binding(14) var bvh_material: texture_2d<u32>;

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

@group(3) @binding(0) var oit_background: texture_2d<f32>;
@group(3) @binding(1) var oit_transparentOut: texture_storage_2d<rgba16float, write>;

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

fn oitShadowTransmittance(origin: vec3f, dir: vec3f, tMax: f32, triEps: f32) -> f32 {
  return traceSceneAlphaTransmittanceTextured(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    &bvh_index,
    &bvh_position,
    &bvh,
    &tlasNodes,
    &tlasInstanceIndices,
    &tlasBlasRoots,
    &tlasInstanceWorldToLocal,
    &tlasInstanceLocalToWorld,
    origin,
    dir,
    max(tMax, 0.0),
    triEps,
    true,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
  );
}

fn oitSpotConeFalloff(lightDir: vec3f, wi: vec3f, cosInner: f32, cosOuter: f32) -> f32 {
  let axisLen2 = dot(lightDir, lightDir);
  if (axisLen2 <= 0.01) { return 1.0; }
  let axis = lightDir * inverseSqrt(axisLen2);
  let cosTheta = dot(-axis, wi);
  if (cosTheta < cosOuter) { return 0.0; }
  if (abs(cosInner - cosOuter) < 1e-5) {
    return select(0.0, 1.0, cosTheta >= cosOuter);
  }
  return smoothstep(cosOuter, cosInner, cosTheta);
}

fn oitPointSpotAttenuation(dist: f32, cutoffDistance: f32, decay: f32, dist2Floor: f32) -> f32 {
  var attenuation = 1.0;
  if (decay > 0.01) {
    if (abs(decay - 2.0) < 1e-5) {
      attenuation = 1.0 / (dist * dist + dist2Floor);
    } else {
      attenuation = 1.0 / max(pow(max(dist, 1.0), decay), max(dist2Floor, 1e-6));
    }
  }
  if (cutoffDistance > 0.0) {
    let x = clamp(1.0 - pow(dist / cutoffDistance, 4.0), 0.0, 1.0);
    attenuation = attenuation * x * x;
  }
  return attenuation;
}

struct OitLayerNormals {
  smoothNormal: vec3f,
  shadingNormal: vec3f,
};

fn oitLayerNormals(hit: IntersectionResult) -> OitLayerNormals {
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = hit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < arrayLength(&tlasInstanceWorldToLocal);
  let n_i = select(0u, n_base, n_ok);
  let n0 = bvh_normal[hit.indices.x];
  let n1 = bvh_normal[hit.indices.y];
  let n2 = bvh_normal[hit.indices.z];
  let smoothNormal = smoothShadingNormal(
    hit,
    hit.normal,
    n0.xyz,
    n1.xyz,
    n2.xyz,
    n_ok,
    tlasInstanceWorldToLocal[n_i],
    tlasInstanceWorldToLocal[n_i + 1u],
    tlasInstanceWorldToLocal[n_i + 2u],
  );
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  var normals: OitLayerNormals;
  normals.smoothNormal = smoothNormal;
  normals.shadingNormal = applyBumpMapForHit(hit, normalMapped);
  return normals;
}

fn oitEnvSampleDir(n: vec3f, tangentScale: f32, bitangentScale: f32) -> vec3f {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.95) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  let tangent = safe_normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  return safe_normalize(n + tangent * tangentScale + bitangent * bitangentScale);
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

fn oitLayerSkyRadiance(payload: RestirDIMaterialPayload, normal: vec3f, wo: vec3f) -> vec3f {
  // Deterministic five-tap hemisphere estimate for transparent layers. This
  // keeps the pass temporally stable while letting clearcoat, sheen, anisotropy,
  // iridescence, and envMapIntensity affect camera-visible alpha-blended sky
  // light for both HDRI-backed scenes and no-HDRI scalar/procedural sky fallback.
  let d0 = normal;
  let d1 = oitEnvSampleDir(normal,  0.70,  0.00);
  let d2 = oitEnvSampleDir(normal, -0.70,  0.00);
  let d3 = oitEnvSampleDir(normal,  0.00,  0.70);
  let d4 = oitEnvSampleDir(normal,  0.00, -0.70);
  let avg =
      oitLayerEnvSampleRadiance(payload, normal, wo, d0)
    + oitLayerEnvSampleRadiance(payload, normal, wo, d1)
    + oitLayerEnvSampleRadiance(payload, normal, wo, d2)
    + oitLayerEnvSampleRadiance(payload, normal, wo, d3)
    + oitLayerEnvSampleRadiance(payload, normal, wo, d4);
  return avg * (2.0 * PI / 5.0);
}

fn oitLayerAnalyticNEE(
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  payload: RestirDIMaterialPayload,
  wo: vec3f,
) -> vec3f {
  let analyticDims = textureDimensions(analytic_lights);
  let analyticHeader = textureLoad(analytic_lights, vec2i(0, 0), 0);
  let count = u32(max(analyticHeader.x, 0.0));
  if (count == 0u) { return vec3f(0.0); }

  var Lo = vec3f(0.0);
  for (var li = 0u; li < count; li++) {
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
    if (dist < 1e-4) { continue; }
    let wi = toL / dist;
    let nDotL = dot(normal, wi);
    if (nDotL <= 0.0) { continue; }

    let cone = oitSpotConeFalloff(lightDir, wi, cosInner, cosOuter);
    if (cone <= 0.0) { continue; }

    var shadowT = 1.0;
    if (!castShadowDisabled) {
      shadowT = oitShadowTransmittance(
        hitPos + geoNormal * 1e-3,
        wi,
        max(dist - 2e-3, 0.0),
        ubo.triIntersectEpsilon,
      );
      if (shadowT <= 0.001) { continue; }
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
    Lo += lightLe * brdf * cone * attenuation * shadowT;
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
) -> vec3f {
  let count = min(ubo.emitterCount, arrayLength(&emitters));
  if (count == 0u) { return vec3f(0.0); }

  var Lo = vec3f(0.0);
  // Deterministic area estimate: four fixed stratified uniform-area samples per
  // emitter. This is still camera-visible OIT lighting, not transparent
  // ReSTIR/GI participation, but it avoids the old centroid-only blind spot on
  // large or emissive-textured finite emitters while staying temporally stable.
  // This gives transparent layers native finite-emitter visibility without
  // coupling the OIT pass to the opaque ReSTIR-DI reservoir state.
  let sampleWeight = 1.0 / f32(OIT_AREA_EMITTER_SAMPLE_COUNT);
  for (var lid = 0u; lid < count; lid = lid + 1u) {
    let e = emitters[lid];
    for (var si = 0u; si < OIT_AREA_EMITTER_SAMPLE_COUNT; si = si + 1u) {
      let xi = oitAreaEmitterXi(si);
      let ls = sampleEmitterPoint(e, xi);
      let toL = ls.pos - hitPos;
      let dist2 = dot(toL, toL);
      if (dist2 < 1e-8 || ls.area <= 0.0) { continue; }

      let dist = sqrt(dist2);
      let wi = toL / dist;
      let nDotL = max(0.0, dot(normal, wi));
      let nlDotL = max(0.0, dot(-ls.normal, wi));
      if (nDotL < 1e-6 || nlDotL < 1e-6) { continue; }

      var shadowT = 1.0;
      if (e.castShadowDisabled < 0.5) {
        shadowT = oitShadowTransmittance(
          hitPos + geoNormal * 1e-3,
          wi,
          max(dist - 2e-3, 0.0),
          ubo.triIntersectEpsilon,
        );
        if (shadowT <= 0.001) { continue; }
      }

      let G = emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor);
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
      Lo += Le * brdf * G * ls.area * shadowT * sampleWeight;
    }
  }
  return Lo;
}

fn oitLayerRadiance(hit: IntersectionResult, hitPos: vec3f, rayDir: vec3f, materialWord: u32) -> vec3f {
  let scalarBase = decodeMaterialColor(hit.matColorPacked).rgb;
  let uv1 = materialAtlasUv1ForHit(hit);
  let normals = oitLayerNormals(hit);
  let normal = normals.shadingNormal;
  let payload = sampleRestirDIMaterialPayloadForHit(hit, normals.smoothNormal, normal, scalarBase, materialWord);
  let wo = safe_normalize(-rayDir);

  let emitCoord = vec2u(hit.indices.w % BVH_MATERIAL_TEX_WIDTH, hit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let emissive = sampleEmissiveMap(
    hit.indices.w,
    hit.uv,
    uv1,
    textureLoad(bvh_emissive, vec2i(emitCoord), 0).rgb,
  );
  let baked = sampleLightMap(hit.indices.w, hit.uv, uv1);

  let skyAmbient = oitLayerSkyRadiance(payload, normal, wo);
  let analyticDirect = oitLayerAnalyticNEE(hitPos, normal, payload.clearcoatNormal, hit.normal, payload, wo);
  let areaDirect = oitLayerAreaEmitterNEE(hitPos, normal, payload.clearcoatNormal, hit.normal, payload, wo);
  let sunBase = safe_normalize(ubo.sunDirection);
  let sunXi = worldHash2(hitPos, hit.indices.w ^ 0x4f495431u);
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
  var sunVisibility = 1.0;
  if ((ubo.stainedGlassFlags & SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) == 0u) {
    sunVisibility = oitShadowTransmittance(
      hitPos + hit.normal * 1e-3,
      toSun,
      1e6,
      ubo.triIntersectEpsilon,
    );
  }
  let sunDirect = vec3f(ubo.sunIntensity) * sunBrdf * sunVisibility;
  let viewFacing = 0.35 + 0.65 * abs(dot(normal, -rayDir));
  return (skyAmbient + sunDirect + analyticDirect + areaDirect) * viewFacing + emissive + baked;
}

@compute @workgroup_size(8, 8, 1)
fn transparentOitMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(oit_transparentOut);
  let pix = gid.xy;
  if (any(pix >= dims)) { return; }

  let background = textureLoad(oit_background, vec2i(pix), 0).rgb;
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let primaryRay = generatePrimaryRay_common(pix.x, pix.y, dims.x, dims.y, ubo.cameraPos, invVP);

  var walkRay = primaryRay;
  var traveled = 0.0;
  var transmittance = 1.0;
  var accum = vec3f(0.0);
  let step = max(1e-4, ubo.triIntersectEpsilon * 4.0);

  for (var layer = 0u; layer < 32u; layer = layer + 1u) {
    let hit = traceSceneFirstHit(
      ubo.bvhMode,
      ubo.tlasNodeCount,
      &bvh_index,
      &bvh_position,
      &bvh,
      &tlasNodes,
      &tlasInstanceIndices,
      &tlasBlasRoots,
      &tlasInstanceWorldToLocal,
      &tlasInstanceLocalToWorld,
      walkRay,
      ubo.triIntersectEpsilon,
    );
    if (!hit.didHit || transmittance <= 0.001) {
      break;
    }

    let word = oitMaterialWord(hit.indices.w);
    let alpha = materialAlphaCoverageForHit(hit, word);
    if (oitHitIsMaskDiscarded(hit, alpha)) {
      traveled = traveled + hit.dist + step;
      walkRay.origin = primaryRay.origin + primaryRay.direction * traveled;
      continue;
    }

    if (alpha.mode == 2u && alpha.coverage <= 0.001) {
      traveled = traveled + hit.dist + step;
      walkRay.origin = primaryRay.origin + primaryRay.direction * traveled;
      continue;
    }

    if (alpha.mode == 2u && alpha.coverage < 0.999) {
      let a = clamp(alpha.coverage, 0.0, 1.0);
      let hitPos = walkRay.origin + walkRay.direction * hit.dist;
      let layerRadiance = oitLayerRadiance(hit, hitPos, primaryRay.direction, word);
      accum = accum + layerRadiance * a * transmittance;
      transmittance = transmittance * (1.0 - a);
      traveled = traveled + hit.dist + step;
      walkRay.origin = primaryRay.origin + primaryRay.direction * traveled;
      continue;
    }

    break;
  }

  textureStore(oit_transparentOut, pix, vec4f(accum + background * transmittance, 1.0));
}
`;

export const TRANSPARENT_OIT_MODULE: WgslModule = {
  name: 'transparentOit',
  source: TRANSPARENT_OIT_WGSL,
  requires: ['common', 'materialAtlas', 'environmentSample', 'ggxBrdf', 'emitterLeAtXi'],
};
