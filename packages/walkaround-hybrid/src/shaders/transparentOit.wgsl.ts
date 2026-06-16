/**
 * Camera-visible transparent composition for walkaround-hybrid.
 *
 * The opaque shade/denoise chain now skips fractional `alphaMode:'blend'`
 * surfaces. This pass walks the same primary ray front-to-back, collects those
 * transparent layers using the atlas-backed alpha coverage helper, and
 * composites them over the already-denoised opaque/background radiance.
 *
 * Lighting policy: transparent layer radiance is an intentionally cheap
 * camera-visible approximation. The direct sun term uses the same atlas-backed
 * material-lobe BRDF as opaque shade/ReSTIR material scoring, with the same
 * castShadow-aware scene visibility query as opaque direct sun. Sky ambient,
 * emissive, and light-map terms remain first-hit camera-visible approximations.
 * ReSTIR/GI participation remains handled by the existing stochastic traversal
 * path until transparent GI has its own validation row.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const TRANSPARENT_OIT_WGSL = /* wgsl */ `

@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(1) @binding(12) var bvh_emissive: texture_2d<f32>;
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

fn oitLayerRadiance(hit: IntersectionResult, hitPos: vec3f, rayDir: vec3f, materialWord: u32) -> vec3f {
  let scalarBase = decodeMaterialColor(hit.matColorPacked).rgb;
  let uv1 = materialAtlasUv1ForHit(hit);
  let normals = oitLayerNormals(hit);
  let normal = normals.shadingNormal;
  let payload = sampleRestirDIMaterialPayloadForHit(hit, normals.smoothNormal, normal, scalarBase, materialWord);

  let emitCoord = vec2u(hit.indices.w % BVH_MATERIAL_TEX_WIDTH, hit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let emissive = sampleEmissiveMap(
    hit.indices.w,
    hit.uv,
    uv1,
    textureLoad(bvh_emissive, vec2i(emitCoord), 0).rgb,
  );
  let baked = sampleLightMap(hit.indices.w, hit.uv, uv1);

  let skyAmbient = envRadiance(normal) * payload.albedo * INV_PI;
  let wo = safe_normalize(-rayDir);
  let toSun = safe_normalize(ubo.sunDirection);
  let sunBrdf = evalGGXWithSpecularClearcoatSheen(
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
    normal,
    payload.clearcoatNormal,
    wo,
    toSun,
  );
  var sunVisibility = 1.0;
  if ((ubo.stainedGlassFlags & SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) == 0u) {
    let sunOccluded = traceSceneAnyCastMask(
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
      hitPos + hit.normal * 1e-3,
      toSun,
      1e6,
      ubo.triIntersectEpsilon,
      true,
      bvh_material,
      BVH_MATERIAL_TEX_WIDTH,
    );
    sunVisibility = select(1.0, 0.0, sunOccluded);
  }
  let sunDirect = vec3f(ubo.sunIntensity) * sunBrdf * sunVisibility;
  let viewFacing = 0.35 + 0.65 * abs(dot(normal, -rayDir));
  return (skyAmbient + sunDirect) * viewFacing + emissive + baked;
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

    if (alpha.mode == 2u && alpha.coverage > 0.001 && alpha.coverage < 0.999) {
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
  requires: ['common', 'materialAtlas', 'environmentSample', 'ggxBrdf'],
};
