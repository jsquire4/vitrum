/**
 * Canonical ReSTIR primary-surface cast.
 *
 * W2-C9 canonicalisation (premium-grade-refactor-20260517 §W2):
 *   Pre-refactor, `castPrimary` (spatial.wgsl.ts:47) and `castPrimary_t`
 *   (temporal.wgsl.ts:47) were bit-identical 18-line functions modulo
 *   the parameter name for the camera position. Shade.wgsl inlines a
 *   similar but semantically distinct cast (it consumes raw Hit fields
 *   that PrimarySurface does not carry — see note below).
 *
 *   Spatial and temporal both consume the cast result through the same
 *   `PrimarySurface` shape; this module provides the single declaration
 *   site, parameterised on `camPos` to match temporal's pre-refactor
 *   signature. Spatial gets the cheaper signature for free since it
 *   already had `ubo.cameraPos` lexically available.
 *
 *   Shade keeps its inline primary cast because it additionally needs
 *   `hit.triIndex`, `hit.uv`, `hit.matColorPacked` for the glass-surface
 *   texture-id lookup, the per-tri Beer-Lambert visible-colour read
 *   (`bvh_beer[triIndex]`), and `decodeIsMetal` / `decodeSurfaceTextureId`
 *   — fields that `PrimarySurface` does not carry. Replacing shade's cast
 *   with this helper would require either widening PrimarySurface (touches
 *   every other consumer) or threading the Hit struct alongside, neither
 *   of which is the C9 cleanup. Shade's inline cast is documented in
 *   shade.wgsl.ts.
 *
 *   RIS likewise inlines its primary cast because the surface-decode is
 *   load-bearing for the subsequent M_LIGHT loop (where `albedo`/`rough`/
 *   `metal` feed per-candidate p̂ with sampled emitter point `ls.pos`,
 *   not the centroid used by the canonical helper); RIS has no
 *   `castPrimary*` function to remove.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESTIR_CAST_PRIMARY_WGSL = /* wgsl */ `

// B1 — per-triangle roughness+metalness (r32uint texture). Decoded into the
// PrimarySurface rough/metal so temporal+spatial DI reuse evaluate the GGX p̂
// with the real authored material (was hardcoded rough=0.85/0.05, metal=0).
@group(1) @binding(14) var bvh_material: texture_2d<u32>;

// ============================================================
// Canonical primary-surface cast used by temporal + spatial.
//   Generates a primary ray for (px, dims) through (camPos, invVP),
//   intersects against the BVH, and decodes the hit into a
//   PrimarySurface (defined in reservoirGi.wgsl).
//
//   See restirCastPrimary.wgsl.ts header for why shade and ris
//   intentionally retain inline primary casts.
// ============================================================
fn primarySurfaceFromRay(ray: Ray) -> PrimarySurface {
  var s: PrimarySurface;
  let hit = traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(
    ubo.bvhMode, ubo.tlasNodeCount,

    ray, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, 0u);
  s.hit = hit.didHit;
  if (!hit.didHit) {
    return s;
  }
  s.pos    = ray.origin + ray.direction * hit.dist;
  let geoNormal = hit.normal;
  s.geoNormal = geoNormal;
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = hit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < tlasWorldToLocalColumnCount();
  let n_i = select(0u, n_base, n_ok);
  let smoothNormal = smoothShadingNormal(
    hit, geoNormal,
    sceneLoadBvhNormal(hit.indices.x).xyz, sceneLoadBvhNormal(hit.indices.y).xyz, sceneLoadBvhNormal(hit.indices.z).xyz,
    n_ok,
    tlasLoadWorldToLocalColumn(n_i), tlasLoadWorldToLocalColumn(n_i + 1u), tlasLoadWorldToLocalColumn(n_i + 2u),
  );
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  s.normal = applyBumpMapForHit(hit, normalMapped);
  s.wo     = -ray.direction;
  let scalarMatColor = decodeMaterialColor(hit.matColorPacked);
  let matColor = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(hit, scalarMatColor.a),
  );
  // B1 — real authored roughness/metalness from the per-tri bvh_material texture
  // (was hardcoded select(0.85,0.05,isGlass) / metal 0). The diffuse-default
  // invariant keeps default-diffuse surfaces at 0.85 / glass at 0.05.
  let rmCoord = vec2u(hit.indices.w % BVH_MATERIAL_TEX_WIDTH, hit.indices.w / BVH_MATERIAL_TEX_WIDTH);
  let materialWord = textureLoad(bvh_material, vec2i(rmCoord), 0).r;
  let payload = sampleRestirDIMaterialPayloadForHit(hit, smoothNormal, s.normal, matColor.rgb, materialWord, s.wo);
  s.clearcoatNormal = payload.clearcoatNormal;
  s.albedo = payload.albedo;
  s.rough  = payload.rough;
  s.metal  = payload.metal;
  s.transmission = matColor.a;
  s.isGlass = materialHasTransmission(matColor.a);
  s.specular = payload.specular;
  s.anisotropy = payload.anisotropy;
  s.anisotropyTangent = payload.anisotropyTangent;
  s.anisotropyBitangent = payload.anisotropyBitangent;
  s.iridescence = payload.iridescence;
  s.clearcoat = payload.clearcoat;
  s.sheen = payload.sheen;
  s.sheenRoughness = payload.sheenRoughness;
  s.reflectionLayerTransmission = payload.reflectionLayerTransmission;
  s.layerTransmission = payload.layerTransmission;
  s.volumeScattering = payload.volumeScattering;
  s.bulkThickness = payload.bulkThickness;
  s.envMapIntensity = payload.envMapIntensity;
  s.depth  = hit.dist;
  s.triangleId = hit.indices.w;
  s.instanceId = select(0u, hit.instanceIndex, ubo.bvhMode == 1u);
  s.materialKey = restir_gi_receiver_domain_key(
    hit.matColorPacked,
    materialWord,
    hit.indices.w,
    s.instanceId,
    payload,
    matColor.a,
  );
  return s;
}

fn castPrimary(px: vec2u, dims: vec2u, camPos: vec3f, invVP: mat4x4f) -> PrimarySurface {
  let ray = generatePrimaryRay_common(px.x, px.y, dims.x, dims.y, camPos, invVP);
  return primarySurfaceFromRay(ray);
}

// Re-cast a pixel from a historical camera represented only by its inverse VP.
// The near-plane ray origin avoids inventing a previous camera position and
// lets temporal reuse validate correspondence against the CURRENT BVH.
fn castPrimaryFromInvVP(px: vec2u, dims: vec2u, invVP: mat4x4f) -> PrimarySurface {
  let ray = generatePrimaryRayFromInvVP_common(px.x, px.y, dims.x, dims.y, invVP);
  return primarySurfaceFromRay(ray);
}
`;

/** W2-C9 — declarative include-graph entry for the canonical primary cast. */
export const RESTIR_CAST_PRIMARY_MODULE: WgslModule = {
  name: 'restirCastPrimary',
  source: RESTIR_CAST_PRIMARY_WGSL,
  // Focused closure: UBO + traversal/types + PrimarySurface + shared math +
  // material atlas payloads + camera ray generation.
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialAtlas', 'restirGiMaterial', 'cameraRays'],
};
