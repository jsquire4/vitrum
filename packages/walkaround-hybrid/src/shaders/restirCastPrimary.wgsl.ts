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

// ============================================================
// Canonical primary-surface cast used by temporal + spatial.
//   Generates a primary ray for (px, dims) through (camPos, invVP),
//   intersects against the BVH, and decodes the hit into a
//   PrimarySurface (defined in common.wgsl).
//
//   See restirCastPrimary.wgsl.ts header for why shade and ris
//   intentionally retain inline primary casts.
// ============================================================
fn castPrimary(px: vec2u, dims: vec2u, camPos: vec3f, invVP: mat4x4f) -> PrimarySurface {
  var s: PrimarySurface;
  let ray = generatePrimaryRay_common(px.x, px.y, dims.x, dims.y, camPos, invVP);
  let hit = traceSceneFirstHit(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    ray, ubo.triIntersectEpsilon);
  s.hit = hit.didHit;
  if (!hit.didHit) {
    return s;
  }
  s.pos    = ray.origin + ray.direction * hit.dist;
  s.normal = hit.normal;
  s.wo     = -ray.direction;
  let matColor = decodeMaterialColor(hit.matColorPacked);
  let isGlass  = matColor.a > 0.3;
  s.albedo = matColor.rgb;
  s.rough  = select(0.85, 0.05, isGlass);
  s.metal  = 0.0;
  s.depth  = hit.dist;
  return s;
}
`;

/** W2-C9 — declarative include-graph entry for the canonical primary cast. */
export const RESTIR_CAST_PRIMARY_MODULE: WgslModule = {
  name: 'restirCastPrimary',
  source: RESTIR_CAST_PRIMARY_WGSL,
  // Depends on `common` for PrimarySurface, bvh bindings, ubo binding,
  // generatePrimaryRay_common, bvhIntersectFirstHit, decodeMaterialColor.
  requires: ['common'],
};
