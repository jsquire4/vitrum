/**
 * Scene traversal — canonical BVH/TLAS intersection helpers + the
 * merged-world-BVH-vs-TLAS+local-BLAS dispatch wrappers (PR-3).
 *
 * Split out of common.wgsl.ts (T9-stepA). Injects the canonical
 * `BVH_INTERSECT_WGSL` + `TLAS_TRAVERSAL_WGSL` from `@vitrum/shared-bvh`
 * (single source of truth for BVHNode, Ray, IntersectionResult,
 * intersectTriangle, bvhIntersectFirstHit/Any, traceTlas*), then defines
 * `traceSceneFirstHit` plus the cast-mask-aware shadow wrapper which pick the
 * path from `ubo.bvhMode` / `ubo.tlasNodeCount`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  BVH_CAST_SHADOW_MASK_WGSL,
  BVH_INTERSECT_CORE_WGSL,
  TLAS_TRAVERSAL_CORE_WGSL,
} from '@vitrum/shared-bvh';
import { SCENE_STORAGE_ARENA_WGSL } from './sceneStorageArena.wgsl.js';

export const SCENE_TRAVERSAL_WGSL = /* wgsl */ `// ============================================================
// BVH structs + intersection helpers — canonical from @vitrum/shared-bvh
// (sweep-20260518/moller-trumbore-canonical). Single source of truth for
// BVHNode, Ray, IntersectionResult, safeInvDir, intersectTriangle,
// bvhIntersectFirstHit, bvhIntersectAny. Pre-canonical inline copies were
// here (lines 128-164 + 480-735 in the pre-refactor file).
//
// Migration notes:
//   - The canonical return type is IntersectionResult (superset). The
//     pre-canonical HitResult is gone; its bary field is now barycoord,
//     and triIndex is now indices.w (matches DDGI / RC conventions).
//   - intersectTriangle now returns IntersectionResult (not f32). The
//     one remaining inline caller (bvhTraceTintedVisibility in
//     surfaceTextures.wgsl) unwraps .dist / .didHit at the call site.
//   - Cast-mask-aware any-hit traversal carries both skipGlass and packed
//     cast-shadow/alpha flags. Tinted transmission uses the dedicated
//     per-channel visibility walk in surfaceTextures.wgsl.
// ============================================================
${BVH_INTERSECT_CORE_WGSL}
${SCENE_STORAGE_ARENA_WGSL}
${TLAS_TRAVERSAL_CORE_WGSL}

// Scene traversal — merged world BVH vs TLAS+local BLAS (PR-3).
fn traceSceneFirstHit(
  bvhMode: u32,
  tlasNodeCount: u32,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  if (bvhMode == 1u && tlasNodeCount > 0u) {
    return traceTlasFirstHit(tlasNodeCount, ray, triEps);
  }
  return bvhIntersectFirstHit(ray, triEps);
}

${BVH_CAST_SHADOW_MASK_WGSL}

// SHADOW-01 — castShadow-aware occlusion wrapper for the ReSTIR **DI** shadow
// predicates (ris.wgsl candidate visibility, ReSTIR-GI visibility, GRIS
// reconnection visibility, and shadingTerms.wgsl shading / analytic / sun
// visibility). The leaf loops skip triangles whose bvh_material word has bit 0 set
// (castShadow:false — packBVHRoughMetalFromCore) or bit 2 set
// (scalar alpha discarded). Callers pass the
// module-scope bvh_material texture + BVH_MATERIAL_TEX_WIDTH so this module
// stays binding-free. DDGI / RC use the sibling predicate-backed shared-bvh
// traversal because those passes carry material flags through MaterialEntry
// buffers rather than this texture.
fn traceSceneAnyCastMask(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
  castMask: texture_2d<u32>,
  castMaskWidth: u32,
) -> bool {
  if (bvhMode == 1u && tlasNodeCount > 0u) {
    return traceTlasAnyCastMask(
      tlasNodeCount,
      origin,
      dir,
      tMax,
      triEps,
      skipGlass,
      castMask,
      castMaskWidth,
    );
  }
  return bvhIntersectAnyAtRootCastMask(origin, dir, tMax, triEps, skipGlass, 0u, castMask, castMaskWidth);
}

// ─── WS1 (2026-05-29) — smooth shading normal via barycentric per-vertex blend ─
//
// Mirrors the DDGI precedent (probeUpdateRays.wgsl.ts:443-454): interpolate the
// per-vertex normals at the hit's barycentric coordinate, normalize, and apply
// the hit side so back-face hits get a consistently-oriented normal —
//   n = normalize(w·n0 + u·n1 + v·n2) · side.
//
// This replaces the faceted geometric face normal (hit.normal, cross(e1,e2))
// for SHADING. The geometric normal must still be used for ray-origin offsets /
// backface bias by the caller (a smooth normal can point into the surface near
// a silhouette edge, which would self-intersect the offset ray).
//
// TLAS mode (V21): the geometry arena holds LOCAL-space BLAS normals, so the blended
// shading normal is transformed to WORLD by the hit instance's inverse-transpose
// (tlasTransformNormalFromLocalCols with the instance world-to-local columns —
// the SAME transform traceTlasFirstHit applies to the geometric normal). The
// caller passes isTlas + the three world-to-local columns (read from the
// arena-backed world-to-local loader at instanceIndex*4); merged
// mode passes isTlas=false and the blend is already world-space. (The earlier
// wave kept the geometric normal in TLAS — that left smooth shading dormant on
// every multi-mesh / instanced scene, which all auto-select TLAS.)
//
// Degenerate guard: if the blended vector collapses (antipodal vertex normals
// across a thin/folded triangle) we fall back to the geometric face normal so
// the result stays finite + unit-length.
// Takes the three per-vertex normals BY VALUE (n0/n1/n2) rather than the
// arena-backed normal stream by pointer: Naga (wgpu-native / Firefox) rejects
// storage-buffer pointer function parameters, so a value-arg signature is naga-native
// and needs no shader-rewrite shim. Callers load sceneLoadBvhNormal(hit.indices.xyz)
// inline at the call site (indexing a module-scope storage global is fine; only
// passing it as a storage-buffer pointer param is the Naga gap). Caught by the wsl-gpu
// T1 smoke gate (lavapipe/naga) — the prior ptr-param form failed to compile.
// In TLAS mode the per-vertex normals (n0/n1/n2) are LOCAL-space BLAS normals, so
// the blended shading normal is transformed to world by the SAME inverse-transpose
// the geometric normal uses (tlasTransformNormalFromLocalCols with the instance's
// world-to-local columns). The caller reads those columns through the shared
// TLAS-arena loader (instanceIndex*4) and passes them BY VALUE —
// Naga rejects storage-buffer pointer params, but value vec4f args + a bool are naga-native.
// In merged-world mode isTlas is false and the blend is already world-space.
fn smoothShadingNormal(
  hit: IntersectionResult,
  geoNormal: vec3f,
  n0: vec3f,
  n1: vec3f,
  n2: vec3f,
  isTlas: bool,
  w2l0: vec4f,
  w2l1: vec4f,
  w2l2: vec4f,
) -> vec3f {
  let blended =
    hit.barycoord.x * n0 +
    hit.barycoord.y * n1 +
    hit.barycoord.z * n2;
  let len = length(blended);
  if (len < 1e-6) { return geoNormal; }
  var n = blended / len;
  if (isTlas) {
    let worldN = tlasTransformNormalFromLocalCols(w2l0, w2l1, w2l2, n);
    let wl = length(worldN);
    if (wl < 1e-6) { return geoNormal; }
    // hit.side is world-winding parity corrected by TLAS traversal. Apply
    // the transform parity to the authored local normal before multiplying by
    // that side so the final shading normal remains face-forward on mirrored
    // instances as well as non-mirrored ones.
    n = (worldN / wl) * tlasLinearOrientationSign(w2l0, w2l1, w2l2);
  }
  return n * hit.side;
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const SCENE_TRAVERSAL_MODULE: WgslModule = {
  name: "sceneTraversal",
  source: SCENE_TRAVERSAL_WGSL,
  requires: [],
};
