/**
 * Scene traversal — canonical BVH/TLAS intersection helpers + the
 * merged-world-BVH-vs-TLAS+local-BLAS dispatch wrappers (PR-3).
 *
 * Split out of common.wgsl.ts (T9-stepA). Injects the canonical
 * `BVH_INTERSECT_WGSL` + `TLAS_TRAVERSAL_WGSL` from `@vitrum/shared-bvh`
 * (single source of truth for BVHNode, Ray, IntersectionResult,
 * intersectTriangle, bvhIntersectFirstHit/Any, traceTlas*), then defines
 * `traceSceneFirstHit` / `traceSceneAny` which pick the path from
 * `ubo.bvhMode` / `ubo.tlasNodeCount`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { BVH_INTERSECT_WGSL, TLAS_TRAVERSAL_WGSL } from '@vitrum/shared-bvh';

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
//   - bvhIntersectAny gains a skipGlass: bool parameter. All ReSTIR
//     call sites pass true (matches the pre-canonical glass-transmissive
//     shadow behaviour — light passes through, tint is applied by the
//     per-channel bvhTraceTintedVisibility helper in shade).
// ============================================================
${BVH_INTERSECT_WGSL}
${TLAS_TRAVERSAL_WGSL}

// Scene traversal — merged world BVH vs TLAS+local BLAS (PR-3).
fn traceSceneFirstHit(
  bvhMode: u32,
  tlasNodeCount: u32,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  bvh: ptr<storage, array<BVHNode>, read>,
  tlasNodes: ptr<storage, array<BVHNode>, read>,
  tlasInstanceIndices: ptr<storage, array<u32>, read>,
  tlasBlasRoots: ptr<storage, array<u32>, read>,
  tlasInstanceWorldToLocal: ptr<storage, array<vec4f>, read>,
  tlasInstanceLocalToWorld: ptr<storage, array<vec4f>, read>,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  if (bvhMode == 1u && tlasNodeCount > 0u) {
    return traceTlasFirstHit(
      tlasNodes,
      tlasInstanceIndices,
      tlasBlasRoots,
      tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld,
      tlasNodeCount,
      bvh_index,
      bvh_position,
      bvh,
      ray,
      triEps,
    );
  }
  return bvhIntersectFirstHit(bvh_index, bvh_position, bvh, ray, triEps);
}

fn traceSceneAny(
  bvhMode: u32,
  tlasNodeCount: u32,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  bvh: ptr<storage, array<BVHNode>, read>,
  tlasNodes: ptr<storage, array<BVHNode>, read>,
  tlasInstanceIndices: ptr<storage, array<u32>, read>,
  tlasBlasRoots: ptr<storage, array<u32>, read>,
  tlasInstanceWorldToLocal: ptr<storage, array<vec4f>, read>,
  tlasInstanceLocalToWorld: ptr<storage, array<vec4f>, read>,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
) -> bool {
  if (bvhMode == 1u && tlasNodeCount > 0u) {
    return traceTlasAny(
      tlasNodes,
      tlasInstanceIndices,
      tlasBlasRoots,
      tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld,
      tlasNodeCount,
      bvh_index,
      bvh_position,
      bvh,
      origin,
      dir,
      tMax,
      triEps,
      skipGlass,
    );
  }
  return bvhIntersectAny(bvh_index, bvh_position, bvh, origin, dir, tMax, triEps, skipGlass);
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const SCENE_TRAVERSAL_MODULE: WgslModule = {
  name: "sceneTraversal",
  source: SCENE_TRAVERSAL_WGSL,
  requires: [],
};
