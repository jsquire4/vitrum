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
// TLAS mode (V21): bvh_normal holds LOCAL-space BLAS normals, so the blended
// shading normal is transformed to WORLD by the hit instance's inverse-transpose
// (tlasTransformNormalFromLocalCols with the instance world-to-local columns —
// the SAME transform traceTlasFirstHit applies to the geometric normal). The
// caller passes isTlas + the three world-to-local columns (read from the
// module-scope tlasInstanceWorldToLocal binding at instanceIndex*4); merged
// mode passes isTlas=false and the blend is already world-space. (The earlier
// wave kept the geometric normal in TLAS — that left smooth shading dormant on
// every multi-mesh / instanced scene, which all auto-select TLAS.)
//
// Degenerate guard: if the blended vector collapses (antipodal vertex normals
// across a thin/folded triangle) we fall back to the geometric face normal so
// the result stays finite + unit-length.
// Takes the three per-vertex normals BY VALUE (n0/n1/n2) rather than the
// bvh_normal storage buffer by pointer: Naga (wgpu-native / Firefox) rejects
// ptr<storage> function parameters, so a value-arg signature is naga-native
// and needs no shader-rewrite shim. Callers load bvh_normal[hit.indices.xyz]
// inline at the call site (indexing a module-scope storage global is fine; only
// passing it AS a ptr<storage> param is the Naga gap). Caught by the wsl-gpu
// T1 smoke gate (lavapipe/naga) — the prior ptr-param form failed to compile.
// In TLAS mode the per-vertex normals (n0/n1/n2) are LOCAL-space BLAS normals, so
// the blended shading normal is transformed to world by the SAME inverse-transpose
// the geometric normal uses (tlasTransformNormalFromLocalCols with the instance's
// world-to-local columns). The caller reads those columns from the module-scope
// tlasInstanceWorldToLocal binding (instanceIndex*4) and passes them BY VALUE —
// Naga rejects ptr<storage> params, but value vec4f args + a bool are naga-native.
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
    n = worldN / wl;
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
