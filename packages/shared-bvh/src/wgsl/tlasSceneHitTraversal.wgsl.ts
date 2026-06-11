/**
 * TLAS traverse-into-BLAS for pt-webgpu `SceneHit` path tracing.
 *
 * Requires the including module to define `SceneHit`, `Ray`, `params`,
 * `tlasNodes`, `tlasInstanceIndices`, `tlasBlasRoots`, instance transform
 * buffers, `LEAFNODE_FLAG`, `INFINITY`, `intersectAabb`, `traceMeshBvh`, and
 * `transformPointCols` / `transformDirectionCols` /
 * `transformNormalFromWorldToLocalCols`.
 *
 * @see packages/pt-webgpu/src/wgsl/pathTrace/intersection.wgsl.ts
 */

import { TLAS_TRAVERSAL_STACK_DEPTH } from './tlasTraversal.wgsl.js';

/**
 * String anchor for the SceneHit initialisation block inside
 * `TLAS_SCENE_HIT_TRAVERSAL_WGSL`.  Callers that augment the traversal
 * function (e.g. pt-webgpu's `tlasSceneHitTraversalWithInstanceIndex`) must
 * match this exact substring so their `String.replace()` targets are stable.
 *
 * Mirrors the WGSL text at lines 24–25 of the exported string below.
 * If you change the normal-init or stack declaration, update this constant too.
 */
export const TLAS_SCENE_HIT_INIT_ANCHOR =
  `  (*hit).normal = vec3f(0.0, 1.0, 0.0);\n  var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;`;

/**
 * String anchor for the SceneHit per-instance assignment block inside
 * `TLAS_SCENE_HIT_TRAVERSAL_WGSL`.  Same stability contract as
 * `TLAS_SCENE_HIT_INIT_ANCHOR` above.
 *
 * Mirrors the WGSL text at lines 68–71 of the exported string below.
 */
export const TLAS_SCENE_HIT_ASSIGNMENT_ANCHOR =
  `          (*hit).triIndex = localHit.triIndex;\n          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);\n          // Barycentric weights are space-invariant`;

/**
 * WGSL snippet that provides `traceTlasClosest` and `traceTlasAny`.
 *
 * Required symbols — the including module MUST define ALL of these before this
 * snippet is composed in:
 *
 *   Types:
 *     SceneHit  — struct with fields: didHit (bool), dist (f32), triIndex (u32),
 *                 normal (vec3f), baryVW (vec2f)
 *     Ray       — struct with fields: origin (vec3f), direction (vec3f)
 *
 *   Uniforms / globals:
 *     params                        — uniform with field tlasNodeCount: u32
 *     tlasNodes                     — storage array<BvhNode>
 *     tlasInstanceIndices           — storage array<u32>
 *     tlasBlasRoots                 — storage array<u32>
 *     tlasInstanceWorldToLocal      — storage array<vec4f>
 *     tlasInstanceLocalToWorld      — storage array<vec4f>
 *
 *   Constants:
 *     LEAFNODE_FLAG   — u32 flag bit that marks leaf BVH nodes
 *     INFINITY        — f32 max representable value
 *
 *   Functions:
 *     intersectAabb(ray, bmin, bmax, tMin, tMax) -> bool
 *     traceMeshBvh(ray, tMin, tMax, closest, hit, blasRoot, fullGeom) -> bool
 *     transformPointCols(c0,c1,c2,c3, p)   -> vec3f
 *     transformDirectionCols(c0,c1,c2, d)  -> vec3f
 *     transformNormalFromWorldToLocalCols(c0,c1,c2, n) -> vec3f
 *
 * @see packages/pt-webgpu/src/wgsl/pathTrace/intersection.wgsl.ts — reference consumer
 */
export const TLAS_SCENE_HIT_TRAVERSAL_WGSL = /* wgsl */ `

fn traceTlasClosest(ray: Ray, tMin: f32, tMax: f32, hit: ptr<function, SceneHit>) -> bool {
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceMeshBvh(ray, tMin, tMax, true, hit, 0u, true);
  }
  (*hit).didHit = false;
  (*hit).dist = tMax;
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(params.tlasNodeCount, arrayLength(&tlasNodes))) { continue; }
    let node = tlasNodes[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!intersectAabb(ray, bmin, bmax, tMin, (*hit).dist)) { continue; }
    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= arrayLength(&tlasInstanceIndices)) { continue; }
        let instIdx = tlasInstanceIndices[permIdx];
        let m = instIdx * 4u;
        if (m + 3u >= arrayLength(&tlasInstanceWorldToLocal) || m + 3u >= arrayLength(&tlasInstanceLocalToWorld)) { continue; }
        let w2l0 = tlasInstanceWorldToLocal[m];
        let w2l1 = tlasInstanceWorldToLocal[m + 1u];
        let w2l2 = tlasInstanceWorldToLocal[m + 2u];
        let w2l3 = tlasInstanceWorldToLocal[m + 3u];
        let l2w0 = tlasInstanceLocalToWorld[m];
        let l2w1 = tlasInstanceLocalToWorld[m + 1u];
        let l2w2 = tlasInstanceLocalToWorld[m + 2u];
        let l2w3 = tlasInstanceLocalToWorld[m + 3u];
        var localRay: Ray;
        localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
        localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
        var localHit: SceneHit;
        let blasRoot = select(0u, tlasBlasRoots[instIdx], instIdx < arrayLength(&tlasBlasRoots));
        _ = traceMeshBvh(localRay, tMin, INFINITY, true, &localHit, blasRoot, true);
        if (localHit.didHit && localHit.dist > tMin && localHit.dist < (*hit).dist) {
          let localHitPos = localRay.origin + localRay.direction * localHit.dist;
          let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
          let worldDist = dot(worldHitPos - ray.origin, ray.direction);
          if (worldDist <= tMin || worldDist >= (*hit).dist) { continue; }
          (*hit).didHit = true;
          (*hit).dist = worldDist;
          (*hit).triIndex = localHit.triIndex;
          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);
          // Barycentric weights are space-invariant — propagate verbatim (the
          // BLAS triangle + its per-vertex UVs are the same in any instance). (P2)
          (*hit).baryVW = localHit.baryVW;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < ${TLAS_TRAVERSAL_STACK_DEPTH}u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        return (*hit).didHit;
      }
    }
  }
  return (*hit).didHit;
}

fn traceTlasAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    var meshHit: SceneHit;
    return traceMeshBvh(ray, tMin, tMax, false, &meshHit, 0u, false);
  }
  var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(params.tlasNodeCount, arrayLength(&tlasNodes))) { continue; }
    let node = tlasNodes[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!intersectAabb(ray, bmin, bmax, tMin, tMax)) { continue; }
    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= arrayLength(&tlasInstanceIndices)) { continue; }
        let instIdx = tlasInstanceIndices[permIdx];
        let m = instIdx * 4u;
        if (m + 3u >= arrayLength(&tlasInstanceWorldToLocal)) { continue; }
        let w2l0 = tlasInstanceWorldToLocal[m];
        let w2l1 = tlasInstanceWorldToLocal[m + 1u];
        let w2l2 = tlasInstanceWorldToLocal[m + 2u];
        let w2l3 = tlasInstanceWorldToLocal[m + 3u];
        var localRay: Ray;
        localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
        localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
        var localTMax = tMax;
        if (tMax < INFINITY * 0.5) {
          let localEnd = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin + ray.direction * tMax);
          localTMax = max(dot(localEnd - localRay.origin, localRay.direction), tMin);
        }
        var localHit: SceneHit;
        let blasRoot = select(0u, tlasBlasRoots[instIdx], instIdx < arrayLength(&tlasBlasRoots));
        if (traceMeshBvh(localRay, tMin, localTMax, false, &localHit, blasRoot, false)) {
          return true;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < ${TLAS_TRAVERSAL_STACK_DEPTH}u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        // Conservative any-hit overflow policy: prefer occlusion over light leak.
        return true;
      }
    }
  }
  return false;
}
`;
