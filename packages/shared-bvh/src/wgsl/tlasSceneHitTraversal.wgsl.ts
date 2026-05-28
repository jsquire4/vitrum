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

export const TLAS_SCENE_HIT_TRAVERSAL_WGSL = /* wgsl */ `

fn traceTlasClosest(ray: Ray, tMin: f32, tMax: f32, hit: ptr<function, SceneHit>) -> bool {
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceMeshBvh(ray, tMin, tMax, true, hit, 0u, true);
  }
  (*hit).didHit = false;
  (*hit).dist = tMax;
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  var stack: array<u32, 64>;
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
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < 64u) {
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
  var stack: array<u32, 64>;
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
        _ = traceMeshBvh(localRay, tMin, INFINITY, true, &localHit, blasRoot, false);
        if (!localHit.didHit || localHit.dist <= tMin) { continue; }
        let localHitPos = localRay.origin + localRay.direction * localHit.dist;
        let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
        let worldDist = dot(worldHitPos - ray.origin, ray.direction);
        if (worldDist > tMin && worldDist < tMax) {
          return true;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < 64u) {
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
