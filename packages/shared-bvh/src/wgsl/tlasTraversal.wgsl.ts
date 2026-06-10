/**
 * TLAS + per-root BLAS traversal for ReSTIR (vec4 storage).
 *
 * Ported from pt-webgpu `pathTrace/intersection.wgsl.ts` (Heitz / instance
 * transform conservative world-t). Used by walkaround-hybrid when
 * `WalkaroundUBO.bvhMode == 1`.
 *
 * @see packages/pt-webgpu/src/wgsl/pathTrace/intersection.wgsl.ts
 */

export const TLAS_TRAVERSAL_WGSL = /* wgsl */ `

fn tlasTransformPointCols(c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, p: vec3f) -> vec3f {
  let r = c0 * p.x + c1 * p.y + c2 * p.z + c3;
  return r.xyz / max(abs(r.w), 1e-8);
}

fn tlasTransformDirectionCols(c0: vec4f, c1: vec4f, c2: vec4f, d: vec3f) -> vec3f {
  return safe_normalize((c0 * d.x + c1 * d.y + c2 * d.z).xyz);
}

fn tlasTransformNormalFromLocalCols(w2l0: vec4f, w2l1: vec4f, w2l2: vec4f, nLocal: vec3f) -> vec3f {
  return safe_normalize(vec3f(
    dot(vec3f(w2l0.x, w2l1.x, w2l2.x), nLocal),
    dot(vec3f(w2l0.y, w2l1.y, w2l2.y), nLocal),
    dot(vec3f(w2l0.z, w2l1.z, w2l2.z), nLocal),
  ));
}

fn tlasIntersectAabb(ray: Ray, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32) -> bool {
  let invDir = safeInvDir(ray.direction);
  let t1 = (bmin - ray.origin) * invDir;
  let t2 = (bmax - ray.origin) * invDir;
  let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  let tFar = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
  return !(tNear > tFar || tFar < tMin || tNear > tMax);
}

fn traceTlasFirstHit(
  tlasNodes: ptr<storage, array<BVHNode>, read>,
  tlasInstanceIndices: ptr<storage, array<u32>, read>,
  tlasBlasRoots: ptr<storage, array<u32>, read>,
  tlasInstanceWorldToLocal: ptr<storage, array<vec4f>, read>,
  tlasInstanceLocalToWorld: ptr<storage, array<vec4f>, read>,
  tlasNodeCount: u32,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  bvh: ptr<storage, array<BVHNode>, read>,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  if (tlasNodeCount == 0u || arrayLength(tlasNodes) == 0u || arrayLength(tlasInstanceIndices) == 0u) {
    return bvhIntersectFirstHit(bvh_index, bvh_position, bvh, ray, triEps);
  }

  var best: IntersectionResult;
  best.didHit = false;
  best.dist = BVH_INTERSECT_INFINITY;
  best.matColorPacked = 0u;
  best.uv = vec2f(0.0);

  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(tlasNodeCount, arrayLength(tlasNodes))) { continue; }
    let node = (*tlasNodes)[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!tlasIntersectAabb(ray, bmin, bmax, 0.0, best.dist)) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= arrayLength(tlasInstanceIndices)) { continue; }
        let instIdx = (*tlasInstanceIndices)[permIdx];
        let m = instIdx * 4u;
        if (m + 3u >= arrayLength(tlasInstanceWorldToLocal) || m + 3u >= arrayLength(tlasInstanceLocalToWorld)) { continue; }
        let w2l0 = (*tlasInstanceWorldToLocal)[m];
        let w2l1 = (*tlasInstanceWorldToLocal)[m + 1u];
        let w2l2 = (*tlasInstanceWorldToLocal)[m + 2u];
        let w2l3 = (*tlasInstanceWorldToLocal)[m + 3u];
        let l2w0 = (*tlasInstanceLocalToWorld)[m];
        let l2w1 = (*tlasInstanceLocalToWorld)[m + 1u];
        let l2w2 = (*tlasInstanceLocalToWorld)[m + 2u];
        let l2w3 = (*tlasInstanceLocalToWorld)[m + 3u];
        var localRay: Ray;
        localRay.origin = tlasTransformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
        localRay.direction = tlasTransformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
        let blasRoot = select(0u, (*tlasBlasRoots)[instIdx], instIdx < arrayLength(tlasBlasRoots));
        // closest-hit traversal — glass always occludes (skipGlass=false),
        // preserving the pre-H32 semantics of traceTlasFirstHit.
        let localHit = bvhIntersectFirstHitAtRoot(
          bvh_index, bvh_position, bvh, localRay, triEps, blasRoot, false,
        );
        if (localHit.didHit && localHit.dist > 0.0 && localHit.dist < best.dist) {
          let localHitPos = localRay.origin + localRay.direction * localHit.dist;
          let worldHitPos = tlasTransformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
          let worldDist = dot(worldHitPos - ray.origin, ray.direction);
          if (worldDist <= 0.0 || worldDist >= best.dist) { continue; }
          best = localHit;
          best.dist = worldDist;
          best.normal = tlasTransformNormalFromLocalCols(w2l0, w2l1, w2l2, localHit.normal);
          // Carry the hit instance so the caller can transform the SMOOTH
          // (barycentric, local-space) shading normal to world by the SAME
          // inverse-transpose used for the geometric normal above. V21 TLAS.
          best.instanceIndex = instIdx;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < 64u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        return best;
      }
    }
  }
  return best;
}

fn traceTlasAny(
  tlasNodes: ptr<storage, array<BVHNode>, read>,
  tlasInstanceIndices: ptr<storage, array<u32>, read>,
  tlasBlasRoots: ptr<storage, array<u32>, read>,
  tlasInstanceWorldToLocal: ptr<storage, array<vec4f>, read>,
  tlasInstanceLocalToWorld: ptr<storage, array<vec4f>, read>,
  tlasNodeCount: u32,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  bvh: ptr<storage, array<BVHNode>, read>,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
) -> bool {
  if (tlasNodeCount == 0u || arrayLength(tlasNodes) == 0u || arrayLength(tlasInstanceIndices) == 0u) {
    return bvhIntersectAny(bvh_index, bvh_position, bvh, origin, dir, tMax, triEps, skipGlass);
  }

  var ray: Ray;
  ray.origin = origin;
  ray.direction = dir;

  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(tlasNodeCount, arrayLength(tlasNodes))) { continue; }
    let node = (*tlasNodes)[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!tlasIntersectAabb(ray, bmin, bmax, 0.0, tMax)) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= arrayLength(tlasInstanceIndices)) { continue; }
        let instIdx = (*tlasInstanceIndices)[permIdx];
        let m = instIdx * 4u;
        if (m + 3u >= arrayLength(tlasInstanceWorldToLocal) || m + 3u >= arrayLength(tlasInstanceLocalToWorld)) { continue; }
        let w2l0 = (*tlasInstanceWorldToLocal)[m];
        let w2l1 = (*tlasInstanceWorldToLocal)[m + 1u];
        let w2l2 = (*tlasInstanceWorldToLocal)[m + 2u];
        let w2l3 = (*tlasInstanceWorldToLocal)[m + 3u];
        let l2w0 = (*tlasInstanceLocalToWorld)[m];
        let l2w1 = (*tlasInstanceLocalToWorld)[m + 1u];
        let l2w2 = (*tlasInstanceLocalToWorld)[m + 2u];
        let l2w3 = (*tlasInstanceLocalToWorld)[m + 3u];
        var localRay: Ray;
        localRay.origin = tlasTransformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
        localRay.direction = tlasTransformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
        let blasRoot = select(0u, (*tlasBlasRoots)[instIdx], instIdx < arrayLength(tlasBlasRoots));
        // H32 — single glass-aware closest-hit traversal replaces the old
        // any-hit pre-test + closest-hit follow-up pair.  The any-hit path
        // respected skipGlass but the follow-up bvhIntersectFirstHitAtRoot did
        // NOT, so glass primitives wrongly occluded TLAS shadow rays even when
        // skipGlass=true.  The double traversal was also needlessly expensive.
        // One closest-hit call with the same skipGlass flag fixes both.
        let localHit = bvhIntersectFirstHitAtRoot(
          bvh_index, bvh_position, bvh, localRay, triEps, blasRoot, skipGlass,
        );
        if (!localHit.didHit || localHit.dist <= 0.0) { continue; }
        let localHitPos = localRay.origin + localRay.direction * localHit.dist;
        let worldHitPos = tlasTransformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
        let worldDist = dot(worldHitPos - ray.origin, ray.direction);
        if (worldDist > 1e-4 && worldDist < tMax) {
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
        return true;
      }
    }
  }
  return false;
}

`;
