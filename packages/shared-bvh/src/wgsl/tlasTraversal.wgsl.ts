/**
 * TLAS + per-root BLAS traversal for ReSTIR (vec4 storage).
 *
 * Ported from pt-webgpu `pathTrace/intersection.wgsl.ts` (Heitz / instance
 * transform conservative world-t). Used by walkaround-hybrid when
 * `WalkaroundUBO.bvhMode == 1`.
 *
 * @see packages/pt-webgpu/src/wgsl/pathTrace/intersection.wgsl.ts
 */
import { TLAS_TRAVERSAL_STACK_DEPTH as SHARED_TLAS_TRAVERSAL_STACK_DEPTH } from '../strides.js';


/**
 * TLAS traversal stack depth. Single TS source of truth — interpolated into
 * the WGSL below (WGSL array sizes must be literal const-expressions).
 *
 * INTENTIONALLY 64, not the BLAS traversal's 60 (bvhIntersect.wgsl.ts).
 * Every internally built TLAS is accepted only after validateTlasBuild proves
 * its exact maximum DFS stack occupancy is <= this capacity.
 */
export const TLAS_TRAVERSAL_STACK_DEPTH = SHARED_TLAS_TRAVERSAL_STACK_DEPTH;
export const TLAS_TRAVERSAL_STATUS_COMPLETE = 0 as const;
export const TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW = 1 as const;
export const TLAS_TRAVERSAL_STATUS_FALLBACK = 2 as const;

/**
 * Default loaders for the ordinary module-global TLAS buffers. Packed-arena
 * consumers compose the core with their own functions of these signatures.
 */
export const TLAS_TRAVERSAL_GLOBAL_LOADERS_WGSL = /* wgsl */ `
fn tlasLoadNode(index: u32) -> BVHNode { return tlasNodes[index]; }
fn tlasNodeCapacity() -> u32 { return arrayLength(&tlasNodes); }
fn tlasLoadInstanceIndex(index: u32) -> u32 { return tlasInstanceIndices[index]; }
fn tlasInstanceIndexCount() -> u32 { return arrayLength(&tlasInstanceIndices); }
fn tlasLoadBlasRoot(index: u32) -> u32 { return tlasBlasRoots[index]; }
fn tlasBlasRootCount() -> u32 { return arrayLength(&tlasBlasRoots); }
fn tlasLoadWorldToLocalColumn(index: u32) -> vec4f {
  return tlasInstanceWorldToLocal[index];
}
fn tlasWorldToLocalColumnCount() -> u32 {
  return arrayLength(&tlasInstanceWorldToLocal);
}
fn tlasLoadLocalToWorldColumn(index: u32) -> vec4f {
  return tlasInstanceLocalToWorld[index];
}
fn tlasLocalToWorldColumnCount() -> u32 {
  return arrayLength(&tlasInstanceLocalToWorld);
}
`;

/** Binding-agnostic TLAS traversal over value-return loader functions. */
export const TLAS_TRAVERSAL_CORE_WGSL = /* wgsl */ `

const TLAS_TRAVERSAL_STATUS_COMPLETE: u32 = ${TLAS_TRAVERSAL_STATUS_COMPLETE}u;
const TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW: u32 = ${TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW}u;
const TLAS_TRAVERSAL_STATUS_FALLBACK: u32 = ${TLAS_TRAVERSAL_STATUS_FALLBACK}u;
var<private> tlasTraversalStatusCode: u32;

fn tlasLastTraversalStatus() -> u32 {
  return tlasTraversalStatusCode;
}

fn tlasSafeNormalize(v: vec3f) -> vec3f {
  let lengthSquared = dot(v, v);
  if (lengthSquared <= 1e-20) { return vec3f(0.0); }
  return v * inverseSqrt(lengthSquared);
}

fn tlasTransformPointCols(c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, p: vec3f) -> vec3f {
  let r = c0 * p.x + c1 * p.y + c2 * p.z + c3;
  return r.xyz / max(abs(r.w), 1e-8);
}

fn tlasTransformDirectionCols(c0: vec4f, c1: vec4f, c2: vec4f, d: vec3f) -> vec3f {
  return tlasSafeNormalize((c0 * d.x + c1 * d.y + c2 * d.z).xyz);
}

fn tlasTransformNormalFromLocalCols(w2l0: vec4f, w2l1: vec4f, w2l2: vec4f, nLocal: vec3f) -> vec3f {
  // Matrices are column-major and points use c0*x + c1*y + c2*z + c3.
  // A local normal therefore transforms by transpose(worldToLocal). Each
  // output component is a dot with one W2L column. Assembling W2L rows here
  // would multiply by W2L and is only accidentally correct for diagonals.
  return tlasSafeNormalize(vec3f(
    dot(w2l0.xyz, nLocal),
    dot(w2l1.xyz, nLocal),
    dot(w2l2.xyz, nLocal),
  ));
}

// Orientation parity of a linear instance transform. A negative determinant
// reverses triangle winding, so local Moller-Trumbore side must be flipped to
// describe the authored world-space front face. The inverse has the same sign,
// therefore callers holding either L2W or W2L columns may use this helper.
fn tlasLinearOrientationSign(c0: vec4f, c1: vec4f, c2: vec4f) -> f32 {
  let determinant = dot(c0.xyz, cross(c1.xyz, c2.xyz));
  return select(-1.0, 1.0, determinant >= 0.0);
}

fn tlasIntersectAabb(ray: Ray, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32) -> bool {
  let invDir = safeInvDir(ray.direction);
  let t1 = (bmin - ray.origin) * invDir;
  let t2 = (bmax - ray.origin) * invDir;
  let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  let tFar = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
  return !(tNear > tFar || tFar < tMin || tNear > tMax);
}

fn tlasEmptyFirstHit() -> IntersectionResult {
  var hit: IntersectionResult;
  hit.didHit = false;
  hit.dist = BVH_INTERSECT_INFINITY;
  hit.matColorPacked = 0u;
  hit.uv = vec2f(0.0);
  return hit;
}

fn tlasTraceInstanceFirstHit(
  instIdx: u32,
  ray: Ray,
  triEps: f32,
  best: ptr<function, IntersectionResult>,
) {
  let m = instIdx * 4u;
  if (m + 3u >= tlasWorldToLocalColumnCount() || m + 3u >= tlasLocalToWorldColumnCount()) { return; }
  let w2l0 = tlasLoadWorldToLocalColumn(m);
  let w2l1 = tlasLoadWorldToLocalColumn(m + 1u);
  let w2l2 = tlasLoadWorldToLocalColumn(m + 2u);
  let w2l3 = tlasLoadWorldToLocalColumn(m + 3u);
  let l2w0 = tlasLoadLocalToWorldColumn(m);
  let l2w1 = tlasLoadLocalToWorldColumn(m + 1u);
  let l2w2 = tlasLoadLocalToWorldColumn(m + 2u);
  let l2w3 = tlasLoadLocalToWorldColumn(m + 3u);
  var localRay: Ray;
  localRay.origin = tlasTransformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
  localRay.direction = tlasTransformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
  var blasRoot = 0u;
  if (instIdx < tlasBlasRootCount()) { blasRoot = tlasLoadBlasRoot(instIdx); }
  let localHit = bvhIntersectFirstHitAtRoot(localRay, triEps, blasRoot, false);
  if (localHit.didHit && localHit.dist > 0.0) {
    let localHitPos = localRay.origin + localRay.direction * localHit.dist;
    let worldHitPos = tlasTransformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
    let worldDist = dot(worldHitPos - ray.origin, ray.direction);
    if (worldDist > 0.0 && worldDist < (*best).dist) {
      *best = localHit;
      (*best).dist = worldDist;
      (*best).normal = tlasTransformNormalFromLocalCols(w2l0, w2l1, w2l2, localHit.normal);
      (*best).side = localHit.side * tlasLinearOrientationSign(l2w0, l2w1, l2w2);
      (*best).instanceIndex = instIdx;
    }
  }
}

fn tlasFirstHitFallback(ray: Ray, triEps: f32) -> IntersectionResult {
  var best = tlasEmptyFirstHit();
  for (var permIdx = 0u; permIdx < tlasInstanceIndexCount(); permIdx = permIdx + 1u) {
    tlasTraceInstanceFirstHit(tlasLoadInstanceIndex(permIdx), ray, triEps, &best);
  }
  tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_FALLBACK;
  return best;
}

fn tlasTraceInstanceAny(
  instIdx: u32,
  ray: Ray,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
) -> bool {
  let m = instIdx * 4u;
  if (m + 3u >= tlasWorldToLocalColumnCount() || m + 3u >= tlasLocalToWorldColumnCount()) { return false; }
  let w2l0 = tlasLoadWorldToLocalColumn(m);
  let w2l1 = tlasLoadWorldToLocalColumn(m + 1u);
  let w2l2 = tlasLoadWorldToLocalColumn(m + 2u);
  let w2l3 = tlasLoadWorldToLocalColumn(m + 3u);
  let l2w0 = tlasLoadLocalToWorldColumn(m);
  let l2w1 = tlasLoadLocalToWorldColumn(m + 1u);
  let l2w2 = tlasLoadLocalToWorldColumn(m + 2u);
  let l2w3 = tlasLoadLocalToWorldColumn(m + 3u);
  var localRay: Ray;
  localRay.origin = tlasTransformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
  localRay.direction = tlasTransformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
  var blasRoot = 0u;
  if (instIdx < tlasBlasRootCount()) { blasRoot = tlasLoadBlasRoot(instIdx); }
  let localHit = bvhIntersectFirstHitAtRoot(localRay, triEps, blasRoot, skipGlass);
  if (!localHit.didHit || localHit.dist <= 0.0) { return false; }
  let localHitPos = localRay.origin + localRay.direction * localHit.dist;
  let worldHitPos = tlasTransformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
  let worldDist = dot(worldHitPos - ray.origin, ray.direction);
  return worldDist > 1e-4 && worldDist < tMax;
}

fn tlasAnyFallback(ray: Ray, tMax: f32, triEps: f32, skipGlass: bool) -> bool {
  for (var permIdx = 0u; permIdx < tlasInstanceIndexCount(); permIdx = permIdx + 1u) {
    if (tlasTraceInstanceAny(tlasLoadInstanceIndex(permIdx), ray, tMax, triEps, skipGlass)) {
      tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_FALLBACK;
      return true;
    }
  }
  tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_FALLBACK;
  return false;
}

fn traceTlasFirstHit(
  tlasNodeCount: u32,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_COMPLETE;
  if (tlasNodeCount == 0u || tlasNodeCapacity() == 0u || tlasInstanceIndexCount() == 0u) {
    return bvhIntersectFirstHit(ray, triEps);
  }

  var best = tlasEmptyFirstHit();

  var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(tlasNodeCount, tlasNodeCapacity())) { continue; }
    let node = tlasLoadNode(nodeIdx);
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!tlasIntersectAabb(ray, bmin, bmax, 0.0, best.dist)) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= tlasInstanceIndexCount()) { continue; }
        tlasTraceInstanceFirstHit(tlasLoadInstanceIndex(permIdx), ray, triEps, &best);
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u <= ${TLAS_TRAVERSAL_STACK_DEPTH}u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;
        return tlasFirstHitFallback(ray, triEps);
      }
    }
  }
  return best;
}

fn traceTlasAny(
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
) -> bool {
  tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_COMPLETE;
  if (tlasNodeCount == 0u || tlasNodeCapacity() == 0u || tlasInstanceIndexCount() == 0u) {
    return bvhIntersectAny(origin, dir, tMax, triEps, skipGlass);
  }

  var ray: Ray;
  ray.origin = origin;
  ray.direction = dir;

  var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(tlasNodeCount, tlasNodeCapacity())) { continue; }
    let node = tlasLoadNode(nodeIdx);
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!tlasIntersectAabb(ray, bmin, bmax, 0.0, tMax)) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= tlasInstanceIndexCount()) { continue; }
        if (tlasTraceInstanceAny(tlasLoadInstanceIndex(permIdx), ray, tMax, triEps, skipGlass)) {
          return true;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u <= ${TLAS_TRAVERSAL_STACK_DEPTH}u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;
        return tlasAnyFallback(ray, tMax, triEps, skipGlass);
      }
    }
  }
  return false;
}

`;

/** Convenience composition for the ordinary module-global TLAS buffers. */
export const TLAS_TRAVERSAL_WGSL = /* wgsl */ `
${TLAS_TRAVERSAL_CORE_WGSL}
${TLAS_TRAVERSAL_GLOBAL_LOADERS_WGSL}
`;
