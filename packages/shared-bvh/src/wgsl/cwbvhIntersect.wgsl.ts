/**
 * CWBVH traversal for the compressed-wide BVH layout.
 *
 * This module mirrors `compressedWideBvh.ts`:
 *   - one node has 8 child slots;
 *   - parent bounds are six f32 words (`CwbvhNodeBounds`);
 *   - child bounds are six u16 words, uploaded to WGSL as three packed u32
 *     words (`lo16 | hi16 << 16`);
 *   - child metadata is three u32 words (`kind`, `nodeIndexOrTriOffset`,
 *     `triCount`).
 */

import { CWBVH_TRAVERSAL_STACK_DEPTH } from '../strides.js';
import { MOLLER_TRUMBORE_WGSL, SAFE_INV_DIR_WGSL } from './bvhIntersect.wgsl.js';
import { buildMaterialTransmissionPredicatesWGSL } from './materialTransmission.wgsl.js';

const CWBVH_MATERIAL_TRANSMISSION_PREDICATE_WGSL =
  buildMaterialTransmissionPredicatesWGSL({
    packedFunctionName: 'cwbvhPackedMaterialHasTransmission',
  });

export const CWBVH_INTERSECT_STACK_DEPTH = CWBVH_TRAVERSAL_STACK_DEPTH;

/** Default value-return loaders for the ordinary module-global buffers. */
export const CWBVH_INTERSECT_GLOBAL_LOADERS_WGSL = /* wgsl */ `
fn cwbvhLoadNodeBounds(index: u32) -> CwbvhNodeBounds {
  return cwbvhNodeBounds[index];
}
fn cwbvhNodeBoundsCount() -> u32 { return arrayLength(&cwbvhNodeBounds); }
fn cwbvhLoadChildBoundsWord(index: u32) -> u32 {
  return cwbvhChildBoundsPacked[index];
}
fn cwbvhChildBoundsWordCount() -> u32 {
  return arrayLength(&cwbvhChildBoundsPacked);
}
fn cwbvhLoadChildMeta(index: u32) -> CwbvhChildMeta {
  return cwbvhChildMeta[index];
}
fn cwbvhChildMetaCount() -> u32 { return arrayLength(&cwbvhChildMeta); }
fn cwbvhLoadChildCount(index: u32) -> u32 { return cwbvhChildCount[index]; }
fn cwbvhChildCountCount() -> u32 { return arrayLength(&cwbvhChildCount); }
fn cwbvhLoadIndex(index: u32) -> vec4u { return bvh_index[index]; }
fn cwbvhIndexCount() -> u32 { return arrayLength(&bvh_index); }
fn cwbvhLoadPosition(index: u32) -> vec4f { return bvh_position[index]; }
fn cwbvhPositionCount() -> u32 { return arrayLength(&bvh_position); }
`;

export const CWBVH_INTERSECT_CORE_WGSL = /* wgsl */ `
${CWBVH_MATERIAL_TRANSMISSION_PREDICATE_WGSL}

const CWBVH_CHILDREN: u32 = 8u;
const CWBVH_CHILD_BOUNDS_PACKED_U32: u32 = 3u;
const CWBVH_INTERSECT_STACK_DEPTH: u32 = ${CWBVH_INTERSECT_STACK_DEPTH}u;
const CWBVH_INTERSECT_INFINITY: f32 = 1e20;
const CWBVH_CHILD_EMPTY: u32 = 0u;
const CWBVH_CHILD_NODE: u32 = 1u;
const CWBVH_CHILD_LEAF: u32 = 2u;
const CWBVH_STATUS_COMPLETE: u32 = 0u;
const CWBVH_STATUS_STACK_OVERFLOW: u32 = 1u;
const CWBVH_STATUS_INVALID_LAYOUT: u32 = 2u;

struct CwbvhNodeBounds {
  boundsMin: array<f32, 3>,
  boundsMax: array<f32, 3>,
};

struct CwbvhChildMeta {
  kind: u32,
  indexOrOffset: u32,
  triCount: u32,
};

struct CwbvhRay {
  origin: vec3f,
  direction: vec3f,
};

struct CwbvhAabb {
  boundsMin: vec3f,
  boundsMax: vec3f,
};

struct CwbvhIntersectionResult {
  status: u32,
  didHit: bool,
  dist: f32,
  triIndex: u32,
  indices: vec4u,
  barycoord: vec3f,
  normal: vec3f,
  side: f32,
  matColorPacked: u32,
  uv: vec2f,
};

fn cwbvhMiss() -> CwbvhIntersectionResult {
  var result: CwbvhIntersectionResult;
  result.status = CWBVH_STATUS_COMPLETE;
  result.didHit = false;
  result.dist = CWBVH_INTERSECT_INFINITY;
  result.triIndex = 0xffffffffu;
  result.indices = vec4u(0u);
  result.barycoord = vec3f(0.0);
  result.normal = vec3f(0.0);
  result.side = 0.0;
  result.matColorPacked = 0u;
  result.uv = vec2f(0.0);
  return result;
}

fn cwbvhUnpackLo16(word: u32) -> u32 {
  return word & 0xffffu;
}

fn cwbvhUnpackHi16(word: u32) -> u32 {
  return (word >> 16u) & 0xffffu;
}

fn cwbvhDecodeBound(q: u32, parentMin: f32, parentMax: f32) -> f32 {
  if (!(parentMax > parentMin)) {
    return parentMin;
  }
  if (q == 0u) { return parentMin; }
  if (q >= 65535u) { return parentMax; }
  let t = f32(q) / 65535.0;
  // Convex interpolation avoids overflow in parentMax - parentMin for
  // opposite-sign finite f32 bounds near the representable limits.
  return parentMin * (1.0 - t) + parentMax * t;
}

fn cwbvhLoadChildBounds(
  nodeIndex: u32,
  slot: u32,
) -> CwbvhAabb {
  let node = cwbvhLoadNodeBounds(nodeIndex);
  let pMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
  let pMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
  let base = nodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_PACKED_U32 + slot * CWBVH_CHILD_BOUNDS_PACKED_U32;
  let w0 = cwbvhLoadChildBoundsWord(base + 0u);
  let w1 = cwbvhLoadChildBoundsWord(base + 1u);
  let w2 = cwbvhLoadChildBoundsWord(base + 2u);

  var bounds: CwbvhAabb;
  bounds.boundsMin = vec3f(
    cwbvhDecodeBound(cwbvhUnpackLo16(w0), pMin.x, pMax.x),
    cwbvhDecodeBound(cwbvhUnpackHi16(w0), pMin.y, pMax.y),
    cwbvhDecodeBound(cwbvhUnpackLo16(w1), pMin.z, pMax.z),
  );
  bounds.boundsMax = vec3f(
    cwbvhDecodeBound(cwbvhUnpackHi16(w1), pMin.x, pMax.x),
    cwbvhDecodeBound(cwbvhUnpackLo16(w2), pMin.y, pMax.y),
    cwbvhDecodeBound(cwbvhUnpackHi16(w2), pMin.z, pMax.z),
  );
  return bounds;
}

fn cwbvhAabbEntry(
  origin: vec3f,
  invDir: vec3f,
  bmin: vec3f,
  bmax: vec3f,
  tMax: f32,
) -> f32 {
  let t0 = (bmin - origin) * invDir;
  let t1 = (bmax - origin) * invDir;
  let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tNear > tFar || tFar < 0.0 || tNear > tMax) {
    return CWBVH_INTERSECT_INFINITY;
  }
  return tNear;
}

fn cwbvhBoundsAreValid(bmin: vec3f, bmax: vec3f) -> bool {
  // Ordered comparisons reject NaN and the finite ceiling rejects +/-Inf.
  // Layout corruption must surface as INVALID_LAYOUT so renderer wrappers can
  // restart the canonical binary traversal instead of silently dropping hits.
  let finiteMax = vec3f(3.402823e38);
  return all(bmin <= bmax) &&
    all(abs(bmin) <= finiteMax) &&
    all(abs(bmax) <= finiteMax);
}

fn cwbvhIntersectFirstHitRangeFromRoot(
  ray: CwbvhRay,
  triEps: f32,
  tMin: f32,
  tMax: f32,
  nodeCount: u32,
  rootNode: u32,
  skipGlass: bool,
) -> CwbvhIntersectionResult {
  var best = cwbvhMiss();
  best.dist = tMax;
  if (nodeCount == 0u) {
    return best;
  }
  if (rootNode >= nodeCount) {
    best.status = CWBVH_STATUS_INVALID_LAYOUT;
    return best;
  }
  if (
    nodeCount > cwbvhNodeBoundsCount() ||
    nodeCount > cwbvhChildBoundsWordCount() / (CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_PACKED_U32) ||
    nodeCount > cwbvhChildMetaCount() / CWBVH_CHILDREN ||
    nodeCount > cwbvhChildCountCount()
  ) {
    best.status = CWBVH_STATUS_INVALID_LAYOUT;
    return best;
  }

  var stack: array<u32, ${CWBVH_INTERSECT_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = rootNode;
  stackPtr = stackPtr + 1u;

  let invDir = safeInvDir(ray.direction);

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIndex = stack[stackPtr];
    if (nodeIndex >= nodeCount) {
      best.status = CWBVH_STATUS_INVALID_LAYOUT;
      return best;
    }

    let rawCount = cwbvhLoadChildCount(nodeIndex);
    if (rawCount > CWBVH_CHILDREN) {
      best.status = CWBVH_STATUS_INVALID_LAYOUT;
      return best;
    }
    let parent = cwbvhLoadNodeBounds(nodeIndex);
    let parentMin = vec3f(parent.boundsMin[0], parent.boundsMin[1], parent.boundsMin[2]);
    let parentMax = vec3f(parent.boundsMax[0], parent.boundsMax[1], parent.boundsMax[2]);
    if (!cwbvhBoundsAreValid(parentMin, parentMax)) {
      best.status = CWBVH_STATUS_INVALID_LAYOUT;
      return best;
    }
    let count = rawCount;
    for (var slot = 0u; slot < count; slot = slot + 1u) {
      let childIndex = nodeIndex * CWBVH_CHILDREN + slot;
      let childInfo = cwbvhLoadChildMeta(childIndex);
      if (childInfo.kind == CWBVH_CHILD_EMPTY) {
        best.status = CWBVH_STATUS_INVALID_LAYOUT;
        return best;
      }

      let bounds = cwbvhLoadChildBounds(nodeIndex, slot);
      if (!cwbvhBoundsAreValid(bounds.boundsMin, bounds.boundsMax)) {
        best.status = CWBVH_STATUS_INVALID_LAYOUT;
        return best;
      }
      let childT = cwbvhAabbEntry(ray.origin, invDir, bounds.boundsMin, bounds.boundsMax, best.dist);
      if (childT == CWBVH_INTERSECT_INFINITY) {
        continue;
      }

      if (childInfo.kind == CWBVH_CHILD_NODE) {
        if (childInfo.indexOrOffset >= nodeCount) {
          best.status = CWBVH_STATUS_INVALID_LAYOUT;
          return best;
        }
        if (stackPtr >= CWBVH_INTERSECT_STACK_DEPTH) {
          best.status = CWBVH_STATUS_STACK_OVERFLOW;
          return best;
        }
        stack[stackPtr] = childInfo.indexOrOffset;
        stackPtr = stackPtr + 1u;
      } else if (childInfo.kind == CWBVH_CHILD_LEAF) {
        let indexCount = cwbvhIndexCount();
        if (childInfo.triCount == 0u || childInfo.indexOrOffset > indexCount || childInfo.triCount > indexCount - childInfo.indexOrOffset) {
          best.status = CWBVH_STATUS_INVALID_LAYOUT;
          return best;
        }
        for (var i = 0u; i < childInfo.triCount; i = i + 1u) {
          let triIdx = childInfo.indexOrOffset + i;
          let idxEntry = cwbvhLoadIndex(triIdx);
          if (skipGlass) {
            if (cwbvhPackedMaterialHasTransmission(idxEntry.w)) {
              continue;
            }
          }
          let idx = idxEntry.xyz;
          if (idx.x >= cwbvhPositionCount() || idx.y >= cwbvhPositionCount() || idx.z >= cwbvhPositionCount()) {
            best.status = CWBVH_STATUS_INVALID_LAYOUT;
            return best;
          }
          let pa4 = cwbvhLoadPosition(idx.x);
          let pb4 = cwbvhLoadPosition(idx.y);
          let pc4 = cwbvhLoadPosition(idx.z);
          let tri = mollerTrumboreCore(ray.origin, ray.direction, pa4.xyz, pb4.xyz, pc4.xyz, triEps);
          if (tri.hit && tri.t > tMin && tri.t < best.dist) {
            best.didHit = true;
            best.dist = tri.t;
            best.triIndex = triIdx;
            best.indices = vec4u(idx, triIdx);
            best.barycoord = tri.bary;
            best.normal = normalize(cross(pb4.xyz - pa4.xyz, pc4.xyz - pa4.xyz)) * sign(tri.det);
            best.side = sign(tri.det);
            best.matColorPacked = idxEntry.w;
            let uvA = unpack2x16float(bitcast<u32>(pa4.w));
            let uvB = unpack2x16float(bitcast<u32>(pb4.w));
            let uvC = unpack2x16float(bitcast<u32>(pc4.w));
            best.uv = tri.bary.x * uvA + tri.bary.y * uvB + tri.bary.z * uvC;
          }
        }
      } else {
        best.status = CWBVH_STATUS_INVALID_LAYOUT;
        return best;
      }
    }
  }

  return best;
}

fn cwbvhIntersectFirstHitFromRoot(
  ray: CwbvhRay,
  triEps: f32,
  nodeCount: u32,
  rootNode: u32,
  skipGlass: bool,
) -> CwbvhIntersectionResult {
  return cwbvhIntersectFirstHitRangeFromRoot(
    ray,
    triEps,
    triEps,
    CWBVH_INTERSECT_INFINITY,
    nodeCount,
    rootNode,
    skipGlass,
  );
}

fn cwbvhIntersectFirstHit(
  ray: CwbvhRay,
  triEps: f32,
  nodeCount: u32,
  skipGlass: bool,
) -> CwbvhIntersectionResult {
  return cwbvhIntersectFirstHitFromRoot(
    ray,
    triEps,
    nodeCount,
    0u,
    skipGlass,
  );
}

`;

export const CWBVH_INTERSECT_WGSL = /* wgsl */ `
${SAFE_INV_DIR_WGSL}
${MOLLER_TRUMBORE_WGSL}
${CWBVH_INTERSECT_CORE_WGSL}
${CWBVH_INTERSECT_GLOBAL_LOADERS_WGSL}
`;
