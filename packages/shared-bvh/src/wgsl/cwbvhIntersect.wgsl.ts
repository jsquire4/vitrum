/**
 * CWBVH traversal prototype for the compressed-wide BVH layout.
 *
 * This module mirrors `compressedWideBvh.ts`:
 *   - one node has 8 child slots;
 *   - parent bounds are six f32 words (`CwbvhNodeBounds`);
 *   - child bounds are six u16 words, uploaded to WGSL as three packed u32
 *     words (`lo16 | hi16 << 16`);
 *   - child metadata is three u32 words (`kind`, `nodeIndexOrTriOffset`,
 *     `triCount`).
 *
 * It intentionally lives in `shared-bvh` without being wired into renderer
 * defaults. Backends should opt into this only after parity/perf A/B against
 * the canonical binary-BVH traversal lands.
 */

import { MOLLER_TRUMBORE_WGSL, SAFE_INV_DIR_WGSL } from './bvhIntersect.wgsl.js';

export const CWBVH_INTERSECT_STACK_DEPTH = 64;

export const CWBVH_INTERSECT_CORE_WGSL = /* wgsl */ `
const CWBVH_CHILDREN: u32 = 8u;
const CWBVH_CHILD_BOUNDS_PACKED_U32: u32 = 3u;
const CWBVH_INTERSECT_STACK_DEPTH: u32 = ${CWBVH_INTERSECT_STACK_DEPTH}u;
const CWBVH_INTERSECT_INFINITY: f32 = 1e20;
const CWBVH_CHILD_EMPTY: u32 = 0u;
const CWBVH_CHILD_NODE: u32 = 1u;
const CWBVH_CHILD_LEAF: u32 = 2u;

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
  let extent = parentMax - parentMin;
  if (!(extent > 0.0)) {
    return parentMin;
  }
  return parentMin + (f32(q) / 65535.0) * extent;
}

fn cwbvhLoadChildBounds(
  cwbvhNodeBounds: ptr<storage, array<CwbvhNodeBounds>, read>,
  cwbvhChildBoundsPacked: ptr<storage, array<u32>, read>,
  nodeIndex: u32,
  slot: u32,
) -> CwbvhAabb {
  let node = (*cwbvhNodeBounds)[nodeIndex];
  let pMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
  let pMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
  let base = nodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_PACKED_U32 + slot * CWBVH_CHILD_BOUNDS_PACKED_U32;
  let w0 = (*cwbvhChildBoundsPacked)[base + 0u];
  let w1 = (*cwbvhChildBoundsPacked)[base + 1u];
  let w2 = (*cwbvhChildBoundsPacked)[base + 2u];

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

fn cwbvhIntersectFirstHitRangeFromRoot(
  cwbvhNodeBounds: ptr<storage, array<CwbvhNodeBounds>, read>,
  cwbvhChildBoundsPacked: ptr<storage, array<u32>, read>,
  cwbvhChildMeta: ptr<storage, array<CwbvhChildMeta>, read>,
  cwbvhChildCount: ptr<storage, array<u32>, read>,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
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
  if (nodeCount == 0u || rootNode >= nodeCount) {
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
      continue;
    }

    let count = min((*cwbvhChildCount)[nodeIndex], CWBVH_CHILDREN);
    for (var slot = 0u; slot < count; slot = slot + 1u) {
      let childIndex = nodeIndex * CWBVH_CHILDREN + slot;
      let childInfo = (*cwbvhChildMeta)[childIndex];
      if (childInfo.kind == CWBVH_CHILD_EMPTY) {
        continue;
      }

      let bounds = cwbvhLoadChildBounds(cwbvhNodeBounds, cwbvhChildBoundsPacked, nodeIndex, slot);
      let childT = cwbvhAabbEntry(ray.origin, invDir, bounds.boundsMin, bounds.boundsMax, best.dist);
      if (childT == CWBVH_INTERSECT_INFINITY) {
        continue;
      }

      if (childInfo.kind == CWBVH_CHILD_NODE) {
        if (stackPtr >= CWBVH_INTERSECT_STACK_DEPTH) {
          return best;
        }
        stack[stackPtr] = childInfo.indexOrOffset;
        stackPtr = stackPtr + 1u;
      } else if (childInfo.kind == CWBVH_CHILD_LEAF) {
        for (var i = 0u; i < childInfo.triCount; i = i + 1u) {
          let triIdx = childInfo.indexOrOffset + i;
          let idxEntry = (*bvh_index)[triIdx];
          if (skipGlass) {
            let trans4 = (idxEntry.w >> 4u) & 0xFu;
            if (trans4 > 4u) {
              continue;
            }
          }
          let idx = idxEntry.xyz;
          let pa4 = (*bvh_position)[idx.x];
          let pb4 = (*bvh_position)[idx.y];
          let pc4 = (*bvh_position)[idx.z];
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
            let uvA = unpack2x16unorm(bitcast<u32>(pa4.w));
            let uvB = unpack2x16unorm(bitcast<u32>(pb4.w));
            let uvC = unpack2x16unorm(bitcast<u32>(pc4.w));
            best.uv = tri.bary.x * uvA + tri.bary.y * uvB + tri.bary.z * uvC;
          }
        }
      }
    }
  }

  return best;
}

fn cwbvhIntersectFirstHitFromRoot(
  cwbvhNodeBounds: ptr<storage, array<CwbvhNodeBounds>, read>,
  cwbvhChildBoundsPacked: ptr<storage, array<u32>, read>,
  cwbvhChildMeta: ptr<storage, array<CwbvhChildMeta>, read>,
  cwbvhChildCount: ptr<storage, array<u32>, read>,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  ray: CwbvhRay,
  triEps: f32,
  nodeCount: u32,
  rootNode: u32,
  skipGlass: bool,
) -> CwbvhIntersectionResult {
  return cwbvhIntersectFirstHitRangeFromRoot(
    cwbvhNodeBounds,
    cwbvhChildBoundsPacked,
    cwbvhChildMeta,
    cwbvhChildCount,
    bvh_index,
    bvh_position,
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
  cwbvhNodeBounds: ptr<storage, array<CwbvhNodeBounds>, read>,
  cwbvhChildBoundsPacked: ptr<storage, array<u32>, read>,
  cwbvhChildMeta: ptr<storage, array<CwbvhChildMeta>, read>,
  cwbvhChildCount: ptr<storage, array<u32>, read>,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  ray: CwbvhRay,
  triEps: f32,
  nodeCount: u32,
  skipGlass: bool,
) -> CwbvhIntersectionResult {
  return cwbvhIntersectFirstHitFromRoot(
    cwbvhNodeBounds,
    cwbvhChildBoundsPacked,
    cwbvhChildMeta,
    cwbvhChildCount,
    bvh_index,
    bvh_position,
    ray,
    triEps,
    nodeCount,
    0u,
    skipGlass,
  );
}

fn cwbvhIntersectAnyFromRoot(
  cwbvhNodeBounds: ptr<storage, array<CwbvhNodeBounds>, read>,
  cwbvhChildBoundsPacked: ptr<storage, array<u32>, read>,
  cwbvhChildMeta: ptr<storage, array<CwbvhChildMeta>, read>,
  cwbvhChildCount: ptr<storage, array<u32>, read>,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  nodeCount: u32,
  rootNode: u32,
  skipGlass: bool,
) -> bool {
  if (nodeCount == 0u || rootNode >= nodeCount) {
    return false;
  }

  var stack: array<u32, ${CWBVH_INTERSECT_STACK_DEPTH}>;
  var stackPtr = 0u;
  stack[stackPtr] = rootNode;
  stackPtr = stackPtr + 1u;

  let invDir = safeInvDir(dir);

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIndex = stack[stackPtr];
    if (nodeIndex >= nodeCount) {
      continue;
    }

    let count = min((*cwbvhChildCount)[nodeIndex], CWBVH_CHILDREN);
    for (var slot = 0u; slot < count; slot = slot + 1u) {
      let childIndex = nodeIndex * CWBVH_CHILDREN + slot;
      let childInfo = (*cwbvhChildMeta)[childIndex];
      if (childInfo.kind == CWBVH_CHILD_EMPTY) {
        continue;
      }

      let bounds = cwbvhLoadChildBounds(cwbvhNodeBounds, cwbvhChildBoundsPacked, nodeIndex, slot);
      let childT = cwbvhAabbEntry(origin, invDir, bounds.boundsMin, bounds.boundsMax, tMax);
      if (childT == CWBVH_INTERSECT_INFINITY) {
        continue;
      }

      if (childInfo.kind == CWBVH_CHILD_NODE) {
        if (stackPtr >= CWBVH_INTERSECT_STACK_DEPTH) {
          return false;
        }
        stack[stackPtr] = childInfo.indexOrOffset;
        stackPtr = stackPtr + 1u;
      } else if (childInfo.kind == CWBVH_CHILD_LEAF) {
        for (var i = 0u; i < childInfo.triCount; i = i + 1u) {
          let triIdx = childInfo.indexOrOffset + i;
          let idxEntry = (*bvh_index)[triIdx];
          if (skipGlass) {
            let trans4 = (idxEntry.w >> 4u) & 0xFu;
            if (trans4 > 4u) {
              continue;
            }
          }
          let idx = idxEntry.xyz;
          let tri = mollerTrumboreCore(
            origin,
            dir,
            (*bvh_position)[idx.x].xyz,
            (*bvh_position)[idx.y].xyz,
            (*bvh_position)[idx.z].xyz,
            triEps,
          );
          if (tri.hit && tri.t > 1e-4 && tri.t < tMax) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

fn cwbvhIntersectAny(
  cwbvhNodeBounds: ptr<storage, array<CwbvhNodeBounds>, read>,
  cwbvhChildBoundsPacked: ptr<storage, array<u32>, read>,
  cwbvhChildMeta: ptr<storage, array<CwbvhChildMeta>, read>,
  cwbvhChildCount: ptr<storage, array<u32>, read>,
  bvh_index: ptr<storage, array<vec4u>, read>,
  bvh_position: ptr<storage, array<vec4f>, read>,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  nodeCount: u32,
  skipGlass: bool,
) -> bool {
  return cwbvhIntersectAnyFromRoot(
    cwbvhNodeBounds,
    cwbvhChildBoundsPacked,
    cwbvhChildMeta,
    cwbvhChildCount,
    bvh_index,
    bvh_position,
    origin,
    dir,
    tMax,
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
`;
