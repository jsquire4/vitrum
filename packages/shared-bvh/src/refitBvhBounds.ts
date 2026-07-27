import { isLeafSplit } from './buildArrayBvh.js';
import { BVH_NODE_FLOATS } from './strides.js';
function preflightRefitInputs(
  bvhNodes: Float32Array, indices: Uint32Array, positions: Float32Array,
  positionStrideFloats: 3 | 4, indexStride: 3 | 4,
): number {
  if (bvhNodes.length % BVH_NODE_FLOATS !== 0) {
    throw new Error('[@vitrum/shared-bvh/refitBvhBounds] node buffer is not node-stride-aligned.');
  }
  if (positionStrideFloats !== 3 && positionStrideFloats !== 4) {
    throw new RangeError('[@vitrum/shared-bvh/refitBvhBounds] position stride must be 3 or 4.');
  }
  if (indexStride !== 3 && indexStride !== 4) {
    throw new RangeError('[@vitrum/shared-bvh/refitBvhBounds] index stride must be 3 or 4.');
  }
  if (positions.length % positionStrideFloats !== 0 || indices.length % indexStride !== 0) {
    throw new Error('[@vitrum/shared-bvh/refitBvhBounds] geometry buffer is not stride-aligned.');
  }
  const totalNodes = bvhNodes.length / BVH_NODE_FLOATS;
  if (totalNodes === 0) return 0;
  const triangleCount = indices.length / indexStride;
  const vertexCount = positions.length / positionStrideFloats;
  const words = new Uint32Array(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.length);
  const seen = new Uint8Array(totalNodes);
  const pending = [0];
  let visited = 0;
  while (pending.length > 0) {
    const nodeIndex = pending.pop()!;
    if (nodeIndex >= totalNodes || seen[nodeIndex] !== 0) {
      throw new Error(`[@vitrum/shared-bvh/refitBvhBounds] invalid or repeated topology node ${nodeIndex}.`);
    }
    seen[nodeIndex] = 1;
    visited += 1;
    const base = nodeIndex * BVH_NODE_FLOATS;
    const splitOrCount = words[base + 7]!;
    if (!isLeafSplit(splitOrCount)) {
      const left = nodeIndex + 1;
      const right = nodeIndex + words[base + 6]!;
      if (splitOrCount > 2 || left >= totalNodes || right <= nodeIndex || right >= totalNodes) {
        throw new Error(`[@vitrum/shared-bvh/refitBvhBounds] interior node ${nodeIndex} is corrupt.`);
      }
      pending.push(right, left);
      continue;
    }
    const leafCount = splitOrCount & 0xffff;
    const triangleOffset = words[base + 6]!;
    if (triangleOffset > triangleCount || leafCount > triangleCount - triangleOffset) {
      throw new Error(`[@vitrum/shared-bvh/refitBvhBounds] leaf ${nodeIndex} triangle range is corrupt.`);
    }
    for (let tri = triangleOffset; tri < triangleOffset + leafCount; tri += 1) {
      for (let lane = 0; lane < 3; lane += 1) {
        const vertex = indices[tri * indexStride + lane]!;
        if (vertex >= vertexCount) {
          throw new Error(`[@vitrum/shared-bvh/refitBvhBounds] triangle ${tri} has an invalid vertex.`);
        }
        const p = vertex * positionStrideFloats;
        if (!Number.isFinite(positions[p]) || !Number.isFinite(positions[p + 1]) || !Number.isFinite(positions[p + 2])) {
          throw new Error(`[@vitrum/shared-bvh/refitBvhBounds] vertex ${vertex} is non-finite.`);
        }
      }
    }
  }
  if (visited !== totalNodes) {
    throw new Error('[@vitrum/shared-bvh/refitBvhBounds] topology contains unreachable nodes.');
  }
  return totalNodes;
}


/**
 * In-place BVH **refit**: recompute every node's AABB bounds without rebuilding
 * tree topology. Used by transform/position update paths that preserve leaf
 * membership and only need fresh min/max fields.
 *
 * @param indexStride  Words per triangle in `indices`. Default `3`
 *   (`array<vec3u>` form, used by all current callers — RC, DDGI, walkaround).
 *   Pass `4` when the source is stride-4 (pt-webgpu `packSceneFromCore` output)
 *   to avoid a separate collapse step; the `.w` payload lane is ignored.
 */
export function refitBvhBounds(
  bvhNodes: Float32Array,
  indices: Uint32Array,
  positions: Float32Array,
  positionStrideFloats: 3 | 4,
  indexStride: 3 | 4 = 3,
): void {
  const UINT32_PER_NODE = BVH_NODE_FLOATS;
  const totalNodes = preflightRefitInputs(bvhNodes, indices, positions, positionStrideFloats, indexStride);
  if (totalNodes === 0) return;
  const u32 = new Uint32Array(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.length);
  const f32 = bvhNodes;

  const order = new Int32Array(totalNodes);
  let orderLen = 0;
  const stack = new Int32Array(totalNodes * 2);
  let sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const entry = stack[--sp]!;
    const isSecondVisit = (entry & 0x80000000) !== 0;
    const nodeIdx = entry & 0x7fffffff;
    if (isSecondVisit) {
      order[orderLen++] = nodeIdx;
      continue;
    }
    const splitOrCount = u32[nodeIdx * UINT32_PER_NODE + 7]!;
    const isLeaf = isLeafSplit(splitOrCount);
    if (isLeaf) {
      order[orderLen++] = nodeIdx;
      continue;
    }
    stack[sp++] = nodeIdx | 0x80000000;
    const leftChild = nodeIdx + 1;
    const rightChild = nodeIdx + u32[nodeIdx * UINT32_PER_NODE + 6]!;
    stack[sp++] = rightChild;
    stack[sp++] = leftChild;
  }

  for (let oi = 0; oi < orderLen; oi += 1) {
    const nodeIdx = order[oi]!;
    const base = nodeIdx * UINT32_PER_NODE;
    const splitOrCount = u32[base + 7]!;
    const isLeaf = isLeafSplit(splitOrCount);

    let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
    let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;

    if (isLeaf) {
      const triCount = splitOrCount & 0xffff;
      const triOffset = u32[base + 6]!;
      // The canonical empty BVH is one valid leaf with count 0. Keep its finite
      // point bounds instead of writing the untouched +/-Infinity accumulators.
      if (triCount === 0) {
        mnX = 0; mnY = 0; mnZ = 0;
        mxX = 0; mxY = 0; mxZ = 0;
      }
      for (let t = 0; t < triCount; t += 1) {
        const triIdx = triOffset + t;
        const i0 = indices[triIdx * indexStride + 0]!;
        const i1 = indices[triIdx * indexStride + 1]!;
        const i2 = indices[triIdx * indexStride + 2]!;

        let off = i0 * positionStrideFloats;
        let x = positions[off + 0]!, y = positions[off + 1]!, z = positions[off + 2]!;
        if (x < mnX) mnX = x; if (x > mxX) mxX = x;
        if (y < mnY) mnY = y; if (y > mxY) mxY = y;
        if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
        off = i1 * positionStrideFloats;
        x = positions[off + 0]!; y = positions[off + 1]!; z = positions[off + 2]!;
        if (x < mnX) mnX = x; if (x > mxX) mxX = x;
        if (y < mnY) mnY = y; if (y > mxY) mxY = y;
        if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
        off = i2 * positionStrideFloats;
        x = positions[off + 0]!; y = positions[off + 1]!; z = positions[off + 2]!;
        if (x < mnX) mnX = x; if (x > mxX) mxX = x;
        if (y < mnY) mnY = y; if (y > mxY) mxY = y;
        if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
      }
    } else {
      const leftChild = nodeIdx + 1;
      const rightChild = nodeIdx + u32[base + 6]!;
      let cBase = leftChild * UINT32_PER_NODE;
      let cMnX = f32[cBase + 0]!, cMnY = f32[cBase + 1]!, cMnZ = f32[cBase + 2]!;
      let cMxX = f32[cBase + 3]!, cMxY = f32[cBase + 4]!, cMxZ = f32[cBase + 5]!;
      if (cMnX < mnX) mnX = cMnX; if (cMxX > mxX) mxX = cMxX;
      if (cMnY < mnY) mnY = cMnY; if (cMxY > mxY) mxY = cMxY;
      if (cMnZ < mnZ) mnZ = cMnZ; if (cMxZ > mxZ) mxZ = cMxZ;
      cBase = rightChild * UINT32_PER_NODE;
      cMnX = f32[cBase + 0]!; cMnY = f32[cBase + 1]!; cMnZ = f32[cBase + 2]!;
      cMxX = f32[cBase + 3]!; cMxY = f32[cBase + 4]!; cMxZ = f32[cBase + 5]!;
      if (cMnX < mnX) mnX = cMnX; if (cMxX > mxX) mxX = cMxX;
      if (cMnY < mnY) mnY = cMnY; if (cMxY > mxY) mxY = cMxY;
      if (cMnZ < mnZ) mnZ = cMnZ; if (cMxZ > mxZ) mxZ = cMxZ;
    }

    f32[base + 0] = mnX; f32[base + 1] = mnY; f32[base + 2] = mnZ;
    f32[base + 3] = mxX; f32[base + 4] = mxY; f32[base + 5] = mxZ;
  }
}
