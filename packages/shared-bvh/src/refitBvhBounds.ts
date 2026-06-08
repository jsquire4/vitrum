import { isLeafSplit } from './buildArrayBvh.js';

/**
 * In-place BVH **refit**: recompute every node's AABB bounds without rebuilding
 * tree topology. Used by transform/position update paths that preserve leaf
 * membership and only need fresh min/max fields.
 */
export function refitBvhBounds(
  bvhNodes: Float32Array,
  indices: Uint32Array,
  positions: Float32Array,
  positionStrideFloats: 3 | 4,
): void {
  const UINT32_PER_NODE = 8;
  const totalNodes = bvhNodes.length / UINT32_PER_NODE;
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
      for (let t = 0; t < triCount; t += 1) {
        const triIdx = triOffset + t;
        const i0 = indices[triIdx * 3 + 0]!;
        const i1 = indices[triIdx * 3 + 1]!;
        const i2 = indices[triIdx * 3 + 2]!;

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
