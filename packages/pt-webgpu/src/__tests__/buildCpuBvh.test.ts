import { describe, expect, it } from 'vitest';
import { buildCpuBvh } from '../scene/buildCpuBvh.js';

function makePositions(vertexCount: number): Float32Array {
  const out = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i += 1) {
    out[i * 4] = i;
    out[i * 4 + 1] = (i % 3) * 0.25;
    out[i * 4 + 2] = (i % 5) * 0.5;
    out[i * 4 + 3] = 0;
  }
  return out;
}

describe('buildCpuBvh', () => {
  it('builds a single-leaf node for small triangle sets', () => {
    const positions = makePositions(3);
    const indices = new Uint32Array([0, 1, 2, 0]);
    const triMaterialIds = new Uint32Array([7]);
    const built = buildCpuBvh(positions, indices, triMaterialIds);

    expect(built.bvhNodes.length).toBe(8);
    const nodeU32 = new Uint32Array(built.bvhNodes.buffer);
    expect(nodeU32[6]).toBe(0); // leaf triangle offset
    expect(nodeU32[7]).toBe(0xffff0001); // leaf flag + triangle count
    expect(Array.from(built.reorderedIndices)).toEqual(Array.from(indices));
    expect(Array.from(built.reorderedTriMaterialIds)).toEqual([7]);
  });

  it('builds multiple nodes and preserves triangle/material multiset', () => {
    const positions = makePositions(18);
    const indices = new Uint32Array([
      0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 0, 12, 13, 14, 0, 15, 16, 17, 0,
    ]);
    const triMaterialIds = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const built = buildCpuBvh(positions, indices, triMaterialIds);

    expect(built.bvhNodes.length).toBeGreaterThan(8);
    expect(built.bvhNodes.length % 8).toBe(0);
    expect(built.reorderedIndices.length).toBe(indices.length);
    expect(built.reorderedTriMaterialIds.length).toBe(triMaterialIds.length);

    const sortedMaterials = Array.from(built.reorderedTriMaterialIds).sort((a, b) => a - b);
    expect(sortedMaterials).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('encodes interior nodes with relative right-child offsets (1 ≤ offset < totalNodes)', () => {
    // Build a BVH large enough to have multiple interior nodes.
    const positions = makePositions(18);
    const indices = new Uint32Array([
      0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 0, 12, 13, 14, 0, 15, 16, 17, 0,
    ]);
    const triMaterialIds = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const built = buildCpuBvh(positions, indices, triMaterialIds);

    const LEAFNODE_FLAG = 0xffff;
    const nodeU32 = new Uint32Array(built.bvhNodes.buffer);
    const totalNodes = built.bvhNodes.length / 8; // 8 u32 per node

    let foundInterior = false;
    for (let i = 0; i < totalNodes; i++) {
      const splitOrCount = nodeU32[i * 8 + 7] ?? 0;
      const isLeaf = splitOrCount >>> 16 === LEAFNODE_FLAG;
      if (isLeaf) continue;
      foundInterior = true;
      const offset = nodeU32[i * 8 + 6] ?? 0;
      expect(offset).toBeGreaterThanOrEqual(1);
      expect(offset).toBeLessThan(totalNodes);
    }
    // Sanity: a 6-triangle BVH must have at least one interior node.
    expect(foundInterior).toBe(true);
  });
});
