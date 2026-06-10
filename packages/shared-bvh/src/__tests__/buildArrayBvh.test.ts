/**
 * W2-C2 — buildArrayBvh smoke + invariant tests.
 *
 * Verifies the THREE-independent CPU BVH builder hoisted from pt-webgpu:
 *   1. Empty input returns a single zero-filled node + the unmodified inputs.
 *   2. Single-leaf case for tiny input encodes LEAFNODE_FLAG | triCount.
 *   3. Multi-leaf case preserves the triangle / material multiset.
 *   4. Interior nodes carry valid RELATIVE right-child offsets.
 *   5. Stride-3 indices round-trip with the same canonical node layout.
 *   6. Output for the same input is byte-identical run-to-run (determinism).
 */

import { describe, expect, it } from 'vitest';
import { buildArrayBvh, validateBvhEncoding } from '../index.js';

const LEAFNODE_FLAG = 0xffff;

function makeStride4Positions(vertexCount: number): Float32Array {
  const out = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i += 1) {
    out[i * 4] = i;
    out[i * 4 + 1] = (i % 3) * 0.25;
    out[i * 4 + 2] = (i % 5) * 0.5;
    out[i * 4 + 3] = 0;
  }
  return out;
}

function makeStride3Positions(vertexCount: number): Float32Array {
  const out = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i += 1) {
    out[i * 3] = i;
    out[i * 3 + 1] = (i % 3) * 0.25;
    out[i * 3 + 2] = (i % 5) * 0.5;
  }
  return out;
}

describe('buildArrayBvh', () => {
  it('1. returns a single empty node for zero triangles', () => {
    const positions = new Float32Array(0);
    const indices = new Uint32Array(0);
    const triMaterialIds = new Uint32Array(0);
    const built = buildArrayBvh(positions, indices, triMaterialIds);

    expect(built.bvhNodes.length).toBe(8); // single 8 × u32 node
    expect(built.reorderedIndices.length).toBe(0);
    expect(built.reorderedTriMaterialIds.length).toBe(0);
  });

  it('2. emits a single-leaf node for a single triangle (default stride 4)', () => {
    const positions = makeStride4Positions(3);
    const indices = new Uint32Array([0, 1, 2, 0]);
    const triMaterialIds = new Uint32Array([7]);
    const built = buildArrayBvh(positions, indices, triMaterialIds);

    expect(built.bvhNodes.length).toBe(8); // one node, 8 × u32
    const nodeU32 = new Uint32Array(built.bvhNodes.buffer);
    expect(nodeU32[6]).toBe(0); // leaf triangle offset
    expect(nodeU32[7]).toBe(0xffff0001); // leaf flag | triangle count
    expect(Array.from(built.reorderedIndices)).toEqual(Array.from(indices));
    expect(Array.from(built.reorderedTriMaterialIds)).toEqual([7]);
  });

  it('3. preserves triangle / material multiset across SAH reorder', () => {
    const positions = makeStride4Positions(18);
    const indices = new Uint32Array([
      0, 1, 2, 0,
      3, 4, 5, 0,
      6, 7, 8, 0,
      9, 10, 11, 0,
      12, 13, 14, 0,
      15, 16, 17, 0,
    ]);
    const triMaterialIds = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const built = buildArrayBvh(positions, indices, triMaterialIds);

    expect(built.bvhNodes.length).toBeGreaterThan(8); // multi-node tree
    expect(built.bvhNodes.length % 8).toBe(0);
    expect(built.reorderedIndices.length).toBe(indices.length);
    expect(built.reorderedTriMaterialIds.length).toBe(triMaterialIds.length);
    expect(Array.from(built.reorderedTriMaterialIds).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('4. interior nodes carry valid relative right-child offsets', () => {
    const positions = makeStride4Positions(18);
    const indices = new Uint32Array([
      0, 1, 2, 0,
      3, 4, 5, 0,
      6, 7, 8, 0,
      9, 10, 11, 0,
      12, 13, 14, 0,
      15, 16, 17, 0,
    ]);
    const triMaterialIds = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const built = buildArrayBvh(positions, indices, triMaterialIds);

    const totalNodes = built.bvhNodes.length / 8;
    // validateBvhEncoding (sibling export) enforces 1 ≤ offset < totalNodes.
    expect(() => validateBvhEncoding(built.bvhNodes, totalNodes)).not.toThrow();

    const u32 = new Uint32Array(built.bvhNodes.buffer);
    let foundInterior = false;
    for (let i = 0; i < totalNodes; i++) {
      const splitOrCount = u32[i * 8 + 7] ?? 0;
      const isLeaf = (splitOrCount >>> 16) === LEAFNODE_FLAG;
      if (isLeaf) continue;
      foundInterior = true;
      const offset = u32[i * 8 + 6] ?? 0;
      expect(offset).toBeGreaterThanOrEqual(1);
      expect(offset).toBeLessThan(totalNodes);
    }
    expect(foundInterior).toBe(true);
  });

  it('5. stride-3 positions + stride-3 indices round-trip with canonical node layout', () => {
    const positions = makeStride3Positions(6);
    // 2 triangles, stride-3 indices (no padding lane).
    const indices = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const triMaterialIds = new Uint32Array([11, 22]);
    const built = buildArrayBvh(positions, indices, triMaterialIds, {
      positionStride: 3,
      indexStride: 3,
    });

    expect(built.bvhNodes.length).toBe(8); // 2 tris ≤ maxLeafTriangles → single leaf
    expect(built.reorderedIndices.length).toBe(indices.length);
    // Stride-3 outputs have no padding lane to zero-fill.
    expect(Array.from(built.reorderedIndices)).toEqual(Array.from(indices));
    expect(Array.from(built.reorderedTriMaterialIds)).toEqual([11, 22]);
  });

  it('7. planar SAH (B7): a coplanar floor grid builds a balanced tree (depth ~log n)', () => {
    // Regression pin for B7. A coplanar floor (all triangles on the y=0 plane,
    // dy=0 for every node AABB) used to zero `surfaceArea` for every split, so the
    // SAH lost all discrimination and the builder produced a near-linked-list
    // (verified depth 45 for 2000 tris before the half-perimeter fix). The fix
    // ranks splits along the in-plane axes, restoring a balanced tree (depth 9).
    const NQUADS = 1000;
    const side = Math.ceil(Math.sqrt(NQUADS));
    const positions: number[] = [];
    const indices: number[] = [];
    const mats: number[] = [];
    let v = 0; let q = 0;
    for (let i = 0; i < side && q < NQUADS; i += 1) {
      for (let j = 0; j < side && q < NQUADS; j += 1, q += 1) {
        positions.push(i, 0, j, 0, i + 1, 0, j, 0, i + 1, 0, j + 1, 0, i, 0, j + 1, 0);
        indices.push(v, v + 1, v + 2, 0, v, v + 2, v + 3, 0);
        mats.push(0, 0);
        v += 4;
      }
    }
    const built = buildArrayBvh(new Float32Array(positions), new Uint32Array(indices), new Uint32Array(mats));

    const u32 = new Uint32Array(built.bvhNodes.buffer);
    let maxDepth = 0; let maxLeaf = 0; let leafTris = 0;
    const walk = (idx: number, d: number): void => {
      maxDepth = Math.max(maxDepth, d);
      const split = u32[idx * 8 + 7]!;
      if ((split >>> 16) === LEAFNODE_FLAG) {
        const c = split & 0xffff;
        maxLeaf = Math.max(maxLeaf, c);
        leafTris += c;
        return;
      }
      walk(idx + 1, d + 1);
      walk(idx + (u32[idx * 8 + 6] ?? 0), d + 1);
    };
    walk(0, 0);

    const triCount = NQUADS * 2; // 2000
    expect(leafTris).toBe(triCount);          // set-preservation
    expect(maxLeaf).toBeLessThanOrEqual(4);   // leaves stay bounded
    expect(maxDepth).toBeLessThanOrEqual(16); // balanced (pre-fix was 45)
  });

  it('6. deterministic: identical inputs → byte-identical bvhNodes', () => {
    const positions = makeStride4Positions(18);
    const indices = new Uint32Array([
      0, 1, 2, 0,
      3, 4, 5, 0,
      6, 7, 8, 0,
      9, 10, 11, 0,
      12, 13, 14, 0,
      15, 16, 17, 0,
    ]);
    const triMaterialIds = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const a = buildArrayBvh(positions, indices, triMaterialIds);
    const b = buildArrayBvh(positions, indices, triMaterialIds);

    expect(a.bvhNodes.length).toBe(b.bvhNodes.length);
    for (let i = 0; i < a.bvhNodes.length; i++) {
      expect(a.bvhNodes[i]).toBe(b.bvhNodes[i]);
    }
    expect(Array.from(a.reorderedIndices)).toEqual(Array.from(b.reorderedIndices));
    expect(Array.from(a.reorderedTriMaterialIds)).toEqual(Array.from(b.reorderedTriMaterialIds));
  });
});
