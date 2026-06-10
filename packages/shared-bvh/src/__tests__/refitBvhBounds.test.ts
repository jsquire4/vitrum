/**
 * refitBvhBounds — traversal coverage (item 26).
 *
 * Strategy: build a small BVH via buildArrayBvh (which is already
 * well-tested), then exercise refitBvhBounds by:
 *   (a) re-running on the unchanged positions → byte-identical ("no-op" refit).
 *   (b) mutating vertex positions → asserting that every leaf AABB exactly
 *       bounds its triangles (independently recomputed), every interior node
 *       exactly bounds its children, and the root equals the brute-force
 *       scene AABB.
 *   (c) degenerate cases: single triangle, and the empty-BVH contract.
 *
 * Node layout (matches buildArrayBvh.ts file-header comment):
 *   f32[0..2]  boundsMin xyz
 *   f32[3..5]  boundsMax xyz
 *   u32[6]     rightChildOrTriOffset
 *              - leaf:     absolute triangle offset into reorderedIndices
 *              - interior: relative offset to right child
 *   u32[7]     splitAxisOrTriCount
 *              - leaf:     0xFFFF0000 | triangleCount
 *              - interior: split axis (0=X, 1=Y, 2=Z)
 */

import { describe, expect, it } from 'vitest';
import { buildArrayBvh, isLeafSplit, refitBvhBounds } from '../index.js';

// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * Build a stride-3 position + stride-3 index BVH around a set of
 * triangle vertices given as flat [x,y,z, x,y,z, ...].
 *
 * Returns { bvhNodes, reorderedIndices, positions } where positions is
 * the ORIGINAL (pre-BVH-reorder) buffer so callers can mutate it
 * independently.
 */
function buildTestBvh(rawPositions: number[], triangles: [number, number, number][]) {
  const positions = new Float32Array(rawPositions);
  const flatIdx: number[] = [];
  for (const [a, b, c] of triangles) flatIdx.push(a, b, c);
  const indices = new Uint32Array(flatIdx);
  const triMat = new Uint32Array(triangles.length);
  const result = buildArrayBvh(positions, indices, triMat, {
    positionStride: 3,
    indexStride: 3,
  });
  return { bvhNodes: result.bvhNodes, reorderedIndices: result.reorderedIndices, positions };
}

/** Read a node's AABB from the Float32Array buffer. */
function readNodeAabb(f32: Float32Array, nodeIdx: number) {
  const b = nodeIdx * 8;
  return {
    minX: f32[b + 0]!, minY: f32[b + 1]!, minZ: f32[b + 2]!,
    maxX: f32[b + 3]!, maxY: f32[b + 4]!, maxZ: f32[b + 5]!,
  };
}

/** Read node slot 6 (rightChildOrTriOffset) as u32. */
function readSlot6(f32: Float32Array, nodeIdx: number): number {
  return new Uint32Array(f32.buffer)[nodeIdx * 8 + 6]!;
}

/** Read node slot 7 (splitAxisOrTriCount) as u32. */
function readSlot7(f32: Float32Array, nodeIdx: number): number {
  return new Uint32Array(f32.buffer)[nodeIdx * 8 + 7]!;
}

/**
 * Independently recompute the tight AABB for a leaf node by reading
 * triangles from reorderedIndices (stride 3) and positions.
 */
function leafAabbFromGeometry(
  f32: Float32Array,
  nodeIdx: number,
  reorderedIndices: Uint32Array,
  positions: Float32Array,
) {
  const triOffset = readSlot6(f32, nodeIdx);
  const triCount = readSlot7(f32, nodeIdx) & 0xffff;
  let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
  let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
  for (let t = 0; t < triCount; t++) {
    const i0 = reorderedIndices[(triOffset + t) * 3 + 0]!;
    const i1 = reorderedIndices[(triOffset + t) * 3 + 1]!;
    const i2 = reorderedIndices[(triOffset + t) * 3 + 2]!;
    for (const vi of [i0, i1, i2]) {
      const x = positions[vi * 3 + 0]!, y = positions[vi * 3 + 1]!, z = positions[vi * 3 + 2]!;
      if (x < mnX) mnX = x; if (x > mxX) mxX = x;
      if (y < mnY) mnY = y; if (y > mxY) mxY = y;
      if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
    }
  }
  return { minX: mnX, minY: mnY, minZ: mnZ, maxX: mxX, maxY: mxY, maxZ: mxZ };
}

/** Brute-force scene AABB from positions (stride 3). */
function bruteForceSaxbb(positions: Float32Array) {
  let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
  let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
  for (let i = 0; i < positions.length / 3; i++) {
    const x = positions[i * 3]!, y = positions[i * 3 + 1]!, z = positions[i * 3 + 2]!;
    if (x < mnX) mnX = x; if (x > mxX) mxX = x;
    if (y < mnY) mnY = y; if (y > mxY) mxY = y;
    if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
  }
  return { minX: mnX, minY: mnY, minZ: mnZ, maxX: mxX, maxY: mxY, maxZ: mxZ };
}

/**
 * Walk every node in the BVH after a refit and assert:
 *   - leaf nodes: AABB exactly matches independent triangle read.
 *   - interior nodes: AABB exactly matches union of children's AABBs.
 * Returns the root AABB for the caller to compare against brute-force.
 */
function validateTree(
  f32: Float32Array,
  reorderedIndices: Uint32Array,
  positions: Float32Array,
) {
  const totalNodes = f32.length / 8;
  // Post-order traversal — same strategy refitBvhBounds itself uses.
  const stack: { node: number; isSecondVisit: boolean }[] = [{ node: 0, isSecondVisit: false }];
  while (stack.length > 0) {
    const entry = stack.pop()!;
    const { node: nodeIdx, isSecondVisit } = entry;
    const slot7 = readSlot7(f32, nodeIdx);
    const leaf = isLeafSplit(slot7);
    if (leaf) {
      // Leaf: assert against independent geometry scan.
      const got = readNodeAabb(f32, nodeIdx);
      const expected = leafAabbFromGeometry(f32, nodeIdx, reorderedIndices, positions);
      expect(got.minX).toBeCloseTo(expected.minX, 5);
      expect(got.minY).toBeCloseTo(expected.minY, 5);
      expect(got.minZ).toBeCloseTo(expected.minZ, 5);
      expect(got.maxX).toBeCloseTo(expected.maxX, 5);
      expect(got.maxY).toBeCloseTo(expected.maxY, 5);
      expect(got.maxZ).toBeCloseTo(expected.maxZ, 5);
    } else if (isSecondVisit) {
      // Interior: assert against union of children.
      const leftChild = nodeIdx + 1;
      const rightChild = nodeIdx + readSlot6(f32, nodeIdx);
      const lA = readNodeAabb(f32, leftChild);
      const rA = readNodeAabb(f32, rightChild);
      const expectedMin = {
        minX: Math.min(lA.minX, rA.minX),
        minY: Math.min(lA.minY, rA.minY),
        minZ: Math.min(lA.minZ, rA.minZ),
      };
      const expectedMax = {
        maxX: Math.max(lA.maxX, rA.maxX),
        maxY: Math.max(lA.maxY, rA.maxY),
        maxZ: Math.max(lA.maxZ, rA.maxZ),
      };
      const got = readNodeAabb(f32, nodeIdx);
      expect(got.minX).toBeCloseTo(expectedMin.minX, 5);
      expect(got.minY).toBeCloseTo(expectedMin.minY, 5);
      expect(got.minZ).toBeCloseTo(expectedMin.minZ, 5);
      expect(got.maxX).toBeCloseTo(expectedMax.maxX, 5);
      expect(got.maxY).toBeCloseTo(expectedMax.maxY, 5);
      expect(got.maxZ).toBeCloseTo(expectedMax.maxZ, 5);
    } else {
      // First visit of interior: schedule second visit, then children.
      stack.push({ node: nodeIdx, isSecondVisit: true });
      const leftChild = nodeIdx + 1;
      const rightChild = nodeIdx + readSlot6(f32, nodeIdx);
      stack.push({ node: rightChild, isSecondVisit: false });
      stack.push({ node: leftChild, isSecondVisit: false });
    }
    void totalNodes; // silence unused var warning
  }
  return readNodeAabb(f32, 0);
}

// ── geometry factories ────────────────────────────────────────────────────────

/**
 * A simple box mesh: 8 vertices, 12 triangles (two per face).
 * Vertices are the 8 corners of an axis-aligned box [lx,hx] × [ly,hy] × [lz,hz].
 * Returns flat position array (stride 3) and triangle index list.
 */
function makeBox(lx: number, hx: number, ly: number, hy: number, lz: number, hz: number): {
  positions: number[];
  triangles: [number, number, number][];
} {
  const positions = [
    lx, ly, lz,  hx, ly, lz,  hx, hy, lz,  lx, hy, lz,   // z=lz face
    lx, ly, hz,  hx, ly, hz,  hx, hy, hz,  lx, hy, hz,   // z=hz face
  ];
  const triangles: [number, number, number][] = [
    [0, 1, 2], [0, 2, 3],   // -z face
    [4, 6, 5], [4, 7, 6],   // +z face
    [0, 5, 1], [0, 4, 5],   // -y face
    [2, 6, 7], [2, 7, 3],   // +y face
    [0, 3, 7], [0, 7, 4],   // -x face
    [1, 5, 6], [1, 6, 2],   // +x face
  ];
  return { positions, triangles };
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe('refitBvhBounds', () => {

  it('(a) no-op refit: refitting with unchanged positions is byte-identical', () => {
    const { positions, triangles } = makeBox(0, 2, 0, 3, 0, 1);
    const { bvhNodes, reorderedIndices } = buildTestBvh(positions, triangles);

    // Snapshot the buffer bytes before refit.
    const before = bvhNodes.slice();

    // Refit with the exact same positions.
    refitBvhBounds(bvhNodes, reorderedIndices, new Float32Array(positions), 3);

    expect(Array.from(bvhNodes)).toEqual(Array.from(before));
  });

  it('(b) leaf AABBs exactly bound their triangles after position mutation', () => {
    // Build a multi-leaf BVH (enough tris to force at least 2 leaves with maxLeafTriangles=4).
    const { positions, triangles } = makeBox(-1, 1, -1, 1, -1, 1);
    const { bvhNodes, reorderedIndices } = buildTestBvh(positions, triangles);

    // Mutate: scale all positions by 2.
    const newPositions = new Float32Array(positions.length);
    for (let i = 0; i < positions.length; i++) newPositions[i] = (positions[i] ?? 0) * 2;

    refitBvhBounds(bvhNodes, reorderedIndices, newPositions, 3);

    validateTree(bvhNodes, reorderedIndices, newPositions);
  });

  it('(c) every interior node exactly bounds its children after mutation', () => {
    // Use a larger mesh so the SAH produces interior nodes.
    const { positions: p1, triangles: t1 } = makeBox(0, 1, 0, 1, 0, 1);
    const { positions: p2, triangles: t2 } = makeBox(5, 6, 5, 6, 5, 6);
    // Combine: append the second box's vertices with an offset.
    const vOffset = p1.length / 3;
    const combinedPos = [...p1, ...p2];
    const combinedTris: [number, number, number][] = [
      ...t1,
      ...t2.map(([a, b, c]) => [a + vOffset, b + vOffset, c + vOffset] as [number, number, number]),
    ];
    const { bvhNodes, reorderedIndices } = buildTestBvh(combinedPos, combinedTris);

    // Shift one box far away.
    const newPositions = new Float32Array(combinedPos);
    for (let i = vOffset; i < combinedPos.length / 3; i++) {
      newPositions[i * 3 + 0] = (newPositions[i * 3 + 0] ?? 0) + 100;
      newPositions[i * 3 + 1] = (newPositions[i * 3 + 1] ?? 0) + 100;
      newPositions[i * 3 + 2] = (newPositions[i * 3 + 2] ?? 0) + 100;
    }

    refitBvhBounds(bvhNodes, reorderedIndices, newPositions, 3);

    validateTree(bvhNodes, reorderedIndices, newPositions);
  });

  it('(d) root AABB equals brute-force scene AABB after mutation', () => {
    const { positions, triangles } = makeBox(0, 4, -2, 2, -3, 3);
    const { bvhNodes, reorderedIndices } = buildTestBvh(positions, triangles);

    // Non-uniform scale.
    const newPositions = new Float32Array(positions.length);
    for (let i = 0; i < positions.length / 3; i++) {
      newPositions[i * 3 + 0] = (positions[i * 3 + 0] ?? 0) * 3;
      newPositions[i * 3 + 1] = (positions[i * 3 + 1] ?? 0) * 0.5;
      newPositions[i * 3 + 2] = (positions[i * 3 + 2] ?? 0) * 2 + 10;
    }

    refitBvhBounds(bvhNodes, reorderedIndices, newPositions, 3);

    const rootAabb = readNodeAabb(bvhNodes, 0);
    const brute = bruteForceSaxbb(newPositions);

    expect(rootAabb.minX).toBeCloseTo(brute.minX, 5);
    expect(rootAabb.minY).toBeCloseTo(brute.minY, 5);
    expect(rootAabb.minZ).toBeCloseTo(brute.minZ, 5);
    expect(rootAabb.maxX).toBeCloseTo(brute.maxX, 5);
    expect(rootAabb.maxY).toBeCloseTo(brute.maxY, 5);
    expect(rootAabb.maxZ).toBeCloseTo(brute.maxZ, 5);
  });

  it('(e) single-triangle BVH: leaf AABB is exact after mutation', () => {
    // One triangle → single leaf node (no interior nodes).
    const positions = new Float32Array([0, 0, 0,  1, 0, 0,  0, 1, 0]);
    const indices = new Uint32Array([0, 1, 2]);
    const triMat = new Uint32Array([0]);
    const { bvhNodes, reorderedIndices } = buildArrayBvh(
      positions, indices, triMat, { positionStride: 3, indexStride: 3 },
    );

    // Move the triangle to a new location.
    const newPositions = new Float32Array([10, 20, 30,  11, 20, 30,  10, 21, 30]);

    refitBvhBounds(bvhNodes, reorderedIndices, newPositions, 3);

    const aabb = readNodeAabb(bvhNodes, 0);
    expect(aabb.minX).toBeCloseTo(10, 5);
    expect(aabb.minY).toBeCloseTo(20, 5);
    expect(aabb.minZ).toBeCloseTo(30, 5);
    expect(aabb.maxX).toBeCloseTo(11, 5);
    expect(aabb.maxY).toBeCloseTo(21, 5);
    expect(aabb.maxZ).toBeCloseTo(30, 5);
  });

  it('(f) empty BVH (zero nodes) returns without error', () => {
    // The zero-triangle empty node from buildArrayBvh has totalNodes=1 but
    // slot7=0 (NOT a leaf flag) since the empty node is zero-filled.
    // refitBvhBounds itself guards `if (totalNodes === 0) return`.
    // This test exercises the guard by passing a zero-length buffer.
    const emptyBvhNodes = new Float32Array(0);
    const emptyIndices = new Uint32Array(0);
    const emptyPos = new Float32Array(0);
    // Must not throw.
    expect(() => refitBvhBounds(emptyBvhNodes, emptyIndices, emptyPos, 3)).not.toThrow();
  });

  it('(g) stride-4 positions: refitBvhBounds reads xyz and ignores w', () => {
    // stride-4: each vertex is [x, y, z, w] — w must be ignored.
    // Build with stride 4 (buildArrayBvh default).
    const pos4 = new Float32Array([
      0, 0, 0, 999,   // vertex 0 — w=999 is trash
      1, 0, 0, 999,
      0, 1, 0, 999,
      0, 0, 1, 999,
      1, 1, 1, 999,
      2, 2, 2, 999,
    ]);
    const indices4 = new Uint32Array([
      0, 1, 2, 0,
      3, 4, 5, 0,
    ]);
    const triMat = new Uint32Array([0, 0]);
    const { bvhNodes, reorderedIndices } = buildArrayBvh(pos4, indices4, triMat, {
      positionStride: 4,
      indexStride: 4,
    });

    // Mutate: move second triangle's verts.
    const newPos = new Float32Array(pos4);
    newPos[3 * 4 + 0] = 10; newPos[4 * 4 + 0] = 10; newPos[5 * 4 + 0] = 10;
    newPos[3 * 4 + 1] = 10; newPos[4 * 4 + 1] = 10; newPos[5 * 4 + 1] = 10;
    newPos[3 * 4 + 2] = 10; newPos[4 * 4 + 2] = 10; newPos[5 * 4 + 2] = 10;

    expect(() => refitBvhBounds(bvhNodes, reorderedIndices, newPos, 4)).not.toThrow();

    // Root AABB must bound both clusters: [0,1]³ and [10]³.
    const rootAabb = readNodeAabb(bvhNodes, 0);
    expect(rootAabb.minX).toBeLessThanOrEqual(0);
    expect(rootAabb.minY).toBeLessThanOrEqual(0);
    expect(rootAabb.minZ).toBeLessThanOrEqual(0);
    expect(rootAabb.maxX).toBeGreaterThanOrEqual(10);
    expect(rootAabb.maxY).toBeGreaterThanOrEqual(10);
    expect(rootAabb.maxZ).toBeGreaterThanOrEqual(10);

    // The w lane (999) must NOT have inflated any bound.
    expect(rootAabb.maxX).toBeLessThanOrEqual(11);
    expect(rootAabb.maxY).toBeLessThanOrEqual(11);
    expect(rootAabb.maxZ).toBeLessThanOrEqual(11);
  });

});
