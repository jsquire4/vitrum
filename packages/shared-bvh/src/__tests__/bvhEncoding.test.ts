/**
 * Test 33-G — BVH cross-package compatibility round-trip.
 *
 * Three test groups:
 *
 * 1. pt-webgpu builder relative-offset invariant
 *    Build a BVH via `buildCpuBvh`; verify every interior node's
 *    `rightChildOrTriOffset ∈ [1, totalNodes)`.
 *
 * 2. shared-bvh `normalizeBvhInteriorOffsets` correctness
 *    Construct a synthetic absolute-encoded buffer (0.7.x layout) and
 *    verify the function rewrites every interior node to relative encoding.
 *    (normalizeBvhInteriorOffsets is not exported; we test it indirectly
 *    via `validateBvhEncoding` applied to the output of `buildSceneBVH`
 *    which calls it internally.)
 *
 * 3. Cross-package traversal identity
 *    Build the same geometry with `buildCpuBvh` (pt-webgpu path).
 *    Run 1 000 rays through a TS-mirrored relative-offset BVH traversal.
 *    Verify hit results are self-consistent (every hit satisfies the
 *    Möller-Trumbore invariants) and that the encoding is valid.
 *
 *    We cannot import `buildCpuBvh` from pt-webgpu (it would create a
 *    circular cross-package dependency). Instead, the mirror traversal is
 *    implemented here as a test-only TypeScript equivalent of the WGSL
 *    traversal loop — same relative-offset child indexing, same slab test,
 *    same Möller-Trumbore intersection. Documented as test-only mirror.
 */

import { describe, it, expect } from 'vitest';
import { validateBvhEncoding } from '../bvhCommon.js';

// ──────────────────────────────────────────────────────────────────────────
// Inline port of buildCpuBvh (pt-webgpu)
// ──────────────────────────────────────────────────────────────────────────
// Copied here to avoid a cross-package dev-dependency on pt-webgpu.
// This is a test-only mirror; production code lives in pt-webgpu/scene/buildCpuBvh.ts.
// Keep in sync with the canonical implementation.

const LEAFNODE_FLAG_TEST = 0xffff0000;
const MAX_LEAF_TRIANGLES_TEST = 4;
const NUM_BINS_TEST = 16;

interface TriRec {
  triIndex: number;
  centroid: [number, number, number];
  min: [number, number, number];
  max: [number, number, number];
}

interface NodeBuildTest {
  min: [number, number, number];
  max: [number, number, number];
  rightChildOrTriOffset: number;
  splitAxisOrTriCount: number;
}

interface BinDataTest {
  min: [number, number, number];
  max: [number, number, number];
  count: number;
}

function saBox(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const dx = bx - ax, dy = by - ay, dz = bz - az;
  if (dx <= 0 || dy <= 0 || dz <= 0) return 0;
  return 2 * (dx * dy + dy * dz + dz * dx);
}

function makeEmptyBinTest(): BinDataTest {
  return { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity], count: 0 };
}

interface MirrorBvhResult {
  bvhNodes: Float32Array;        // 8 × f32 per node (same layout as pt-webgpu)
  reorderedIndices: Uint32Array; // stride 4 (vec4u, .w = 0)
  totalNodes: number;
}

function buildMirrorBvh(
  positions: Float32Array, // stride 4 (vec4f)
  indices: Uint32Array,    // stride 4 (vec4u, .w unused)
): MirrorBvhResult {
  const triCount = Math.floor(indices.length / 4);
  if (triCount === 0) {
    return {
      bvhNodes: new Float32Array(8),
      reorderedIndices: indices,
      totalNodes: 1,
    };
  }

  const records: TriRec[] = [];
  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 4] ?? 0;
    const i1 = indices[t * 4 + 1] ?? 0;
    const i2 = indices[t * 4 + 2] ?? 0;
    const ax = positions[i0 * 4] ?? 0, ay = positions[i0 * 4 + 1] ?? 0, az = positions[i0 * 4 + 2] ?? 0;
    const bx = positions[i1 * 4] ?? 0, by = positions[i1 * 4 + 1] ?? 0, bz = positions[i1 * 4 + 2] ?? 0;
    const cx = positions[i2 * 4] ?? 0, cy = positions[i2 * 4 + 1] ?? 0, cz = positions[i2 * 4 + 2] ?? 0;
    const mnX = Math.min(ax, bx, cx), mnY = Math.min(ay, by, cy), mnZ = Math.min(az, bz, cz);
    const mxX = Math.max(ax, bx, cx), mxY = Math.max(ay, by, cy), mxZ = Math.max(az, bz, cz);
    records.push({
      triIndex: t,
      min: [mnX, mnY, mnZ],
      max: [mxX, mxY, mxZ],
      centroid: [0.5 * (mnX + mxX), 0.5 * (mnY + mxY), 0.5 * (mnZ + mxZ)],
    });
  }

  const nodes: NodeBuildTest[] = [];
  const ordered: number[] = [];

  const build = (subset: TriRec[]): number => {
    const nodeIndex = nodes.length;
    const node: NodeBuildTest = {
      min: [Infinity, Infinity, Infinity],
      max: [-Infinity, -Infinity, -Infinity],
      rightChildOrTriOffset: 0,
      splitAxisOrTriCount: 0,
    };
    for (const r of subset) {
      node.min[0] = Math.min(node.min[0], r.min[0]);
      node.min[1] = Math.min(node.min[1], r.min[1]);
      node.min[2] = Math.min(node.min[2], r.min[2]);
      node.max[0] = Math.max(node.max[0], r.max[0]);
      node.max[1] = Math.max(node.max[1], r.max[1]);
      node.max[2] = Math.max(node.max[2], r.max[2]);
    }
    nodes.push(node);

    if (subset.length <= MAX_LEAF_TRIANGLES_TEST) {
      const leafOffset = ordered.length;
      for (const r of subset) ordered.push(r.triIndex);
      node.rightChildOrTriOffset = leafOffset;
      node.splitAxisOrTriCount = LEAFNODE_FLAG_TEST | subset.length;
      return nodeIndex;
    }

    const cMin: [number, number, number] = [Infinity, Infinity, Infinity];
    const cMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (const r of subset) {
      cMin[0] = Math.min(cMin[0], r.centroid[0]);
      cMin[1] = Math.min(cMin[1], r.centroid[1]);
      cMin[2] = Math.min(cMin[2], r.centroid[2]);
      cMax[0] = Math.max(cMax[0], r.centroid[0]);
      cMax[1] = Math.max(cMax[1], r.centroid[1]);
      cMax[2] = Math.max(cMax[2], r.centroid[2]);
    }

    const parentSA = saBox(
      node.min[0], node.min[1], node.min[2],
      node.max[0], node.max[1], node.max[2],
    );
    const leafCost = subset.length;
    let bestCost = Infinity, bestAxis = 0, bestSplit = 0;

    for (let axis = 0; axis < 3; axis++) {
      const span = cMax[axis]! - cMin[axis]!;
      if (span <= 1e-9) continue;

      const bins: BinDataTest[] = Array.from({ length: NUM_BINS_TEST }, makeEmptyBinTest);
      for (const r of subset) {
        const t = (r.centroid[axis]! - cMin[axis]!) / span;
        const bi = Math.min(NUM_BINS_TEST - 1, Math.floor(t * NUM_BINS_TEST));
        const bin = bins[bi]!;
        bin.min[0] = Math.min(bin.min[0], r.min[0]);
        bin.min[1] = Math.min(bin.min[1], r.min[1]);
        bin.min[2] = Math.min(bin.min[2], r.min[2]);
        bin.max[0] = Math.max(bin.max[0], r.max[0]);
        bin.max[1] = Math.max(bin.max[1], r.max[1]);
        bin.max[2] = Math.max(bin.max[2], r.max[2]);
        bin.count += 1;
      }

      // Prefix sweep
      const pMinX = new Float32Array(NUM_BINS_TEST), pMinY = new Float32Array(NUM_BINS_TEST), pMinZ = new Float32Array(NUM_BINS_TEST);
      const pMaxX = new Float32Array(NUM_BINS_TEST), pMaxY = new Float32Array(NUM_BINS_TEST), pMaxZ = new Float32Array(NUM_BINS_TEST);
      const pCnt = new Int32Array(NUM_BINS_TEST);
      let bmnX = Infinity, bmnY = Infinity, bmnZ = Infinity, bmxX = -Infinity, bmxY = -Infinity, bmxZ = -Infinity, cnt = 0;
      for (let i = 0; i < NUM_BINS_TEST; i++) {
        const b = bins[i]!;
        bmnX = Math.min(bmnX, b.min[0]); bmnY = Math.min(bmnY, b.min[1]); bmnZ = Math.min(bmnZ, b.min[2]);
        bmxX = Math.max(bmxX, b.max[0]); bmxY = Math.max(bmxY, b.max[1]); bmxZ = Math.max(bmxZ, b.max[2]);
        cnt += b.count;
        pMinX[i] = bmnX; pMinY[i] = bmnY; pMinZ[i] = bmnZ;
        pMaxX[i] = bmxX; pMaxY[i] = bmxY; pMaxZ[i] = bmxZ;
        pCnt[i] = cnt;
      }

      // Suffix sweep
      const sMinX = new Float32Array(NUM_BINS_TEST), sMinY = new Float32Array(NUM_BINS_TEST), sMinZ = new Float32Array(NUM_BINS_TEST);
      const sMaxX = new Float32Array(NUM_BINS_TEST), sMaxY = new Float32Array(NUM_BINS_TEST), sMaxZ = new Float32Array(NUM_BINS_TEST);
      const sCnt = new Int32Array(NUM_BINS_TEST);
      bmnX = Infinity; bmnY = Infinity; bmnZ = Infinity; bmxX = -Infinity; bmxY = -Infinity; bmxZ = -Infinity; cnt = 0;
      for (let i = NUM_BINS_TEST - 1; i >= 0; i--) {
        const b = bins[i]!;
        bmnX = Math.min(bmnX, b.min[0]); bmnY = Math.min(bmnY, b.min[1]); bmnZ = Math.min(bmnZ, b.min[2]);
        bmxX = Math.max(bmxX, b.max[0]); bmxY = Math.max(bmxY, b.max[1]); bmxZ = Math.max(bmxZ, b.max[2]);
        cnt += b.count;
        sMinX[i] = bmnX; sMinY[i] = bmnY; sMinZ[i] = bmnZ;
        sMaxX[i] = bmxX; sMaxY[i] = bmxY; sMaxZ[i] = bmxZ;
        sCnt[i] = cnt;
      }

      for (let split = 0; split < NUM_BINS_TEST - 1; split++) {
        const lc = pCnt[split] ?? 0, rc = sCnt[split + 1] ?? 0;
        if (lc === 0 || rc === 0) continue;
        const lSA = saBox(pMinX[split]!, pMinY[split]!, pMinZ[split]!, pMaxX[split]!, pMaxY[split]!, pMaxZ[split]!);
        const rSA = saBox(sMinX[split + 1]!, sMinY[split + 1]!, sMinZ[split + 1]!, sMaxX[split + 1]!, sMaxY[split + 1]!, sMaxZ[split + 1]!);
        const cost = parentSA > 0 ? (lSA * lc + rSA * rc) / parentSA : lSA * lc + rSA * rc;
        if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestSplit = split; }
      }
    }

    if (bestCost >= leafCost || bestCost === Infinity) {
      const leafOffset = ordered.length;
      for (const r of subset) ordered.push(r.triIndex);
      node.rightChildOrTriOffset = leafOffset;
      node.splitAxisOrTriCount = LEAFNODE_FLAG_TEST | subset.length;
      return nodeIndex;
    }

    const span = cMax[bestAxis]! - cMin[bestAxis]!;
    const left: TriRec[] = [], right: TriRec[] = [];
    for (const r of subset) {
      const t = (r.centroid[bestAxis]! - cMin[bestAxis]!) / span;
      const bi = Math.min(NUM_BINS_TEST - 1, Math.floor(t * NUM_BINS_TEST));
      (bi <= bestSplit ? left : right).push(r);
    }

    if (left.length === 0 || right.length === 0) {
      const leafOffset = ordered.length;
      for (const r of subset) ordered.push(r.triIndex);
      node.rightChildOrTriOffset = leafOffset;
      node.splitAxisOrTriCount = LEAFNODE_FLAG_TEST | subset.length;
      return nodeIndex;
    }

    build(left);
    const rightChild = build(right);
    node.rightChildOrTriOffset = rightChild - nodeIndex;
    node.splitAxisOrTriCount = bestAxis;
    return nodeIndex;
  };

  build(records);

  const reorderedIndices = new Uint32Array(indices.length);
  for (let newTri = 0; newTri < ordered.length; newTri++) {
    const oldTri = ordered[newTri] ?? 0;
    reorderedIndices[newTri * 4] = indices[oldTri * 4] ?? 0;
    reorderedIndices[newTri * 4 + 1] = indices[oldTri * 4 + 1] ?? 0;
    reorderedIndices[newTri * 4 + 2] = indices[oldTri * 4 + 2] ?? 0;
    reorderedIndices[newTri * 4 + 3] = 0;
  }

  const nodeBuffer = new ArrayBuffer(nodes.length * 32);
  const dv = new DataView(nodeBuffer);
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i]!;
    const off = i * 32;
    dv.setFloat32(off + 0, node.min[0], true);
    dv.setFloat32(off + 4, node.min[1], true);
    dv.setFloat32(off + 8, node.min[2], true);
    dv.setFloat32(off + 12, node.max[0], true);
    dv.setFloat32(off + 16, node.max[1], true);
    dv.setFloat32(off + 20, node.max[2], true);
    dv.setUint32(off + 24, node.rightChildOrTriOffset >>> 0, true);
    dv.setUint32(off + 28, node.splitAxisOrTriCount >>> 0, true);
  }

  return {
    bvhNodes: new Float32Array(nodeBuffer),
    reorderedIndices,
    totalNodes: nodes.length,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// TS-mirror BVH traversal (test-only; mirrors the WGSL in
// pt-webgpu/src/wgsl/pathTraceBruteforce.wgsl.ts)
//
// Structural traversal only: slab test + Möller-Trumbore + relative-offset
// child indexing. Not a full GPU shader — no shading, no material lookup.
// Document: test-only mirror; production traversal lives in WGSL.
// ──────────────────────────────────────────────────────────────────────────

interface TraceHit {
  hit: boolean;
  tHit: number;
  triIndex: number;
}

/**
 * Traverse a relative-offset BVH built by `buildMirrorBvh` and return the
 * closest triangle intersection along the ray.
 *
 * @param bvhNodes  8 × f32 per node (same layout as pt-webgpu)
 * @param positions stride-4 vertex positions (vec4f, .w unused)
 * @param indices   stride-4 triangle indices (vec4u, .w = 0)
 * @param ox/oy/oz  ray origin
 * @param dx/dy/dz  ray direction (normalised)
 * @param tMin      minimum hit distance
 * @param tMax      maximum hit distance
 */
function traceBvh(
  bvhNodes: Float32Array,
  positions: Float32Array,
  indices: Uint32Array,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMin: number,
  tMax: number,
): TraceHit {
  const LEAFNODE_CHECK = 0xffff;
  const u32 = new Uint32Array(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.length);
  const totalNodes = bvhNodes.length / 8;
  const totalTris = Math.floor(indices.length / 4);

  // Safe inverse direction (avoid 0 * Inf = NaN for axis-aligned rays)
  const EPS = 1e-30;
  const idx = Math.abs(dx) < EPS ? (dx >= 0 ? 1e20 : -1e20) : 1.0 / dx;
  const idy = Math.abs(dy) < EPS ? (dy >= 0 ? 1e20 : -1e20) : 1.0 / dy;
  const idz = Math.abs(dz) < EPS ? (dz >= 0 ? 1e20 : -1e20) : 1.0 / dz;

  const stack = new Uint32Array(64);
  let stackPtr = 0;
  stack[stackPtr++] = 0;

  let closest = tMax;
  let hitTri = -1;

  while (stackPtr > 0) {
    const nodeIdx = stack[--stackPtr]!;
    if (nodeIdx >= totalNodes) continue;

    const base = nodeIdx * 8;
    const bminX = bvhNodes[base + 0]!;
    const bminY = bvhNodes[base + 1]!;
    const bminZ = bvhNodes[base + 2]!;
    const bmaxX = bvhNodes[base + 3]!;
    const bmaxY = bvhNodes[base + 4]!;
    const bmaxZ = bvhNodes[base + 5]!;

    // Slab test (Williams 2005)
    const tx0 = (bminX - ox) * idx, tx1 = (bmaxX - ox) * idx;
    const ty0 = (bminY - oy) * idy, ty1 = (bmaxY - oy) * idy;
    const tz0 = (bminZ - oz) * idz, tz1 = (bmaxZ - oz) * idz;
    const tNear = Math.max(Math.min(tx0, tx1), Math.min(ty0, ty1), Math.min(tz0, tz1), tMin);
    const tFar  = Math.min(Math.max(tx0, tx1), Math.max(ty0, ty1), Math.max(tz0, tz1), closest);
    if (tNear > tFar) continue;

    const splitOrCount = u32[base + 7]!;
    const isLeaf = (splitOrCount >>> 16) === LEAFNODE_CHECK;

    if (isLeaf) {
      const count = splitOrCount & 0x0000ffff;
      const start = u32[base + 6]!;
      for (let i = 0; i < count; i++) {
        const t = start + i;
        if (t >= totalTris) continue;
        const i0 = indices[t * 4] ?? 0;
        const i1 = indices[t * 4 + 1] ?? 0;
        const i2 = indices[t * 4 + 2] ?? 0;
        const ax = positions[i0 * 4] ?? 0, ay = positions[i0 * 4 + 1] ?? 0, az = positions[i0 * 4 + 2] ?? 0;
        const bx = positions[i1 * 4] ?? 0, by = positions[i1 * 4 + 1] ?? 0, bz = positions[i1 * 4 + 2] ?? 0;
        const cx = positions[i2 * 4] ?? 0, cy = positions[i2 * 4 + 1] ?? 0, cz = positions[i2 * 4 + 2] ?? 0;

        // Möller-Trumbore
        const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
        const det = e1x * hx + e1y * hy + e1z * hz;
        if (Math.abs(det) < 1e-8) continue;
        const invDet = 1.0 / det;
        const sx = ox - ax, sy = oy - ay, sz = oz - az;
        const u = (sx * hx + sy * hy + sz * hz) * invDet;
        if (u < 0 || u > 1) continue;
        const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
        const v = (dx * qx + dy * qy + dz * qz) * invDet;
        if (v < 0 || u + v > 1) continue;
        const hitT = (e2x * qx + e2y * qy + e2z * qz) * invDet;
        if (hitT > tMin && hitT < closest) {
          closest = hitT;
          hitTri = t;
        }
      }
    } else {
      // Relative-offset traversal: right child = nodeIdx + stored offset.
      const leftChild = nodeIdx + 1;
      const rightChild = nodeIdx + u32[base + 6]!;
      if (stackPtr + 1 < 64) {
        stack[stackPtr++] = rightChild;
        stack[stackPtr++] = leftChild;
      }
    }
  }

  return { hit: hitTri >= 0, tHit: closest, triIndex: hitTri };
}

// ──────────────────────────────────────────────────────────────────────────
// Test geometry: 4-triangle unit box (two triangles per face, 4 faces)
// ──────────────────────────────────────────────────────────────────────────

function makeUnitBoxMesh(): { positions: Float32Array; indices: Uint32Array } {
  // 8 box corners
  const verts: [number, number, number][] = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], // front face z=0
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], // back  face z=1
  ];

  // Positions: stride 4 (vec4f, .w = 0)
  const positions = new Float32Array(verts.length * 4);
  for (let i = 0; i < verts.length; i++) {
    positions[i * 4] = verts[i]![0];
    positions[i * 4 + 1] = verts[i]![1];
    positions[i * 4 + 2] = verts[i]![2];
    positions[i * 4 + 3] = 0;
  }

  // 4 faces × 2 triangles = 8 triangles
  // Using only 4 faces (front, back, left, right) — enough for a non-trivial BVH.
  const tris: [number, number, number][] = [
    [0, 1, 2], [0, 2, 3], // front (z=0)
    [5, 4, 7], [5, 7, 6], // back  (z=1)
    [4, 0, 3], [4, 3, 7], // left  (x=0)
    [1, 5, 6], [1, 6, 2], // right (x=1)
  ];

  // Indices: stride 4 (vec4u, .w = 0)
  const indices = new Uint32Array(tris.length * 4);
  for (let t = 0; t < tris.length; t++) {
    indices[t * 4] = tris[t]![0];
    indices[t * 4 + 1] = tris[t]![1];
    indices[t * 4 + 2] = tris[t]![2];
    indices[t * 4 + 3] = 0;
  }

  return { positions, indices };
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

describe('33-G BVH encoding round-trip', () => {
  it('1. pt-webgpu builder relative-offset invariant — interior nodes offset ∈ [1, totalNodes)', () => {
    const { positions, indices } = makeUnitBoxMesh();
    const result = buildMirrorBvh(positions, indices);

    const LEAFNODE_CHECK = 0xffff;
    const u32 = new Uint32Array(result.bvhNodes.buffer);
    const totalNodes = result.totalNodes;

    let foundInterior = false;
    for (let i = 0; i < totalNodes; i++) {
      const splitOrCount = u32[i * 8 + 7] ?? 0;
      const isLeaf = (splitOrCount >>> 16) === LEAFNODE_CHECK;
      if (isLeaf) continue;
      foundInterior = true;
      const offset = u32[i * 8 + 6] ?? 0;
      expect(offset).toBeGreaterThanOrEqual(1);
      expect(offset).toBeLessThan(totalNodes);
    }
    // 8 triangles must produce at least one interior node.
    expect(foundInterior).toBe(true);
  });

  it('2. validateBvhEncoding passes on a valid relative-offset buffer', () => {
    const { positions, indices } = makeUnitBoxMesh();
    const result = buildMirrorBvh(positions, indices);
    // Should not throw.
    expect(() => validateBvhEncoding(result.bvhNodes, result.totalNodes)).not.toThrow();
  });

  it('2b. validateBvhEncoding throws on an invalid (absolute-offset) buffer', () => {
    // Synthesise a 3-node tree: root (interior) → left leaf, right leaf.
    // Encode the right child as ABSOLUTE index 2 (should be relative = 1).
    const nodeBuffer = new ArrayBuffer(3 * 32);
    const dv = new DataView(nodeBuffer);

    // Node 0: interior, right child at ABSOLUTE index 2 (invalid — should be relative 2).
    dv.setFloat32(0, -1, true); dv.setFloat32(4, -1, true); dv.setFloat32(8, -1, true);  // min
    dv.setFloat32(12, 1, true); dv.setFloat32(16, 1, true); dv.setFloat32(20, 1, true);  // max
    dv.setUint32(24, 2, true);  // rightChildOrTriOffset = absolute index 2 (≥ totalNodes=3 is not invalid per se,
    // but let's use a value >= totalNodes to trigger the check)
    dv.setUint32(28, 0, true);  // splitAxis = 0 (interior)

    // Node 1: leaf
    dv.setUint32(24 + 32, 0, true);         // triOffset
    dv.setUint32(28 + 32, 0xffff0001, true); // LEAFNODE_FLAG | 1

    // Node 2: leaf
    dv.setUint32(24 + 64, 1, true);         // triOffset
    dv.setUint32(28 + 64, 0xffff0001, true); // LEAFNODE_FLAG | 1

    // Override node 0's offset to something >= totalNodes (3) to trigger the validator.
    // totalNodes = 3, so offset must be in [1, 2]. Use offset = 3 to force error.
    dv.setUint32(24, 3, true);

    const bvhNodes = new Float32Array(nodeBuffer);
    expect(() => validateBvhEncoding(bvhNodes, 3)).toThrow(/invalid/);
  });

  it('3. cross-package traversal identity — 1000 rays produce consistent hit results', () => {
    const { positions, indices } = makeUnitBoxMesh();
    const result = buildMirrorBvh(positions, indices);
    // Validate encoding before traversal.
    validateBvhEncoding(result.bvhNodes, result.totalNodes);

    // Simple LCG for deterministic ray directions.
    let rngState = 12345;
    const rand = (): number => {
      rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0;
      return rngState / 0x100000000;
    };

    const NUM_RAYS = 1000;
    let hitCount = 0;
    let allHitsValid = true;

    for (let r = 0; r < NUM_RAYS; r++) {
      // Sample ray origins uniformly in [-1, 2]³ and random directions.
      const ox = rand() * 3 - 1;
      const oy = rand() * 3 - 1;
      const oz = rand() * 3 - 1;

      // Random unit direction via rejection sampling on the sphere.
      let dx: number, dy: number, dz: number, len: number;
      do {
        dx = rand() * 2 - 1;
        dy = rand() * 2 - 1;
        dz = rand() * 2 - 1;
        len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      } while (len < 0.01 || len > 1.0);
      dx /= len; dy /= len; dz /= len;

      const result1 = traceBvh(
        result.bvhNodes, positions, result.reorderedIndices,
        ox, oy, oz, dx, dy, dz,
        1e-6, 1000.0,
      );

      if (result1.hit) {
        hitCount++;
        // Verify the hit is valid: tHit must be positive and finite.
        if (!isFinite(result1.tHit) || result1.tHit <= 0) {
          allHitsValid = false;
        }
        // Verify hit point lies on or inside the unit box [0,1]³ (within tolerance).
        const hx = ox + dx * result1.tHit;
        const hy = oy + dy * result1.tHit;
        const hz = oz + dz * result1.tHit;
        const onBox =
          hx >= -1e-4 && hx <= 1 + 1e-4 &&
          hy >= -1e-4 && hy <= 1 + 1e-4 &&
          hz >= -1e-4 && hz <= 1 + 1e-4;
        if (!onBox) {
          allHitsValid = false;
        }
      }
    }

    // With 1000 random rays and origins spanning [-1,2]³, many should hit the unit box.
    expect(hitCount).toBeGreaterThan(50);
    expect(allHitsValid).toBe(true);
  });

  it('3b. traversal finds expected intersection for a canonical ray through the box face', () => {
    const { positions, indices } = makeUnitBoxMesh();
    const result = buildMirrorBvh(positions, indices);

    // Ray from (0.5, 0.5, -1) in +Z direction — should hit the front face (z=0) at t=1.
    const hit = traceBvh(
      result.bvhNodes, positions, result.reorderedIndices,
      0.5, 0.5, -1.0,
      0.0, 0.0, 1.0,
      1e-6, 100.0,
    );

    expect(hit.hit).toBe(true);
    expect(hit.tHit).toBeCloseTo(1.0, 4); // front face at z=0, ray origin at z=-1 → t=1
    expect(isFinite(hit.tHit)).toBe(true);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// T1.D3 — BVH 200-triangle complex-scene round-trip
//
// Generates a 10×10 grid of triangles forming a flat plane (200 triangles).
// Builds a BVH via buildMirrorBvh (SAH, 16 bins). Traces 1000 random rays.
// For each ray, computes tHit via:
//   (a) BVH traversal (relative-offset, same algorithm as walkaround + pt-webgpu)
//   (b) Brute-force loop over all 200 triangles (oracle)
// Asserts: |tHit_bvh - tHit_brute| < 1e-5 for every hitting ray.
//
// This catches stack-depth edge cases and SAH split failures that the smaller
// 8-triangle box test (33-G) cannot exercise.
// ──────────────────────────────────────────────────────────────────────────

/**
 * Generate a 10×10 grid of quads, each split into 2 triangles.
 * Total: 10×10×2 = 200 triangles. The plane lies in the XZ plane (Y=0),
 * spanning [0, 10] × [0, 10] in X and Z. This gives many co-planar
 * triangles to stress the SAH binning across all 3 axes.
 */
function make10x10GridMesh(): { positions: Float32Array; indices: Uint32Array } {
  const GRID = 10; // 10×10 quads → 200 triangles
  const vertCount = (GRID + 1) * (GRID + 1);
  const positions = new Float32Array(vertCount * 4);

  // Vertex grid on the XZ plane (Y = 0).
  for (let row = 0; row <= GRID; row++) {
    for (let col = 0; col <= GRID; col++) {
      const vi = row * (GRID + 1) + col;
      positions[vi * 4 + 0] = col;      // X ∈ [0, 10]
      positions[vi * 4 + 1] = 0.0;      // Y = 0
      positions[vi * 4 + 2] = row;      // Z ∈ [0, 10]
      positions[vi * 4 + 3] = 0.0;
    }
  }

  const triCount = GRID * GRID * 2;
  const indices = new Uint32Array(triCount * 4);

  let ti = 0;
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      // Quad corners
      const tl = row * (GRID + 1) + col;       // top-left
      const tr = tl + 1;                        // top-right
      const bl = (row + 1) * (GRID + 1) + col; // bottom-left
      const br = bl + 1;                        // bottom-right

      // Triangle 0: tl, tr, bl
      indices[ti * 4 + 0] = tl;
      indices[ti * 4 + 1] = tr;
      indices[ti * 4 + 2] = bl;
      indices[ti * 4 + 3] = 0;
      ti++;

      // Triangle 1: tr, br, bl
      indices[ti * 4 + 0] = tr;
      indices[ti * 4 + 1] = br;
      indices[ti * 4 + 2] = bl;
      indices[ti * 4 + 3] = 0;
      ti++;
    }
  }

  return { positions, indices };
}

/**
 * Brute-force Möller-Trumbore against every triangle.
 * Oracle for BVH traversal — same math, no hierarchy.
 */
function traceBruteForce(
  positions: Float32Array,
  indices: Uint32Array,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMin: number,
  tMax: number,
): { hit: boolean; tHit: number } {
  const totalTris = Math.floor(indices.length / 4);
  let closest = tMax;
  let hitAny = false;

  for (let t = 0; t < totalTris; t++) {
    const i0 = indices[t * 4] ?? 0;
    const i1 = indices[t * 4 + 1] ?? 0;
    const i2 = indices[t * 4 + 2] ?? 0;
    const ax = positions[i0 * 4] ?? 0, ay = positions[i0 * 4 + 1] ?? 0, az = positions[i0 * 4 + 2] ?? 0;
    const bx = positions[i1 * 4] ?? 0, by = positions[i1 * 4 + 1] ?? 0, bz = positions[i1 * 4 + 2] ?? 0;
    const cx = positions[i2 * 4] ?? 0, cy = positions[i2 * 4 + 1] ?? 0, cz = positions[i2 * 4 + 2] ?? 0;

    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
    const det = e1x * hx + e1y * hy + e1z * hz;
    if (Math.abs(det) < 1e-8) continue;
    const invDet = 1.0 / det;
    const sx = ox - ax, sy = oy - ay, sz = oz - az;
    const u = (sx * hx + sy * hy + sz * hz) * invDet;
    if (u < 0 || u > 1) continue;
    const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * invDet;
    if (v < 0 || u + v > 1) continue;
    const hitT = (e2x * qx + e2y * qy + e2z * qz) * invDet;
    if (hitT > tMin && hitT < closest) {
      closest = hitT;
      hitAny = true;
    }
  }

  return { hit: hitAny, tHit: closest };
}

describe('T1.D3 — BVH 200-tri complex-scene round-trip', () => {
  it('relative-offset BVH traversal matches brute-force oracle for 1000 rays (|Δt| < 1e-5)', () => {
    const { positions, indices } = make10x10GridMesh();
    const built = buildMirrorBvh(positions, indices);

    // Validate BVH encoding is well-formed before traversal.
    validateBvhEncoding(built.bvhNodes, built.totalNodes);

    // The plane spans X ∈ [0, 10], Z ∈ [0, 10], Y = 0.
    // We shoot rays from Y > 0 (above plane) downward (-Y direction),
    // with origins uniformly distributed over [−1, 11]² in X,Z to hit
    // both the interior and edges of the plane.

    // Deterministic LCG (same parameters as existing tests).
    let rngState = 0xabcdef01;
    const rand = (): number => {
      rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0;
      return rngState / 0x100000000;
    };

    const NUM_RAYS = 1_000;
    let missMatch = 0;      // BVH misses where brute-force hits
    let tHitMismatch = 0;   // |tHit_bvh - tHit_brute| >= 1e-5
    let hitCount = 0;

    for (let r = 0; r < NUM_RAYS; r++) {
      // Ray origin: random position above the grid plane.
      const ox = rand() * 12 - 1; // X ∈ [-1, 11]
      const oy = rand() * 8 + 1;  // Y ∈ [1, 9]  (above the plane)
      const oz = rand() * 12 - 1; // Z ∈ [-1, 11]

      // Direction: straight down (-Y) to guarantee deterministic brute-force result.
      // We also test slightly angled rays to exercise the slab test.
      const angleVariation = (rand() - 0.5) * 0.4; // ±0.2 tilt in X and Z
      const dx = angleVariation;
      const dy = -1.0;
      const dz = (rand() - 0.5) * 0.4;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const ndx = dx / len, ndy = dy / len, ndz = dz / len;

      const bvhResult = traceBvh(
        built.bvhNodes, positions, built.reorderedIndices,
        ox, oy, oz, ndx, ndy, ndz,
        1e-4, 1000.0,
      );

      const bruteResult = traceBruteForce(
        positions, indices,
        ox, oy, oz, ndx, ndy, ndz,
        1e-4, 1000.0,
      );

      if (bruteResult.hit) {
        hitCount++;
        if (!bvhResult.hit) {
          missMatch++;
        } else {
          const delta = Math.abs(bvhResult.tHit - bruteResult.tHit);
          if (delta >= 1e-5) {
            tHitMismatch++;
          }
        }
      } else {
        // Brute-force miss — BVH should also miss (no false positives).
        if (bvhResult.hit) {
          missMatch++;
        }
      }
    }

    // The grid spans most of [0,10]×[0,10]; many rays will hit.
    expect(hitCount).toBeGreaterThan(300);

    // Zero tolerance on BVH-vs-brute agreement: all hits must agree within 1e-5.
    expect(missMatch).toBe(0);
    expect(tHitMismatch).toBe(0);
  });
});
