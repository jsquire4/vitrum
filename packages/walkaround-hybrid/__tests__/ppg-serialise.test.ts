/**
 * W9 — PPG tree serialisation tests.
 *
 * Cover the CPU `serialiseDTree` + `serialiseSTree` producers and their
 * GPU-equivalent traversal helpers. These are the load-bearing oracle
 * tests: any divergence between the CPU `findDTreeLeaf` and the flat-buffer
 * traversal (which the WGSL kernel mirrors literally) would manifest as
 * wrong guide samples on the GPU.
 *
 * Reference: Müller et al. 2017 §3.1 (sTree), §3.2 (dTree).
 */

import { describe, it, expect } from 'vitest';
import {
  serialiseDTree,
  serialiseSTree,
  gpuTraverseDTreeLeaf,
  gpuTraverseSTreeLeaf,
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  STREE_HEADER_F32,
  STREE_NODE_F32,
} from '../src/ppg/serialise.js';
import {
  buildEmptyDTree,
  findDTreeLeaf,
  refineDTree,
} from '../src/ppg/dTree.js';
import {
  buildSTree,
  findSTreeLeaf,
  sTreeAccumulate,
  splitOverflowLeaves,
} from '../src/ppg/sTree.js';
import {
  PPG_CELL_SPLIT_THRESHOLD,
  PPG_DTREE_INITIAL_DEPTH,
} from '../src/ppg/ppgConstants.js';
import type { DTree } from '../src/ppg/types.js';

const SCENE_AABB: import('../src/ppg/types.js').AABB = {
  min: [-10, -10, -10],
  max: [10, 10, 10],
};

// ── Test 1: serialiseDTree shape + header ───────────────────────────────────

describe('serialiseDTree — buffer shape and header', () => {
  it('produces a buffer of the documented size', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    const buf = serialiseDTree(dTree);
    const expected = DTREE_HEADER_F32 + dTree.nodes.length * DTREE_NODE_F32;
    expect(buf.length).toBe(expected);
  });

  it('header carries nodeCount, leafCount, totalFlux', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    // Set some flux so totalFlux is non-zero.
    dTree.totalFlux = 7.5;
    const buf = serialiseDTree(dTree);
    expect(buf[0]).toBe(dTree.nodes.length);
    expect(buf[2]).toBeCloseTo(7.5, 6);
    const leafCount = dTree.nodes.filter((n) => n.isLeaf).length;
    expect(buf[1]).toBe(leafCount);
  });

  it('every leaf packs solidAngle and flux correctly', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    for (let i = 0; i < dTree.nodes.length; i++) {
      const n = dTree.nodes[i]!;
      n.flux = i * 0.5;
    }
    dTree.totalFlux = 999;
    const buf = serialiseDTree(dTree);
    for (let i = 0; i < dTree.nodes.length; i++) {
      const n = dTree.nodes[i]!;
      const base = DTREE_HEADER_F32 + i * DTREE_NODE_F32;
      expect(buf[base + 0]).toBeCloseTo(n.u0, 6);
      expect(buf[base + 1]).toBeCloseTo(n.v0, 6);
      expect(buf[base + 2]).toBeCloseTo(n.u1, 6);
      expect(buf[base + 3]).toBeCloseTo(n.v1, 6);
      expect(buf[base + 4]).toBeCloseTo(n.flux, 6);
      expect(buf[base + 5]).toBeCloseTo(n.solidAngle, 6);
      // firstChild
      expect(buf[base + 6]).toBe(n.firstChild);
      // isLeaf flag
      expect(buf[base + 7]).toBe(n.isLeaf ? 1.0 : 0.0);
    }
  });
});

// ── Test 2: round-trip CPU findDTreeLeaf ↔ GPU traversal ────────────────────

describe('serialiseDTree — CPU/GPU traversal oracle', () => {
  function leafCentre(n: import('../src/ppg/types.js').DTreeNode): [number, number] {
    return [(n.u0 + n.u1) * 0.5, (n.v0 + n.v1) * 0.5];
  }

  it('CPU findDTreeLeaf and GPU traversal return the same leaf for every leaf-centre query', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    const buf = serialiseDTree(dTree);
    for (let i = 0; i < dTree.nodes.length; i++) {
      const n = dTree.nodes[i]!;
      if (!n.isLeaf) continue;
      const uv = leafCentre(n);
      const cpuLeafIdx = findDTreeLeaf(dTree, uv);
      const gpuLeafBase = gpuTraverseDTreeLeaf(buf, uv);
      const cpuBase = DTREE_HEADER_F32 + cpuLeafIdx * DTREE_NODE_F32;
      expect(gpuLeafBase).toBe(cpuBase);
      // Verify the leaf returned actually contains the query UV.
      expect(buf[gpuLeafBase + 0]!).toBeLessThanOrEqual(uv[0]);
      expect(buf[gpuLeafBase + 2]!).toBeGreaterThanOrEqual(uv[0]);
      expect(buf[gpuLeafBase + 1]!).toBeLessThanOrEqual(uv[1]);
      expect(buf[gpuLeafBase + 3]!).toBeGreaterThanOrEqual(uv[1]);
    }
  });

  it('CPU/GPU agree after adaptive refinement (deeper tree)', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    // Spike a leaf and refine to force splits.
    const leaves = dTree.nodes.filter((n) => n.isLeaf);
    leaves[0]!.flux = 100;
    for (let i = 1; i < leaves.length; i++) leaves[i]!.flux = 0.0001;
    dTree.totalFlux = 100 + 0.0001 * (leaves.length - 1);
    refineDTree(dTree);

    const buf = serialiseDTree(dTree);
    // Sample 81 random UVs deterministically across [0,1]² and require parity.
    let state = 0x12345;
    function rng(): number {
      state = (state * 1664525 + 1013904223) | 0;
      return ((state >>> 0) % 1024) / 1024;
    }
    for (let q = 0; q < 81; q++) {
      const u = rng();
      const v = rng();
      const cpuIdx = findDTreeLeaf(dTree, [u, v]);
      const gpuBase = gpuTraverseDTreeLeaf(buf, [u, v]);
      const cpuBase = DTREE_HEADER_F32 + cpuIdx * DTREE_NODE_F32;
      expect(gpuBase).toBe(cpuBase);
    }
  });

  it('GPU traversal returns a leaf (flag == 1.0) for any in-bounds UV', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    const buf = serialiseDTree(dTree);
    for (const uv of [[0.0, 0.0], [0.99, 0.99], [0.5, 0.5], [0.25, 0.75]] as const) {
      const base = gpuTraverseDTreeLeaf(buf, uv);
      expect(buf[base + 7]).toBe(1.0); // isLeaf flag
    }
  });
});

// ── Test 3: serialiseSTree shape, header, and offsets ───────────────────────

describe('serialiseSTree — buffer shape and offset table', () => {
  it('returns sTreeBuf, dTreeBuf, and dTreeOffsets of consistent sizes', () => {
    const sTree = buildSTree(SCENE_AABB);
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(sTree);
    expect(sTreeBuf.length).toBe(STREE_HEADER_F32 + sTree.nodes.length * STREE_NODE_F32);
    expect(dTreeOffsets.length).toBe(sTree.dTrees.length);
    // First (and only) dTree starts at offset 0.
    expect(dTreeOffsets[0]).toBe(0);
    // dTreeBuf is exactly the size of the single root dTree.
    const expectedDt = DTREE_HEADER_F32 + sTree.dTrees[0]!.nodes.length * DTREE_NODE_F32;
    expect(dTreeBuf.length).toBeGreaterThanOrEqual(expectedDt);
  });

  it('per-cell dTree blocks are placed at the documented offsets after a split', () => {
    const sTree = buildSTree(SCENE_AABB);
    // Force a split: accumulate enough samples then split.
    for (let i = 0; i < PPG_CELL_SPLIT_THRESHOLD + 1; i++) {
      sTreeAccumulate(sTree, [0, 0, 0], [0.5, 0.5], 1.0);
    }
    splitOverflowLeaves(sTree);
    expect(sTree.dTrees.length).toBeGreaterThan(1);

    const { dTreeBuf, dTreeOffsets } = serialiseSTree(sTree);
    // For each offset, verify the header at that position matches the dTree node count.
    for (let k = 0; k < sTree.dTrees.length; k++) {
      const off = dTreeOffsets[k]!;
      expect(dTreeBuf[off]).toBe(sTree.dTrees[k]!.nodes.length);
    }
  });
});

// ── Test 4: sTree CPU/GPU traversal oracle ──────────────────────────────────

describe('serialiseSTree — CPU/GPU traversal oracle', () => {
  it('a fresh single-cell sTree returns the root for any in-bounds query', () => {
    const sTree = buildSTree(SCENE_AABB);
    const { sTreeBuf } = serialiseSTree(sTree);
    const positions: [number, number, number][] = [[0, 0, 0], [-9, -9, -9], [9, 9, 9]];
    for (const pos of positions) {
      const cpuIdx = findSTreeLeaf(sTree, pos);
      const gpuBase = gpuTraverseSTreeLeaf(sTreeBuf, pos);
      expect(cpuIdx).toBe(0);
      expect(gpuBase).toBe(STREE_HEADER_F32);
    }
  });

  it('after split, CPU/GPU agree on which leaf contains a query position', () => {
    const sTree = buildSTree(SCENE_AABB);
    for (let i = 0; i < PPG_CELL_SPLIT_THRESHOLD + 1; i++) {
      sTreeAccumulate(sTree, [0, 0, 0], [0.5, 0.5], 1.0);
    }
    splitOverflowLeaves(sTree);
    const { sTreeBuf } = serialiseSTree(sTree);
    // Query a handful of positions and require parity.
    const probes: Array<[number, number, number]> = [
      [-5, -5, -5], [5, 5, 5], [0, -7, 3], [-2, 4, -6], [7, -1, 2],
    ];
    for (const pos of probes) {
      const cpuIdx = findSTreeLeaf(sTree, pos);
      const gpuBase = gpuTraverseSTreeLeaf(sTreeBuf, pos);
      const cpuBase = STREE_HEADER_F32 + cpuIdx * STREE_NODE_F32;
      expect(gpuBase).toBe(cpuBase);
    }
  });
});

// ── Test 5: integer-encoded fields round-trip through f32 losslessly ────────

describe('serialise — integer round-trip through f32', () => {
  it('firstChild indices up to 2^20 round-trip exactly', () => {
    // The dTree currently caps at 4^4 = 256 nodes (MAX_DEPTH=4), so 2^20 is
    // a generous bound. This test pins the lossless-encoding invariant for
    // any future depth bump up to ~2^24.
    const buf = new Float32Array(DTREE_HEADER_F32 + DTREE_NODE_F32);
    for (const v of [0, 1, 100, 1024, 4095, 1 << 16, 1 << 20]) {
      buf[DTREE_HEADER_F32 + 6] = v;
      const decoded = buf[DTREE_HEADER_F32 + 6]! | 0;
      expect(decoded).toBe(v);
    }
  });

  it('serialise + deserialise of leaf flux is numerically stable across small/large values', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    const sample = dTree.nodes.find((n) => n.isLeaf)!;
    for (const v of [0, 1e-6, 1e-3, 1, 1e3, 1e6]) {
      sample.flux = v;
      const buf = serialiseDTree(dTree);
      const leafIdx = dTree.nodes.indexOf(sample);
      const base = DTREE_HEADER_F32 + leafIdx * DTREE_NODE_F32;
      // f32 has ~7 decimal digits; tolerate 1e-5 relative for the big values.
      const eps = Math.max(1e-6, Math.abs(v) * 1e-5);
      expect(Math.abs(buf[base + 4]! - v)).toBeLessThan(eps);
    }
  });
});

// ── Test 6: Item A — host-config overflow clamp (maxNodes / maxDTreeNodesPerCell)
//
// A host that allocates the PPG GPU buffers with `maxDTreeNodesPerCell < 341`
// (below the depth-4 quadtree's 341-node cap) would, WITHOUT the clamp, have
// `serialiseSTree` emit a `dTreeBuf` LARGER than the GPU allocation — the
// `writeBuffer` upload throws / truncates. The clamp serialises only the first
// `maxNodes` nodes AND promotes any served interior node whose four children
// fall outside `[0, maxNodes)` to a leaf, so the truncated buffer (a) fits the
// allocation and (b) leaves NO dangling firstChild pointer that the WGSL
// block-contiguous traversal could follow into the next cell's data.
//
// The clamp must match the GPU UPDATE path, which bounds the per-cell node
// range identically via `nodeLimit = min(dTree.nodes.length, maxDTreeNodesPerCell)`
// (`PPGCoordinator._mergeFluxAndRefine`).

/** Grow a dTree to its full depth-4 cap (341 nodes) by spiking every leaf and
 *  refining until it stops growing. Mirrors the probe used to verify the
 *  341-node ceiling. */
function buildFullDepthDTree(): DTree {
  const d = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
  for (let pass = 0; pass < 6; pass++) {
    const before = d.nodes.length;
    for (const n of d.nodes) if (n.isLeaf) n.flux = 1.0;
    d.totalFlux = d.nodes.filter((n) => n.isLeaf).length;
    refineDTree(d);
    if (d.nodes.length === before) break; // converged at the depth cap
  }
  return d;
}

describe('serialiseDTree — Item A overflow clamp (maxNodes)', () => {
  it('a full depth-4 dTree has exactly 341 nodes (the default cap — never clamps)', () => {
    const d = buildFullDepthDTree();
    expect(d.nodes.length).toBe(341);
    // Default config: omitting maxNodes (or cap ≥ 341) is the historical path.
    expect(serialiseDTree(d).length).toBe(DTREE_HEADER_F32 + 341 * DTREE_NODE_F32);
    expect(serialiseDTree(d, 341).length).toBe(DTREE_HEADER_F32 + 341 * DTREE_NODE_F32);
    // cap ≥ node count is a no-op clamp — identical bytes to the unclamped buffer.
    expect(Array.from(serialiseDTree(d, 1000))).toEqual(Array.from(serialiseDTree(d)));
  });

  it('cap < node count: buffer fits a GPU allocation sized for exactly `cap` nodes', () => {
    const d = buildFullDepthDTree(); // 341 nodes
    for (const cap of [1, 5, 21, 85, 200, 340]) {
      const buf = serialiseDTree(d, cap);
      // The GPU dTree slot for a cell is `header + cap × stride` f32. The
      // serialised buffer must be EXACTLY that — never larger (no overflow).
      const expectedF32 = DTREE_HEADER_F32 + cap * DTREE_NODE_F32;
      expect(buf.length).toBe(expectedF32);
      // Header nodeCount reflects the served (clamped) count, matching the GPU
      // update-path bound `min(nodes.length, cap)`.
      expect(buf[0]).toBe(Math.min(d.nodes.length, cap));
    }
  });

  it('clamp matches the GPU update-path node bound exactly', () => {
    const d = buildFullDepthDTree();
    // The PPGCoordinator update path clamps per-cell to
    // `nodeLimit = min(dTree.nodes.length, maxDTreeNodesPerCell)`. The served
    // node count MUST equal that bound for every cap.
    for (const cap of [1, 7, 50, 85, 341, 9999]) {
      const buf = serialiseDTree(d, cap);
      const gpuNodeLimit = Math.min(d.nodes.length, cap);
      expect(buf[0]).toBe(gpuNodeLimit);
    }
  });

  it('NO served interior node points to a child outside the served prefix (no cross-cell read)', () => {
    const d = buildFullDepthDTree();
    for (const cap of [1, 5, 21, 85, 200]) {
      const buf = serialiseDTree(d, cap);
      const N = buf[0]!;
      for (let i = 0; i < N; i++) {
        const base = DTREE_HEADER_F32 + i * DTREE_NODE_F32;
        const isLeaf = buf[base + 7]! > 0.5;
        const firstChild = buf[base + 6]!;
        if (isLeaf) {
          // Promoted-or-genuine leaf: firstChild cleared to −1.
          expect(firstChild).toBe(-1);
        } else {
          // Interior: all four children MUST fall inside [0, N) — otherwise the
          // WGSL `firstChild + ci` jump would read past this cell's block.
          expect(firstChild).toBeGreaterThanOrEqual(0);
          expect(firstChild + 3).toBeLessThan(N);
        }
      }
    }
  });

  it('GPU traversal on a clamped buffer still terminates at a valid leaf for any UV', () => {
    const d = buildFullDepthDTree();
    const buf = serialiseDTree(d, 21); // aggressively clamped
    for (const uv of [[0, 0], [0.99, 0.99], [0.5, 0.5], [0.25, 0.75], [0.1, 0.9]] as const) {
      const base = gpuTraverseDTreeLeaf(buf, uv);
      // Must land on a leaf flag (1.0) inside the served region — proves the
      // descent never followed a dangling pointer.
      expect(buf[base + 7]).toBe(1.0);
      expect(base).toBeGreaterThanOrEqual(DTREE_HEADER_F32);
      expect(base).toBeLessThan(buf.length);
    }
  });
});

describe('serialiseSTree — Item A overflow clamp (maxDTreeNodesPerCell)', () => {
  it('concatenated dTreeBuf stays within the GPU allocation when cells exceed the cap', () => {
    // Build a multi-cell sTree, then grow every per-cell dTree to the full
    // 341-node cap so the clamp is exercised on each cell.
    const sTree = buildSTree(SCENE_AABB);
    for (let i = 0; i < PPG_CELL_SPLIT_THRESHOLD + 1; i++) {
      sTreeAccumulate(sTree, [0, 0, 0], [0.5, 0.5], 1.0);
    }
    splitOverflowLeaves(sTree);
    expect(sTree.dTrees.length).toBeGreaterThan(1);
    // Force every dTree to full depth.
    for (const d of sTree.dTrees) {
      for (let pass = 0; pass < 6; pass++) {
        const before = d.nodes.length;
        for (const n of d.nodes) if (n.isLeaf) n.flux = 1.0;
        d.totalFlux = d.nodes.filter((n) => n.isLeaf).length;
        refineDTree(d);
        if (d.nodes.length === before) break;
      }
    }

    const cap = 85; // sub-341 host config
    const { dTreeBuf, dTreeOffsets } = serialiseSTree(sTree, cap);

    // Simulate the GPU allocation: `maxSpatialCells × (header + cap × stride)`.
    // The concatenated buffer must fit it exactly (each cell clamped to `cap`).
    const perCellF32 = DTREE_HEADER_F32 + cap * DTREE_NODE_F32;
    expect(dTreeBuf.length).toBe(sTree.dTrees.length * perCellF32);
    // Every cell's header reports the clamped node count, and successive
    // offsets are spaced by exactly the clamped per-cell stride.
    for (let k = 0; k < sTree.dTrees.length; k++) {
      const off = dTreeOffsets[k]!;
      expect(off).toBe(k * perCellF32);
      expect(dTreeBuf[off]).toBe(Math.min(sTree.dTrees[k]!.nodes.length, cap));
    }
  });

  it('omitting the cap reproduces the historical (unclamped) buffer for default-sized trees', () => {
    const sTree = buildSTree(SCENE_AABB);
    const unclamped = serialiseSTree(sTree);
    // A fresh single-cell sTree's dTree is well under 341 nodes, so a generous
    // cap is a no-op — byte-identical to the unclamped path.
    const clampedGenerous = serialiseSTree(sTree, 341);
    expect(Array.from(clampedGenerous.dTreeBuf)).toEqual(Array.from(unclamped.dTreeBuf));
    expect(Array.from(clampedGenerous.dTreeOffsets)).toEqual(Array.from(unclamped.dTreeOffsets));
  });
});
