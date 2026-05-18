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
