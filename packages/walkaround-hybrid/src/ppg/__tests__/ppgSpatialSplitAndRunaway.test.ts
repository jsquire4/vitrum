/**
 * ppgSpatialSplitAndRunaway.test.ts — road-to-100 A2 (spatial sTree adaptivity)
 * + the PPG refine-loop runaway bound.
 *
 * A2 wires the GPU per-cell sample counter into `splitOverflowLeaves` so the
 * spatial tree finally subdivides (previously it was one global cell forever:
 * `sTreeAccumulate` had no GPU-path callers, so every cell's `sampleCount` was
 * eternally zero). These tests exercise the CPU side of that machinery directly:
 *
 *   1. SPLIT FIRES — synthetic per-cell counts over threshold subdivide the
 *      named leaf; both children inherit a CLONE of the parent dTree (Müller
 *      §3.1 directional prior); the grown tree serialises within the GPU stride
 *      and round-trips through the flat-buffer traversal oracle.
 *   2. MULTI-CELL LOOKUP — after splits the serialised sTree routes query
 *      positions to the correct leaf cell (a χ²-style coverage check that the
 *      kd-descent in the flat buffer agrees with the CPU `findSTreeLeaf`).
 *   3. RUNAWAY BOUND — the per-window flux decay (`decayAccumulators` /
 *      PPG_FLUX_DECAY) keeps the persistent flux accumulator bounded under
 *      steady synthetic input, where the no-decay regime diverges linearly.
 */

import { describe, it, expect } from 'vitest';
import {
  buildSTree,
  splitOverflowLeaves,
  findSTreeLeaf,
  decayAccumulators,
} from '../sTree.js';
import {
  serialiseSTree,
  gpuTraverseSTreeLeaf,
  gpuTraverseDTreeLeaf,
  STREE_HEADER_F32,
  STREE_NODE_F32,
} from '../serialise.js';
import { dTreeAccumulateFlux } from '../dTree.js';
import { PPG_CELL_SPLIT_THRESHOLD, PPG_FLUX_DECAY } from '../ppgConstants.js';
import type { AABB, STree } from '../types.js';

const SCENE: AABB = { min: [-4, -3, -2], max: [4, 3, 2] };

/** Count leaf nodes in an sTree. */
function leafCount(t: STree): number {
  return t.nodes.filter(n => n.splitAxis === -1).length;
}

/** Build a per-cell sample-count array (indexed by dTreeIndex). */
function countsFor(t: STree, fn: (dTreeIndex: number) => number): Uint32Array {
  const c = new Uint32Array(t.dTrees.length);
  for (const node of t.nodes) {
    if (node.splitAxis === -1) c[node.dTreeIndex] = fn(node.dTreeIndex);
  }
  return c;
}

describe('A2 — sTree spatial split fed by GPU per-cell sample counts', () => {
  it('does NOT split when no counts are supplied (regression: pre-A2 path)', () => {
    const t = buildSTree(SCENE);
    // CPU node.sampleCount is never written on the GPU path → stays 0 → no split.
    splitOverflowLeaves(t);
    expect(leafCount(t)).toBe(1);
  });

  it('splits the root leaf when its GPU cell count exceeds threshold', () => {
    const t = buildSTree(SCENE);
    // Mark the single root cell (dTreeIndex 0) over threshold.
    const counts = new Uint32Array([PPG_CELL_SPLIT_THRESHOLD + 1]);
    splitOverflowLeaves(t, undefined, 16_384, counts);
    expect(leafCount(t)).toBe(2);
    // The split axis is the longest scene axis (X: 8 units).
    const root = t.nodes[0]!;
    expect(root.splitAxis).toBe(0);
    expect(root.splitValue).toBeCloseTo((SCENE.min[0] + SCENE.max[0]) * 0.5, 6);
  });

  it('does NOT split a cell below threshold even with counts supplied', () => {
    const t = buildSTree(SCENE);
    const counts = new Uint32Array([PPG_CELL_SPLIT_THRESHOLD]); // == threshold, not >
    splitOverflowLeaves(t, undefined, 16_384, counts);
    expect(leafCount(t)).toBe(1);
  });

  it('respects maxCells cap (no growth past the GPU buffer capacity)', () => {
    const t = buildSTree(SCENE);
    // Cap at 1 cell: a single root leaf already meets the cap → no split.
    const counts = new Uint32Array([PPG_CELL_SPLIT_THRESHOLD + 1]);
    splitOverflowLeaves(t, undefined, 1, counts);
    expect(leafCount(t)).toBe(1);
  });

  it('children inherit a CLONE of the parent dTree (directional prior, Müller §3.1)', () => {
    const t = buildSTree(SCENE);
    // Deposit some directional flux into the root cell's dTree so the prior is
    // non-trivial and we can detect that the children copied it.
    dTreeAccumulateFlux(t.dTrees[0]!, [0.7, 0.3], 5.0);
    const parentTotal = t.dTrees[0]!.totalFlux;
    expect(parentTotal).toBeGreaterThan(0);

    const counts = new Uint32Array([PPG_CELL_SPLIT_THRESHOLD + 1]);
    splitOverflowLeaves(t, undefined, 16_384, counts);

    // Two leaves now, with dTreeIndex slots that both carry the parent's flux.
    const leaves = t.nodes.filter(n => n.splitAxis === -1);
    expect(leaves.length).toBe(2);
    for (const leaf of leaves) {
      const dt = t.dTrees[leaf.dTreeIndex]!;
      expect(dt.totalFlux).toBeCloseTo(parentTotal, 6);
    }
    // It must be a CLONE, not an alias: mutating one child's dTree leaves the
    // other unchanged (no shared object reference).
    const [a, b] = leaves;
    const dtA = t.dTrees[a!.dTreeIndex]!;
    const dtB = t.dTrees[b!.dTreeIndex]!;
    expect(dtA).not.toBe(dtB);
    dtA.totalFlux += 99;
    expect(dtB.totalFlux).toBeCloseTo(parentTotal, 6);
  });

  it('grown tree serialises within the GPU per-cell stride and round-trips', () => {
    const t = buildSTree(SCENE);
    // Three rounds of splitting (each round can split every over-threshold leaf
    // once) to build a multi-level kd-tree.
    for (let round = 0; round < 3; round++) {
      const counts = countsFor(t, () => PPG_CELL_SPLIT_THRESHOLD + 1);
      splitOverflowLeaves(t, undefined, 16_384, counts);
    }
    expect(leafCount(t)).toBeGreaterThan(2);
    // dTrees.length must equal leaf count (one dTree per leaf — the split reuses
    // the parent slot for the left child + pushes one for the right).
    expect(t.dTrees.length).toBe(leafCount(t));

    // Serialise with the default per-cell node cap (341) and verify the sTree
    // node buffer length matches the header node count × stride.
    const maxNodesPerCell = 341;
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(t, maxNodesPerCell);
    expect(sTreeBuf[0]).toBe(t.nodes.length); // header nodeCount
    expect(sTreeBuf.length).toBe(STREE_HEADER_F32 + t.nodes.length * STREE_NODE_F32);
    expect(dTreeOffsets.length).toBe(t.dTrees.length);

    // GPU stride respected: every dTree block fits within maxNodesPerCell.
    for (let k = 0; k < dTreeOffsets.length; k++) {
      const off = dTreeOffsets[k]!;
      const nodeCount = dTreeBuf[off]!; // dTree header [0] = nodeCount
      expect(nodeCount).toBeLessThanOrEqual(maxNodesPerCell);
    }

    // Round-trip: the flat-buffer sTree descent agrees with the CPU findSTreeLeaf
    // for a grid of query positions inside the scene.
    let mismatches = 0;
    const samples = 2000;
    let seed = 0x1234;
    const rand = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; };
    for (let i = 0; i < samples; i++) {
      const p: [number, number, number] = [
        SCENE.min[0] + rand() * (SCENE.max[0] - SCENE.min[0]),
        SCENE.min[1] + rand() * (SCENE.max[1] - SCENE.min[1]),
        SCENE.min[2] + rand() * (SCENE.max[2] - SCENE.min[2]),
      ];
      const cpuIdx = findSTreeLeaf(t, p);
      const cpuBase = STREE_HEADER_F32 + cpuIdx * STREE_NODE_F32;
      const gpuBase = gpuTraverseSTreeLeaf(sTreeBuf, p);
      if (gpuBase !== cpuBase) mismatches++;
      // The leaf must have a valid dTree we can descend (no dangling pointer).
      const dTreeIndex = sTreeBuf[gpuBase + 10]!;
      const dOff = dTreeOffsets[dTreeIndex]!;
      const leafBase = gpuTraverseDTreeLeaf(dTreeBuf.subarray(dOff), [rand(), rand()]);
      expect(Number.isFinite(leafBase)).toBe(true);
    }
    expect(mismatches).toBe(0);
  });

  it('multi-cell coverage: every grown leaf cell is reachable by some position', () => {
    const t = buildSTree(SCENE);
    for (let round = 0; round < 3; round++) {
      const counts = countsFor(t, () => PPG_CELL_SPLIT_THRESHOLD + 1);
      splitOverflowLeaves(t, undefined, 16_384, counts);
    }
    const nLeaves = leafCount(t);
    expect(nLeaves).toBeGreaterThan(4);

    // Dense grid sweep; every leaf cell must receive at least one hit (χ²-style
    // "no empty bins" coverage — a kd-tree over a box partitions it fully).
    const hitCells = new Set<number>();
    const N = 24;
    for (let ix = 0; ix < N; ix++) {
      for (let iy = 0; iy < N; iy++) {
        for (let iz = 0; iz < N; iz++) {
          const p: [number, number, number] = [
            SCENE.min[0] + (ix + 0.5) / N * (SCENE.max[0] - SCENE.min[0]),
            SCENE.min[1] + (iy + 0.5) / N * (SCENE.max[1] - SCENE.min[1]),
            SCENE.min[2] + (iz + 0.5) / N * (SCENE.max[2] - SCENE.min[2]),
          ];
          const idx = findSTreeLeaf(t, p);
          hitCells.add(t.nodes[idx]!.dTreeIndex);
        }
      }
    }
    expect(hitCells.size).toBe(nLeaves);
  });
});

describe('PPG refine-loop runaway bound (per-window flux decay)', () => {
  // Simulate the persistent-accumulator update the coordinator performs each
  // window: flux ← decay·flux + freshDeposit, with decayAccumulators applied to
  // the carried-over tree before the fresh deposit. Under STEADY synthetic input
  // F this must converge to the bounded geometric steady state F/(1−decay);
  // the no-decay regime (decay=1) diverges linearly.
  function simulate(decay: number, windows: number): number[] {
    const t = buildSTree(SCENE);
    const dTree = t.dTrees[0]!;
    const F = 1000; // steady per-window deposited flux
    const totals: number[] = [];
    for (let w = 0; w < windows; w++) {
      // Decay the carried-over accumulator (no-op on the very first window).
      // `decayAccumulators` deliberately rejects 1 because production decay=1
      // is the unbounded regime. Skip decay entirely for the negative-control
      // branch below so that test can still demonstrate the historical runaway.
      if (w > 0 && decay < 1) decayAccumulators(t, decay);
      // Fresh deposit for this window into a fixed direction.
      dTreeAccumulateFlux(dTree, [0.7, 0.3], F);
      totals.push(dTree.totalFlux);
    }
    return totals;
  }

  it('PPG_FLUX_DECAY (0.5) bounds steady-input flux at the geometric steady state', () => {
    const totals = simulate(PPG_FLUX_DECAY, 40);
    const tail = totals[totals.length - 1]!;
    const analyticSteadyState = 1000 / (1 - PPG_FLUX_DECAY); // = 2000
    expect(tail).toBeCloseTo(analyticSteadyState, 0);
    // Bounded: late windows are flat (ratio → 1.0), not growing.
    expect(totals[39]! / totals[20]!).toBeLessThan(1.001);
    expect(totals[39]! / totals[20]!).toBeGreaterThan(0.999);
  });

  it('no-decay (decay=1) diverges linearly — confirms the runaway it fixes', () => {
    const totals = simulate(1.0, 40);
    // win39 ≈ 40·F, win20 ≈ 21·F → ratio ≈ 1.9 (unbounded linear growth).
    expect(totals[39]!).toBeCloseTo(40 * 1000, 0);
    expect(totals[39]! / totals[20]!).toBeGreaterThan(1.5);
  });

  it('decay=0 reproduces the historical full reset (no carry-over)', () => {
    const totals = simulate(0.0, 10);
    // Each window starts from zero carry-over → total == F every window.
    for (const tot of totals) expect(tot).toBeCloseTo(1000, 6);
  });
});
