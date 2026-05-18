/**
 * sTree refinement scheduler — W9 follow-up tests.
 *
 * Covers:
 *   1. Cadence — `shouldReadback` returns true exactly on the
 *      `intervalFrames` cadence (and never on frame 0).
 *   2. Decode — `decodeAtomicsToSnapshot` correctly maps atomic slots
 *      back to per-cell flux totals using `dTreeOffsets`.
 *   3. Heuristic gate — `consumeReadback` accepts when total flux grows,
 *      rejects when atomics are static.
 *   4. Split — `applySplit` increments `refinementCount` exactly when
 *      `splitOverflowLeaves` actually grew the tree.
 *   5. History reset — `resetHistory` clears prior-frame baselines so the
 *      next readback doesn't see a spurious huge delta after a scene swap.
 *   6. Refinement count surface — exported `refinementCount` matches the
 *      number of growth events.
 *   7. Ping-pong — `acquireStaging` advances through both buffers and
 *      blocks a second in-flight readback before the first is consumed.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import {
  STreeRefinementScheduler,
  decodeAtomicsToSnapshot,
  PPG_REFINEMENT_INTERVAL_DEFAULT,
} from '../src/ppg/refinementScheduler.js';
import { buildSTree, splitOverflowLeaves } from '../src/ppg/sTree.js';
import { serialiseSTree } from '../src/ppg/serialise.js';
import { PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS } from '../src/ppg/ppgConstants.js';
import type { AABB } from '../src/ppg/types.js';

installWebGPUPolyfills();

// ── Helpers ─────────────────────────────────────────────────────────────────

function unitAABB(): AABB {
  return { min: [-1, -1, -1], max: [1, 1, 1] };
}

// Encode `lum * 65536` like the WGSL kernel does. Tests build a fake
// atomics buffer by writing encoded counts into the slot positions that
// correspond to each cell's dTree leaves.
function encodeFluxU32(lum: number): number {
  return Math.min(Math.floor(lum * 65536), 0xFFFFFFFF);
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('STreeRefinementScheduler — cadence (shouldReadback)', () => {
  it('returns false on frame 0 (kernel has not yet written)', () => {
    const sched = new STreeRefinementScheduler();
    expect(sched.shouldReadback(0)).toBe(false);
  });

  it('returns true on every Nth frame at the default cadence', () => {
    const sched = new STreeRefinementScheduler();
    for (let f = 1; f < PPG_REFINEMENT_INTERVAL_DEFAULT; f++) {
      expect(sched.shouldReadback(f)).toBe(false);
    }
    expect(sched.shouldReadback(PPG_REFINEMENT_INTERVAL_DEFAULT)).toBe(true);
    expect(sched.shouldReadback(PPG_REFINEMENT_INTERVAL_DEFAULT * 2)).toBe(true);
    expect(sched.shouldReadback(PPG_REFINEMENT_INTERVAL_DEFAULT * 3)).toBe(true);
  });

  it('honours an override cadence', () => {
    const sched = new STreeRefinementScheduler({ intervalFrames: 5 });
    expect(sched.shouldReadback(1)).toBe(false);
    expect(sched.shouldReadback(4)).toBe(false);
    expect(sched.shouldReadback(5)).toBe(true);
    expect(sched.shouldReadback(10)).toBe(true);
  });
});

describe('decodeAtomicsToSnapshot', () => {
  it('decodes fixed-point flux correctly for the single-cell root', () => {
    const sTree = buildSTree(unitAABB());
    const { dTreeOffsets } = serialiseSTree(sTree);
    // The root cell has dTreeOffsets[0] = 0 → slotBase = 0. The dTree at
    // INITIAL_DEPTH=2 has 1 + 4 + 16 = 21 nodes; only the 16 leaves
    // accumulate flux, but the slot mapping covers all nodes (the
    // interior slots stay zero in practice — the kernel only writes leaves).
    const slots = sTree.dTrees[0]!.nodes.length;
    const atomics = new Uint32Array(slots);
    // Write a known flux value into a few leaf slots.
    atomics[3] = encodeFluxU32(0.5);   // half-unit flux
    atomics[7] = encodeFluxU32(1.25);  // 1.25-unit flux
    atomics[15] = encodeFluxU32(0.25);

    const snap = decodeAtomicsToSnapshot(sTree, dTreeOffsets, atomics.buffer);
    expect(snap.totalFlux).toBeCloseTo(0.5 + 1.25 + 0.25, 5);
    expect(snap.perCellFlux.length).toBe(1);
    expect(snap.perCellFlux[0]).toBeCloseTo(2.0, 5);
    // Std-dev of a single sample is 0 (no variance for N=1).
    expect(snap.fluxStdDev).toBe(0);
  });

  it('decodes split cells using the dTreeOffsets table', () => {
    const sTree = buildSTree(unitAABB());
    // Force a split by maxing out the root sample count.
    sTree.nodes[0]!.sampleCount = PPG_CELL_SPLIT_THRESHOLD + 1;
    splitOverflowLeaves(sTree, PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS);
    const { dTreeOffsets } = serialiseSTree(sTree);

    // After split: 1 interior + 2 leaves; 3 dTrees total (original + 2 clones).
    expect(sTree.nodes.length).toBe(3);
    expect(sTree.dTrees.length).toBe(3);
    expect(dTreeOffsets.length).toBe(3);

    // Build an atomics buffer big enough for all slots: each dTree
    // contributes one slot per node. The slot offset for cell k starts
    // at floor(dTreeOffsets[k] / 8). Write 1.0 flux into the FIRST leaf
    // slot of cell 1 (left child) and 2.0 into cell 2 (right child).
    // The root dTree (index 0) becomes orphaned but its slots still
    // exist in the buffer.
    const totalSlots = sTree.dTrees.reduce((acc, dT) => acc + dT.nodes.length, 0) + 32;
    const atomics = new Uint32Array(totalSlots);
    const leftDTreeBase = Math.floor(dTreeOffsets[1]! / 8);
    const rightDTreeBase = Math.floor(dTreeOffsets[2]! / 8);
    atomics[leftDTreeBase + 1] = encodeFluxU32(1.0);
    atomics[rightDTreeBase + 1] = encodeFluxU32(2.0);

    const snap = decodeAtomicsToSnapshot(sTree, dTreeOffsets, atomics.buffer);
    expect(snap.perCellFlux.length).toBe(2); // two leaves now
    // Total should be 1.0 + 2.0 = 3.0 (within fixed-point precision).
    expect(snap.totalFlux).toBeCloseTo(3.0, 4);
    // One leaf has 1.0, the other has 2.0 — order depends on the leaf
    // traversal order in the flat sTree node array. Assert membership.
    const sorted = Array.from(snap.perCellFlux).sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(1.0, 4);
    expect(sorted[1]).toBeCloseTo(2.0, 4);
    // Std-dev of {1, 2} = 1/sqrt(2) ≈ 0.707 (Bessel's correction).
    expect(snap.fluxStdDev).toBeCloseTo(Math.sqrt(0.5), 3);
  });
});

describe('STreeRefinementScheduler — heuristic gate', () => {
  it('accepts the first readback when flux is non-zero (samples increasing branch)', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    const { dTreeOffsets } = serialiseSTree(sTree);
    const slots = sTree.dTrees[0]!.nodes.length;
    const atomics = new Uint32Array(slots);
    atomics[3] = encodeFluxU32(0.5);

    const snap = sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer);
    expect(snap).not.toBeNull();
    expect(snap!.totalFlux).toBeCloseTo(0.5, 4);
  });

  it('rejects a static readback (total flux unchanged) after the first', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    const { dTreeOffsets } = serialiseSTree(sTree);
    const slots = sTree.dTrees[0]!.nodes.length;
    const atomics = new Uint32Array(slots);
    atomics[3] = encodeFluxU32(0.5);

    const first = sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer);
    expect(first).not.toBeNull();
    // Second readback with the SAME atomic state — no new samples →
    // gate must reject.
    // (We need to acquire/release the in-flight flag.)
    sched.acquireStaging?.(); // no-op when no staging is allocated
    const second = sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer);
    expect(second).toBeNull();
  });

  it('rejects an all-zero readback (no kernel writes yet)', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    const { dTreeOffsets } = serialiseSTree(sTree);
    const slots = sTree.dTrees[0]!.nodes.length;
    const atomics = new Uint32Array(slots);
    const snap = sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer);
    expect(snap).toBeNull();
  });

  it('writes per-cell sample-count proxy onto sTree.nodes when accepted', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    const { dTreeOffsets } = serialiseSTree(sTree);
    const slots = sTree.dTrees[0]!.nodes.length;
    const atomics = new Uint32Array(slots);
    // Push the encoded flux above PPG_CELL_SPLIT_THRESHOLD so the
    // subsequent split actually fires.
    atomics[3] = encodeFluxU32(PPG_CELL_SPLIT_THRESHOLD + 100);

    const snap = sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer);
    expect(snap).not.toBeNull();
    expect(sTree.nodes[0]!.sampleCount).toBeGreaterThan(PPG_CELL_SPLIT_THRESHOLD);
  });
});

describe('STreeRefinementScheduler — applySplit', () => {
  it('increments refinementCount when the tree grows', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    // Pre-load the threshold so splitOverflowLeaves actually splits.
    sTree.nodes[0]!.sampleCount = PPG_CELL_SPLIT_THRESHOLD + 1;
    expect(sched.refinementCount).toBe(0);
    const grew = sched.applySplit(sTree, PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS);
    expect(grew).toBe(true);
    expect(sched.refinementCount).toBe(1);
    expect(sched.lastRefinementLeafCount).toBe(2);
  });

  it('does NOT increment refinementCount when no leaf overflows', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    sTree.nodes[0]!.sampleCount = 0; // below threshold
    const grew = sched.applySplit(sTree, PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS);
    expect(grew).toBe(false);
    expect(sched.refinementCount).toBe(0);
    expect(sched.lastRefinementLeafCount).toBe(1);
  });

  it('accumulates refinementCount across multiple cycles', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    sTree.nodes[0]!.sampleCount = PPG_CELL_SPLIT_THRESHOLD + 1;
    sched.applySplit(sTree, PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS);
    // Force a second split on the (now leaf) child of the previous split.
    for (const n of sTree.nodes) {
      if (n.splitAxis === -1) n.sampleCount = PPG_CELL_SPLIT_THRESHOLD + 1;
    }
    sched.applySplit(sTree, PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS);
    expect(sched.refinementCount).toBe(2);
  });
});

describe('STreeRefinementScheduler — resetHistory', () => {
  it('clears prior-frame baselines so a fresh-scene readback is accepted', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    const { dTreeOffsets } = serialiseSTree(sTree);
    const slots = sTree.dTrees[0]!.nodes.length;
    const atomics = new Uint32Array(slots);
    atomics[3] = encodeFluxU32(1.0);

    // First readback — accepted (samples increasing).
    expect(sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer)).not.toBeNull();
    // Second identical readback — rejected (gate would reject as static).
    expect(sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer)).toBeNull();
    // After reset, even the same atomics buffer is accepted again because
    // the baseline went back to zero.
    sched.resetHistory();
    expect(sched.consumeReadback(sTree, dTreeOffsets, atomics.buffer)).not.toBeNull();
  });

  it('does NOT zero out the cumulative refinementCount', () => {
    const sched = new STreeRefinementScheduler();
    const sTree = buildSTree(unitAABB());
    sTree.nodes[0]!.sampleCount = PPG_CELL_SPLIT_THRESHOLD + 1;
    sched.applySplit(sTree, PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS);
    expect(sched.refinementCount).toBe(1);
    sched.resetHistory();
    expect(sched.refinementCount).toBe(1); // unchanged
  });
});

describe('STreeRefinementScheduler — staging ping-pong', () => {
  function makeMockBufferDevice(): GPUDevice {
    let createCount = 0;
    return {
      createBuffer: (desc: GPUBufferDescriptor) => {
        createCount++;
        return {
          __id: createCount,
          size: desc.size,
          usage: desc.usage,
          label: desc.label ?? '',
          destroy: () => {},
        } as unknown as GPUBuffer;
      },
    } as unknown as GPUDevice;
  }

  it('allocates two staging buffers on first ensureStaging call', () => {
    const sched = new STreeRefinementScheduler();
    const dev = makeMockBufferDevice();
    sched.ensureStaging(dev, 1024);
    const a = sched.acquireStaging();
    const b = sched.acquireStaging();
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    expect(a).not.toBe(b); // ping-pong distinct
  });

  it('blocks a second concurrent acquireStaging via shouldReadback gate', () => {
    const sched = new STreeRefinementScheduler({ intervalFrames: 1 });
    const dev = makeMockBufferDevice();
    sched.ensureStaging(dev, 1024);
    // Cadence ticks: frames 1, 2, 3 should all match (interval=1, except
    // frame 0 is always skipped).
    expect(sched.shouldReadback(1)).toBe(true);
    sched.acquireStaging();
    // _readbackInFlight is now set; until consumeReadback clears it,
    // shouldReadback must return false on the next cadence tick.
    expect(sched.shouldReadback(2)).toBe(false);
  });

  it('dispose() releases both staging buffers and zeros internal state', () => {
    const sched = new STreeRefinementScheduler();
    const dev = makeMockBufferDevice();
    sched.ensureStaging(dev, 1024);
    sched.dispose();
    // After dispose, acquireStaging must return null (no buffers).
    expect(sched.acquireStaging()).toBeNull();
  });

  it('ensureStaging is idempotent when byteSize is unchanged', () => {
    const sched = new STreeRefinementScheduler();
    const dev = makeMockBufferDevice();
    sched.ensureStaging(dev, 1024);
    const firstAcquired = sched.acquireStaging();
    // Reset ping-pong + clear in-flight flag so we can acquire again
    // after a no-op re-ensure.
    sched.resetHistory();
    sched.ensureStaging(dev, 1024); // same size
    const secondAcquired = sched.acquireStaging();
    // The same underlying buffer should be reused (not a fresh
    // allocation). We tag each createBuffer call with __id starting at 1;
    // a re-allocation would have produced __id=3.
    const firstId = (firstAcquired as unknown as { __id: number }).__id;
    const secondId = (secondAcquired as unknown as { __id: number }).__id;
    expect([1, 2]).toContain(firstId);
    expect([1, 2]).toContain(secondId);
  });

  it('ensureStaging reallocates when byteSize changes (resize path)', () => {
    const sched = new STreeRefinementScheduler();
    const dev = makeMockBufferDevice();
    sched.ensureStaging(dev, 1024);
    sched.ensureStaging(dev, 2048);
    // After the resize, the new ping-pong buffers have __id 3 and 4.
    sched.resetHistory();
    const acquired = sched.acquireStaging();
    const id = (acquired as unknown as { __id: number }).__id;
    expect(id).toBeGreaterThanOrEqual(3);
  });
});

// ── End-to-end through the WalkaroundGPUPipeline accessor ──────────────────

describe('WalkaroundGPUPipeline — ppgRefinementCount accessor (zero by default)', () => {
  it('returns 0 when PPG is disabled', async () => {
    // Importing at top level would force the pipeline module + its
    // many GPU-handle imports; do it inline so the rest of this file
    // stays isolated from any module-level GPU touches.
    const { WalkaroundGPUPipeline } = await import('../src/pipeline/WalkaroundGPUPipeline.js');
    const pipeline = new WalkaroundGPUPipeline({} as GPUDevice, 64, 64);
    expect(pipeline.getPPGRefinementCount()).toBe(0);
  });
});
