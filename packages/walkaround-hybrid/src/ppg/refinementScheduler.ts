/**
 * sTree refinement scheduler — W9 follow-up.
 *
 * Bridges the gap between the GPU `ppgUpdate` kernel (which atomically
 * accumulates per-dTree-leaf flux into `fluxAtomicsBuf` every frame) and
 * the CPU `splitOverflowLeaves` routine (which subdivides over-sampled
 * sTree leaves on Müller §3.1's rebuild schedule).
 *
 * Cadence — refinement runs every `intervalFrames` (default 32) frames.
 * The scheduler maintains a ping-pong of two staging buffers so a readback
 * can be in flight while the next frame's update kernel writes the live
 * atomic buffer.
 *
 * Heuristic gating — once readback completes, the scheduler decides
 * whether the refinement step is worth running:
 *
 *   1. Sample-count proxy is INCREASING (current total flux > prior total
 *      flux by a meaningful delta). If atomics haven't moved, no new
 *      samples have arrived and a split would be premature.
 *   2. OR loss proxy is DECREASING (current per-cell-flux variance has
 *      dropped vs prior — the directional distribution is firming up,
 *      so spatial subdivision can extract more headroom).
 *
 * When neither holds, the readback is consumed (we still need to clear
 * the atomics so next frame starts fresh) but `splitOverflowLeaves` is
 * skipped and no GPU re-upload happens.
 *
 * Output — when refinement DOES run, the CPU sTree is updated in place,
 * re-serialised, and re-uploaded via the caller's hook. The caller also
 * clears `fluxAtomicsBuf` so the next training cycle accumulates afresh.
 *
 * Reference: Müller et al. 2017, Practical Path Guiding §3.1, §5 (the
 * "tree is serialised to a flat GPU buffer each rebuild cycle").
 */

import type { STree } from './types.js';
import { splitOverflowLeaves } from './sTree.js';
import {
  DTREE_NODE_F32,
  DTREE_HEADER_F32,
} from './serialise.js';

/** Inverse of the WGSL `encodeFlux` scale (1 / 65536). */
const FLUX_DECODE: number = 1 / 65536;

/**
 * Refinement cadence — number of rendered frames between successive
 * readback-and-refine attempts. Müller paper suggests "every few thousand
 * samples"; for an interactive walkaround at ~60 fps a 32-frame interval
 * triggers refinement roughly twice a second, which is brisk enough to
 * track scene-relight responsiveness without saturating the CPU with
 * readback work.
 */
export const PPG_REFINEMENT_INTERVAL_DEFAULT: number = 32;

/**
 * Per-call gating thresholds. Tuned to be permissive — we DO want to
 * refine on every readback that shows the atomics have moved, since
 * splitOverflowLeaves itself has a `sampleCount > threshold` guard
 * (`PPG_CELL_SPLIT_THRESHOLD = 12 000` in flux-units after decode).
 *
 * Setting `minTotalFluxDelta` too high would cause stale leaves that
 * sit just under the per-cell threshold to never refine; too low and
 * we waste a readback + re-upload on a no-op. 1e-3 (decoded flux
 * units) is well above floating-point noise and well below any
 * meaningful contribution.
 */
export interface RefinementGatingThresholds {
  /** Lower bound on (total_flux_current − total_flux_prev) to count as
   *  "samples increasing". */
  minTotalFluxDelta: number;
  /** Per-cell flux-variance drop that counts as "loss decreasing". */
  minLossDeltaFraction: number;
}

export const REFINEMENT_DEFAULT_GATING: Readonly<RefinementGatingThresholds> = Object.freeze({
  minTotalFluxDelta: 1e-3,
  minLossDeltaFraction: 1e-3,
});

/**
 * Per-cell statistics decoded from a single atomics readback. Used to drive
 * the heuristic gate; also written back into the CPU `STreeNode.sampleCount`
 * field so `splitOverflowLeaves` can decide which cells to subdivide.
 */
export interface RefinementSnapshot {
  /** Sum of decoded flux across every dTree leaf in every cell. */
  totalFlux: number;
  /** Per-sTree-leaf decoded flux total (length = number of leaves). */
  perCellFlux: Float64Array;
  /** A scalar proxy for "loss": the standard-deviation of perCellFlux,
   *  capturing how unevenly samples landed across cells. Decreasing
   *  std-dev = converging distribution. */
  fluxStdDev: number;
}

/**
 * Stateful scheduler — instantiated once per pipeline, ticked every frame.
 *
 * Two ping-pong staging buffers are owned here so the pipeline doesn't
 * need to worry about destroying them. They are sized to the live
 * `fluxAtomicsBuf` and re-allocated when its size changes (resize).
 */
export class STreeRefinementScheduler {
  // ── Configuration ────────────────────────────────────────────────────
  private readonly _intervalFrames: number;
  private readonly _gating: RefinementGatingThresholds;

  // ── Cadence + history state ──────────────────────────────────────────
  /** Total successful refinement cycles since construction. Surfaced via
   *  `engine.debug.ppgRefinementCount()`. */
  private _refinementCount: number = 0;
  /** Total accumulated readbacks (refined + skipped). Useful for
   *  scheduling debug. */
  private _readbackCount: number = 0;
  /** Last decoded total flux — gating compares against this. */
  private _prevTotalFlux: number = 0;
  /** Last decoded flux std-dev — used for the "loss decreasing" branch. */
  private _prevFluxStdDev: number = 0;
  /** Whether a readback is currently in flight (don't start a second). */
  private _readbackInFlight: boolean = false;
  /** Number of CPU mirror sTree leaves the last refinement was applied
   *  to. Used by tests + the debug surface. */
  private _lastRefinementLeafCount: number = 0;

  // ── GPU staging — owned here so dispose() can release them ───────────
  private _stagingBuffers: GPUBuffer[] = [];
  private _stagingByteSize: number = 0;
  /** Index of the next staging buffer to use. Two buffers ping-pong so a
   *  readback can be in flight while the next dispatch writes the live
   *  atomic buffer. */
  private _stagingIndex: number = 0;

  constructor(opts?: {
    /** Override the default 32-frame cadence. Must be ≥ 1. */
    intervalFrames?: number;
    /** Override the default gating thresholds. */
    gating?: RefinementGatingThresholds;
  }) {
    this._intervalFrames = Math.max(1, opts?.intervalFrames ?? PPG_REFINEMENT_INTERVAL_DEFAULT);
    this._gating = opts?.gating ?? REFINEMENT_DEFAULT_GATING;
  }

  /** Total refinement cycles that actually mutated the sTree. Surfaced
   *  via `engine.debug.ppgRefinementCount()`. */
  get refinementCount(): number { return this._refinementCount; }

  /** Total readbacks attempted (refined + skipped). */
  get readbackCount(): number { return this._readbackCount; }

  /** Leaf count of the sTree as of the last successful refinement.
   *  Helpful for confirming `splitOverflowLeaves` actually grew the tree. */
  get lastRefinementLeafCount(): number { return this._lastRefinementLeafCount; }

  /** Whether `tick()` should attempt a readback this frame. */
  shouldReadback(frameIndex: number): boolean {
    if (this._readbackInFlight) return false;
    // First tick on frame 0 is wasted (the kernel hasn't written anything yet).
    // Skip it to avoid a no-op readback on the very first frame.
    if (frameIndex === 0) return false;
    return (frameIndex % this._intervalFrames) === 0;
  }

  /**
   * Allocate (or reallocate) the two ping-pong staging buffers sized to
   * the live `fluxAtomicsBuf`. Called from the pipeline at PPG init AND
   * on resize. Idempotent when `byteSize` is unchanged.
   *
   * The staging buffers are MAP_READ + COPY_DST — the pipeline issues
   * `copyBufferToBuffer(fluxAtomics → staging)` inside the per-frame
   * encoder, then `mapAsync(GPUMapMode.READ)` outside it.
   */
  ensureStaging(device: GPUDevice, byteSize: number): void {
    if (this._stagingByteSize === byteSize && this._stagingBuffers.length === 2) {
      return;
    }
    // Resize → free + reallocate.
    for (const b of this._stagingBuffers) b.destroy();
    this._stagingBuffers = [
      device.createBuffer({
        label: 'ppg-refinement-staging-0',
        size: byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      device.createBuffer({
        label: 'ppg-refinement-staging-1',
        size: byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
    ];
    this._stagingByteSize = byteSize;
    this._stagingIndex = 0;
    this._readbackInFlight = false;
  }

  /** Acquire the current staging buffer for `copyBufferToBuffer`. Advances
   *  the ping-pong on return — every call must be matched with a
   *  `consumeReadback()` on the same buffer once mapAsync resolves.
   *
   *  Returns null when staging isn't allocated yet. */
  acquireStaging(): GPUBuffer | null {
    if (this._stagingBuffers.length === 0) return null;
    const buf = this._stagingBuffers[this._stagingIndex]!;
    this._readbackInFlight = true;
    this._stagingIndex = (this._stagingIndex + 1) % this._stagingBuffers.length;
    return buf;
  }

  /**
   * Decode the mapped staging buffer + apply the gating heuristic.
   * On accept: writes per-cell sample-count estimates into `sTree.nodes[*].sampleCount`
   * and returns the snapshot so the caller can finish the refine step
   * (splitOverflowLeaves + re-upload). On reject: returns null.
   *
   * The caller MUST unmap the staging buffer when done.
   */
  consumeReadback(
    sTree: STree,
    dTreeOffsets: Uint32Array,
    mappedRange: ArrayBuffer,
  ): RefinementSnapshot | null {
    this._readbackInFlight = false;
    this._readbackCount++;

    // Decode the u32 atomic array → per-cell flux totals.
    const snap = decodeAtomicsToSnapshot(sTree, dTreeOffsets, mappedRange);

    // Heuristic gate — accept if either branch fires.
    const totalFluxDelta = snap.totalFlux - this._prevTotalFlux;
    const fluxStdDevDelta = this._prevFluxStdDev - snap.fluxStdDev; // positive = decreasing
    const samplesIncreasing = totalFluxDelta > this._gating.minTotalFluxDelta;
    // Loss-decreasing branch only fires when we have a meaningful prior
    // std-dev to compare against. The first successful readback always
    // takes the samples-increasing branch.
    const lossDecreasing =
      this._prevFluxStdDev > 0 &&
      fluxStdDevDelta > this._gating.minLossDeltaFraction * this._prevFluxStdDev;

    // Update history BEFORE returning so the next call sees this snapshot
    // as the baseline regardless of whether refinement actually ran.
    this._prevTotalFlux = snap.totalFlux;
    this._prevFluxStdDev = snap.fluxStdDev;

    if (!samplesIncreasing && !lossDecreasing) {
      return null;
    }

    // Write the per-cell sample-count proxy into the CPU mirror so
    // splitOverflowLeaves can use its standard threshold. We use the
    // decoded flux as the sample-count proxy — units differ from a pure
    // count, but splitOverflowLeaves takes the threshold by argument
    // so the caller can pass a calibrated `PPG_CELL_SPLIT_THRESHOLD`
    // pre-scaled to flux units.
    let leafCursor = 0;
    for (const node of sTree.nodes) {
      if (node.splitAxis !== -1) continue;
      node.sampleCount = snap.perCellFlux[leafCursor] ?? 0;
      leafCursor++;
    }

    return snap;
  }

  /** Run the actual split + bookkeeping update. Returns true when at
   *  least one leaf was split. Caller is responsible for re-uploading
   *  the serialised tree afterwards. */
  applySplit(
    sTree: STree,
    splitThreshold: number,
    maxCells: number,
  ): boolean {
    const leavesBefore = countLeaves(sTree);
    splitOverflowLeaves(sTree, splitThreshold, maxCells);
    const leavesAfter = countLeaves(sTree);
    this._lastRefinementLeafCount = leavesAfter;
    if (leavesAfter > leavesBefore) {
      this._refinementCount++;
      return true;
    }
    return false;
  }

  /**
   * Reset the in-flight + history state. Call on `setScene`-style resets
   * where the sTree is rebuilt fresh — without this, the gating
   * heuristic would see a huge spurious "loss" when the new scene's
   * atomic readback comes back with totally different distribution.
   *
   * Does NOT reset `_refinementCount` so the debug counter shows
   * cumulative work across scenes (matches the user's request that the
   * metric proves "refinement is happening").
   */
  resetHistory(): void {
    this._prevTotalFlux = 0;
    this._prevFluxStdDev = 0;
    this._readbackInFlight = false;
    this._stagingIndex = 0;
  }

  /** Release the staging buffers. Idempotent. */
  dispose(): void {
    for (const b of this._stagingBuffers) b.destroy();
    this._stagingBuffers = [];
    this._stagingByteSize = 0;
    this._readbackInFlight = false;
  }
}

/**
 * Decode a raw mapped readback of `fluxAtomicsBuf` into per-cell flux
 * totals + a scalar loss proxy. Exported for direct unit testing.
 *
 * Slot mapping (matches the WGSL update kernel):
 *   slot = (dTreeOffset + DTREE_HEADER_F32 + nodeIdx × DTREE_NODE_F32) / DTREE_NODE_F32
 *        = floor(dTreeOffset / DTREE_NODE_F32) + nodeIdx
 *   (the 4-f32 dTree header maps to an integer-truncated +0.5 inside the
 *   shader's u32 division, which collapses to the formula above).
 */
export function decodeAtomicsToSnapshot(
  sTree: STree,
  dTreeOffsets: Uint32Array,
  mappedRange: ArrayBuffer,
): RefinementSnapshot {
  const atomics = new Uint32Array(mappedRange);

  const leafIndices: number[] = [];
  for (let i = 0; i < sTree.nodes.length; i++) {
    if (sTree.nodes[i]!.splitAxis === -1) leafIndices.push(i);
  }
  const leafCount = leafIndices.length;

  const perCellFlux = new Float64Array(leafCount);
  let totalFlux = 0;

  for (let li = 0; li < leafCount; li++) {
    const nodeIdx = leafIndices[li]!;
    const node = sTree.nodes[nodeIdx]!;
    const dTreeIdx = node.dTreeIndex;
    const dTree = sTree.dTrees[dTreeIdx];
    if (dTree === undefined) continue;

    const dTreeOffsetF32 = dTreeOffsets[dTreeIdx] ?? 0;
    // Slot base in the atomic array. Match the WGSL slot calculation:
    //   slot = floor((dTreeOffsetF32 + HEADER) / DTREE_NODE_F32)
    //        = floor(dTreeOffsetF32 / DTREE_NODE_F32) + 0   (header rounds down)
    // The header occupies the first DTREE_HEADER_F32 f32 lanes; the shader's
    // u32 division truncates +0.5 → 0, so we land on a stable per-cell base.
    const slotBase = Math.floor(dTreeOffsetF32 / DTREE_NODE_F32);

    let cellFlux = 0;
    const nNodes = dTree.nodes.length;
    for (let k = 0; k < nNodes; k++) {
      const slot = slotBase + k;
      if (slot >= atomics.length) break;
      // Decode fixed-point: atomic u32 = encodeFlux(lum) = lum × 65536.
      const decoded = atomics[slot]! * FLUX_DECODE;
      cellFlux += decoded;
    }
    perCellFlux[li] = cellFlux;
    totalFlux += cellFlux;
  }

  // Welford-style single-pass std-dev for the loss proxy.
  let mean = 0;
  let m2 = 0;
  for (let i = 0; i < leafCount; i++) {
    const x = perCellFlux[i]!;
    const delta = x - mean;
    mean += delta / (i + 1);
    m2 += delta * (x - mean);
  }
  const variance = leafCount > 1 ? m2 / (leafCount - 1) : 0;
  const fluxStdDev = Math.sqrt(Math.max(0, variance));

  return { totalFlux, perCellFlux, fluxStdDev };
}

/** Count leaves in an sTree. Local helper to avoid exporting from sTree.ts. */
function countLeaves(sTree: STree): number {
  let n = 0;
  for (const node of sTree.nodes) if (node.splitAxis === -1) n++;
  return n;
}

// Re-export to satisfy the linter (DTREE_HEADER_F32 used only in comments).
// Keeping the import provides single-source-of-truth — if the f32 layout
// changes, this file fails to compile, surfacing the dependency.
void DTREE_HEADER_F32;
