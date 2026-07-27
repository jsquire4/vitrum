/**
 * sTree — Adaptive spatial binary tree for Practical Path Guiding.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.1.
 *
 * The spatial domain is partitioned by a binary kd-tree. Cells are split
 * adaptively when the accumulated sample count exceeds
 * `PPG_CELL_SPLIT_THRESHOLD`. Both children inherit the parent cell's
 * directional distribution (dTree) upon splitting, then accumulate
 * independently.
 *
 * ADDRESSES DEVIATION 1 (from plan/archive/sweep-2026-05-11-fixes-engines.md Item 25):
 *   The deleted implementation used a static kd-tree built once from a 4×4×4
 *   uniform grid. This module implements a DYNAMIC adaptive split: after each
 *   training frame's atomic accumulation, `splitOverflowLeaves` traverses the
 *   tree and subdivides any leaf whose `sampleCount > threshold`. The tree
 *   rebuilds by extending its flat node array — no static grid.
 */

import {
  PPG_CELL_SPLIT_THRESHOLD,
  PPG_DTREE_INITIAL_DEPTH,
} from './ppgConstants.js';
import type { AABB, STree, STreeNode, DTree } from './types.js';
import { buildEmptyDTree, dTreeAccumulateFlux } from './dTree.js';

// ────────────────────────────────────────────────────────────────────────────
// AABB helpers
// ────────────────────────────────────────────────────────────────────────────

/** Return the index of the longest axis of an AABB (0=X, 1=Y, 2=Z). */
function longestAxis(aabb: AABB): 0 | 1 | 2 {
  const dx = aabb.max[0] - aabb.min[0];
  const dy = aabb.max[1] - aabb.min[1];
  const dz = aabb.max[2] - aabb.min[2];
  if (dx >= dy && dx >= dz) return 0;
  if (dy >= dz) return 1;
  return 2;
}

/** Clone an AABB (avoids aliasing bugs during split). */
function cloneAABB(a: AABB): AABB {
  return {
    min: [a.min[0], a.min[1], a.min[2]],
    max: [a.max[0], a.max[1], a.max[2]],
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Build
// ────────────────────────────────────────────────────────────────────────────

/**
 * Build a fresh sTree with a single root leaf covering `sceneBounds`.
 * The root gets an empty dTree at initial depth `PPG_DTREE_INITIAL_DEPTH`.
 *
 * Called once at engine init; thereafter `splitOverflowLeaves` keeps the
 * tree current.
 *
 * @param sceneBounds  World-space AABB covering the entire scene.
 */
export function buildSTree(
  sceneBounds: AABB,
  initialDTreeDepth: number = PPG_DTREE_INITIAL_DEPTH,
): STree {
  const rootDTree: DTree = buildEmptyDTree(initialDTreeDepth);
  const rootNode: STreeNode = {
    aabb: cloneAABB(sceneBounds),
    splitAxis: -1,
    splitValue: 0,
    leftChild: -1,
    rightChild: -1,
    dTreeIndex: 0,
    sampleCount: 0,
  };
  return {
    nodes: [rootNode],
    dTrees: [rootDTree],
    sceneBounds: cloneAABB(sceneBounds),
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Traversal
// ────────────────────────────────────────────────────────────────────────────

/**
 * Descend the sTree to find the leaf node whose cell contains `position`.
 * Returns the node index in `sTree.nodes`.
 *
 * Müller §3.1: descent at each interior node compares the query position
 * against the split plane and follows left (< splitValue) or right (≥).
 *
 * ALL training and guide directions live in the WORLD frame (deviation 4 fix).
 * Position must be in world space.
 */
export function findSTreeLeaf(
  sTree: STree,
  position: [number, number, number],
): number {
  let idx = 0;
  while (true) {
    const node = sTree.nodes[idx]!;
    if (node.splitAxis === -1) return idx;
    const axis = node.splitAxis;
    if (position[axis] < node.splitValue) {
      idx = node.leftChild;
    } else {
      idx = node.rightChild;
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Training: accumulate a sample
// ────────────────────────────────────────────────────────────────────────────

/**
 * CPU oracle/test helper: record that a training sample at `position`
 * contributed non-negative `trainingMass` to the direction encoded in `uv`
 * (cylindrical equal-area UV in [0,1]²).
 *
 * The production hybrid path trains on the GPU, reads back per-cell counts /
 * flux, and merges those results in `PPGCoordinator`; this helper remains for
 * CPU-side invariants and legacy path-guiding tests.
 *
 * Semantics:
 *   - `position` = sample position in WORLD space
 *   - `octUV`    = cylindrical equal-area map of the selected direction (world frame)
 *   - `flux`     = non-negative histogram mass. The GPU path uses the initial
 *                  RIS estimator `w_sum / M`, not raw incoming/outgoing radiance.
 *
 * DEVIATION 4 FIX: octUV is computed from direction in WORLD space. No
 * per-surface ONB transform is applied here.
 */
export function sTreeAccumulate(
  sTree: STree,
  position: [number, number, number],
  octUV: [number, number],
  flux: number,
): void {
  if (position.some((component) => !Number.isFinite(component))) {
    throw new RangeError(
      `position must contain finite world coordinates; got ${position.join(',')}`,
    );
  }
  const leafIdx = findSTreeLeaf(sTree, position);
  const node = sTree.nodes[leafIdx];
  if (!node || node.splitAxis !== -1) {
    throw new RangeError(`sTree traversal did not resolve a valid leaf; got ${leafIdx}`);
  }
  if (!Number.isSafeInteger(node.sampleCount)
      || node.sampleCount < 0
      || node.sampleCount >= Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`sTree leaf ${leafIdx} has an invalid sample count ${node.sampleCount}`);
  }
  const dTree = sTree.dTrees[node.dTreeIndex];
  if (!dTree) {
    throw new RangeError(`sTree leaf ${leafIdx} references missing dTree ${node.dTreeIndex}`);
  }
  // Descend dTree to the leaf covering octUV and accumulate flux (dTree.ts).
  dTreeAccumulateFlux(dTree, octUV, flux);
  // Publish the count only after the directional deposit succeeds. Invalid
  // flux/UV/topology must leave both halves of the training sample unchanged.
  node.sampleCount += 1;
}

// ────────────────────────────────────────────────────────────────────────────
// Rebuild: adaptive split after a training frame
// ────────────────────────────────────────────────────────────────────────────

/**
 * Traverse all leaves of `sTree`; split any leaf whose `sampleCount` exceeds
 * `threshold` (default `PPG_CELL_SPLIT_THRESHOLD`).
 *
 * Splitting (Müller §3.1):
 *   1. Choose split axis = longest axis of the leaf AABB.
 *   2. Split value = midpoint along that axis.
 *   3. Create two child nodes, each inheriting a COPY of the parent dTree.
 *   4. Partition the parent's sampleCount in half between children (approximate;
 *      exact partitioning would require per-sample position replay which is
 *      not tracked). Both children start with `sampleCount = 0` and a clone
 *      of the parent dTree so the directional distribution is preserved.
 *   5. The parent node becomes an interior node.
 *
 * This is the CPU-side rebuild step described in Müller §5 ("tree is serialised
 * to a flat GPU buffer each rebuild cycle").
 *
 * @param sTree     The tree to mutate in place.
 * @param threshold Sample-count threshold (default `PPG_CELL_SPLIT_THRESHOLD`).
 * @param maxCells  Hard cap on total leaf count.
 * @param cellSampleCounts  A2 — OPTIONAL per-cell sample counts indexed by
 *   `dTreeIndex` (the GPU-side counter read back by `PPGCoordinator`). When
 *   supplied, each leaf's split decision uses `cellSampleCounts[node.dTreeIndex]`
 *   instead of the CPU-side `node.sampleCount` (which the production training
 *   path never writes — `sTreeAccumulate` has no GPU-path callers). When
 *   omitted, the historical `node.sampleCount` path is used unchanged (keeps the
 *   existing unit tests valid).
 */
export function splitOverflowLeaves(
  sTree: STree,
  threshold: number = PPG_CELL_SPLIT_THRESHOLD,
  maxCells: number = 16_384,
  cellSampleCounts?: ArrayLike<number>,
): void {
  if (!Number.isFinite(threshold) || threshold < 0) {
    throw new RangeError(`threshold must be finite and non-negative; got ${threshold}`);
  }
  if (!Number.isSafeInteger(maxCells) || maxCells < 1) {
    throw new RangeError(`maxCells must be a positive safe integer; got ${maxCells}`);
  }
  // Snapshot leaf count before we start (new leaves added during iteration
  // may themselves be over threshold — we defer them to the next rebuild cycle).
  const initialLen = sTree.nodes.length;
  // Validate the complete source epoch before the first split. A malformed late
  // counter must not leave earlier leaves already subdivided.
  for (let i = 0; i < initialLen; i++) {
    const node = sTree.nodes[i]!;
    if (node.splitAxis !== -1) continue;
    if (!Number.isSafeInteger(node.dTreeIndex)
        || node.dTreeIndex < 0
        || node.dTreeIndex >= sTree.dTrees.length) {
      throw new RangeError(`sTree leaf ${i} references invalid dTree ${node.dTreeIndex}`);
    }
    const sampleCount = cellSampleCounts !== undefined
      ? (cellSampleCounts[node.dTreeIndex] ?? 0)
      : node.sampleCount;
    if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
      throw new RangeError(
        `sTree leaf ${i} has an invalid sample count ${sampleCount}`,
      );
    }
  }
  let leafCount = countLeaves(sTree);

  for (let i = 0; i < initialLen; i++) {
    const node = sTree.nodes[i]!;
    if (node.splitAxis !== -1) continue;          // interior node
    // A2 — prefer the externally-supplied GPU sample count for this cell when
    // available; fall back to the (CPU-path) node.sampleCount otherwise.
    const sampleCount = cellSampleCounts !== undefined
      ? (cellSampleCounts[node.dTreeIndex] ?? 0)
      : node.sampleCount;
    if (sampleCount <= threshold) continue;       // not over threshold
    if (leafCount >= maxCells) break;             // hard cap

    // Split this leaf.
    const axis = longestAxis(node.aabb);
    const splitVal = (node.aabb.min[axis] + node.aabb.max[axis]) * 0.5;

    // Both children inherit a COPY of the parent's directional distribution
    // (Müller §3.1). REUSE the parent's dTree slot for the left child instead
    // of orphaning it: the parent leaf is being promoted to an interior node
    // and no longer needs its own dTree, so we hand that slot to the left
    // child and push exactly ONE new dTree for the right child. This keeps
    // `dTrees.length === leafCount` (each split is net +1 leaf and +1 dTree),
    // which is what the serialise + GPU-buffer-capacity bound assumes.
    //
    // The previous scheme pushed TWO new dTrees and left the parent's slot
    // dangling, so `dTrees.length` grew ~2× faster than the live leaf count —
    // overflowing the GPU dTree buffer (sized for `maxCells` cells) and
    // leaking orphan dTrees into the array unbounded. Sampling is unchanged:
    // both children still descend a clone of the parent distribution; only
    // the internal array index of the left child's dTree differs (traversal
    // reaches it via the node's `dTreeIndex`, never by raw position).
    const parentDTreeIdx = node.dTreeIndex;
    const parentDTree = sTree.dTrees[parentDTreeIdx]!;
    const leftDTreeIdx  = parentDTreeIdx;
    const rightDTreeIdx = sTree.dTrees.length;
    sTree.dTrees[leftDTreeIdx] = cloneDTree(parentDTree);
    sTree.dTrees.push(cloneDTree(parentDTree));

    // Build child AABBs.
    const leftAABB  = cloneAABB(node.aabb);
    const rightAABB = cloneAABB(node.aabb);
    leftAABB.max[axis]  = splitVal;
    rightAABB.min[axis] = splitVal;

    const leftIdx  = sTree.nodes.length;
    const rightIdx = leftIdx + 1;

    sTree.nodes.push({
      aabb: leftAABB,
      splitAxis: -1,
      splitValue: 0,
      leftChild: -1,
      rightChild: -1,
      dTreeIndex: leftDTreeIdx,
      sampleCount: 0,
    });
    sTree.nodes.push({
      aabb: rightAABB,
      splitAxis: -1,
      splitValue: 0,
      leftChild: -1,
      rightChild: -1,
      dTreeIndex: rightDTreeIdx,
      sampleCount: 0,
    });

    // Promote parent to interior node.
    node.splitAxis  = axis;
    node.splitValue = splitVal;
    node.leftChild  = leftIdx;
    node.rightChild = rightIdx;
    node.dTreeIndex = -1;
    node.sampleCount = 0;

    leafCount++;
  }
}

/** Count the number of leaf nodes in the sTree. */
function countLeaves(sTree: STree): number {
  let n = 0;
  for (const node of sTree.nodes) {
    if (node.splitAxis === -1) n++;
  }
  return n;
}

/** Deep-clone a dTree (used when splitting an sTree leaf). */
function cloneDTree(src: DTree): DTree {
  return {
    nodes: src.nodes.map(n => ({ ...n })),
    totalFlux: src.totalFlux,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Reset accumulators after rebuild
// ────────────────────────────────────────────────────────────────────────────

/**
 * CPU oracle/test helper: reset all leaf sampleCounts and all dTree flux
 * accumulators.
 *
 * The production GPU-readback path deliberately keeps persistent dTree flux
 * with decay after merging per-frame readback; do not read this helper as the
 * current runtime training cadence.
 */
export function resetAccumulators(sTree: STree): void {
  for (const node of sTree.nodes) {
    if (node.splitAxis === -1) {
      node.sampleCount = 0;
    }
  }
  for (const dTree of sTree.dTrees) {
    dTree.totalFlux = 0;
    for (const dn of dTree.nodes) {
      dn.flux = 0;
    }
  }
}

/**
 * RUNAWAY FIX — Müller §5 per-iteration flux DECAY (replaces the full reset on
 * the persistent CPU accumulator).
 *
 * The training loop deposits initial-reservoir histogram mass `w_sum / M`
 * summed over selected samples in each window.
 * If the trained flux were simply ACCUMULATED across windows with no reset and
 * no decay, the total grows without bound (linear in window count — verified in
 * the CPU harness: `last/win6 = 2.31x` and climbing). The full reset
 * (`resetAccumulators`) bounds it but throws away ALL temporal history every
 * window → high window-to-window variance and a cold restart each cycle.
 *
 * Müller's practical scheme keeps the SD-tree structure across iterations and
 * combines successive iterations' statistics; vitrum models that as a geometric
 * decay of the persistent flux accumulator: each window scales the carried-over
 * flux by `decay ∈ [0,1)` BEFORE the new window's deposits are added. Under a
 * steady per-window input F this converges to the bounded geometric steady
 * state `F / (1 − decay)` (decay 0.5 ⇒ 2F), instead of diverging — see
 * `__tests__/ppgRunawayBound.test.ts`.
 *
 * `decay === 0` is exactly the historical full reset (no carry-over);
 * `decay === 1` is the divergent no-decay accumulation (rejected). The PPG
 * coordinator uses {@link PPG_FLUX_DECAY} (0.5).
 *
 * sTree leaf `sampleCount`s are split-decision statistics, not radiance; they
 * are produced fresh by the GPU counter each window (A2) and are NOT decayed
 * here (they are zeroed so the next window's GPU readback is the sole source).
 */
export function decayAccumulators(sTree: STree, decay: number): void {
  if (!Number.isFinite(decay) || decay < 0 || decay >= 1) {
    throw new RangeError(`decay must be finite and inside [0,1); got ${decay}`);
  }
  for (const node of sTree.nodes) {
    if (!Number.isSafeInteger(node.sampleCount) || node.sampleCount < 0) {
      throw new RangeError(`sTree contains invalid sample count ${node.sampleCount}`);
    }
  }
  for (const dTree of sTree.dTrees) {
    if (!Number.isFinite(dTree.totalFlux) || dTree.totalFlux < 0) {
      throw new RangeError(`sTree contains invalid dTree total flux ${dTree.totalFlux}`);
    }
    for (const node of dTree.nodes) {
      if (!Number.isFinite(node.flux) || node.flux < 0) {
        throw new RangeError(`sTree contains invalid dTree node flux ${node.flux}`);
      }
    }
  }
  const d = decay;
  for (const node of sTree.nodes) {
    if (node.splitAxis === -1) {
      node.sampleCount = 0;
    }
  }
  for (const dTree of sTree.dTrees) {
    dTree.totalFlux *= d;
    for (const dn of dTree.nodes) {
      dn.flux *= d;
    }
  }
}
