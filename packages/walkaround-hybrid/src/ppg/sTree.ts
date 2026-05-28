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
import { buildEmptyDTree } from './dTree.js';

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
export function buildSTree(sceneBounds: AABB): STree {
  const rootDTree: DTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
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
 * Record that a training sample at `position` contributed `flux` to the
 * directional direction encoded in `octUV` (octahedral UV in [0,1]²).
 *
 * Called after each path bounce:
 *   - `position` = sample position in WORLD space
 *   - `octUV`    = octahedral map of the INCOMING direction ωi (world frame)
 *   - `flux`     = L_i estimate (incoming radiance at the sample point)
 *
 * DEVIATION 3 FIX: the deposit signal is the incoming radiance L_i (path
 * throughput estimate before the BRDF multiply), NOT the shade-pass outgoing
 * radiance L_o.
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
  const leafIdx = findSTreeLeaf(sTree, position);
  const node = sTree.nodes[leafIdx]!;
  node.sampleCount += 1;

  const dTree = sTree.dTrees[node.dTreeIndex]!;
  // Descend dTree to find the leaf covering octUV, accumulate flux there.
  dTreeAccumulateFlux(dTree, octUV, flux);
}

/**
 * Inlined here (NOT a re-export from dTree.ts) to avoid the circular
 * import sTree↔dTree would otherwise create. Mirrors the same descent
 * as `findDTreeLeaf` in dTree.ts; if that traversal changes, this copy
 * must too.
 */
function dTreeAccumulateFlux(
  dTree: DTree,
  octUV: [number, number],
  flux: number,
): void {
  // Inline traversal to avoid circular import: descend the quadtree.
  let idx = 0;
  while (true) {
    const node = dTree.nodes[idx]!;
    if (node.isLeaf) {
      node.flux += flux;
      dTree.totalFlux += flux;
      return;
    }
    // Children: NW, NE, SW, SE quadrants of the parent patch.
    const uMid = (node.u0 + node.u1) * 0.5;
    const vMid = (node.v0 + node.v1) * 0.5;
    const goRight = octUV[0] >= uMid;
    const goDown  = octUV[1] >= vMid;
    // firstChild ordering: 0=NW(left,top), 1=NE(right,top), 2=SW(left,bot), 3=SE(right,bot)
    idx = node.firstChild + (goDown ? 2 : 0) + (goRight ? 1 : 0);
  }
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
 */
export function splitOverflowLeaves(
  sTree: STree,
  threshold: number = PPG_CELL_SPLIT_THRESHOLD,
  maxCells: number = 16_384,
): void {
  // Snapshot leaf count before we start (new leaves added during iteration
  // may themselves be over threshold — we defer them to the next rebuild cycle).
  const initialLen = sTree.nodes.length;
  let leafCount = countLeaves(sTree);

  for (let i = 0; i < initialLen; i++) {
    const node = sTree.nodes[i]!;
    if (node.splitAxis !== -1) continue;          // interior node
    if (node.sampleCount <= threshold) continue;  // not over threshold
    if (leafCount >= maxCells) break;             // hard cap

    // Split this leaf.
    const axis = longestAxis(node.aabb);
    const splitVal = (node.aabb.min[axis] + node.aabb.max[axis]) * 0.5;

    // Clone parent dTree for each child.
    const parentDTree = sTree.dTrees[node.dTreeIndex]!;
    const leftDTreeIdx  = sTree.dTrees.length;
    const rightDTreeIdx = leftDTreeIdx + 1;
    sTree.dTrees.push(cloneDTree(parentDTree));
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
 * Reset all leaf sampleCounts and all dTree flux accumulators after serialising
 * the current tree to the GPU buffer. Called at the start of each training
 * iteration so the next frame accumulates fresh statistics.
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
