/**
 * A5 — dTree orphan-growth regression.
 *
 * Bug: `refineDTree`'s merge pass marks merged children defunct
 * (`isLeaf=false; firstChild=-1`) but never removes them from the flat
 * `dTree.nodes` array, while the split pass appends. Across repeated
 * split/merge refine cycles the array grows without bound and accumulates
 * orphan nodes that are unreachable from the root — polluting the flat array
 * that the GPU flux readback clamps at MAX_DTREE_NODES_PER_CELL.
 *
 * Fix: after the split+merge passes, `refineDTree` compacts `dTree.nodes` to a
 * fresh array (DFS from root, re-emitting only reachable nodes and re-patching
 * `firstChild` to the consecutive-children invariant `buildSubtree` documents).
 *
 * These tests pin:
 *   1. `nodes.length` stays bounded by the true reachable-node count across
 *      ≥20 split/merge cycles (the regression — fails on the pre-compaction code).
 *   2. No orphan / defunct nodes survive (every node is reachable from root).
 *   3. The consecutive-children layout invariant holds post-compaction.
 *   4. Sampling behaviour of live leaves (flux/solidAngle) is preserved.
 */

import { describe, it, expect } from 'vitest';
import {
  buildEmptyDTree,
  refineDTree,
  dTreePdf,
  sumLeafSolidAngles,
} from '../src/ppg/dTree.js';
import { PPG_DTREE_INITIAL_DEPTH } from '../src/ppg/ppgConstants.js';
import type { DTree } from '../src/ppg/types.js';

const FOUR_PI = 4 * Math.PI;

/**
 * Count the nodes actually reachable from the root by following `firstChild`
 * links. This is the "true" node count the flat array should hold once orphans
 * are compacted away. Walks the consecutive-children block [c, c+4).
 */
function countReachableNodes(dTree: DTree): number {
  let count = 0;
  const stack: number[] = [0];
  while (stack.length > 0) {
    const idx = stack.pop()!;
    const node = dTree.nodes[idx];
    if (!node) continue;
    count++;
    if (!node.isLeaf && node.firstChild >= 0) {
      for (let ci = 0; ci < 4; ci++) stack.push(node.firstChild + ci);
    }
  }
  return count;
}

/**
 * Detect orphan / malformed nodes: a node that is neither a valid leaf nor a
 * valid interior node. The pre-fix merge pass leaves children as
 * `{ isLeaf:false, firstChild:-1 }` — an "interior node with no children",
 * which is exactly this malformed shape.
 */
function countMalformedNodes(dTree: DTree): number {
  let n = 0;
  for (const node of dTree.nodes) {
    if (!node.isLeaf && node.firstChild < 0) n++;
  }
  return n;
}

/** Drive one split-then-merge refine cycle deterministically.
 *
 * 1. Spike one live leaf's flux far above the split fraction → forces a split.
 * 2. Refine (split pass fires).
 * 3. Flatten flux to near-uniform-tiny → forces the merge pass to collapse
 *    freshly-split sibling groups back to their parent.
 * 4. Refine again (merge pass fires).
 */
function refineCycle(dTree: DTree, spikeMagnitude: number): void {
  // ── Phase A: spike → split ──
  const leaves = dTree.nodes.filter((n) => n.isLeaf);
  if (leaves.length > 0) {
    leaves[0]!.flux = spikeMagnitude;
    for (let i = 1; i < leaves.length; i++) leaves[i]!.flux = 0.0001;
  }
  let total = 0;
  for (const n of dTree.nodes) if (n.isLeaf) total += n.flux;
  dTree.totalFlux = total;
  refineDTree(dTree);

  // ── Phase B: flatten → merge ──
  // Give every live leaf the same tiny flux so each child sits below the merge
  // fraction, collapsing newly-split groups back to leaves.
  const leaves2 = dTree.nodes.filter((n) => n.isLeaf);
  let total2 = 0;
  for (const n of leaves2) {
    n.flux = 1.0;
    total2 += 1.0;
  }
  // Inflate totalFlux so each leaf's fraction drops below the merge threshold.
  dTree.totalFlux = total2 * 1e6;
  refineDTree(dTree);
}

describe('PPG dTree — refine compaction (A5 orphan-growth)', () => {
  it('keeps nodes.length bounded by the reachable-node count across ≥20 refine cycles', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);

    for (let cycle = 0; cycle < 24; cycle++) {
      refineCycle(dTree, 1000);

      const reachable = countReachableNodes(dTree);
      // The flat array must hold EXACTLY the reachable nodes — no orphans
      // accumulated, and not monotonically growing. Pre-fix this fails: the
      // merge pass strands children in the array, so length > reachable and
      // grows each cycle.
      expect(dTree.nodes.length).toBe(reachable);
    }

    // Final upper bound: with INITIAL_DEPTH=2 (21 nodes) and bounded refinement,
    // a compacted tree stays small. Pre-fix the array balloons past this.
    expect(dTree.nodes.length).toBeLessThanOrEqual(200);
  });

  it('leaves no orphan / defunct (isLeaf=false, firstChild=-1) nodes after refinement', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    for (let cycle = 0; cycle < 24; cycle++) refineCycle(dTree, 1000);
    expect(countMalformedNodes(dTree)).toBe(0);
  });

  it('preserves the consecutive-children layout invariant after compaction', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    for (let cycle = 0; cycle < 12; cycle++) refineCycle(dTree, 1000);

    for (let i = 0; i < dTree.nodes.length; i++) {
      const node = dTree.nodes[i]!;
      if (node.isLeaf) continue;
      // Interior node: firstChild must point to a valid in-range consecutive
      // block of 4 children.
      expect(node.firstChild).toBeGreaterThanOrEqual(0);
      expect(node.firstChild + 3).toBeLessThan(dTree.nodes.length);
      // Children's depth must be exactly one deeper.
      for (let ci = 0; ci < 4; ci++) {
        expect(dTree.nodes[node.firstChild + ci]!.depth).toBe(node.depth + 1);
      }
    }
  });

  it('preserves solid-angle invariant (Σ leaf solid angle ≈ 4π) post-compaction', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    for (let cycle = 0; cycle < 12; cycle++) refineCycle(dTree, 1000);
    expect(sumLeafSolidAngles(dTree)).toBeCloseTo(FOUR_PI, 5);
  });

  it('preserves live-leaf sampling: a spiked direction keeps a positive finite PDF', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    // One split cycle so the tree is refined but not flattened to root.
    const leaves = dTree.nodes.filter((n) => n.isLeaf);
    leaves[0]!.flux = 1000;
    for (let i = 1; i < leaves.length; i++) leaves[i]!.flux = 1;
    let total = 0;
    for (const n of dTree.nodes) if (n.isLeaf) total += n.flux;
    dTree.totalFlux = total;
    refineDTree(dTree);

    // PDF at the centre of the spiked region must be positive + finite — the
    // compaction must not have stranded the live leaves.
    const pdf = dTreePdf(dTree, [0.25, 0.25]);
    expect(pdf).toBeGreaterThan(0);
    expect(Number.isFinite(pdf)).toBe(true);
  });
});
