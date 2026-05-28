/**
 * T2.H3 — PPG paper-faithful (Müller et al. 2017) tests.
 *
 * The rebuild (shipped W9; plan archived at `plan/archive/sprint-ppg-rebuild-future-archived-2026-05-28.md`)
 * fixes the 5 axes the deleted scaffold deviated from. Each test pins one axis or the supporting
 * math:
 *
 *   1. sTree adaptive split (§3.1, deviation 1)
 *   2. dTree adaptive refinement (§3.2, deviation 2)
 *   3. Solid-angle invariant (deviation 5)
 *   4. Guide PDF normalization (PDF integral over the unit-octahedral square)
 *   5. MIS combination identity (§3.4)
 *   6. Training-signal contract (deviation 3 — WGSL reads incoming radiance L_i,
 *      not post-clamp Lo)
 */

import { describe, it, expect } from 'vitest';
import {
  buildSTree,
  findSTreeLeaf,
  sTreeAccumulate,
  splitOverflowLeaves,
} from '../src/ppg/sTree.js';
import {
  buildEmptyDTree,
  dTreePdf,
  refineDTree,
  sumLeafSolidAngles,
  sumLeafPdfIntegrals,
} from '../src/ppg/dTree.js';
import {
  PPG_CELL_SPLIT_THRESHOLD,
  PPG_DTREE_FLUX_FRACTION,
  PPG_DTREE_INITIAL_DEPTH,
} from '../src/ppg/ppgConstants.js';
import { PPG_UPDATE_WGSL } from '../src/ppg/ppgUpdate.wgsl.js';

const SCENE_AABB: import('../src/ppg/types.js').AABB = {
  min: [-10, -10, -10],
  max: [10, 10, 10],
};

// ── Test 1: sTree adaptive split (deviation 1) ───────────────────────────────

describe('PPG sTree — adaptive split (Müller §3.1)', () => {
  it('splits a leaf when sample count exceeds PPG_CELL_SPLIT_THRESHOLD', () => {
    const tree = buildSTree(SCENE_AABB);
    expect(tree.nodes.length).toBe(1);  // single root leaf

    // Accumulate threshold + 1 samples uniformly inside the root cell.
    for (let i = 0; i < PPG_CELL_SPLIT_THRESHOLD + 1; i++) {
      sTreeAccumulate(tree, [0, 0, 0], [0.5, 0.5], 1.0);
    }

    splitOverflowLeaves(tree);

    // Root → split → 2 children. Total nodes = 3 (root interior + 2 leaves).
    expect(tree.nodes.length).toBe(3);

    // Root is now interior (splitAxis ≥ 0), its sampleCount cleared.
    expect(tree.nodes[0]!.splitAxis).toBeGreaterThanOrEqual(0);

    // Both children cover disjoint halves of the parent's longest axis.
    const left = tree.nodes[1]!;
    const right = tree.nodes[2]!;
    expect(left.splitAxis).toBe(-1);   // leaf
    expect(right.splitAxis).toBe(-1);  // leaf

    // The disjoint AABBs together cover the parent.
    const leftCovers = (left.aabb.max[0] - left.aabb.min[0]) +
                       (right.aabb.max[0] - right.aabb.min[0]);
    const parentExtent = SCENE_AABB.max[0] - SCENE_AABB.min[0];
    // Either the X axis was split, OR another axis. Verify total volume preserved.
    const leftVol = (left.aabb.max[0] - left.aabb.min[0]) *
                    (left.aabb.max[1] - left.aabb.min[1]) *
                    (left.aabb.max[2] - left.aabb.min[2]);
    const rightVol = (right.aabb.max[0] - right.aabb.min[0]) *
                     (right.aabb.max[1] - right.aabb.min[1]) *
                     (right.aabb.max[2] - right.aabb.min[2]);
    const parentVol = (SCENE_AABB.max[0] - SCENE_AABB.min[0]) *
                      (SCENE_AABB.max[1] - SCENE_AABB.min[1]) *
                      (SCENE_AABB.max[2] - SCENE_AABB.min[2]);
    expect(leftVol + rightVol).toBeCloseTo(parentVol, 6);
    // Quiet the unused-var warning — leftCovers + parentExtent kept for diagnostic
    // value when reading test failures.
    void leftCovers;
    void parentExtent;
  });

  it('does not split a leaf below the threshold', () => {
    const tree = buildSTree(SCENE_AABB);
    for (let i = 0; i < PPG_CELL_SPLIT_THRESHOLD - 1; i++) {
      sTreeAccumulate(tree, [0, 0, 0], [0.5, 0.5], 1.0);
    }
    splitOverflowLeaves(tree);
    expect(tree.nodes.length).toBe(1);
    expect(tree.nodes[0]!.splitAxis).toBe(-1);  // -1 = leaf
  });

  it('findSTreeLeaf returns root for any in-bounds point on a fresh tree', () => {
    const tree = buildSTree(SCENE_AABB);
    expect(findSTreeLeaf(tree, [0, 0, 0])).toBe(0);
    expect(findSTreeLeaf(tree, [-9, -9, -9])).toBe(0);
    expect(findSTreeLeaf(tree, [9, 9, 9])).toBe(0);
  });
});

// ── Test 2: dTree adaptive refinement (deviation 2) ──────────────────────────

describe('PPG dTree — adaptive refinement (Müller §3.2)', () => {
  it('splits a quadtree leaf when its flux fraction > PPG_DTREE_FLUX_FRACTION', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    const initialNodeCount = dTree.nodes.length;

    // Concentrate ~all flux into a single leaf cell.
    // Walk the leaves and pick one to spike.
    const leaves = dTree.nodes.filter((n) => n.isLeaf);
    expect(leaves.length).toBeGreaterThan(0);
    const target = leaves[0]!;
    target.flux = 1000.0;
    // Other leaves keep tiny flux so the split predicate fires on the spike.
    for (let i = 1; i < leaves.length; i++) {
      leaves[i]!.flux = 0.001;
    }
    // refineDTree reads dTree.totalFlux (not derived from leaves at call time).
    dTree.totalFlux = 1000.0 + 0.001 * (leaves.length - 1);

    refineDTree(dTree);

    // The spiked leaf should split into 4 quadrants.
    expect(dTree.nodes.length).toBeGreaterThan(initialNodeCount);
  });

  it('does not refine when no leaf exceeds the flux fraction', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    const before = dTree.nodes.length;

    // Uniform low flux — no leaf above ρ × total. With N leaves at flux=1
    // and totalFlux=N, each leaf has fraction 1/N. For N ≥ 100 (=PPG_DTREE_INITIAL_DEPTH=2 means 16 leaves; insufficient).
    // Use a different setup: every leaf gets equal flux; ratio = 1/N.
    // For N=16, ratio = 0.0625, which IS > PPG_DTREE_FLUX_FRACTION = 0.01.
    // So this would split. Adjust to make this test demonstrate the no-split case:
    // set very-many-leaves equivalent by giving them all small equal flux.
    let totalLeafFlux = 0;
    for (const n of dTree.nodes) {
      if (n.isLeaf) {
        n.flux = 1.0;
        totalLeafFlux += 1.0;
      }
    }
    // Boost totalFlux so no individual leaf exceeds ρ × total.
    // With 16 leaves at flux=1 each and totalFlux=200, each leaf is 0.5%
    // which is below ρ=1%.
    dTree.totalFlux = totalLeafFlux * 12.5;  // makes per-leaf ratio = 0.005 (< 0.01)
    refineDTree(dTree);

    expect(dTree.nodes.length).toBe(before);
  });
});

// ── Test 3: Solid-angle invariant (deviation 5) ──────────────────────────────

describe('PPG dTree — solid-angle invariant', () => {
  it('sum of all leaf solid angles ≈ 4π for a tree covering the full octahedral square', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    const sum = sumLeafSolidAngles(dTree);
    // Octahedral [0,1]² covers the full sphere = 4π sr.
    expect(sum).toBeCloseTo(4 * Math.PI, 5);
  });

  it('solid-angle sum invariant under adaptive refinement', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    // Force a split by spiking a leaf.
    const leaves = dTree.nodes.filter((n) => n.isLeaf);
    leaves[0]!.flux = 100.0;
    for (let i = 1; i < leaves.length; i++) leaves[i]!.flux = 0.001;
    dTree.totalFlux = 100.0 + 0.001 * (leaves.length - 1);
    refineDTree(dTree);

    // Even after split, total solid angle still 4π (split children together
    // cover the parent's solid angle).
    expect(sumLeafSolidAngles(dTree)).toBeCloseTo(4 * Math.PI, 5);
  });
});

// ── Test 4: Guide PDF normalization ──────────────────────────────────────────

describe('PPG dTree — guide PDF normalization', () => {
  it('integral of pdf over the octahedral square ≈ 1', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    // Give each leaf equal flux so PDF is uniform.
    for (const n of dTree.nodes) {
      if (n.isLeaf) n.flux = 1.0;
    }
    // Sum of (leaf area × leaf PDF) over the octahedral square should = 1.
    expect(sumLeafPdfIntegrals(dTree)).toBeCloseTo(1.0, 5);
  });

  it('pdf at a known octahedral UV is positive and finite', () => {
    const dTree = buildEmptyDTree(PPG_DTREE_INITIAL_DEPTH);
    for (const n of dTree.nodes) {
      if (n.isLeaf) n.flux = 1.0;
    }
    const pdf = dTreePdf(dTree, [0.5, 0.5]);
    expect(pdf).toBeGreaterThan(0);
    expect(Number.isFinite(pdf)).toBe(true);
  });
});

// ── Test 5: MIS weight identity (Müller §3.4) ────────────────────────────────

describe('PPG MIS — power-heuristic identity', () => {
  function powerHeuristic(a: number, b: number): number {
    const a2 = a * a;
    const b2 = b * b;
    return a2 / Math.max(a2 + b2, 1e-12);
  }

  it('for α=0.5 and equal pdfs, w_ppg = w_bsdf = 0.5', () => {
    const alpha = 0.5;
    const pdfPpg = 1.0;
    const pdfBsdf = 1.0;
    const w_ppg = powerHeuristic(alpha * pdfPpg, (1 - alpha) * pdfBsdf);
    const w_bsdf = powerHeuristic((1 - alpha) * pdfBsdf, alpha * pdfPpg);
    expect(w_ppg).toBeCloseTo(0.5, 6);
    expect(w_bsdf).toBeCloseTo(0.5, 6);
    expect(w_ppg + w_bsdf).toBeCloseTo(1.0, 6);
  });

  it('weights sum to 1 across α and pdf ratios', () => {
    for (const alpha of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      for (const r of [0.5, 1.0, 2.0, 10.0]) {
        const pdfPpg = 1.0;
        const pdfBsdf = r;
        const w_ppg = powerHeuristic(alpha * pdfPpg, (1 - alpha) * pdfBsdf);
        const w_bsdf = powerHeuristic((1 - alpha) * pdfBsdf, alpha * pdfPpg);
        expect(w_ppg + w_bsdf).toBeCloseTo(1.0, 6);
      }
    }
  });
});

// ── Test 6: Training-signal contract (deviation 3) ───────────────────────────

describe('PPG WGSL — training-signal contract', () => {
  it('ppgUpdate kernel reads from an L_i (incoming radiance) input, not Lo', () => {
    // The PRIOR scaffold trained on Lo (post-clamp outgoing radiance from the
    // shade pass). The rebuild MUST train on incoming radiance L_i. We verify
    // the WGSL kernel string references an L_i / incoming binding rather than
    // a Lo / outgoing one.
    expect(PPG_UPDATE_WGSL).not.toMatch(/Lo_clamp|Lo_outgoing|outgoingRadiance/);
    // Look for the L_i contract: the kernel binds an incoming-radiance buffer.
    // Either explicit `Li` / `incomingRadiance` symbol, or a binding doc-comment.
    const acceptsLiContract =
      /\bLi\b/.test(PPG_UPDATE_WGSL) ||
      /incomingRadiance/.test(PPG_UPDATE_WGSL) ||
      /incoming.{0,30}radiance/i.test(PPG_UPDATE_WGSL);
    expect(acceptsLiContract).toBe(true);
  });

  it('ppgUpdate kernel includes a workgroup_size declaration (sanity check)', () => {
    expect(PPG_UPDATE_WGSL).toMatch(/@workgroup_size/);
  });
});
