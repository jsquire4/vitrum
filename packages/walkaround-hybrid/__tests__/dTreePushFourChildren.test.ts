/**
 * dTreePushFourChildren.test.ts — Task 4.5 Theme-I behavior pin for the dTree
 * `pushFourChildren` helper extraction (#6).
 *
 * buildSubtree + buildSubtreeChildrenOnly both computed identical child extents
 * + pushed four child nodes with identical shape; that push is now factored into
 * a shared `pushFourChildren` helper. The build must remain BIT-IDENTICAL — same
 * node array, same firstChild wiring, same per-leaf solid angles, same BFS-by-
 * level / DFS-by-subtree append order.
 *
 * GOLDEN: the full DTreeNode array for several initial depths is pinned here
 * (computed from the spec — 4^d leaves, consecutive children, octahedral patch
 * solid angles). If the refactor perturbs ANY field of ANY node this fails.
 */

import { describe, it, expect } from 'vitest';
import { buildEmptyDTree } from '../src/ppg/dTree.js';
import type { DTreeNode } from '../src/ppg/types.js';

const FOUR_PI = 4 * Math.PI;

// Independent reference builder (the layout contract, computed a DIFFERENT way
// than the production code so this is a true oracle, not a copy). Builds the
// node array level-by-level: at each level push 4 children per non-leaf parent,
// patching firstChild. This reproduces the documented BFS-by-level /
// DFS-by-subtree order via an explicit recursion identical in OUTPUT.
function refBuild(maxDepth: number): DTreeNode[] {
  const nodes: DTreeNode[] = [];
  function rec(u0: number, u1: number, v0: number, v1: number, depth: number): number {
    const idx = nodes.length;
    const isLeaf = depth >= maxDepth;
    nodes.push({
      isLeaf, u0, v0, u1, v1,
      solidAngle: isLeaf ? FOUR_PI * (u1 - u0) * (v1 - v0) : -1,
      flux: 0, firstChild: -1, depth,
    });
    if (isLeaf) return idx;
    const uMid = (u0 + u1) * 0.5, vMid = (v0 + v1) * 0.5;
    const ext: Array<[number, number, number, number]> = [
      [u0, uMid, v0, vMid], [uMid, u1, v0, vMid],
      [u0, uMid, vMid, v1], [uMid, u1, vMid, v1],
    ];
    const childIsLeaf = (depth + 1) >= maxDepth;
    const firstChild = nodes.length;
    nodes[idx]!.firstChild = firstChild;
    for (let ci = 0; ci < 4; ci++) {
      const [cu0, cu1, cv0, cv1] = ext[ci]!;
      nodes.push({
        isLeaf: childIsLeaf, u0: cu0, u1: cu1, v0: cv0, v1: cv1,
        solidAngle: childIsLeaf ? FOUR_PI * (cu1 - cu0) * (cv1 - cv0) : -1,
        flux: 0, firstChild: -1, depth: depth + 1,
      });
    }
    if (!childIsLeaf) {
      for (let ci = 0; ci < 4; ci++) {
        const [cu0, cu1, cv0, cv1] = ext[ci]!;
        const childIdx = firstChild + ci;
        nodes[childIdx]!.firstChild = nodes.length;
        recChildrenOnly(cu0, cu1, cv0, cv1, depth + 1);
      }
    }
    return idx;
  }
  function recChildrenOnly(u0: number, u1: number, v0: number, v1: number, depth: number): void {
    const uMid = (u0 + u1) * 0.5, vMid = (v0 + v1) * 0.5;
    const childIsLeaf = (depth + 1) >= maxDepth;
    const ext: Array<[number, number, number, number]> = [
      [u0, uMid, v0, vMid], [uMid, u1, v0, vMid],
      [u0, uMid, vMid, v1], [uMid, u1, vMid, v1],
    ];
    const firstChild = nodes.length;
    for (let ci = 0; ci < 4; ci++) {
      const [cu0, cu1, cv0, cv1] = ext[ci]!;
      nodes.push({
        isLeaf: childIsLeaf, u0: cu0, u1: cu1, v0: cv0, v1: cv1,
        solidAngle: childIsLeaf ? FOUR_PI * (cu1 - cu0) * (cv1 - cv0) : -1,
        flux: 0, firstChild: -1, depth: depth + 1,
      });
    }
    if (!childIsLeaf) {
      for (let ci = 0; ci < 4; ci++) {
        const [cu0, cu1, cv0, cv1] = ext[ci]!;
        const childIdx = firstChild + ci;
        nodes[childIdx]!.firstChild = nodes.length;
        recChildrenOnly(cu0, cu1, cv0, cv1, depth + 1);
      }
    }
  }
  rec(0, 1, 0, 1, 0);
  return nodes;
}

describe('dTree pushFourChildren — build stays bit-identical (Task 4.5 #6)', () => {
  for (const depth of [0, 1, 2, 3, 4]) {
    it(`buildEmptyDTree(${depth}) produces the golden node array (4^${depth} leaves)`, () => {
      const built = buildEmptyDTree(depth);
      const golden = refBuild(depth);

      expect(built.totalFlux).toBe(0);
      expect(built.nodes.length).toBe(golden.length);
      // Total node count: Σ_{k=0..depth} 4^k.
      const expectedCount = depth === 0 ? 1 : (Math.pow(4, depth + 1) - 1) / 3;
      expect(built.nodes.length).toBe(expectedCount);
      // Full-array structural equality (every field of every node).
      expect(built.nodes).toEqual(golden);

      // Leaf count = 4^depth, each leaf's solid angle sums to 4π.
      const leaves = built.nodes.filter(n => n.isLeaf);
      expect(leaves.length).toBe(Math.pow(4, depth));
      const totalSA = leaves.reduce((a, n) => a + n.solidAngle, 0);
      expect(totalSA).toBeCloseTo(FOUR_PI, 9);

      // Consecutive-children invariant: every interior node's children are at
      // firstChild..firstChild+3 with the matching depth.
      for (const n of built.nodes) {
        if (n.isLeaf) continue;
        expect(n.firstChild).toBeGreaterThanOrEqual(0);
        for (let ci = 0; ci < 4; ci++) {
          expect(built.nodes[n.firstChild + ci]!.depth).toBe(n.depth + 1);
        }
      }
    });
  }
});
