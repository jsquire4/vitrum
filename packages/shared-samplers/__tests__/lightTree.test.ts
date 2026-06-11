/**
 * lightTree.test.ts — Unit tests for Sprint 3 CPU-side light tree construction.
 *
 * Covers:
 *  - 1 emitter → trivial leaf tree
 *  - 2 emitters → 3 nodes (internal + 2 leaves), correct power sum
 *  - 4 emitters with linear power [1, 2, 4, 8] → root totalPower = 15,
 *    debug._powerPrefixSumDebug is monotonically increasing (values may exceed 1.0 — see
 *    buildLightTree JSDoc: it is an unnormalised node-power prefix-sum, not a true CDF)
 *  - Doubling all powers → all node totalPowers double (Sprint 2 round-trip)
 *  - packLightTreeForGPU → correct float layout
 *  - co-located centroids → power-based fallback split, no crash, balanced tree
 *
 * Deliberately does NOT test GPU binary-search traversal — that lives in
 * the fork's GLSL and is verified via the variance benchmark in
 * plan/sprint-3-benchmark.md.
 */

import { describe, it, expect } from 'vitest';
import { buildLightTree, packLightTreeForGPU } from '../src/lightTree.js';
import type { LightTreeBuildInput } from '../src/lightTree.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Unit AABB centred at a point — side length 1. */
function pointAabb(
  cx: number,
  cy: number,
  cz: number,
): { min: readonly [number, number, number]; max: readonly [number, number, number] } {
  return {
    min: [cx - 0.5, cy - 0.5, cz - 0.5],
    max: [cx + 0.5, cy + 0.5, cz + 0.5],
  };
}

/** Build minimal LightTreeBuildInput for N emitters at distinct positions. */
function makeInput(powers: number[]): LightTreeBuildInput {
  const _n = powers.length; // available for bounds assertions
  return {
    powers,
    centroids: powers.map((_, i) => [i * 2, 0, 0] as const),
    aabbs: powers.map((_, i) => pointAabb(i * 2, 0, 0)),
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('buildLightTree', () => {
  it('1 emitter → single leaf node', () => {
    const input = makeInput([5.0]);
    const { nodes, debug: { _powerPrefixSumDebug } } = buildLightTree(input);

    expect(nodes).toHaveLength(1);
    const node = nodes[0]!;

    // Leaf properties
    expect(node.emitterIndex).toBe(0);
    expect(node.totalPower).toBeCloseTo(5.0);
    expect(node.leftChild).toBe(-1);
    expect(node.rightChild).toBe(-1);

    // _powerPrefixSumDebug of a single node is [1.0] (root power / root power)
    expect(_powerPrefixSumDebug).toHaveLength(1);
    expect(_powerPrefixSumDebug[0]).toBeCloseTo(1.0);
  });

  it('2 emitters → 3 nodes (1 internal + 2 leaves), internal totalPower = sum', () => {
    const input = makeInput([3.0, 7.0]);
    const { nodes } = buildLightTree(input);

    // Binary tree with 2 leaves = 3 nodes total
    expect(nodes).toHaveLength(3);

    // Root (index 0) should be the internal node
    const root = nodes[0]!;
    expect(root.emitterIndex).toBe(-1);
    expect(root.totalPower).toBeCloseTo(10.0); // 3 + 7

    // Left and right children are leaves (emitterIndex ≥ 0)
    const leftChild = nodes[root.leftChild]!;
    const rightChild = nodes[root.rightChild]!;
    expect(leftChild.emitterIndex).toBeGreaterThanOrEqual(0);
    expect(rightChild.emitterIndex).toBeGreaterThanOrEqual(0);

    // Both children are leaves
    expect(leftChild.leftChild).toBe(-1);
    expect(leftChild.rightChild).toBe(-1);
    expect(rightChild.leftChild).toBe(-1);
    expect(rightChild.rightChild).toBe(-1);

    // Leaf powers sum to root
    const leafPowerSum = leftChild.totalPower + rightChild.totalPower;
    expect(leafPowerSum).toBeCloseTo(10.0);
  });

  it('4 emitters with powers [1,2,4,8] → root totalPower = 15, debug._powerPrefixSumDebug is monotonically increasing', () => {
    const input = makeInput([1, 2, 4, 8]);
    const { nodes, debug: { _powerPrefixSumDebug } } = buildLightTree(input);

    // 4 leaves → 7 nodes in a full binary tree
    expect(nodes).toHaveLength(7);

    const root = nodes[0]!;
    expect(root.totalPower).toBeCloseTo(15.0);

    // _powerPrefixSumDebug must be monotonically non-decreasing.
    // Values can exceed 1.0 because internal nodes aggregate subtree power and
    // their power is counted before each child's power — this is expected and
    // documented. The array is an unnormalised prefix-sum, not a true CDF.
    for (let i = 1; i < _powerPrefixSumDebug.length; i++) {
      expect(_powerPrefixSumDebug[i]!).toBeGreaterThanOrEqual(_powerPrefixSumDebug[i - 1]!);
    }

    // First entry > 0 (root power / root power — always exactly 1.0 for a positive tree).
    expect(_powerPrefixSumDebug[0]!).toBeCloseTo(1.0);
    // Final entry > 0 since all emitters have positive power.
    expect(_powerPrefixSumDebug[_powerPrefixSumDebug.length - 1]!).toBeGreaterThan(0);
  });

  it('doubling all input powers doubles every node totalPower (Sprint 2 round-trip)', () => {
    const basePowers = [1, 3, 2, 5];
    const { nodes: baseNodes } = buildLightTree(makeInput(basePowers));
    const { nodes: doubledNodes } = buildLightTree(makeInput(basePowers.map((p) => p * 2)));

    expect(doubledNodes).toHaveLength(baseNodes.length);
    for (let i = 0; i < baseNodes.length; i++) {
      expect(doubledNodes[i]!.totalPower).toBeCloseTo(baseNodes[i]!.totalPower * 2);
    }
  });

  it('throws if emitter arrays are empty', () => {
    expect(() =>
      buildLightTree({ powers: [], centroids: [], aabbs: [] }),
    ).toThrow();
  });

  it('throws on length mismatch between powers/centroids/aabbs', () => {
    expect(() =>
      buildLightTree({
        powers: [1, 2],
        centroids: [[0, 0, 0]],
        aabbs: [{ min: [0, 0, 0], max: [1, 1, 1] }],
      }),
    ).toThrow();
  });

  it('co-located centroids (degenerate span) → power-based split, no crash, balanced tree (L-2)', () => {
    // All 4 emitters share the same centroid — centroid-axis span = 0.
    // buildSubtree must fall back to power-based median split rather than
    // centroid sort. The resulting tree should be a valid binary tree with
    // 7 nodes (4 leaves + 3 internal) and root totalPower = sum of all powers.
    const powers = [1, 8, 2, 4];
    const sharedCentroid: readonly [number, number, number] = [0, 0, 0];
    const sharedAabb = { min: [-0.5, -0.5, -0.5] as const, max: [0.5, 0.5, 0.5] as const };
    const input: import('../src/lightTree.js').LightTreeBuildInput = {
      powers,
      centroids: powers.map(() => sharedCentroid),
      aabbs: powers.map(() => sharedAabb),
    };

    // Must not throw
    const { nodes, debug: { _powerPrefixSumDebug } } = buildLightTree(input);

    // Correct node count for 4 leaves
    expect(nodes).toHaveLength(7);

    // Root totalPower = 1 + 8 + 2 + 4 = 15
    expect(nodes[0]!.totalPower).toBeCloseTo(15.0);

    // _powerPrefixSumDebug is still monotonically non-decreasing
    for (let i = 1; i < _powerPrefixSumDebug.length; i++) {
      expect(_powerPrefixSumDebug[i]!).toBeGreaterThanOrEqual(_powerPrefixSumDebug[i - 1]!);
    }

    // All leaves have valid emitterIndex (≥ 0)
    const leaves = nodes.filter((n) => n.emitterIndex >= 0);
    expect(leaves).toHaveLength(4);

    // Every leaf power must match one of the input powers
    const leafPowers = leaves.map((l) => l.totalPower).sort((a, b) => a - b);
    expect(leafPowers).toEqual([1, 2, 4, 8]);

    // Each internal node's power equals the sum of its children's powers
    const internalNodes = nodes.filter((n) => n.emitterIndex === -1);
    for (const inode of internalNodes) {
      const leftPow = nodes[inode.leftChild]!.totalPower;
      const rightPow = nodes[inode.rightChild]!.totalPower;
      expect(inode.totalPower).toBeCloseTo(leftPow + rightPow);
    }
  });
});

// ── 33-F: Light-tree leaf PDF sums to 1 ──────────────────────────────────────

describe('33-F leaf PDF partition', () => {
  /**
   * Enumerate all leaves in a built tree and return their powers.
   * A leaf is any node with emitterIndex >= 0.
   */
  function leafPowers(nodes: ReturnType<typeof buildLightTree>['nodes']): number[] {
    return nodes.filter((n) => n.emitterIndex >= 0).map((n) => n.totalPower);
  }

  /**
   * Simple seeded linear-congruential PRNG returning [0, 1).
   * LCG parameters from Numerical Recipes.
   */
  function makeLcg(seed: number) {
    let s = seed >>> 0;
    return () => {
      s = Math.imul(1664525, s) + 1013904223;
      return (s >>> 0) / 0x1_0000_0000;
    };
  }

  it('leaf PDF partition sums to 1 — small (3 emitters)', () => {
    const { nodes } = buildLightTree(makeInput([1, 2, 3]));
    const root = nodes[0]!;
    const totalPower = root.totalPower;
    const pdfs = leafPowers(nodes).map((p) => p / totalPower);
    const sum = pdfs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5); // tolerance 1e-5
  });

  it('leaf PDF partition sums to 1 — medium (16 emitters)', () => {
    const powers = Array.from({ length: 16 }, (_, i) => i + 1);
    const { nodes } = buildLightTree(makeInput(powers));
    const totalPower = nodes[0]!.totalPower;
    const pdfs = leafPowers(nodes).map((p) => p / totalPower);
    const sum = pdfs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('leaf PDF partition sums to 1 — pathological (1000 emitters)', () => {
    // Use pseudo-random-looking powers (reproducible).
    const rng = makeLcg(0xdeadbeef);
    const powers = Array.from({ length: 1000 }, () => rng() * 10 + 0.001);
    const { nodes } = buildLightTree(makeInput(powers));
    const totalPower = nodes[0]!.totalPower;
    const pdfs = leafPowers(nodes).map((p) => p / totalPower);
    const sum = pdfs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it('zero-power emitter has PDF = 0 alongside non-zero emitters', () => {
    // Emitter at index 1 has power 0.
    const { nodes } = buildLightTree(makeInput([5, 0, 3, 7]));
    const totalPower = nodes[0]!.totalPower;
    const leaves = nodes.filter((n) => n.emitterIndex >= 0);
    const zeroLeaf = leaves.find((n) => n.totalPower === 0);
    expect(zeroLeaf).toBeDefined();
    expect(zeroLeaf!.totalPower / totalPower).toBe(0);
  });

  it('single-emitter tree: pdf_leaf[0] == 1', () => {
    const { nodes } = buildLightTree(makeInput([42]));
    const totalPower = nodes[0]!.totalPower;
    const leaves = nodes.filter((n) => n.emitterIndex >= 0);
    expect(leaves).toHaveLength(1);
    expect(leaves[0]!.totalPower / totalPower).toBeCloseTo(1.0, 10);
  });

  it('2-emitter tree with powers [3, 7]: pdf proportional to power', () => {
    const { nodes } = buildLightTree(makeInput([3, 7]));
    const totalPower = nodes[0]!.totalPower; // must be 10
    expect(totalPower).toBeCloseTo(10.0);
    const leaves = nodes.filter((n) => n.emitterIndex >= 0);
    // Sort by emitterIndex to get deterministic order.
    const sorted = leaves.slice().sort((a, b) => a.emitterIndex - b.emitterIndex);
    expect(sorted[0]!.totalPower / totalPower).toBeCloseTo(0.3, 10);
    expect(sorted[1]!.totalPower / totalPower).toBeCloseTo(0.7, 10);
  });

  it('CDF reconstruction: selection frequencies match pdf_leaf within 2 SEs (N=10 000)', () => {
    const powers = [1, 4, 2, 8, 5];
    const { nodes } = buildLightTree(makeInput(powers));
    const totalPower = nodes[0]!.totalPower;
    // Sort leaves by emitterIndex to reconstruct a stable CDF order.
    const leaves = nodes
      .filter((n) => n.emitterIndex >= 0)
      .sort((a, b) => a.emitterIndex - b.emitterIndex);
    const pdfs = leaves.map((l) => l.totalPower / totalPower);

    // Build CDF (leaf-only, in emitterIndex order).
    const cdf: number[] = [];
    let running = 0;
    for (const p of pdfs) {
      running += p;
      cdf.push(running);
    }

    // Sample 10 000 times using deterministic LCG.
    const N = 10_000;
    const counts = new Array<number>(pdfs.length).fill(0);
    const rng = makeLcg(0xcafe_1234);
    for (let s = 0; s < N; s++) {
      const u = rng();
      // Binary search for the first cdf[i] >= u.
      let lo = 0, hi = cdf.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cdf[mid]! < u) lo = mid + 1;
        else hi = mid;
      }
      counts[lo]!++;
    }

    // Verify each bucket is within ±2 binomial standard errors.
    for (let i = 0; i < pdfs.length; i++) {
      const expected = pdfs[i]! * N;
      const se = Math.sqrt(pdfs[i]! * (1 - pdfs[i]!) * N);
      const actual = counts[i]!;
      expect(actual).toBeGreaterThanOrEqual(expected - 2 * se);
      expect(actual).toBeLessThanOrEqual(expected + 2 * se);
    }
  });
});

describe('packLightTreeForGPU', () => {
  it('produces 16 floats per node (4 RGBA texels — B8 cone-carrying stride)', () => {
    const { nodes } = buildLightTree(makeInput([4.0, 6.0]));
    const packed = packLightTreeForGPU(nodes);

    // 3 nodes × 16 floats = 48 floats
    expect(packed).toHaveLength(nodes.length * 16);
  });

  it('leaf node has emitterIndex in slot 0, totalPower in slot 1, children -1 in slots 2+3', () => {
    const { nodes } = buildLightTree(makeInput([9.0]));
    const packed = packLightTreeForGPU(nodes);

    // Node 0 is the single leaf
    expect(packed[0]).toBeCloseTo(0); // emitterIndex = 0
    expect(packed[1]).toBeCloseTo(9.0); // totalPower
    expect(packed[2]).toBeCloseTo(-1); // leftChild
    expect(packed[3]).toBeCloseTo(-1); // rightChild
  });

  it('unoriented (default) nodes pack a full-sphere cone: axis 0, both cosines -1', () => {
    // No cones supplied ⇒ every node is full-sphere. Cone slots [10..12] are the
    // zero axis and [13]/[14] are cos(π) = -1, so lt_coneFactor returns 1 (no
    // orientation culling) — byte-identical descent to the pre-B8 tree.
    const { nodes } = buildLightTree(makeInput([1.0, 2.0]));
    const packed = packLightTreeForGPU(nodes);

    for (let i = 0; i < nodes.length; i++) {
      const b = i * 16;
      expect(packed[b + 10]).toBe(0); // cone.axis.x
      expect(packed[b + 11]).toBe(0); // cone.axis.y
      expect(packed[b + 12]).toBe(0); // cone.axis.z
      expect(packed[b + 13]).toBeCloseTo(-1, 6); // cos(thetaO) = cos(π)
      expect(packed[b + 14]).toBeCloseTo(-1, 6); // cos(thetaO+thetaE) = cos(π)
      expect(packed[b + 15]).toBe(0); // padding
    }
  });
});
