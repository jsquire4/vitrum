/**
 * lightTree.test.ts — Unit tests for Sprint 3 CPU-side light tree construction.
 *
 * Covers:
 *  - 1 emitter → trivial leaf tree
 *  - 2 emitters → 3 nodes (internal + 2 leaves), correct power sum
 *  - 4 emitters with linear power [1, 2, 4, 8] → root totalPower = 15,
 *    _powerPrefixSumDebug is monotonically increasing (values may exceed 1.0 — see
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
  const n = powers.length;
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
    const { nodes, _powerPrefixSumDebug } = buildLightTree(input);

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

  it('4 emitters with powers [1,2,4,8] → root totalPower = 15, _powerPrefixSumDebug is monotonically increasing', () => {
    const input = makeInput([1, 2, 4, 8]);
    const { nodes, _powerPrefixSumDebug } = buildLightTree(input);

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
    const { nodes, _powerPrefixSumDebug } = buildLightTree(input);

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

describe('packLightTreeForGPU', () => {
  it('produces 12 floats per node (3 RGBA texels)', () => {
    const { nodes } = buildLightTree(makeInput([4.0, 6.0]));
    const packed = packLightTreeForGPU(nodes);

    // 3 nodes × 12 floats = 36 floats
    expect(packed).toHaveLength(nodes.length * 12);
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

  it('padding slots 10 and 11 are zero', () => {
    const { nodes } = buildLightTree(makeInput([1.0, 2.0]));
    const packed = packLightTreeForGPU(nodes);

    for (let i = 0; i < nodes.length; i++) {
      expect(packed[i * 12 + 10]).toBe(0);
      expect(packed[i * 12 + 11]).toBe(0);
    }
  });
});
