/**
 * lightTreeTraversal.test.ts — the spatially-aware light-tree SELECTION traversal.
 *
 * These cover the importance-sampling descent (`sampleLightTreeCPU`) and its
 * deterministic pdf inverse (`lightTreePdfCPU`) added to wire the light tree
 * into walkaround ReSTIR-DI. The WGSL `sampleLightTree` mirrors this CPU code
 * branch-for-branch, so verifying the CPU reference pins the GPU behaviour too.
 *
 * The correctness-critical invariant for ReSTIR unbiasedness: the selection
 * pmf the descent assigns to a leaf must integrate to 1 over the emitter set
 * (a proper probability tree) for EVERY shading point. The RIS weight divides
 * p̂ by exactly this pmf, so a pmf that does not sum to 1 biases the estimator.
 */

import { describe, it, expect } from 'vitest';
import {
  buildLightTree,
  sampleLightTreeCPU,
  lightTreePdfCPU,
  type LightTreeBuildInput,
} from '../src/lightTree.js';

const FLOOR = 0.01;

function pointAabb(
  cx: number,
  cy: number,
  cz: number,
): { min: readonly [number, number, number]; max: readonly [number, number, number] } {
  return { min: [cx - 0.5, cy - 0.5, cz - 0.5], max: [cx + 0.5, cy + 0.5, cz + 0.5] };
}

/** N emitters spread along +x at 2-unit spacing, with the given powers. */
function makeInput(powers: number[], spacing = 2): LightTreeBuildInput {
  return {
    powers,
    centroids: powers.map((_, i) => [i * spacing, 0, 0] as const),
    aabbs: powers.map((_, i) => pointAabb(i * spacing, 0, 0)),
  };
}

/** Seeded LCG (Numerical Recipes) → [0, 1). */
function makeLcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = Math.imul(1664525, s) + 1013904223;
    return (s >>> 0) / 0x1_0000_0000;
  };
}

describe('lightTreePdfCPU — pmf partitions to 1 over the emitter set', () => {
  // The decisive unbiasedness property: for ANY shading point, summing the
  // per-emitter selection pdf over all emitters must equal 1.
  function assertPmfSumsToOne(input: LightTreeBuildInput, x: [number, number, number]): void {
    const { nodes } = buildLightTree(input);
    let sum = 0;
    for (let e = 0; e < input.powers.length; e++) {
      sum += lightTreePdfCPU(nodes, x, FLOOR, e);
    }
    expect(sum).toBeCloseTo(1.0, 6);
  }

  it('4 emitters — pmf sums to 1 at several shading points', () => {
    const input = makeInput([1, 2, 4, 8]);
    for (const x of [
      [0, 0, 0],
      [3, 0, 0],
      [100, 50, -20],
      [-5, -5, -5],
    ] as [number, number, number][]) {
      assertPmfSumsToOne(input, x);
    }
  });

  it('16 emitters with random powers — pmf sums to 1', () => {
    const rng = makeLcg(0x1234abcd);
    const powers = Array.from({ length: 16 }, () => rng() * 5 + 0.01);
    assertPmfSumsToOne(makeInput(powers), [7, 3, 1]);
  });

  it('64 emitters in a 3D grid — pmf sums to 1', () => {
    const powers: number[] = [];
    const centroids: [number, number, number][] = [];
    const aabbs: {
      min: readonly [number, number, number];
      max: readonly [number, number, number];
    }[] = [];
    let idx = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 4; k++) {
          powers.push(1 + (idx % 7));
          centroids.push([i * 3, j * 3, k * 3]);
          aabbs.push(pointAabb(i * 3, j * 3, k * 3));
          idx++;
        }
      }
    }
    assertPmfSumsToOne({ powers, centroids, aabbs }, [4.5, 4.5, 4.5]);
  });

  it('single-emitter tree — pdf is exactly 1 (the build is gated off below 2, but pdf is well-defined)', () => {
    const { nodes } = buildLightTree(makeInput([7]));
    expect(lightTreePdfCPU(nodes, [10, 0, 0], FLOOR, 0)).toBeCloseTo(1.0, 10);
  });

  it('emitter not in the tree → pdf 0', () => {
    const { nodes } = buildLightTree(makeInput([1, 2, 3]));
    expect(lightTreePdfCPU(nodes, [0, 0, 0], FLOOR, 99)).toBe(0);
  });
});

describe('sampleLightTreeCPU — draws agree with lightTreePdfCPU and terminate at leaves', () => {
  it('the drawn pdf equals the deterministic pdf for the chosen emitter', () => {
    const input = makeInput([3, 1, 5, 2]);
    const { nodes } = buildLightTree(input);
    const x: [number, number, number] = [3.3, 0.2, -1];
    const rng = makeLcg(0xfeed);
    for (let t = 0; t < 200; t++) {
      const s = sampleLightTreeCPU(nodes, x, FLOOR, rng);
      expect(s.emitterIndex).toBeGreaterThanOrEqual(0);
      const det = lightTreePdfCPU(nodes, x, FLOOR, s.emitterIndex);
      // The pdf carried out of the draw must equal the deterministic path pdf
      // for that leaf — they are the same branch-probability product.
      expect(s.pdf).toBeCloseTo(det, 10);
      expect(s.pdf).toBeGreaterThan(0);
    }
  });

  it('empirical selection frequencies match the pmf within 3 SE (N=20000)', () => {
    const input = makeInput([1, 4, 2, 8, 5]);
    const { nodes } = buildLightTree(input);
    const x: [number, number, number] = [4, 0, 0];
    const expectedPdf = input.powers.map((_, e) => lightTreePdfCPU(nodes, x, FLOOR, e));

    const N = 20_000;
    const counts = new Array<number>(input.powers.length).fill(0);
    const rng = makeLcg(0xc0ffee);
    for (let s = 0; s < N; s++) {
      const r = sampleLightTreeCPU(nodes, x, FLOOR, rng);
      counts[r.emitterIndex]!++;
    }
    for (let e = 0; e < expectedPdf.length; e++) {
      const p = expectedPdf[e]!;
      const expected = p * N;
      const se = Math.sqrt(Math.max(p * (1 - p) * N, 1));
      expect(counts[e]!).toBeGreaterThanOrEqual(expected - 3 * se);
      expect(counts[e]!).toBeLessThanOrEqual(expected + 3 * se);
    }
  });
});

describe('spatial importance — near/bright lights are selected more often than the flat power CDF', () => {
  // Two equal-power lights: one at x=0, one far away at x=100. A pure
  // power-weighted CDF would pick each 50/50. The distance-weighted tree must
  // favour whichever is NEAR the shading point.
  it('equal-power near vs far: the near light dominates selection', () => {
    const input = makeInput([1, 1], 100); // emitters at x=0 and x=100
    const { nodes } = buildLightTree(input);

    // Shading point near the FIRST emitter.
    const xNear: [number, number, number] = [0, 0, 0];
    const pdfNear0 = lightTreePdfCPU(nodes, xNear, FLOOR, 0);
    const pdfNear1 = lightTreePdfCPU(nodes, xNear, FLOOR, 1);
    expect(pdfNear0).toBeGreaterThan(pdfNear1); // near light favoured
    // Flat power CDF would give 0.5/0.5; the tree must be far more peaked.
    expect(pdfNear0).toBeGreaterThan(0.9);

    // Shading point near the SECOND emitter — symmetric.
    const xFar: [number, number, number] = [100, 0, 0];
    expect(lightTreePdfCPU(nodes, xFar, FLOOR, 1)).toBeGreaterThan(
      lightTreePdfCPU(nodes, xFar, FLOOR, 0),
    );
  });

  it('high-power light beats a dim distant one when both are far', () => {
    // Bright light (power 100) at x=0, dim (power 1) at x=10. Shading point
    // equidistant-ish but the bright one wins on power.
    const input = makeInput([100, 1], 10);
    const { nodes } = buildLightTree(input);
    const x: [number, number, number] = [5, 0, 0];
    expect(lightTreePdfCPU(nodes, x, FLOOR, 0)).toBeGreaterThan(
      lightTreePdfCPU(nodes, x, FLOOR, 1),
    );
  });

  it('selection pmf differs from the flat power CDF pmf (the tree adds spatial info)', () => {
    // 3 equal-power lights at x ∈ {0, 5, 50}. Flat CDF pmf = 1/3 each. Near
    // the cluster {0,5}, those two together must exceed 2/3 of the selection
    // mass — proving the tree's pmf is NOT the flat power pmf.
    const input: LightTreeBuildInput = {
      powers: [1, 1, 1],
      centroids: [[0, 0, 0], [5, 0, 0], [50, 0, 0]],
      aabbs: [pointAabb(0, 0, 0), pointAabb(5, 0, 0), pointAabb(50, 0, 0)],
    };
    const { nodes } = buildLightTree(input);
    const x: [number, number, number] = [2.5, 0, 0];
    const near = lightTreePdfCPU(nodes, x, FLOOR, 0) + lightTreePdfCPU(nodes, x, FLOOR, 1);
    const far = lightTreePdfCPU(nodes, x, FLOOR, 2);
    expect(near).toBeGreaterThan(2 / 3);
    expect(far).toBeLessThan(1 / 3);
    // Sanity: still a partition.
    expect(near + far).toBeCloseTo(1.0, 6);
  });
});

describe('degenerate guards — descent always terminates with a positive pdf', () => {
  it('all-zero-power subtree falls back to a 50/50 split (no infinite RIS weight)', () => {
    // Two zero-power emitters → importance sum is 0 at every node. The descent
    // must still terminate at a leaf with pdf > 0 (the 50/50 fallback) so the
    // RIS weight p̂/p_source can never divide by zero.
    const input = makeInput([0, 0]);
    const { nodes } = buildLightTree(input);
    const rng = makeLcg(7);
    const s = sampleLightTreeCPU(nodes, [0, 0, 0], FLOOR, rng);
    expect(s.emitterIndex).toBeGreaterThanOrEqual(0);
    expect(s.pdf).toBeGreaterThan(0);
    expect(s.pdf).toBeCloseTo(0.5, 6); // single internal node, 50/50
  });
});
