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
  LIGHT_TREE_BUCKET_COUNT,
  FULL_SPHERE_CONE,
  packLightTreeForGPU,
  packedLightTreeNodeImportanceCPU,
  packedLightTreePdfCPU,
  samplePackedLightTreeCPU,
  sampleLightTreeCPU,
  lightTreePdfCPU,
  nodeImportance,
  type LightTreeBuildInput,
  type LightTreeNode,
} from '../src/lightTree.js';
import { lightTreeWgsl } from '../src/wgsl/lightTree.wgsl.js';

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

/** A physically consistent, 320-level power comb (each split is 50/50). */
function makeDeepPowerComb(leafCount: number): LightTreeNode[] {
  const leafPowers = Array.from({ length: leafCount }, (_, index) =>
    index === leafCount - 1 ? 2 ** -(leafCount - 1) : 2 ** -(index + 1));
  const nodes: LightTreeNode[] = [];
  const append = (firstLeaf: number): number => {
    const nodeIndex = nodes.length;
    if (firstLeaf === leafCount - 1) {
      nodes.push({
        emitterIndex: firstLeaf,
        totalPower: leafPowers[firstLeaf]!,
        aabbMin: [0, 0, 0],
        aabbMax: [0, 0, 0],
        leftChild: -1,
        rightChild: -1,
        cone: FULL_SPHERE_CONE,
      });
      return nodeIndex;
    }
    nodes.push(undefined as unknown as LightTreeNode);
    const leftChild = nodes.length;
    nodes.push({
      emitterIndex: firstLeaf,
      totalPower: leafPowers[firstLeaf]!,
      aabbMin: [0, 0, 0],
      aabbMax: [0, 0, 0],
      leftChild: -1,
      rightChild: -1,
      cone: FULL_SPHERE_CONE,
    });
    const rightChild = append(firstLeaf + 1);
    nodes[nodeIndex] = {
      emitterIndex: -1,
      totalPower: nodes[leftChild]!.totalPower + nodes[rightChild]!.totalPower,
      aabbMin: [0, 0, 0],
      aabbMax: [0, 0, 0],
      leftChild,
      rightChild,
      cone: FULL_SPHERE_CONE,
    };
    return nodeIndex;
  };
  append(0);
  return nodes;
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

describe('24-bit represented proposal — support, exact PMFs, and packed parity', () => {
  it('preserves min/max positive powers and returns exact bucket-multiple PMFs', () => {
    const { nodes } = buildLightTree(makeInput([Number.MIN_VALUE, Number.MAX_VALUE]));
    const packed = packLightTreeForGPU(nodes);
    const pdfs = [0, 1].map((emitter) =>
      packedLightTreePdfCPU(packed, [0, 0, 0], FLOOR, emitter));
    expect(pdfs[0]).toBeGreaterThanOrEqual(1 / LIGHT_TREE_BUCKET_COUNT);
    expect(pdfs[1]).toBeGreaterThanOrEqual(1 / LIGHT_TREE_BUCKET_COUNT);
    expect(pdfs[0]! + pdfs[1]!).toBe(1);
    for (const pdf of pdfs) {
      expect(Number.isInteger(pdf * LIGHT_TREE_BUCKET_COUNT)).toBe(true);
    }
  });

  it('selects the exact leaf interval on both sides of a represented boundary', () => {
    const { nodes } = buildLightTree(makeInput([Number.MIN_VALUE, Number.MAX_VALUE]));
    const packed = packLightTreeForGPU(nodes);
    const leftBuckets =
      packedLightTreePdfCPU(packed, [0, 0, 0], FLOOR, 0) *
      LIGHT_TREE_BUCKET_COUNT;
    expect(leftBuckets).toBeGreaterThanOrEqual(1);
    const left = samplePackedLightTreeCPU(
      packed,
      [0, 0, 0],
      FLOOR,
      () => (leftBuckets - 0.5) / LIGHT_TREE_BUCKET_COUNT,
    );
    const right = samplePackedLightTreeCPU(
      packed,
      [0, 0, 0],
      FLOOR,
      () => (leftBuckets + 0.5) / LIGHT_TREE_BUCKET_COUNT,
    );
    expect(left.emitterIndex).toBe(0);
    expect(right.emitterIndex).toBe(1);
    expect(left.pdf).toBe(packedLightTreePdfCPU(packed, [0, 0, 0], FLOOR, 0));
    expect(right.pdf).toBe(packedLightTreePdfCPU(packed, [0, 0, 0], FLOOR, 1));
  });

  it('keeps every leaf positive through a 320-level tree without PDF products', () => {
    const nodes = makeDeepPowerComb(320);
    const packed = packLightTreeForGPU(nodes);
    let sum = 0;
    for (let emitter = 0; emitter < 320; emitter++) {
      const pdf = packedLightTreePdfCPU(packed, [0, 0, 0], FLOOR, emitter);
      expect(pdf).toBeGreaterThanOrEqual(1 / LIGHT_TREE_BUCKET_COUNT);
      expect(Number.isInteger(pdf * LIGHT_TREE_BUCKET_COUNT)).toBe(true);
      sum += pdf;
    }
    expect(sum).toBe(1);
    expect(packed[15]).toBe(320);
  });

  it('reads normalized cone axes and extreme AABBs from packed f32 lanes', () => {
    const max = Math.fround(3.4028234663852886e38);
    const rawAxis = [1e-30, -2e-30, 3e-30] as const;
    const { nodes } = buildLightTree({
      powers: [1, 1],
      centroids: [[-max, 0, 0], [max, 0, 0]],
      aabbs: [
        { min: [-max, 0, 0], max: [-max, 0, 0] },
        { min: [max, 0, 0], max: [max, 0, 0] },
      ],
      cones: [
        { axis: rawAxis, thetaO: 0, thetaE: Math.PI / 2 },
        { axis: [0, 0, 0] },
      ],
    });
    const packed = packLightTreeForGPU(nodes);
    const orientedIndex = nodes.findIndex((node) => node.emitterIndex === 0);
    const base = orientedIndex * 16;
    const length = Math.hypot(...rawAxis);
    expect(Array.from(packed.slice(base + 10, base + 13))).toEqual([
      Math.fround(rawAxis[0] / length),
      Math.fround(rawAxis[1] / length),
      Math.fround(rawAxis[2] / length),
    ]);
    for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
      const q = packedLightTreeNodeImportanceCPU(
        packed,
        nodeIndex,
        [0, 0, 0],
        FLOOR,
      );
      expect(Number.isFinite(q)).toBe(true);
      expect(q).toBeGreaterThanOrEqual(0);
    }
  });

  it('consumes no RNG for empty or single-leaf trees', () => {
    let draws = 0;
    const rand = (): number => {
      draws++;
      return 0.25;
    };
    expect(samplePackedLightTreeCPU(new Float32Array(0), [0, 0, 0], FLOOR, rand))
      .toEqual({ emitterIndex: -1, pdf: 0 });
    const { nodes } = buildLightTree(makeInput([1]));
    expect(sampleLightTreeCPU(nodes, [0, 0, 0], FLOOR, rand))
      .toEqual({ emitterIndex: 0, pdf: 1 });
    expect(draws).toBe(0);
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

// ── B8 — orientation-cone selection ──────────────────────────────────────────
describe('B8 orientation cones — oriented emitters culled from behind', () => {
  // Two equal-power emitters at x=±4 (y=z=0). Emitter 0 (x=-4) emits toward +x
  // (axis (1,0,0), narrow lobe); emitter 1 (x=+4) emits toward +x as well. A
  // shading point at x=0 sits in FRONT of emitter 0 (it shines toward +x, the
  // point is at +x of it) but BEHIND emitter 1 (it shines toward +x, away from
  // the point which is at -x of it). With cones, emitter 1 must get near-zero
  // selection probability; emitter 0 takes essentially all of it.
  function orientedInput(): LightTreeBuildInput {
    return {
      powers: [10, 10],
      centroids: [[-4, 0, 0], [4, 0, 0]],
      aabbs: [pointAabb(-4, 0, 0), pointAabb(4, 0, 0)],
      // Both emit toward +x with a narrow ±10° lobe (thetaE small). thetaO=0
      // (single sharp axis per leaf).
      cones: [
        { axis: [1, 0, 0], thetaO: 0, thetaE: Math.PI / 18 },
        { axis: [1, 0, 0], thetaO: 0, thetaE: Math.PI / 18 },
      ],
    };
  }

  it('a directional emitter pointing AWAY from the point gets near-zero selection pdf', () => {
    const { nodes } = buildLightTree(orientedInput());
    const x: [number, number, number] = [0, 0, 0];
    const pFront = lightTreePdfCPU(nodes, x, FLOOR, 0); // x=-4, faces toward point
    const pBack = lightTreePdfCPU(nodes, x, FLOOR, 1);  // x=+4, faces away
    // The back-facing emitter is culled (cone factor 0): pdf ~0; front takes all.
    expect(pBack).toBeLessThan(1e-6);
    expect(pFront).toBeGreaterThan(0.999);
    // Still a partition.
    expect(pFront + pBack).toBeCloseTo(1.0, 6);
  });

  it('never culls a lit point in the cited non-axis-aligned triangle slab', () => {
    // Triangle a=(0,0,0), b=(1,1,0), c=(0,1,1) has unit normal
    // (1,-1,1)/sqrt(3). Its AABB centre lies +0.5/sqrt(3) in front of the
    // emitter plane. A point only 0.05 along the normal is still on the lit
    // side, but the old centre-only cone test saw the centre-to-point vector as
    // back-facing and assigned this emitter exactly zero selection mass.
    const invSqrt3 = 1 / Math.sqrt(3);
    const normal = [invSqrt3, -invSqrt3, invSqrt3] as const;
    const { nodes } = buildLightTree({
      powers: [1, 1],
      centroids: [[0.5, 0.5, 0.5], [10, 0, 0]],
      aabbs: [
        { min: [0, 0, 0], max: [1, 1, 1] },
        pointAabb(10, 0, 0),
      ],
      cones: [
        { axis: normal, thetaO: 0, thetaE: Math.PI / 2 },
        { axis: [0, 0, 0] },
      ],
    });
    const litPoint: [number, number, number] = [
      normal[0] * 0.05,
      normal[1] * 0.05,
      normal[2] * 0.05,
    ];
    const trianglePdf = lightTreePdfCPU(nodes, litPoint, FLOOR, 0);
    expect(trianglePdf).toBeGreaterThan(0);
    expect(trianglePdf + lightTreePdfCPU(nodes, litPoint, FLOOR, 1)).toBeCloseTo(1, 6);
  });

  it('keeps the CPU/WGSL centre guard scale-invariant for a tiny point node', () => {
    const distance = 1e-20;
    const { nodes } = buildLightTree({
      powers: [1, 1],
      centroids: [[0, 0, 0], [1, 0, 0]],
      aabbs: [
        { min: [0, 0, 0], max: [0, 0, 0] },
        { min: [1, 0, 0], max: [1, 0, 0] },
      ],
      cones: [
        { axis: [1, 0, 0], thetaO: 0, thetaE: Math.PI / 18 },
        { axis: [0, 0, 0] },
      ],
    });
    const pointLeaf = nodes.find((node) => node.emitterIndex === 0);
    expect(pointLeaf).toBeDefined();

    // Every non-zero displacement from a zero-radius emitter is outside it.
    expect(nodeImportance(pointLeaf!, -distance, 0, 0, FLOOR)).toBe(0);

    // The WGSL compares the dimensionless radius/distance ratio, avoiding an
    // absolute world-unit floor and squared-distance underflow.
    const wgsl = lightTreeWgsl({ group: 0, binding: 1 });
    expect(wgsl).toContain('if (logRatio >= 0.0) { return 1.0; }');
    expect(wgsl).not.toContain('radius * radius, 1e-');
  });

  it('full-sphere cones (no orientation) reproduce the spatial-only partition exactly', () => {
    // Same geometry/powers but NO cones → both emitters are full-sphere. The
    // partition must match a build with cones omitted entirely (byte-identical
    // behaviour: the B8 cone term is identically 1).
    const geom = {
      powers: [10, 10],
      centroids: [[-4, 0, 0] as const, [4, 0, 0] as const],
      aabbs: [pointAabb(-4, 0, 0), pointAabb(4, 0, 0)],
    };
    const noCones = buildLightTree(geom);
    const fullSphere = buildLightTree({
      ...geom,
      cones: [{ axis: [0, 0, 0] }, { axis: [0, 0, 0] }], // zero axis ⇒ full sphere
    });
    const x: [number, number, number] = [0, 0, 0];
    for (let e = 0; e < 2; e++) {
      expect(lightTreePdfCPU(fullSphere.nodes, x, FLOOR, e))
        .toBeCloseTo(lightTreePdfCPU(noCones.nodes, x, FLOOR, e), 10);
    }
    // Symmetric geometry, equal power, both full-sphere ⇒ 50/50.
    expect(lightTreePdfCPU(noCones.nodes, x, FLOOR, 0)).toBeCloseTo(0.5, 6);
  });

  it('the descent still terminates with a positive pdf when ALL emitters face away', () => {
    // Both emitters face +x; the point is at +x of BOTH (so both face it) —
    // flip: put the point BEHIND both (at large +x while both face -x).
    const { nodes } = buildLightTree({
      powers: [10, 10],
      centroids: [[-4, 0, 0], [4, 0, 0]],
      aabbs: [pointAabb(-4, 0, 0), pointAabb(4, 0, 0)],
      cones: [
        { axis: [-1, 0, 0], thetaO: 0, thetaE: Math.PI / 18 },
        { axis: [-1, 0, 0], thetaO: 0, thetaE: Math.PI / 18 },
      ],
    });
    const x: [number, number, number] = [100, 0, 0]; // far +x, both face -x ⇒ culled
    const rng = makeLcg(11);
    const s = sampleLightTreeCPU(nodes, x, FLOOR, rng);
    // Both cone factors 0 ⇒ importance sum 0 at the root ⇒ 50/50 fallback. The
    // descent must still return a reachable leaf with a strictly-positive pdf
    // (unbiasedness: the selection pdf is divided out, never 0 for a hit leaf).
    expect(s.emitterIndex).toBeGreaterThanOrEqual(0);
    expect(s.pdf).toBeGreaterThan(0);
  });
});

describe('lightTreeWgsl RNG-state specialization', () => {
  it('keeps u32 as the shared default and accepts the pt-webgpu state type', () => {
    const defaultWgsl = lightTreeWgsl({ group: 0, binding: 1 });
    expect(defaultWgsl).toContain(
      'rng: ptr<function, u32>',
    );
    expect(defaultWgsl).toContain(
      'let sinThetaU = clamp(radiusOverDistance, 0.0, 1.0);',
    );
    expect(defaultWgsl).not.toContain('acos(');
    expect(defaultWgsl).not.toContain('asin(');
    const specialized = lightTreeWgsl({
      group: 3,
      binding: 0,
      rngStateType: 'PtRngState',
    });
    expect(specialized).toContain('@group(3) @binding(0)');
    expect(specialized).toContain('rng: ptr<function, PtRngState>');
    expect(specialized).not.toContain('rng: ptr<function, u32>');
  });

  it('guards an empty tree before storage reads and defers the sole RNG draw until an internal root', () => {
    const wgsl = lightTreeWgsl({ group: 0, binding: 1 });
    expect(wgsl.indexOf('if (nodeCount == 0u)')).toBeLessThan(
      wgsl.indexOf('let rootLeft = i32(lightTree[2u])'),
    );
    expect(wgsl).toContain('if (rootLeft >= 0 && rootRight >= 0)');
    expect((wgsl.match(/rand_f32\(rng\)/g) ?? [])).toHaveLength(1);
    expect(wgsl).toContain('result.pdf = f32(currentBuckets) / f32(lt_ROOT_BUCKETS)');
  });

  it('rejects a non-identifier RNG type instead of emitting malformed WGSL', () => {
    expect(() => lightTreeWgsl({
      group: 0,
      binding: 0,
      rngStateType: 'u32); bad(',
    })).toThrow(TypeError);
  });
});
