/**
 * regir.test.ts — ReGIR represented per-cell WRS + effective log density.
 *
 * Correctness-critical for ReSTIR unbiasedness: lane 1 stores
 * `log2(M * r_i * p_tree_i)`, with `r_i` measured from the actual integer
 * replace/keep buckets. The tests pin extreme support and estimator identities.
 */

import { describe, it, expect } from 'vitest';
import {
  buildLightTree,
  packLightTreeForGPU,
  packedLightTreeNodeImportanceCPU,
  packedLightTreePdfCPU,
  type LightTreeBuildInput,
} from '../src/lightTree.js';
import {
  REGIR_LOG2_PSEL_INVALID,
  REGIR_MAX_CANDIDATES_PER_CELL,
  regirBuildSurvivorCPU,
  regirCellTargetFromTree,
  regirCellPmfExact,
} from '../src/regir.js';
import * as pkgIndex from '../src/index.js';

const FLOOR = 0.01;

function pointAabb(
  cx: number,
  cy: number,
  cz: number,
): { min: readonly [number, number, number]; max: readonly [number, number, number] } {
  return { min: [cx - 0.5, cy - 0.5, cz - 0.5], max: [cx + 0.5, cy + 0.5, cz + 0.5] };
}

/** N emitters spread along +x at the given spacing, with the given powers. */
function makeInput(powers: number[], spacing = 4): LightTreeBuildInput {
  return {
    powers,
    centroids: powers.map((_, i) => [i * spacing, 0, 0] as const),
    aabbs: powers.map((_, i) => pointAabb(i * spacing, 0, 0)),
  };
}

/** Deterministic LCG so the WRS draws are reproducible per test. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

describe('ReGIR public surface — re-export equivalence', () => {
  it('the ABI constants and helpers remain importable from the package index', () => {
    // After the lightTree.ts → regir.ts split the public surface must be UNCHANGED:
    // consumers import these from '@vitrum/shared-samplers' (the index), not deep.
    expect(pkgIndex.REGIR_FLOATS_PER_SURVIVOR).toBe(2);
    expect(pkgIndex.REGIR_LOG2_PSEL_INVALID).toBe(REGIR_LOG2_PSEL_INVALID);
    expect(pkgIndex.REGIR_MAX_CANDIDATES_PER_CELL).toBe(REGIR_MAX_CANDIDATES_PER_CELL);
    expect(pkgIndex.regirBuildSurvivorCPU).toBe(regirBuildSurvivorCPU);
    expect(pkgIndex.regirCellTargetFromTree).toBe(regirCellTargetFromTree);
    expect(pkgIndex.regirCellPmfExact).toBe(regirCellPmfExact);
  });

  it('the index-imported helpers produce identical output to the direct ./regir.js imports on a fixture', () => {
    const powers = [1, 2, 4, 8, 3];
    const { nodes } = buildLightTree(makeInput(powers));
    const xc: readonly [number, number, number] = [6, 0, 0];
    // regirCellPmfExact — identical map contents.
    const pmfDirect = regirCellPmfExact(nodes, xc, FLOOR);
    const pmfIndex = pkgIndex.regirCellPmfExact(nodes, xc, FLOOR);
    expect([...pmfIndex.entries()]).toEqual([...pmfDirect.entries()]);
    // regirCellTargetFromTree — identical per-emitter target.
    const tgtDirect = regirCellTargetFromTree(nodes, xc, FLOOR);
    const tgtIndex = pkgIndex.regirCellTargetFromTree(nodes, xc, FLOOR);
    for (const e of powers.keys()) expect(tgtIndex(e)).toBe(tgtDirect(e));
    // regirBuildSurvivorCPU — identical survivor for the same RNG seed.
    const sDirect = regirBuildSurvivorCPU(nodes, xc, FLOOR, 32, tgtDirect, lcg(0xbeef));
    const sIndex = pkgIndex.regirBuildSurvivorCPU(nodes, xc, FLOOR, 32, tgtIndex, lcg(0xbeef));
    expect(sIndex).toEqual(sDirect);
  });
});

describe('ReGIR normalized represented target diagnostic', () => {
  it('integrates to 1 over the emitter set for every cell centroid', () => {
    const powers = [1, 2, 4, 8, 3, 5, 6, 7];
    const { nodes } = buildLightTree(makeInput(powers));
    // Probe several cell centroids spanning + outside the emitter spread.
    for (const xc of [
      [-2, 0, 0],
      [0, 0, 0],
      [6, 0, 0],
      [14, 0, 0],
      [28, 0, 0],
      [50, 2, -3],
    ] as const) {
      const pmf = regirCellPmfExact(nodes, xc, FLOOR);
      let sum = 0;
      for (const p of pmf.values()) sum += p;
      // Every emitter must have a finite, non-negative pmf entry.
      for (const p of pmf.values()) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(p)).toBe(true);
      }
      expect(sum).toBeCloseTo(1, 10);
      expect(pmf.size).toBe(powers.length);
    }
  });

  it('concentrates on the nearest emitter when the cell centroid sits next to one light', () => {
    // 4 equal-power emitters at x = 0, 4, 8, 12. A cell centroid AT x=8 should
    // assign the highest pmf to emitter index 2 (the co-located light).
    const powers = [1, 1, 1, 1];
    const { nodes } = buildLightTree(makeInput(powers));
    const pmf = regirCellPmfExact(nodes, [8, 0, 0], FLOOR);
    const nearest = pmf.get(2)!;
    for (const [idx, p] of pmf) {
      if (idx !== 2) expect(nearest).toBeGreaterThan(p);
    }
  });

  it('a brighter-but-farther light can outrank a dim near light — power × proximity', () => {
    // Emitter 0 dim (power 1) near; emitter 1 very bright (power 1000) farther.
    // Probe a cell roughly midway: the bright light should dominate.
    const { nodes } = buildLightTree(makeInput([1, 1000], 8));
    const pmf = regirCellPmfExact(nodes, [4, 0, 0], FLOOR);
    expect(pmf.get(1)!).toBeGreaterThan(pmf.get(0)!);
  });
});

describe('ReGIR per-cell represented WRS survivor', () => {
  it('keeps qHat/pdf finite through the one-bucket/max-f32 adversarial case', () => {
    const { nodes } = buildLightTree(makeInput([Number.MIN_VALUE, Number.MAX_VALUE]));
    const draws = [0, 0]; // choose the first root bucket, then accept the WRS item
    let cursor = 0;
    const survivor = regirBuildSurvivorCPU(
      nodes,
      [0, 0, 0],
      FLOOR,
      1,
      () => Math.fround(3.4028234663852886e38),
      () => draws[cursor++] ?? 0,
    );
    expect(survivor.emitterIndex).toBe(0);
    expect(survivor.log2PSel).toBe(-24);
    expect(Number.isFinite(survivor.log2PSel)).toBe(true);
  });

  it('the represented survivor distribution remains close to the normalized target', () => {
    const powers = [1, 3, 9, 2, 5];
    const { nodes } = buildLightTree(makeInput(powers));
    const xc: readonly [number, number, number] = [6, 0, 0];
    const exactPmf = regirCellPmfExact(nodes, xc, FLOOR);
    const target = regirCellTargetFromTree(nodes, xc, FLOOR);
    const S = powers.reduce((acc, _, i) => acc + target(i), 0);

    const rand = lcg(0xc0ffee);
    const M = 64; // candidates per sub-reservoir
    const TRIALS = 40000;
    const survivorCounts = new Map<number, number>();
    for (let t = 0; t < TRIALS; t++) {
      const s = regirBuildSurvivorCPU(nodes, xc, FLOOR, M, target, rand);
      if (s.emitterIndex < 0) continue;
      survivorCounts.set(s.emitterIndex, (survivorCounts.get(s.emitterIndex) ?? 0) + 1);
      const exact = exactPmf.get(s.emitterIndex)!;
      // The stored value is an occurrence correction, not this target PMF.
      expect(s.log2PSel).toBeGreaterThan(REGIR_LOG2_PSEL_INVALID);
      expect(Number.isFinite(s.log2PSel)).toBe(true);
      expect(exact).toBeGreaterThan(0);
      expect(S).toBeGreaterThan(0);
    }

    // Survivor frequency converges to the exact normalized cell pmf.
    let totalSurvivors = 0;
    for (const c of survivorCounts.values()) totalSurvivors += c;
    for (const i of powers.keys()) {
      const empirical = (survivorCounts.get(i) ?? 0) / totalSurvivors;
      expect(empirical).toBeCloseTo(exactPmf.get(i)!, 1); // ±0.05
    }
  });

  it('stores the one-bucket newcomer occurrence as log2(M*r*pTree)', () => {
    const { nodes } = buildLightTree(makeInput([1, 1]));
    const lastRootBucket = (2 ** 24 - 1) / 2 ** 24;
    // Draw emitter 0 with a huge WRS weight, then emitter 1 with a tiny one.
    // Ticket zero selects the newcomer through its one represented bucket.
    const draws = [0, lastRootBucket, 0];
    let cursor = 0;
    const survivor = regirBuildSurvivorCPU(
      nodes,
      [0, 0, 0],
      FLOOR,
      2,
      (emitterIndex) => (emitterIndex === 0 ? 3.4028234663852886e38 : 1.1754943508222875e-38),
      () => draws[cursor++] ?? 0.5,
    );

    expect(survivor.emitterIndex).toBe(1);
    const selectedTreePmf = packedLightTreePdfCPU(packLightTreeForGPU(nodes), [0, 0, 0], FLOOR, 1);
    expect(survivor.log2PSel).toBe(Math.fround(1 - 24 + Math.log2(selectedTreePmf)));
    expect(cursor).toBe(3);
  });

  it('stores the B-1-bucket newcomer probability without rounding it to one', () => {
    const { nodes } = buildLightTree(makeInput([1, 1]));
    const lastRootBucket = (2 ** 24 - 1) / 2 ** 24;
    const draws = [0, lastRootBucket, 0];
    let cursor = 0;
    const survivor = regirBuildSurvivorCPU(
      nodes,
      [0, 0, 0],
      FLOOR,
      2,
      (emitterIndex) => (emitterIndex === 0 ? 1.1754943508222875e-38 : 3.4028234663852886e38),
      () => draws[cursor++] ?? 0.5,
    );

    expect(survivor.emitterIndex).toBe(1);
    const selectedTreePmf = packedLightTreePdfCPU(packLightTreeForGPU(nodes), [0, 0, 0], FLOOR, 1);
    const representedLogR = Math.fround(Math.log2(Math.fround((2 ** 24 - 1) / 2 ** 24)));
    expect(representedLogR).toBeLessThan(0);
    expect(survivor.log2PSel).toBe(Math.fround(1 + representedLogR + Math.log2(selectedTreePmf)));
  });

  it('the represented occurrence correction recovers the target-mass integral', () => {
    const powers = [2, 4, 1, 8, 3];
    const { nodes } = buildLightTree(makeInput(powers));
    const xc: readonly [number, number, number] = [5, 0, 0];
    const target = regirCellTargetFromTree(nodes, xc, FLOOR);
    const S = powers.reduce((acc, _, i) => acc + target(i), 0);

    const rand = lcg(0x1234567);
    const M = 32;
    const TRIALS = 50000;
    let sumEst = 0;
    let n = 0;
    for (let t = 0; t < TRIALS; t++) {
      const s = regirBuildSurvivorCPU(nodes, xc, FLOOR, M, target, rand);
      if (s.emitterIndex < 0) continue;
      const pSel = 2 ** s.log2PSel;
      expect(pSel).toBeGreaterThan(0);
      sumEst += target(s.emitterIndex) / pSel;
      n++;
    }
    expect(n).toBeGreaterThan(0);
    expect(sumEst / n).toBeCloseTo(S, 0); // within ~1 of S_c
  });

  it('keeps a positive-power leaf selectable when its centroid cone target is zero', () => {
    const input: LightTreeBuildInput = {
      powers: [1],
      centroids: [[0, 0, 0]],
      aabbs: [pointAabb(0, 0, 0)],
      cones: [{ axis: [1, 0, 0], thetaO: 0, thetaE: Math.PI / 2 }],
    };
    const { nodes } = buildLightTree(input);
    const xc: readonly [number, number, number] = [-10, 0, 0];
    const packed = packLightTreeForGPU(nodes);
    expect(packedLightTreeNodeImportanceCPU(packed, 0, xc, FLOOR)).toBe(0);

    const target = regirCellTargetFromTree(nodes, xc, FLOOR);
    expect(target(0)).toBe(1.1754943508222875e-38);
    let randomCalls = 0;
    const survivor = regirBuildSurvivorCPU(nodes, xc, FLOOR, 1, target, () => {
      randomCalls++;
      return 0;
    });
    expect(survivor).toEqual({ emitterIndex: 0, log2PSel: 0 });
    expect(randomCalls).toBe(0);
  });

  it('rejects candidate loops above the shared bounded serial limit', () => {
    const { nodes } = buildLightTree(makeInput([1]));
    const target = regirCellTargetFromTree(nodes, [0, 0, 0], FLOOR);
    expect(() =>
      regirBuildSurvivorCPU(
        nodes,
        [0, 0, 0],
        FLOOR,
        REGIR_MAX_CANDIDATES_PER_CELL + 1,
        target,
        () => 0,
      ),
    ).toThrow(/\[1, 4096\]/);
  });

  it('returns an empty survivor (skipped, never an infinite weight) when no positive-power emitter is reachable', () => {
    // All zero-power emitters → q̂_c = 0 everywhere → no survivor.
    const { nodes } = buildLightTree(makeInput([0, 0, 0, 0]));
    const target = regirCellTargetFromTree(nodes, [0, 0, 0], FLOOR);
    const s = regirBuildSurvivorCPU(nodes, [0, 0, 0], FLOOR, 16, target, lcg(7));
    expect(s.emitterIndex).toBe(-1);
    expect(s.log2PSel).toBe(REGIR_LOG2_PSEL_INVALID);
  });
});
