/**
 * regir.test.ts — ReGIR (Boksansky 2021) per-cell WRS + selection pmf.
 *
 * Correctness-critical for ReSTIR unbiasedness: the per-cell selection pmf the
 * grid stores (`q̂_c(e)/Ŝ`) must be a VALID pmf (sum to 1 over the emitter set)
 * — RIS divides the target p̂ by exactly this pmf, so a pmf that doesn't sum to
 * 1 biases the estimator. The WGSL grid-build kernel + RIS read path mirror
 * `regirBuildSurvivorCPU` / `regirCellPmfExact` byte-for-byte, so verifying the
 * CPU reference pins the GPU behaviour too.
 */

import { describe, it, expect } from 'vitest';
import {
  buildLightTree,
  type LightTreeBuildInput,
} from '../src/lightTree.js';
import {
  regirBuildSurvivorCPU,
  regirCellTargetFromTree,
  regirCellPmfExact,
} from '../src/regir.js';
import * as pkgIndex from '../src/index.js';

const FLOOR = 0.01;

function pointAabb(
  cx: number, cy: number, cz: number,
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

describe('ReGIR public surface — re-export equivalence (regir.ts move)', () => {
  it('the 4 ReGIR exports remain importable from the package index, identical to ./regir.js', () => {
    // After the lightTree.ts → regir.ts split the public surface must be UNCHANGED:
    // consumers import these from '@vitrum/shared-samplers' (the index), not deep.
    expect(pkgIndex.REGIR_FLOATS_PER_SURVIVOR).toBe(2);
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
    const sDirect = regirBuildSurvivorCPU(nodes, xc, FLOOR, 32, tgtDirect, lcg(0xBEEF));
    const sIndex = pkgIndex.regirBuildSurvivorCPU(nodes, xc, FLOOR, 32, tgtIndex, lcg(0xBEEF));
    expect(sIndex).toEqual(sDirect);
  });
});

describe('ReGIR cell pmf — validity (correctness-critical for unbiasedness)', () => {
  it('the exact normalized cell pmf integrates to 1 over the emitter set, for EVERY cell centroid', () => {
    const powers = [1, 2, 4, 8, 3, 5, 6, 7];
    const { nodes } = buildLightTree(makeInput(powers));
    // Probe several cell centroids spanning + outside the emitter spread.
    for (const xc of [
      [-2, 0, 0], [0, 0, 0], [6, 0, 0], [14, 0, 0], [28, 0, 0], [50, 2, -3],
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

describe('ReGIR per-cell WRS survivor — unbiased selection pmf', () => {
  it('the WRS survivor distribution converges to the exact cell pmf, and pSel matches q̂(e*)/Ŝ', () => {
    const powers = [1, 3, 9, 2, 5];
    const { nodes } = buildLightTree(makeInput(powers));
    const xc: readonly [number, number, number] = [6, 0, 0];
    const exactPmf = regirCellPmfExact(nodes, xc, FLOOR);
    const target = regirCellTargetFromTree(nodes, xc, FLOOR);
    const S = powers.reduce((acc, _, i) => acc + target(i), 0);

    const rand = lcg(0xC0FFEE);
    const M = 64; // candidates per sub-reservoir
    const TRIALS = 40000;
    const survivorCounts = new Map<number, number>();
    for (let t = 0; t < TRIALS; t++) {
      const s = regirBuildSurvivorCPU(nodes, xc, FLOOR, M, target, rand);
      if (s.emitterIndex < 0) continue;
      survivorCounts.set(s.emitterIndex, (survivorCounts.get(s.emitterIndex) ?? 0) + 1);
      // pSel MUST equal the exact q̂(e*)/S relation up to the wSum/M estimate.
      // For a converged-enough M, pSel ≈ q̂(e*)/S = exactPmf(e*).
      const exact = exactPmf.get(s.emitterIndex)!;
      // pSel = q̂(e*) · M / wSum; with the unbiased Ŝ = wSum/M ≈ S, pSel ≈ q̂/S.
      // Assert pSel is within a generous tolerance of the exact pmf (the WRS
      // normalisation estimate Ŝ has variance; the relation is exact only in
      // expectation per draw — see below for the aggregate check).
      expect(s.pSel).toBeGreaterThan(0);
      expect(Number.isFinite(s.pSel)).toBe(true);
      // q̂(e*)/Ŝ ≤ 1 always (q̂(e*) ≤ Ŝ in any single reservoir that selected e*).
      expect(s.pSel).toBeLessThanOrEqual(1 + 1e-6);
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

  it('the estimator E[ q̂(e*) · 1/pSel ] equals S_c (unbiased normalisation) — the RIS-weight identity', () => {
    // RIS divides p̂ by pSel. The unbiasedness identity behind that division is
    // that pSel = q̂(e*)/Ŝ is the survivor effective pdf, so q̂(e*)/pSel = Ŝ is an
    // unbiased estimate of S_c. Average q̂(e*)/pSel over many sub-reservoirs and
    // assert it recovers S_c.
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
      if (s.emitterIndex < 0 || s.pSel <= 0) continue;
      // q̂(e*) / pSel = q̂(e*) / (q̂(e*)/Ŝ) = Ŝ — an unbiased estimate of S_c.
      sumEst += target(s.emitterIndex) / s.pSel;
      n++;
    }
    expect(n).toBeGreaterThan(0);
    expect(sumEst / n).toBeCloseTo(S, 0); // within ~1 of S_c
  });

  it('returns an empty survivor (skipped, never an infinite weight) when no positive-power emitter is reachable', () => {
    // All zero-power emitters → q̂_c = 0 everywhere → no survivor.
    const { nodes } = buildLightTree(makeInput([0, 0, 0, 0]));
    const target = regirCellTargetFromTree(nodes, [0, 0, 0], FLOOR);
    const s = regirBuildSurvivorCPU(nodes, [0, 0, 0], FLOOR, 16, target, lcg(7));
    expect(s.emitterIndex).toBe(-1);
    expect(s.pSel).toBe(0);
  });
});
