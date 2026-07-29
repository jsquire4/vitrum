import { afterEach, describe, expect, it, vi } from 'vitest';

import { RIS_WGSL } from '../shaders/ris.wgsl.js';
import { RESERVOIR_DI_WGSL } from '../shaders/reservoirDi.wgsl.js';
import { TEMPORAL_WGSL } from '../shaders/temporal.wgsl.js';
import { SPATIAL_WGSL } from '../shaders/spatial.wgsl.js';
import {
  RESERVOIR_DI_STRIDE_BYTES,
  RESERVOIR_DI_STRIDE_U32,
} from '../restir/reservoirDiLayout.js';
import { createRestirDIFrameResources } from '../pipeline/frameResources/createRestirDIFrameResources.js';

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let z = value;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

/**
 * Null-event proposal oracle. A successful proposal carries weight 1/p, so
 * E[w] is exactly one for every acceptance probability p. Dividing by all
 * scheduled draws therefore stays invariant; dividing by accepted draws is the
 * historical conditional-on-success bias.
 */
function estimateNullEventProposal(
  acceptanceProbability: number,
  seed: number,
): { attemptedMean: number; acceptedOnlyMean: number } {
  const rng = mulberry32(seed);
  const drawsPerTrial = 32;
  const trials = 30_000;
  let attemptedTotal = 0;
  let acceptedOnlyTotal = 0;
  for (let trial = 0; trial < trials; trial++) {
    let weightSum = 0;
    let accepted = 0;
    for (let draw = 0; draw < drawsPerTrial; draw++) {
      if (rng() < acceptanceProbability) {
        weightSum += 1 / acceptanceProbability;
        accepted++;
      }
    }
    attemptedTotal += weightSum / drawsPerTrial;
    acceptedOnlyTotal += accepted === 0 ? 0 : weightSum / accepted;
  }
  return {
    attemptedMean: attemptedTotal / trials,
    acceptedOnlyMean: acceptedOnlyTotal / trials,
  };
}

function between(source: string, begin: string, end: string): string {
  const beginAt = source.indexOf(begin);
  const endAt = source.indexOf(end, beginAt + begin.length);
  expect(beginAt).toBeGreaterThanOrEqual(0);
  expect(endAt).toBeGreaterThan(beginAt);
  return source.slice(beginAt, endAt);
}

interface ScalarReservoir {
  M: number;
  areaM: number;
  envM: number;
  wSum: number;
  W: number;
}

function generalizedFrame(
  areaM: number,
  envM: number,
  acceptanceProbability: number,
  areaIntegral: number,
  envIntegral: number,
  rng: () => number,
): ScalarReservoir {
  const M = areaM + envM;
  let wSum = 0;
  for (let i = 0; i < areaM; i++) {
    if (rng() < acceptanceProbability) {
      wSum += (areaIntegral / acceptanceProbability) * (M / areaM);
    }
  }
  for (let i = 0; i < envM; i++) {
    if (rng() < acceptanceProbability) {
      wSum += (envIntegral / acceptanceProbability) * (M / envM);
    }
  }
  return { M, areaM, envM, wSum, W: wSum / M };
}

function mergeGeneralized(
  current: ScalarReservoir,
  reused: ScalarReservoir,
): ScalarReservoir {
  const areaM = current.areaM + reused.areaM;
  const envM = current.envM + reused.envM;
  const M = areaM + envM;
  const wSum = current.wSum + reused.W * reused.M;
  return { M, areaM, envM, wSum, W: M === 0 ? 0 : wSum / M };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ReSTIR-DI attempted-candidate accounting', () => {
  it('keeps the estimator invariant as null-event probability changes', () => {
    const probabilities = [0.2, 0.5, 0.8];
    const estimates = probabilities.map((p, index) =>
      estimateNullEventProposal(p, 0x1234_0000 + index)
    );

    for (const estimate of estimates) {
      expect(estimate.attemptedMean).toBeGreaterThan(0.98);
      expect(estimate.attemptedMean).toBeLessThan(1.02);
    }
    const attemptedSpread = Math.max(...estimates.map((e) => e.attemptedMean)) -
      Math.min(...estimates.map((e) => e.attemptedMean));
    expect(attemptedSpread).toBeLessThan(0.02);

    // Characterization control: accepted-only M estimates E[w | accepted],
    // which changes approximately as 1/p and is decisively non-invariant.
    expect(estimates[0]!.acceptedOnlyMean).toBeGreaterThan(4.8);
    expect(estimates[2]!.acceptedOnlyMean).toBeLessThan(1.3);
  });

  it('counts each scheduled support-family draw before rejection guards', () => {
    const lightLoop = between(
      RIS_WGSL,
      'for (var i = 0u; i < M_LIGHT; i++) {',
      '// --- M_ENV candidate(s)',
    );
    const envLoop = between(
      RIS_WGSL,
      'for (var ei = 0u; ei < M_ENV; ei++) {',
      '// --- Visibility test on chosen candidate ---',
    );

    const areaCountAt = lightLoop.indexOf('mAreaSupport = mAreaSupport + 1u;');
    expect(areaCountAt).toBeGreaterThanOrEqual(0);
    expect(areaCountAt).toBeLessThan(lightLoop.indexOf('continue;'));
    expect(lightLoop.match(/mAreaSupport = mAreaSupport \+ 1u;/g)).toHaveLength(1);
    expect(RIS_WGSL).toContain('let scheduledAreaM = M_LIGHT;');
    expect(RIS_WGSL).not.toContain('M_BRDF');

    const activationGuard = envLoop.indexOf('if (!envStrategyActive) { continue; }');
    const countAt = envLoop.indexOf('mEnvSupport = mEnvSupport + 1u;');
    const pdfReject = envLoop.indexOf('if (!(envS.pdf > 0.0)) { continue; }');
    expect(activationGuard).toBeGreaterThanOrEqual(0);
    expect(countAt).toBeGreaterThan(activationGuard);
    expect(countAt).toBeLessThan(pdfReject);
    expect(envLoop.match(/mEnvSupport = mEnvSupport \+ 1u;/g)).toHaveLength(1);
  });

  it('estimates the SUM of disjoint-domain integrals for unequal proposal counts', () => {
    const areaIntegral = 2;
    const envIntegral = 3;
    for (const [areaM, envM, p] of [
      [1, 1, 0.2],
      [65, 1, 0.5],
      [4, 17, 0.8],
    ] as const) {
      const rng = mulberry32(0x9000 + areaM * 31 + envM);
      let sum = 0;
      const trials = 40_000;
      for (let i = 0; i < trials; i++) {
        sum += generalizedFrame(areaM, envM, p, areaIntegral, envIntegral, rng).W;
      }
      expect(sum / trials).toBeGreaterThan(4.94);
      expect(sum / trials).toBeLessThan(5.06);
    }
  });

  it('retains all-null frame multiplicity through temporal and spatial reuse', () => {
    const previous: ScalarReservoir = {
      M: 66,
      areaM: 65,
      envM: 1,
      wSum: 330,
      W: 5,
    };
    const nullCurrent: ScalarReservoir = {
      M: 66,
      areaM: 65,
      envM: 1,
      wSum: 0,
      W: 0,
    };

    const temporal = mergeGeneralized(nullCurrent, previous);
    expect(temporal).toMatchObject({ M: 132, areaM: 130, envM: 2, W: 2.5 });
    const spatial = mergeGeneralized(temporal, nullCurrent);
    expect(spatial).toMatchObject({ M: 198, areaM: 195, envM: 3 });
    expect(spatial.W).toBeCloseTo(5 / 3, 12);

    expect(TEMPORAL_WGSL).toContain(
      'combined.areaM = reservoirDiSaturatingAddU32(current.areaM, previous.areaM);',
    );
    expect(TEMPORAL_WGSL).toContain(
      'combined.M = reservoirDiSaturatingAddU32(current.M, previous.M);',
    );
    expect(SPATIAL_WGSL).toContain(
      'areaSupport = reservoirDiSaturatingAddU32(areaSupport, source.areaM);',
    );
    expect(SPATIAL_WGSL).toContain('output.M = representedAttempts;');
  });

  it('keeps reused estimates invariant as whole-frame null probability changes', () => {
    for (const nullProbability of [0.2, 0.5, 0.8]) {
      const rng = mulberry32(0xa500 + Math.floor(nullProbability * 100));
      let total = 0;
      const trials = 20_000;
      for (let trial = 0; trial < trials; trial++) {
        let history: ScalarReservoir = { M: 0, areaM: 0, envM: 0, wSum: 0, W: 0 };
        for (let frame = 0; frame < 4; frame++) {
          const nonNull = rng() >= nullProbability;
          const value = nonNull ? 5 / (1 - nullProbability) : 0;
          const current: ScalarReservoir = {
            M: 66,
            areaM: 65,
            envM: 1,
            wSum: value * 66,
            W: value,
          };
          history = mergeGeneralized(current, history);
        }
        total += history.W;
      }
      expect(total / trials).toBeGreaterThan(4.9);
      expect(total / trials).toBeLessThan(5.1);
    }
  });

  it('packs support counts and allocates the matching 32-byte host stride', () => {
    expect(RESERVOIR_DI_STRIDE_U32).toBe(8);
    expect(RESERVOIR_DI_STRIDE_BYTES).toBe(32);
    expect(RESERVOIR_DI_WGSL).toContain('const RESERVOIR_DI_STRIDE = 8u;');
    expect(RESERVOIR_DI_WGSL).toMatch(/r\.areaM,\s*r\.envM,/);
    expect(RESERVOIR_DI_WGSL).toContain('words[6u],');
    expect(RESERVOIR_DI_WGSL).toContain('words[7u],');

    vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4 });
    const sizes: number[] = [];
    const device = {
      createBuffer: ({ size }: { size: number }) => {
        sizes.push(size);
        return {};
      },
    } as unknown as GPUDevice;
    createRestirDIFrameResources(device, 5, 7);
    expect(sizes).toEqual([5 * 7 * 32, 5 * 7 * 32, 5 * 7 * 32]);
  });

  it('pins generalized source scaling and null/occluded M persistence', () => {
    expect(RIS_WGSL).toContain(
      'let areaRisScale = f32(scheduledTotalM) / f32(max(1u, scheduledAreaM));',
    );
    expect(RIS_WGSL).toContain(
      'let envRisScale = f32(scheduledTotalM) / f32(max(1u, scheduledEnvM));',
    );
    expect(RIS_WGSL).toContain('r.areaM = mAreaSupport;');
    expect(RIS_WGSL).toContain('r.envM = mEnvSupport;');
    expect(RIS_WGSL).toContain('r.M = mAreaSupport + mEnvSupport;');
    expect(RIS_WGSL).toMatch(
      /r\.M = mAreaSupport \+ mEnvSupport;[\s\S]*\/\/ --- Visibility test/,
    );
  });
});
