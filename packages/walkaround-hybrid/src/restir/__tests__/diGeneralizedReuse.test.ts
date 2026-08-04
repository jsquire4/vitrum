import { describe, expect, it } from 'vitest';
import {
  finalizeGeneralizedDiReservoirLogWeight,
  generalizedDiReuseCandidateLogWeight,
  generalizedDiReuseCandidateWeight,
  generalizedTalbotMisWeight,
} from './support/diGeneralizedReuse.js';
import { RESERVOIR_DI_WGSL } from '../../shaders/reservoirDi.wgsl.js';

interface CpuReservoir {
  readonly selected: number;
  readonly weight: number;
}

interface StratifiedCpuReservoir extends CpuReservoir {
  readonly kind: 'area' | 'environment';
}

function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function sampleDiscrete(probabilities: readonly number[], random: () => number): number {
  const target = random();
  let cumulative = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index]!;
    if (target < cumulative) return index;
  }
  return probabilities.length - 1;
}

function buildReservoir(
  target: readonly number[],
  proposal: readonly number[],
  attempts: number,
  random: () => number,
): CpuReservoir {
  let selected = 0;
  let weightSum = 0;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const candidate = sampleDiscrete(proposal, random);
    const candidateWeight = target[candidate]! / proposal[candidate]!;
    weightSum += candidateWeight;
    if (random() * weightSum < candidateWeight) selected = candidate;
  }
  return {
    selected,
    weight: weightSum / (attempts * target[selected]!),
  };
}

function buildStratifiedReservoir(
  target: readonly number[],
  areaProposal: readonly number[],
  environmentProposal: readonly number[],
  areaAttempts: number,
  environmentAttempts: number,
  random: () => number,
): StratifiedCpuReservoir {
  const totalAttempts = areaAttempts + environmentAttempts;
  let selected = 0;
  let kind: StratifiedCpuReservoir['kind'] = 'area';
  let weightSum = 0;
  const consider = (
    candidate: number,
    proposalDensity: number,
    domainAttempts: number,
    candidateKind: StratifiedCpuReservoir['kind'],
  ): void => {
    const candidateWeight =
      (target[candidate]! / proposalDensity) *
      (totalAttempts / domainAttempts);
    weightSum += candidateWeight;
    if (random() * weightSum < candidateWeight) {
      selected = candidate;
      kind = candidateKind;
    }
  };
  for (let attempt = 0; attempt < areaAttempts; attempt += 1) {
    const candidate = sampleDiscrete(areaProposal, random);
    consider(candidate, areaProposal[candidate]!, areaAttempts, 'area');
  }
  for (let attempt = 0; attempt < environmentAttempts; attempt += 1) {
    const localCandidate = sampleDiscrete(environmentProposal, random);
    consider(
      2 + localCandidate,
      environmentProposal[localCandidate]!,
      environmentAttempts,
      'environment',
    );
  }
  return {
    selected,
    kind,
    weight: weightSum / (totalAttempts * target[selected]!),
  };
}

function monteCarloReuseMean(
  targets: readonly (readonly number[])[],
  proposals: readonly (readonly number[])[],
  attempts: readonly number[],
  trialCount: number,
): number {
  const random = lcg(0x4d595df4);
  let estimateSum = 0;
  for (let trial = 0; trial < trialCount; trial += 1) {
    const reservoirs = targets.map((target, index) =>
      buildReservoir(target, proposals[index]!, attempts[index]!, random)
    );
    let outputContribution = 0;
    for (let sourceIndex = 0; sourceIndex < reservoirs.length; sourceIndex += 1) {
      const reservoir = reservoirs[sourceIndex]!;
      outputContribution += generalizedDiReuseCandidateWeight({
        sourceIndex,
        attempts,
        densitiesAtDomains: targets.map(
          (target) => target[reservoir.selected]!,
        ),
        sourceLogEstimatorNumerator: Math.log2(
          reservoir.weight * targets[sourceIndex]![reservoir.selected]!,
        ),
      });
    }
    estimateSum += outputContribution;
  }
  return estimateSum / trialCount;
}

describe('ReSTIR-DI generalized Talbot / GRIS CPU oracle', () => {
  it('uses represented attempts in the all-technique balance heuristic', () => {
    const weight = generalizedTalbotMisWeight({
      sourceIndex: 1,
      attempts: [2, 6, 4],
      densitiesAtDomains: [0.5, 0.25, 1],
    });
    expect(weight).toBeCloseTo((6 * 0.25) / (2 * 0.5 + 6 * 0.25 + 4 * 1), 12);
  });

  it('keeps an extreme but finite six-technique candidate alive', () => {
    const attempts = Array.from({ length: 6 }, () => 0xffff_ffff);
    const densitiesAtDomains = Array.from(
      { length: 6 },
      () => 3.4e38,
    );
    const misWeight = generalizedTalbotMisWeight({
      sourceIndex: 3,
      attempts,
      densitiesAtDomains,
    });
    const candidateWeight = generalizedDiReuseCandidateWeight({
      sourceIndex: 3,
      attempts,
      densitiesAtDomains,
      sourceReservoirWeight: 3.4e38,
    });
    expect(misWeight).toBeCloseTo(1 / 6, 12);
    expect(candidateWeight).toBeGreaterThan(0);
    expect(Number.isFinite(candidateWeight)).toBe(true);
  });

  it('preserves unequal extreme-density ratios instead of equalizing saturated terms', () => {
    const attempts = [0xffff_ffff, 0xffff_ffff] as const;
    const densitiesAtDomains = [3.4e38, 3.4e28] as const;
    const dominantWeight = generalizedTalbotMisWeight({
      sourceIndex: 0,
      attempts,
      densitiesAtDomains,
    });
    const minorWeight = generalizedTalbotMisWeight({
      sourceIndex: 1,
      attempts,
      densitiesAtDomains,
    });
    expect(dominantWeight).toBeCloseTo(1 / (1 + 1e-10), 12);
    expect(minorWeight).toBeCloseTo(1e-10 / (1 + 1e-10), 20);
    expect(dominantWeight).toBeGreaterThan(0.999_999_999);
    expect(minorWeight).toBeLessThan(1e-9);
  });

  it('preserves unequal ratios when two complete finite candidate products overflow', () => {
    const common = {
      attempts: [1, 1] as const,
      densitiesAtDomains: [3e38, 3e38] as const,
    };
    const logWeights = [
      generalizedDiReuseCandidateLogWeight({
        ...common,
        sourceIndex: 0,
        sourceReservoirWeight: 3e38,
      }),
      generalizedDiReuseCandidateLogWeight({
        ...common,
        sourceIndex: 1,
        sourceReservoirWeight: 3e28,
      }),
    ];
    expect(logWeights[0]! - logWeights[1]!).toBeCloseTo(
      Math.log2(1e10),
      12,
    );
    const finalized = finalizeGeneralizedDiReservoirLogWeight(
      logWeights[0]!,
      3e38,
    );
    expect(finalized.H).toBe(logWeights[0]);
    expect(finalized.logW).toBeGreaterThan(0);
    expect(Number.isFinite(finalized.logW)).toBe(true);
  });

  it('reconstructs source logW from H when a linear f32 W would be zero', () => {
    const sourceDensity = 2 ** -100;
    const logWeight = generalizedDiReuseCandidateLogWeight({
      sourceIndex: 0,
      attempts: [1, 1],
      densitiesAtDomains: [sourceDensity, sourceDensity],
      sourceLogEstimatorNumerator: -300,
    });
    expect(logWeight).toBeCloseTo(-301, 12);
    expect(Number.isFinite(logWeight)).toBe(true);
  });

  it('is unbiased for current plus previous reservoirs with unequal support', () => {
    const targets = [
      [0.2, 1.1, 0.7, 0.4],
      [1.0, 0.1, 0.4, 0.9],
    ] as const;
    const proposals = [
      [0.1, 0.3, 0.4, 0.2],
      [0.35, 0.15, 0.2, 0.3],
    ] as const;
    const mean = monteCarloReuseMean(targets, proposals, [1, 5], 60_000);
    expect(mean).toBeCloseTo(targets[0].reduce((sum, value) => sum + value, 0), 1);
  });

  it('is unbiased for center plus five neighbors with heterogeneous support', () => {
    const targets = [
      [0.3, 0.8, 1.2, 0.5],
      [1.1, 0.4, 0.2, 0.9],
      [0.6, 1.3, 0.4, 0.3],
      [0.2, 0.5, 1.4, 0.7],
      [0.9, 0.2, 0.8, 0.6],
      [0.4, 1.0, 0.3, 1.1],
    ] as const;
    const proposals = [
      [0.1, 0.2, 0.4, 0.3],
      [0.3, 0.3, 0.1, 0.3],
      [0.25, 0.35, 0.25, 0.15],
      [0.15, 0.2, 0.45, 0.2],
      [0.4, 0.1, 0.2, 0.3],
      [0.2, 0.4, 0.1, 0.3],
    ] as const;
    const mean = monteCarloReuseMean(
      targets,
      proposals,
      [1, 2, 3, 4, 5, 6],
      50_000,
    );
    expect(mean).toBeCloseTo(targets[0].reduce((sum, value) => sum + value, 0), 1);
  });

  it('uses areaM/envM rather than total M for heterogeneous stratified supports', () => {
    const targets = [
      [0.7, 1.3, 0.4, 0.9],
      [1.2, 0.3, 1.1, 0.2],
      [0.4, 0.8, 0.6, 1.4],
    ] as const;
    const areaProposals = [
      [0.25, 0.75],
      [0.6, 0.4],
      [0.35, 0.65],
    ] as const;
    const environmentProposals = [
      [0.8, 0.2],
      [0.3, 0.7],
      [0.55, 0.45],
    ] as const;
    const areaAttempts = [8, 3, 12] as const;
    const environmentAttempts = [1, 6, 2] as const;
    const random = lcg(0x6d2b79f5);
    let estimateSum = 0;
    const trialCount = 60_000;
    for (let trial = 0; trial < trialCount; trial += 1) {
      const reservoirs = targets.map((target, index) =>
        buildStratifiedReservoir(
          target,
          areaProposals[index]!,
          environmentProposals[index]!,
          areaAttempts[index]!,
          environmentAttempts[index]!,
          random,
        )
      );
      let contribution = 0;
      for (let sourceIndex = 0; sourceIndex < reservoirs.length; sourceIndex += 1) {
        const reservoir = reservoirs[sourceIndex]!;
        contribution += generalizedDiReuseCandidateWeight({
          sourceIndex,
          attempts: areaAttempts.map((areaAttempt, domainIndex) =>
            reservoir.kind === 'area'
              ? areaAttempt
              : environmentAttempts[domainIndex]!
          ),
          densitiesAtDomains: targets.map(
            (target) => target[reservoir.selected]!,
          ),
          sourceLogEstimatorNumerator: Math.log2(
            reservoir.weight * targets[sourceIndex]![reservoir.selected]!,
          ),
        });
      }
      estimateSum += contribution;
    }
    expect(estimateSum / trialCount).toBeCloseTo(
      targets[0].reduce((sum, value) => sum + value, 0),
      1,
    );
  });

  it('finalizes H/logW without dividing by represented attempts twice', () => {
    const finalized = finalizeGeneralizedDiReservoirLogWeight(
      Math.log2(12),
      3,
    );
    expect(finalized.H).toBeCloseTo(Math.log2(12), 15);
    expect(finalized.logW).toBeCloseTo(2, 15);

    const highLog = finalizeGeneralizedDiReservoirLogWeight(200, 2 ** -100);
    expect(highLog.logW).toBe(300);
    expect(highLog.logW).toBeGreaterThan(Math.log2(3.402823466e38));

    expect(finalizeGeneralizedDiReservoirLogWeight(12, 0).logW)
      .toBe(Number.NEGATIVE_INFINITY);
    expect(finalizeGeneralizedDiReservoirLogWeight(Number.NaN, 3).logW)
      .toBe(Number.NEGATIVE_INFINITY);
  });

  it('pins the production WGSL equation and identity area-measure Jacobian', () => {
    expect(RESERVOIR_DI_WGSL).toContain(
      'm_i(y) = M_i pHat_i(y) / sum_j M_j pHat_j(y)',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'RESERVOIR_DI_EMITTER_AREA_SHIFT_JACOBIAN: f32 = 1.0',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'sourceLogEstimatorNumerator - log2(sourceDensity)',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'representedWrsSelectedLogCorrectionParts(wrs)',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'logW:    f32',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      '(*r).logW = min(result, RESERVOIR_DI_MAX_FINITE_F32)',
    );
    expect(RESERVOIR_DI_WGSL).not.toContain(
      'min(result, log2(RESERVOIR_DI_MAX_FINITE_F32))',
    );
    expect(RESERVOIR_DI_WGSL).not.toContain('let rawW = exp2(logW)');
    expect(RESERVOIR_DI_WGSL).toContain(
      'log2(f32(attempts)) + log2(density)',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'exp2(min(0.0, logDensity - maxLogDensity))',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'fn reservoirDiGeneralizedReuseLogWeight(',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'logTechniqueDenominator +',
    );
    expect(RESERVOIR_DI_WGSL).not.toContain(
      'RESERVOIR_DI_MAX_REUSE_WEIGHT',
    );
  });
});
