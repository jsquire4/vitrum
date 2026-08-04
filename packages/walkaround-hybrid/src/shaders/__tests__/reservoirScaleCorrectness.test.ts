import type { Scene } from '@vitrum/core';
import { describe, expect, it } from 'vitest';
import { validateHybridEngineOptions } from '../../HybridEngineConfig.js';
import {
  sceneRequiresFullRateGlassShading,
} from '../../reservoirScalePolicy.js';
import { RESERVOIR_GI_WGSL } from '../reservoirGi.wgsl.js';
import { RESTIR_GI_MATERIAL_WGSL } from '../restirGiMaterial.wgsl.js';
import { RESTIR_PHAT_WGSL } from '../restirPHat.wgsl.js';
import { RESERVOIR_DI_WGSL } from '../reservoirDi.wgsl.js';
import { RIS_WGSL } from '../ris.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { SPATIAL_WGSL } from '../spatial.wgsl.js';
import { SPATIAL_GI_WGSL } from '../spatialGi.wgsl.js';
import { TEMPORAL_WGSL } from '../temporal.wgsl.js';
import { TEMPORAL_GI_WGSL } from '../temporalGi.wgsl.js';

interface CoarseCandidate {
  readonly sourceTarget: number;
  readonly pdf: number;
  readonly strategyCandidateCount: number;
  readonly receiverAContribution: number;
  readonly receiverBContribution: number;
}

function conditionalReservoirExpectation(
  candidates: readonly CoarseCandidate[],
  contribution: (candidate: CoarseCandidate) => number,
): number {
  const totalCandidates = candidates.length;
  const weights = candidates.map((candidate) =>
    (candidate.sourceTarget / candidate.pdf)
    * (totalCandidates / candidate.strategyCandidateCount));
  const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
  return candidates.reduce((expectation, candidate, index) => {
    const selectionProbability = weights[index]! / weightSum;
    const selectedW =
      weightSum / (totalCandidates * candidate.sourceTarget);
    return expectation
      + selectionProbability * contribution(candidate) * selectedW;
  }, 0);
}

function directStratifiedReference(
  candidates: readonly CoarseCandidate[],
  contribution: (candidate: CoarseCandidate) => number,
): number {
  return candidates.reduce((sum, candidate) =>
    sum + contribution(candidate)
      / (candidate.strategyCandidateCount * candidate.pdf), 0);
}

type Vec3 = readonly [number, number, number];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function geometryTerm(receiver: Vec3, sample: Vec3, sampleNormal: Vec3): number {
  const d = subtract(sample, receiver);
  const distanceSquared = dot(d, d);
  const inverseDistance = 1 / Math.sqrt(distanceSquared);
  return Math.abs(dot(sampleNormal, [
    d[0] * inverseDistance,
    d[1] * inverseDistance,
    d[2] * inverseDistance,
  ])) / distanceSquared;
}

function coarseGiCompatible(
  reservoir: {
    readonly receiverMaterialKey: number;
    readonly position: Vec3;
    readonly normal: Vec3;
  },
  receiver: {
    readonly receiverMaterialKey: number;
    readonly position: Vec3;
    readonly normal: Vec3;
  },
  normalMinimum: number,
  coplanarTolerance: number,
): boolean {
  return reservoir.receiverMaterialKey === receiver.receiverMaterialKey
    && dot(receiver.normal, reservoir.normal) >= normalMinimum
    && Math.abs(dot(
      subtract(receiver.position, reservoir.position),
      reservoir.normal,
    )) <= coplanarTolerance;
}

function receiverLocalFallback(
  compatibleContributions: readonly number[],
  ddgiIrradiance: number,
): number {
  if (compatibleContributions.length === 0) {
    return ddgiIrradiance / Math.PI;
  }
  return compatibleContributions.reduce((sum, value) => sum + value, 0)
    / compatibleContributions.length;
}

describe('scaled ReSTIR-DI receiver-independent estimator', () => {
  it('uses exact source PDFs/W and remains correct for occluder-separated receivers', () => {
    // Two area candidates (strategy count 2) plus one environment candidate
    // (strategy count 1). Receiver A and B see disjoint area candidates.
    const candidates: readonly CoarseCandidate[] = [
      {
        sourceTarget: 4,
        pdf: 0.25 * 0.5,
        strategyCandidateCount: 2,
        receiverAContribution: 4,
        receiverBContribution: 0,
      },
      {
        sourceTarget: 2,
        pdf: 0.5 * 0.5,
        strategyCandidateCount: 2,
        receiverAContribution: 0,
        receiverBContribution: 3,
      },
      {
        sourceTarget: 1,
        pdf: 0.2,
        strategyCandidateCount: 1,
        receiverAContribution: 1,
        receiverBContribution: 2,
      },
    ];
    const estimateA = conditionalReservoirExpectation(
      candidates,
      (candidate) => candidate.receiverAContribution,
    );
    const estimateB = conditionalReservoirExpectation(
      candidates,
      (candidate) => candidate.receiverBContribution,
    );
    expect(estimateA).toBeCloseTo(
      directStratifiedReference(
        candidates,
        (candidate) => candidate.receiverAContribution,
      ),
      12,
    );
    expect(estimateB).toBeCloseTo(
      directStratifiedReference(
        candidates,
        (candidate) => candidate.receiverBContribution,
      ),
      12,
    );
    expect(estimateA).toBeCloseTo(21, 12);
    expect(estimateB).toBeCloseTo(16, 12);
    expect(estimateB).toBeGreaterThan(0);
  });

  it('pins source-only proposal support and current-receiver visibility', () => {
    const sourceTargetBody = RESTIR_PHAT_WGSL.slice(
      RESTIR_PHAT_WGSL.indexOf('fn restir_di_coarse_proposal_phat('),
      RESTIR_PHAT_WGSL.indexOf('// ============================================================', 40),
    );
    expect(sourceTargetBody).toContain('sampleEmitterLeAtXi(');
    expect(sourceTargetBody).toContain('envRadiance(');
    expect(sourceTargetBody).not.toMatch(/PrimarySurface|BRDF|Visibility|traceScene/);
    expect(RIS_WGSL).toContain('reservoirDiInitialCandidateLogWeight(');
    expect(RIS_WGSL).toContain('reservoirDiPositiveLog2(emitterSelPmf)');
    expect(RIS_WGSL).toContain(
      'finaliseReservoirDIFromNativeWrs(&r, wrs, pHat)',
    );
    expect(RESERVOIR_DI_WGSL).toContain(
      'r.logEstimatorNumerator + log2(f32(r.M))',
    );
    expect(TEMPORAL_WGSL).toContain(
      'reservoirDiCoarseReuseLogWeight(previous)',
    );
    expect(SPATIAL_WGSL).toContain(
      'reservoirDiCoarseReuseLogWeight(neighbor)',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'let geometryLogW = restirShadeAppendPositiveFactor(r.logW, G);',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'r.logW, envColor, layeredBrdfE, shadowTint',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'geometryLogW, Le, layeredBrdf, shadowTint,',
    );
    expect(SHADING_TERMS_WGSL).toContain('restirShadeDirectionalVolumeLog(');
    expect(SHADING_TERMS_WGSL).not.toMatch(/\br\.W\b/);
  });
});

describe('scaled ReSTIR-GI shift compatibility', () => {
  const stored = {
    receiverMaterialKey: 7,
    position: [0, 0, 0] as Vec3,
    normal: [0, 0, 1] as Vec3,
  };

  it('rejects material/primitive and depth/normal discontinuities without bleed', () => {
    expect(coarseGiCompatible(stored, {
      receiverMaterialKey: 7,
      position: [2, 0, 0],
      normal: [0, 0, 1],
    }, 0.906, 0.05)).toBe(true);
    expect(coarseGiCompatible(stored, {
      receiverMaterialKey: 8,
      position: [2, 0, 0],
      normal: [0, 0, 1],
    }, 0.906, 0.05)).toBe(false);
    expect(coarseGiCompatible(stored, {
      receiverMaterialKey: 7,
      position: [0, 0, 0.051],
      normal: [0, 0, 1],
    }, 0.906, 0.05)).toBe(false);
    expect(coarseGiCompatible(stored, {
      receiverMaterialKey: 7,
      position: [0, 0, 0],
      normal: [0, 0, -1],
    }, 0.906, 0.05)).toBe(false);
  });

  it('uses the exact reconnection Jacobian and a deterministic receiver-local fallback', () => {
    const sourceReceiver: Vec3 = [0, 0, 0];
    const currentReceiver: Vec3 = [1, 0, 0];
    const sample: Vec3 = [0, 0, 2];
    const sampleNormal: Vec3 = [0, 0, -1];
    const expectedJacobian =
      geometryTerm(currentReceiver, sample, sampleNormal)
      / geometryTerm(sourceReceiver, sample, sampleNormal);
    expect(expectedJacobian).toBeCloseTo(8 / (5 * Math.sqrt(5)), 12);

    const fallbackA = receiverLocalFallback([], 3);
    const fallbackB = receiverLocalFallback([], 3);
    expect(fallbackA).toBe(fallbackB);
    expect(fallbackA).toBeCloseTo(3 / Math.PI, 15);
    // An unrelated incompatible reservoir value cannot affect the fallback.
    expect(receiverLocalFallback([], 3)).not.toBe(
      receiverLocalFallback([1_000_000], 3),
    );
  });

  it('pins key propagation, GRIS Jacobian/current visibility, and fallback in WGSL', () => {
    expect(RESERVOIR_GI_WGSL).toContain('receiverMaterialKey: u32');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'fn restir_gi_receiver_domain_key(',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'h = restir_gi_receiver_key_mix(h, triangleId);',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'h = restir_gi_receiver_key_mix(h, instanceId);',
    );
    expect(RIS_GI_WGSL).toContain('r.receiverMaterialKey =');
    expect(TEMPORAL_GI_WGSL).toContain(
      'previous.receiverMaterialKey != current.receiverMaterialKey',
    );
    expect(SPATIAL_GI_WGSL).toContain(
      'q.receiverMaterialKey != rCenter.receiverMaterialKey',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'g.receiverMaterialKey == receiverMaterialKey',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'logDomainJacobian = grisLogDomainToCanonicalJacobian(',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'restirShadeAppendLogFactor(g.logW, logDomainJacobian)',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'grisTint = grisProxyTintAt(',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'Lo_indirect = restirShadeAggregateDiffuseDemodulated(',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'sampleDDGIAtPoint(pos, normal) * INV_PI,',
    );
  });
});

describe('full-rate camera-prefix and learned-layout policy', () => {
  const sceneWithMaterial = (material: Record<string, unknown>): Scene => ({
    primitives: [{ kind: 'mesh', material }] as unknown as Scene['primitives'],
    emitters: [],
    environment: { kind: 'none' },
  });

  it('requires full-rate shading for positive scalar transmission only', () => {
    expect(sceneRequiresFullRateGlassShading(
      sceneWithMaterial({ transmission: 0 }),
    )).toBe(false);
    expect(sceneRequiresFullRateGlassShading(
      sceneWithMaterial({ transmission: 0.01 }),
    )).toBe(true);
    expect(sceneRequiresFullRateGlassShading(
      sceneWithMaterial({ transmission: 0, transmissionMap: {} }),
    )).toBe(false);
  });

  it('rejects explicit coarse scale for PPG/NRC training layouts', () => {
    const base = {
      device: {} as GPUDevice,
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0] as [number, number, number],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1] as [number, number, number],
      skyIrradiance: 1,
    };
    expect(() => validateHybridEngineOptions({
      ...base,
      restirReservoirScale: 2,
      ppgEnabled: true,
    })).toThrow(/incompatible with PPG\/NRC training layouts/);
    expect(() => validateHybridEngineOptions({
      ...base,
      restirReservoirScale: 4,
      nrcEnabled: true,
    })).toThrow(/incompatible with PPG\/NRC training layouts/);
  });
});
