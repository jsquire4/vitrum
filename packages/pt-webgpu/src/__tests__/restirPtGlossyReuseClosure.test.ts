import { describe, expect, it } from 'vitest';

import {
  RESERVOIR_PT_HERO_WGSL,
  RESTIR_PT_PARAMS_FIELDS,
} from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { RESTIR_PT_RESOLVE_WGSL } from '../wgsl/pathTrace/restirPtResolve.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from '../wgsl/pathTrace/restirPtTemporal.wgsl.js';
import { composePtWebgpuCompositeTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';

type Vec3 = readonly [number, number, number];

function addScaled(a: Vec3, b: Vec3, scale: number): Vec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(v: Vec3): number {
  return Math.sqrt(dot(v, v));
}

function normalize(v: Vec3): Vec3 {
  const inverseLength = 1 / length(v);
  return [v[0] * inverseLength, v[1] * inverseLength, v[2] * inverseLength];
}

function direction(from: Vec3, to: Vec3): Vec3 {
  return normalize(sub(to, from));
}

function tangentBasis(normal: Vec3): readonly [Vec3, Vec3] {
  const helper: Vec3 = Math.abs(normal[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const tangent = normalize(cross(helper, normal));
  return [tangent, cross(normal, tangent)];
}

function halfGeometry(preVertex: Vec3, reconnection: Vec3, reconnectionNormal: Vec3): number {
  const edge = sub(preVertex, reconnection);
  const distance = length(edge);
  return Math.abs(dot(reconnectionNormal, edge) / distance) / (distance * distance);
}

function solidAngleAreaFiniteDifference(
  preVertex: Vec3,
  reconnection: Vec3,
  reconnectionNormal: Vec3,
): number {
  const [tangent, bitangent] = tangentBasis(reconnectionNormal);
  const h = 1e-5;
  const tangentPlus = direction(preVertex, addScaled(reconnection, tangent, h));
  const tangentMinus = direction(preVertex, addScaled(reconnection, tangent, -h));
  const bitangentPlus = direction(preVertex, addScaled(reconnection, bitangent, h));
  const bitangentMinus = direction(preVertex, addScaled(reconnection, bitangent, -h));
  const derivativeT: Vec3 = [
    (tangentPlus[0] - tangentMinus[0]) / (2 * h),
    (tangentPlus[1] - tangentMinus[1]) / (2 * h),
    (tangentPlus[2] - tangentMinus[2]) / (2 * h),
  ];
  const derivativeB: Vec3 = [
    (bitangentPlus[0] - bitangentMinus[0]) / (2 * h),
    (bitangentPlus[1] - bitangentMinus[1]) / (2 * h),
    (bitangentPlus[2] - bitangentMinus[2]) / (2 * h),
  ];
  return length(cross(derivativeT, derivativeB));
}

interface SurfaceIdentity {
  materialId: number;
  instanceIndex: number;
  triangleIndex: number;
  surfaceParam: Vec3;
}

function temporalSurfaceIdentityMatches(
  current: SurfaceIdentity,
  previous: SurfaceIdentity,
  triangleCount: number,
): boolean {
  if (
    current.materialId !== previous.materialId
    || current.instanceIndex !== previous.instanceIndex
    || current.triangleIndex !== previous.triangleIndex
  ) {
    return false;
  }
  const delta = sub(current.surfaceParam, previous.surfaceParam);
  if (current.triangleIndex < triangleCount) {
    return Math.hypot(delta[0], delta[1]) <= 0.08;
  }
  const scale = Math.max(1, length(current.surfaceParam));
  return length(delta) <= 0.03 * scale;
}

function temporalNormalIsValid(normal: Vec3): boolean {
  const lengthSquared = dot(normal, normal);
  return normal.every(Number.isFinite)
    && lengthSquared >= 0.5
    && lengthSquared <= 1.5;
}

interface TemporalCandidate extends SurfaceIdentity {
  M: number;
  W: number;
  normal: Vec3;
}

function selectTemporalCandidateOrCurrent(
  current: TemporalCandidate,
  candidates: readonly TemporalCandidate[],
  triangleCount: number,
): TemporalCandidate {
  let selected: TemporalCandidate | undefined;
  let bestSurfaceDistance = Number.POSITIVE_INFINITY;
  let ambiguous = false;
  for (const candidate of candidates) {
    if (
      candidate.M <= 0
      || candidate.W <= 0
      || !temporalNormalIsValid(candidate.normal)
      || !temporalSurfaceIdentityMatches(current, candidate, triangleCount)
    ) {
      continue;
    }
    const delta = sub(current.surfaceParam, candidate.surfaceParam);
    const surfaceDistance = current.triangleIndex < triangleCount
      ? Math.hypot(delta[0], delta[1])
      : length(delta) / Math.max(1, length(current.surfaceParam));
    if (selected == null || surfaceDistance + 1e-6 < bestSurfaceDistance) {
      selected = candidate;
      bestSurfaceDistance = surfaceDistance;
      ambiguous = false;
    } else if (Math.abs(surfaceDistance - bestSurfaceDistance) <= 1e-6) {
      ambiguous = true;
    }
  }
  return selected == null || ambiguous ? current : selected;
}

function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '');
}

function makeLcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

describe('ReSTIR-PT glossy reuse closure — change of variables and reservoir algebra', () => {
  it('matches the prefix-1 solid-angle CoV with the geometry-only half-G ratio', () => {
    const cases: ReadonlyArray<{
      source: Vec3;
      target: Vec3;
      reconnection: Vec3;
      normal: Vec3;
    }> = [
      {
        source: [-1.2, 0.4, 2.8],
        target: [0.7, -0.3, 1.9],
        reconnection: [0.2, 0.1, 0],
        normal: [0, 0, 1],
      },
      {
        source: [2.1, -0.8, 3.4],
        target: [-0.6, 1.3, 2.2],
        reconnection: [0.1, -0.2, 0.3],
        normal: normalize([0.2, -0.3, 0.932]),
      },
      {
        source: [-0.9, -1.4, 1.7],
        target: [1.8, 0.6, 4.1],
        reconnection: [-0.2, 0.5, -0.1],
        normal: normalize([-0.4, 0.1, 0.91]),
      },
    ];

    for (const config of cases) {
      const analytic = halfGeometry(config.target, config.reconnection, config.normal)
        / halfGeometry(config.source, config.reconnection, config.normal);
      const finiteDifference = solidAngleAreaFiniteDifference(
        config.target,
        config.reconnection,
        config.normal,
      ) / solidAngleAreaFiniteDifference(
        config.source,
        config.reconnection,
        config.normal,
      );
      expect(finiteDifference).toBeCloseTo(analytic, 6);

      const sourceProposalPdf = 0.37;
      const targetProposalPdf = 1.41;
      const doubleCountedProposalRatio = analytic * sourceProposalPdf / targetProposalPdf;
      expect(Math.abs(doubleCountedProposalRatio - finiteDifference))
        .toBeGreaterThan(0.25 * finiteDifference);
    }

    expect(RESERVOIR_PT_HERO_WGSL).toContain('fn restirPtReconnectionJacobianForPair(');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('source.xv, targetDomain.xv, source.xs, source.ns,');
    expect(RESERVOIR_PT_HERO_WGSL).not.toContain('pSource / pTarget');
  });

  it('cancels the producer target once and never applies a second proposal ratio', () => {
    const cases = [
      { pHatSource: 0.15, pSource: 0.02, pHatTarget: 2.8, pTargetProposal: 1.2, jacobian: 0.4 },
      { pHatSource: 4.7, pSource: 0.91, pHatTarget: 0.3, pTargetProposal: 0.08, jacobian: 2.1 },
      { pHatSource: 18, pSource: 3.2, pHatTarget: 7.5, pTargetProposal: 0.7, jacobian: 0.12 },
    ];
    for (const input of cases) {
      const candidateWeight = input.pHatSource / input.pSource;
      const sourceReservoirW = candidateWeight / input.pHatSource;
      expect(sourceReservoirW).toBeCloseTo(1 / input.pSource, 14);

      const correctReuseWeight = input.pHatTarget * sourceReservoirW * input.jacobian;
      expect(correctReuseWeight).toBeCloseTo(
        input.pHatTarget * input.jacobian / input.pSource,
        12,
      );
      const incorrectlyReplayedPdf = correctReuseWeight
        * input.pSource / input.pTargetProposal;
      expect(incorrectlyReplayedPdf).not.toBeCloseTo(correctReuseWeight, 8);
    }
  });

  it('normalizes Jacobian-corrected generalized-balance weights in one common measure', () => {
    const random = makeLcg(0x5eed1234);
    for (let domainCount = 2; domainCount <= 8; domainCount += 1) {
      const counts = Array.from(
        { length: domainCount },
        () => 1 + Math.floor(random() * 32),
      );
      const nativeTargets = Array.from(
        { length: domainCount },
        () => 0.01 + random() * 8,
      );
      // J_i maps domain i into the canonical receiver.  The target-measure
      // proxy density is pHat_i / J_i.
      const domainToCanonicalJacobians = Array.from(
        { length: domainCount },
        () => 0.05 + random() * 7,
      );
      const commonMeasureMasses = counts.map(
        (count, domain) =>
          count * nativeTargets[domain]! / domainToCanonicalJacobians[domain]!,
      );
      const commonDenominator = commonMeasureMasses.reduce(
        (sum, mass) => sum + mass,
        0,
      );

      // This is the shader's candidate-native form.  J_i/J_j converts
      // technique j's proxy density into candidate i's native measure.
      const weights = counts.map((count, candidate) => {
        const candidateJacobian = domainToCanonicalJacobians[candidate]!;
        const nativeDenominator = counts.reduce(
          (sum, otherCount, domain) =>
            sum
            + otherCount
              * nativeTargets[domain]!
              * candidateJacobian
              / domainToCanonicalJacobians[domain]!,
          0,
        );
        const nativeNumerator = count * nativeTargets[candidate]!;
        expect(nativeNumerator / nativeDenominator).toBeCloseTo(
          commonMeasureMasses[candidate]! / commonDenominator,
          13,
        );
        return nativeNumerator / nativeDenominator;
      });
      expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 13);
    }

    // Two-domain hand case: previous→current J=4, so current→previous J=1/4.
    const currentMass = 3 * 0.7;
    const previousMass = 11 * 4.3;
    const previousToCurrent = 4;
    const currentToPrevious = 1 / previousToCurrent;
    const currentWeight = currentMass
      / (currentMass + previousMass * currentToPrevious);
    const previousWeight = previousMass
      / (currentMass * previousToCurrent + previousMass);
    expect(currentWeight + previousWeight).toBeCloseTo(1, 14);
    expect(currentWeight).toBeCloseTo(
      currentMass / (currentMass + previousMass / previousToCurrent),
      14,
    );
    // The untransformed heuristic also normalizes, but is not the generalized
    // balance heuristic's Jacobian-corrected density and gives a different mass.
    expect(currentWeight).not.toBeCloseTo(
      currentMass / (currentMass + previousMass),
      6,
    );

    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'fn rptLogWeightedShiftedTarget(',
    );
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'logWeightedTarget + log(sourceToTargetJacobian)',
    );
  });
});

describe('ReSTIR-PT glossy reuse closure — proposals and domain state', () => {
  it('samples the base/clearcoat/sheen source mixture at its normalized weights', () => {
    const clearcoat = 0.65;
    const sheen = 0.35;
    const sum = 1 + clearcoat + sheen;
    const counts = [0, 0, 0];
    const random = makeLcg(0xc0ffee);
    const samples = 100_000;
    for (let i = 0; i < samples; i += 1) {
      const xi = random() * sum;
      const lobe = xi < 1 ? 0 : xi < 1 + clearcoat ? 1 : 2;
      counts[lobe]! += 1;
    }
    expect(counts[0]! / samples).toBeCloseTo(1 / sum, 2);
    expect(counts[1]! / samples).toBeCloseTo(clearcoat / sum, 2);
    expect(counts[2]! / samples).toBeCloseTo(sheen / sum, 2);

    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'return max(1.0 + max(clearcoat, 0.0) + max(sheen, 0.0), 1e-4);',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'let xiSource = rand_f32(rng) * lobeWeightSum;',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'return brdfDirectionalPdfFullSampledWithClearcoatNormal(',
    );
  });

  it('uses each sample domain native wo under camera motion', () => {
    const visible: Vec3 = [0.25, -0.1, 0.4];
    const previousCamera: Vec3 = [-1.5, 0.7, 2.8];
    const currentCamera: Vec3 = [2.2, -0.4, 1.1];
    const previousNativeWo = direction(visible, previousCamera);
    const incorrectCurrentCameraWo = direction(visible, currentCamera);
    expect(length(sub(previousNativeWo, incorrectCurrentCameraWo))).toBeGreaterThan(0.5);

    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('let woCur  = rCur.woV;');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('let woPrev = rPrev.woV;');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('let wo = r.woV;');
    expect(stripComments(RESTIR_PT_RESOLVE_WGSL)).not.toContain('params.cameraPos');
  });

  it('accepts rigid/skinned correspondence but rejects identity and local-surface changes', () => {
    const mesh: SurfaceIdentity = {
      materialId: 17,
      instanceIndex: 5,
      triangleIndex: 42,
      surfaceParam: [0.22, 0.31, 0],
    };
    expect(temporalSurfaceIdentityMatches(
      mesh,
      { ...mesh, surfaceParam: [0.25, 0.28, 0] },
      100,
    )).toBe(true);
    expect(temporalSurfaceIdentityMatches(
      mesh,
      { ...mesh, surfaceParam: [0.45, 0.31, 0] },
      100,
    )).toBe(false);
    expect(temporalSurfaceIdentityMatches(mesh, { ...mesh, materialId: 18 }, 100)).toBe(false);
    expect(temporalSurfaceIdentityMatches(mesh, { ...mesh, instanceIndex: 6 }, 100)).toBe(false);
    expect(temporalSurfaceIdentityMatches(mesh, { ...mesh, triangleIndex: 43 }, 100)).toBe(false);

    const analytic: SurfaceIdentity = {
      materialId: 3,
      instanceIndex: 9,
      triangleIndex: 130,
      surfaceParam: [0.4, -0.2, 1.1],
    };
    expect(temporalSurfaceIdentityMatches(
      analytic,
      { ...analytic, surfaceParam: [0.405, -0.205, 1.105] },
      100,
    )).toBe(true);
    expect(temporalSurfaceIdentityMatches(
      analytic,
      { ...analytic, surfaceParam: [0.7, -0.2, 1.1] },
      100,
    )).toBe(false);

    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('current.materialIdV != previous.materialIdV');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('current.instanceIndexV != previous.instanceIndexV');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('current.triangleIndexV != previous.triangleIndexV');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('const RPT_TEMPORAL_IDENTITY_RADIUS: i32 = 2;');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('let candidate = loadReservoirPTHero_ro(&rpt_resPrev, candidateIdx);');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('surfaceDistance + 1e-6 < bestSurfaceDistance');
    expect(RESTIR_PT_TEMPORAL_WGSL).not.toContain('rCur.xv - rPrev.xv');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('current.surfaceParamV - previous.surfaceParamV');
  });

  it('keeps the finite current candidate on search miss or ambiguous duplicate identity', () => {
    const current: TemporalCandidate = {
      materialId: 17,
      instanceIndex: 5,
      triangleIndex: 42,
      surfaceParam: [0.22, 0.31, 0],
      normal: [0, 0, 1],
      M: 1,
      W: 3.25,
    };
    const miss: TemporalCandidate = {
      ...current,
      instanceIndex: 99,
      M: 64,
      W: 9,
    };
    const missResult = selectTemporalCandidateOrCurrent(current, [miss], 100);
    expect(missResult).toBe(current);
    expect(missResult.M).toBe(1);
    expect(Number.isFinite(missResult.W) && missResult.W > 0).toBe(true);

    const duplicateA: TemporalCandidate = {
      ...current,
      surfaceParam: [0.21, 0.30, 0],
      M: 40,
      W: 1.5,
    };
    const duplicateB: TemporalCandidate = {
      ...current,
      surfaceParam: [0.23, 0.32, 0],
      M: 80,
      W: 7.5,
    };
    const duplicateResult = selectTemporalCandidateOrCurrent(
      current,
      [duplicateA, duplicateB],
      100,
    );
    expect(duplicateResult).toBe(current);
    expect(duplicateResult.M).toBe(1);
    expect(duplicateResult.W).toBe(3.25);

    const farther: TemporalCandidate = {
      ...current,
      surfaceParam: [0.26, 0.31, 0],
      M: 20,
      W: 2,
    };
    const nearer: TemporalCandidate = {
      ...current,
      surfaceParam: [0.225, 0.31, 0],
      M: 30,
      W: 4,
    };
    expect(selectTemporalCandidateOrCurrent(current, [farther, nearer], 100)).toBe(nearer);
    expect(selectTemporalCandidateOrCurrent(current, [nearer, farther], 100)).toBe(nearer);

    const temporalCode = stripComments(RESTIR_PT_TEMPORAL_WGSL);
    expect(temporalCode.indexOf('for (var oy =')).toBeLessThan(temporalCode.indexOf('for (var ox ='));
    expect(temporalCode).toContain('if (!prevFound || prevAmbiguous)');
    expect(temporalCode).toContain('storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);');
  });
    expect(temporalNormalIsValid([0, 0, 1])).toBe(true);
  it('validates normals without an arbitrary inter-frame normal-delta threshold', () => {
    expect(temporalNormalIsValid(normalize([0.8, 0.1, 0.3]))).toBe(true);
    expect(temporalNormalIsValid([Number.NaN, 0, 1])).toBe(false);
    expect(temporalNormalIsValid([0, 0, 0])).toBe(false);
    expect(temporalNormalIsValid([0, 0, 2])).toBe(false);

    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('rptTemporalNormalIsValid(rCur.nv)');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('rptTemporalNormalIsValid(rPrev.nv)');
    expect(RESTIR_PT_TEMPORAL_WGSL).not.toContain('dot(rCur.nv, rPrev.nv)');
    expect(RESTIR_PT_TEMPORAL_WGSL).not.toContain('RPT_NORMAL_DOT_MIN');
  });
});

describe('ReSTIR-PT glossy reuse closure — validation, exclusion, and fallback', () => {
  it('re-evaluates cross-material spatial targets and validates both reuse edges', () => {
    expect(RESTIR_PT_SPATIAL_WGSL).not.toContain('rQ.materialIdV != rCenter.materialIdV');
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'rCenter, qR[i].heroLambdaV, woCenter, qR[i].xs, qR[i].Lo,',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'qR[i], qR[i].heroLambdaV, qWo[i], qR[i].xs, qR[i].Lo,',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'rCenter.xv, rCenter.nv, qR[i].xs, &rng,',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'qR[j].xv, qR[j].nv, qR[i].xs, &rng,',
    );
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'rptReconnectionVisible(rCur.xv, rCur.nv, rPrev.xs, &rng)',
    );
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'rptReconnectionVisible(rPrev.xv, rPrev.nv, rCur.xs, &rng)',
    );
  });

  it('keeps transmission/delta paths on the full-megakernel fallback', () => {
    const composite = composePtWebgpuCompositeTraceWgsl(false);
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'if (transmission > 0.0) { return false; }',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('bsdfHasFiniteConnectionSupport(');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain(
      'rpt_result[pixelIdx] = vec4f(0.0, 0.0, 0.0, 0.0);',
    );
    expect(composite).toContain('let rptProducerContributed = rptComposite.a > 0.5;');
    expect(composite).toContain(
      'let rptCompositeContributed = rptProducerContributed && rptMixtureSelected;',
    );
    expect(composite).toContain('if (rptCompositeContributed) {');
    expect(composite).toContain('outRadiance = outRadiance + rptComposite.rgb;');
  });

  it('has no maturity switch or obsolete cache-lane readers/writers', () => {
    const production = [
      RESERVOIR_PT_HERO_WGSL,
      RESTIR_PT_PRODUCER_WGSL,
      RESTIR_PT_TEMPORAL_WGSL,
      RESTIR_PT_SPATIAL_WGSL,
      RESTIR_PT_RESOLVE_WGSL,
    ].map(stripComments).join('\n');
    for (const obsolete of [
      'allowGlossyReuse',
      'wi_recon',
      'distRecon',
      'cosReconOut',
      'hybridJacCache',
      'hybridShiftPdf',
      'rngSeed',
      '_padHybrid',
    ]) {
      expect(production).not.toContain(obsolete);
    }
    expect(RESTIR_PT_PARAMS_FIELDS.map((field) => field.name)).toEqual([
      'width', 'height', 'mClamp', '_padA',
    ]);
    expect(RESERVOIR_PT_HERO_WGSL).toContain('const RESERVOIR_PT_HERO_STRIDE: u32 = 16u;');
  });
});
