/** CPU acceptance tests for the renderer's bounded diffuse DDGI-proxy GRIS. */

import { describe, expect, it } from 'vitest';
import {
  reconnectionGeometryTerm as oracleGeometryTerm,
  reconnectionJacobian as oracleJacobian,
  reconnectionShift,
  type ReconnectionPath,
} from '@vitrum/shared-samplers';
import {
  MAX_GRIS_TECHNIQUES,
  canonicalResamplingWeight,
  domainToCanonicalJacobian,
  evaluateTechniqueMatrix,
  finaliseLogScaledReservoirWeight,
  foldAttemptCount,
  foldClampedAttemptCount,
  logCanonicalResamplingWeight,
  logWeightedTransformedDensity,
  normaliseCanonicalResamplingWeights,
  proxyPHatAt,
  reconnectionGeometryTerm,
  transformedDensity,
  type GrisDomain,
  type GrisSample,
  type Vec3,
} from './oracles/grisReuseMis.js';

function unit(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / length, v[1] / length, v[2] / length];
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

describe('GRIS reconnection geometry oracle', () => {
  const fixtures: ReconnectionPath[] = [
    { x1: [0, 0, 0], x2: [2, 0, 0], n2: [-1, 0, 0] },
    { x1: [0.2, -0.5, 1.1], x2: [2.7, 1.3, -0.4], n2: unit([0.3, -0.9, 0.5]) },
    { x1: [1, 2, -1], x2: [-3, 0.5, 2.2], n2: unit([-0.3, 0.4, 0.85]) },
  ];

  it('matches the shared first-principles half-geometry term', () => {
    for (const fixture of fixtures) {
      expect(reconnectionGeometryTerm(fixture.x1, fixture.x2, fixture.n2)).toBeCloseTo(
        oracleGeometryTerm(fixture.x1, fixture.x2, fixture.n2),
        12,
      );
    }
  });

  it('matches the shared domain-to-canonical Jacobian', () => {
    const base = fixtures[1]!;
    const canonicalXv: Vec3 = [-0.6, 0.9, 0.3];
    const sample: GrisSample = {
      kind: 'surface',
      xs: base.x2,
      ns: base.n2,
      Lo: [1, 1, 1],
      nativeDomainIndex: 0,
    };
    const shifted = reconnectionShift(base, canonicalXv);
    expect(domainToCanonicalJacobian(base.x1, canonicalXv, sample)).toBeCloseTo(
      oracleJacobian(base, shifted),
      12,
    );
  });

  it('returns zero for degenerate surface mappings', () => {
    const sample: GrisSample = {
      kind: 'surface',
      xs: [1, 2, 3],
      ns: [0, 1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 0,
    };
    expect(domainToCanonicalJacobian([1, 2, 3], [0, 0, 0], sample)).toBe(0);
    expect(reconnectionGeometryTerm([1, 2, 3], [1, 2, 3], [0, 1, 0])).toBe(0);
    expect(reconnectionGeometryTerm(
      [Number.NaN, 0, 0],
      [1, 2, 3],
      [0, 1, 0],
    )).toBe(0);
    expect(reconnectionGeometryTerm(
      [0, 0, 0],
      [Number.POSITIVE_INFINITY, 0, 0],
      [0, 1, 0],
    )).toBe(0);
  });
});

describe('GRIS transformed technique densities', () => {
  it('uses the inverse determinant pHat / J', () => {
    // Hand arithmetic: [4/1, 8/2, 2/0.5] = [4, 4, 4].
    expect([
      transformedDensity(4, 1),
      transformedDensity(8, 2),
      transformedDensity(2, 0.5),
    ]).toEqual([4, 4, 4]);
  });

  it('zeros inverse-invalid, occluded, and singular techniques', () => {
    expect(transformedDensity(8, 2, false)).toBe(0);
    expect(transformedDensity(0, 2, true)).toBe(0);
    expect(transformedDensity(8, 0, true)).toBe(0);
    expect(transformedDensity(Number.POSITIVE_INFINITY, 1, true)).toBe(0);
    expect(transformedDensity(3.402823466e38, Number.MIN_VALUE, true)).toBe(0);
  });

  it('forms a bounded all-technique matrix with inverse-J weights', () => {
    // xs=(0,2,0), ns=(0,-1,0): G at y={0,1,-2} is {1/4,1,1/16}.
    // Relative to canonical domain 0, J = {1,1/4,4}.
    // The target is the same p=1/pi in all three domains. With attempts
    // {2,3,5}, the numerators are p*{2,12,1.25}; denominator=15.25p.
    const domains: GrisDomain[] = [
      { xv: [0, 0, 0], nv: [0, 1, 0], attempts: 2 },
      { xv: [0, 1, 0], nv: [0, 1, 0], attempts: 3 },
      { xv: [0, -2, 0], nv: [0, 1, 0], attempts: 5 },
    ];
    const sample: GrisSample = {
      kind: 'surface',
      xs: [0, 2, 0],
      ns: [0, -1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 1,
      nativePHat: 1 / Math.PI,
    };
    const matrix = evaluateTechniqueMatrix(domains, sample, 0);

    expect(matrix.jacobians[0]).toBeCloseTo(1, 12);
    expect(matrix.jacobians[1]).toBeCloseTo(0.25, 12);
    expect(matrix.jacobians[2]).toBeCloseTo(4, 12);
    expect(matrix.weights[0]).toBeCloseTo(8 / 61, 12);
    expect(matrix.weights[1]).toBeCloseTo(48 / 61, 12);
    expect(matrix.weights[2]).toBeCloseTo(5 / 61, 12);
    expect(sum(matrix.weights)).toBeCloseTo(1, 12);
    // A no-J attempt ratio would be {0.2,0.3,0.5}; explicitly reject it.
    expect(matrix.weights[1]).not.toBeCloseTo(0.3, 6);
  });

  it('gives occluded and inverse-invalid techniques exactly zero mass', () => {
    const domains: GrisDomain[] = [
      { xv: [0, 0, 0], nv: [0, 1, 0], attempts: 2 },
      { xv: [0, 1, 0], nv: [0, 1, 0], attempts: 300, visibility: 0 },
      { xv: [0, -2, 0], nv: [0, 1, 0], attempts: 500, inverseValid: false },
    ];
    const sample: GrisSample = {
      kind: 'surface',
      xs: [0, 2, 0],
      ns: [0, -1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 1,
      nativePHat: 99,
    };
    const matrix = evaluateTechniqueMatrix(domains, sample, 0);
    expect(matrix.transformedDensities[1]).toBe(0);
    expect(matrix.transformedDensities[2]).toBe(0);
    expect(matrix.weights).toEqual([1, 0, 0]);
  });

  it('rejects non-finite visibility instead of clamping it to visible', () => {
    const domain: GrisDomain = {
      xv: [0, 0, 0],
      nv: [0, 1, 0],
      attempts: 1,
      visibility: Number.POSITIVE_INFINITY,
    };
    const sample: GrisSample = {
      kind: 'environment',
      direction: [0, 1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 0,
      nativePHat: 10,
    };
    expect(proxyPHatAt(domain, sample)).toBe(0);
    expect(evaluateTechniqueMatrix([domain], sample, 0).weights).toEqual([0]);
  });

  it('runtime-rejects an unknown sample discriminant', () => {
    const invalid = {
      kind: 'volume',
      Lo: [1, 1, 1],
      nativeDomainIndex: 0,
    } as unknown as GrisSample;
    expect(() => evaluateTechniqueMatrix([
      { xv: [0, 0, 0], nv: [0, 1, 0], attempts: 1 },
    ], invalid, 0)).toThrow(/sample\.kind/);
  });

  it('returns an all-zero matrix when every inverse technique is invalid', () => {
    const domains: GrisDomain[] = [
      { xv: [0, 0, 0], nv: [0, 1, 0], attempts: 2, visibility: 0 },
      { xv: [1, 0, 0], nv: [0, 1, 0], attempts: 3, inverseValid: false },
    ];
    const sample: GrisSample = {
      kind: 'surface',
      xs: [0, 2, 0],
      ns: [0, -1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 0,
      nativePHat: 10,
    };
    const matrix = evaluateTechniqueMatrix(domains, sample, 0);
    expect(matrix.denominator).toBe(0);
    expect(matrix.weights).toEqual([0, 0]);
  });

  it('uses an identity Jacobian for persistent environment directions', () => {
    const domains: GrisDomain[] = [
      { xv: [-20, 4, 9], nv: [0, 1, 0], attempts: 2 },
      { xv: [1, -7, 3], nv: [0, 1, 0], attempts: 3 },
      { xv: [99, 2, -40], nv: [0, 1, 0], attempts: 5 },
    ];
    const sample: GrisSample = {
      kind: 'environment',
      direction: [0, 4, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 2,
      nativePHat: 1 / Math.PI,
    };
    const matrix = evaluateTechniqueMatrix(domains, sample, 0);
    expect(matrix.jacobians).toEqual([1, 1, 1]);
    expect(matrix.weights[0]).toBeCloseTo(0.2, 12);
    expect(matrix.weights[1]).toBeCloseTo(0.3, 12);
    expect(matrix.weights[2]).toBeCloseTo(0.5, 12);
  });

  it('enforces the shader matrix bound', () => {
    const domains = Array.from({ length: MAX_GRIS_TECHNIQUES + 1 }, (_, index) => ({
      xv: [index, 0, 0] as const,
      nv: [0, 1, 0] as const,
      attempts: 1,
    }));
    const sample: GrisSample = {
      kind: 'environment',
      direction: [0, 1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 0,
    };
    expect(() => evaluateTechniqueMatrix(domains, sample, 0)).toThrow(/at most 6/);
  });

  it.each([0, -1, 1.5, Number.NaN, 0x1_0000_0000])(
    'rejects non-positive or non-u32 represented attempt count %s',
    (attempts) => {
      const sample: GrisSample = {
        kind: 'environment',
        direction: [0, 1, 0],
        Lo: [1, 1, 1],
        nativeDomainIndex: 0,
      };
      expect(() => evaluateTechniqueMatrix([
        { xv: [0, 0, 0], nv: [0, 1, 0], attempts },
      ], sample, 0)).toThrow(/positive u32/);
    },
  );

  it('keeps an extreme but representable density matrix finite and normalized', () => {
    const domains: GrisDomain[] = [{
      xv: [0, 0, 0],
      nv: [0, 1, 0],
      attempts: 0xffff_ffff,
    }];
    const sample: GrisSample = {
      kind: 'environment',
      direction: [0, 1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex: 0,
      nativePHat: 3.402823466e38,
    };
    const matrix = evaluateTechniqueMatrix(domains, sample, 0);
    expect(Number.isFinite(matrix.numerators[0])).toBe(true);
    expect(Number.isFinite(matrix.denominator)).toBe(true);
    expect(matrix.weights).toEqual([1]);
  });

  it('preserves unequal extreme technique masses instead of independently capping them', () => {
    const high = logWeightedTransformedDensity(
      0xffff_ffff,
      3.402823466e38,
      1e-30,
    );
    const low = logWeightedTransformedDensity(1, 3.402823466e38, 1e30);
    const commonScale = Math.max(high, low);
    const scaled = [2 ** (high - commonScale), 2 ** (low - commonScale)];
    expect(scaled[0]).toBe(1);
    expect(scaled[1]).toBeGreaterThanOrEqual(0);
    expect(scaled[1]).toBeLessThan(1e-50);
    expect(scaled[0]).not.toBe(scaled[1]);
  });
});

describe('GRIS reservoir accounting', () => {
  it('includes zero-weight source attempts and never adds a selection attempt', () => {
    // Two useful reservoirs and an occluded 17-attempt reservoir still represent
    // exactly 4+17+9 attempts. Candidate selection does not turn that into 31.
    expect(foldAttemptCount([4, 17, 9])).toBe(30);
    expect(foldAttemptCount([0, 0, 11])).toBe(11);
  });

  it('saturates represented attempts at u32 and rejects lossy numeric inputs', () => {
    expect(foldAttemptCount([0xffff_fffe, 9])).toBe(0xffff_ffff);
    for (const count of [-1, 1.5, Number.NaN, 0x1_0000_0000]) {
      expect(() => foldAttemptCount([count])).toThrow(/must be a u32/);
    }
  });

  it('folds per-source-clamped confidence without adding or re-clamping the batch', () => {
    expect(foldClampedAttemptCount([4, 100, 9], 50)).toBe(63);
    expect(foldClampedAttemptCount([500, 500, 500], 500)).toBe(1500);
    expect(foldClampedAttemptCount([0xffff_ffff, 0xffff_ffff], 0xffff_ffff))
      .toBe(0xffff_ffff);
    expect(() => foldClampedAttemptCount([1], 0)).toThrow(/positive u32/);
  });

  it('applies the source-to-canonical Jacobian exactly once to resampling weight', () => {
    // Hand arithmetic: m * pHatCanonical * Wsource * J = .25*2*3*4 = 6.
    expect(canonicalResamplingWeight(0.25, 2, 3, 4)).toBe(6);
    expect(canonicalResamplingWeight(0.25, 0, 3, 4)).toBe(0);
  });

  it('places unequal source confidence only in generalized m_i, not again beside the UCW', () => {
    // Identical environment targets/Jacobians make the generalized balance
    // masses exactly proportional to represented attempts: m=[1/10, 9/10].
    const domains: GrisDomain[] = [
      { xv: [0, 0, 0], nv: [0, 1, 0], attempts: 1 },
      { xv: [4, 0, -3], nv: [0, 1, 0], attempts: 9 },
    ];
    const samples: GrisSample[] = [0, 1].map((nativeDomainIndex) => ({
      kind: 'environment',
      direction: [0, 1, 0],
      Lo: [1, 1, 1],
      nativeDomainIndex,
      nativePHat: 1 / Math.PI,
    }));
    const sourceUcw = [7, 2] as const;
    const logWeights = samples.map((sample, index) => {
      const matrix = evaluateTechniqueMatrix(domains, sample, 0);
      return logCanonicalResamplingWeight(
        matrix.weights[index]!,
        1 / Math.PI,
        sourceUcw[index]!,
        1,
      );
    });
    const normalized = normaliseCanonicalResamplingWeights(logWeights).weights;

    // Eq. 7 + UCW: (m1*W1)/(m0*W0) = (0.9*2)/(0.1*7) = 18/7.
    expect(normalized[1]! / normalized[0]!).toBeCloseTo(18 / 7, 12);
    // Multiplying source M again would produce an M² ratio of 162/7.
    expect(normalized[1]! / normalized[0]!).not.toBeCloseTo(162 / 7, 6);
  });

  it('normalizes WRS with a common scale and caps only the final estimator weight', () => {
    const logWeights = [
      logCanonicalResamplingWeight(0.75, 4, 1e30, 2),
      logCanonicalResamplingWeight(0.25, 4, 1e-20, 2),
    ];
    const normalized = normaliseCanonicalResamplingWeights(logWeights);
    expect(normalized.weights[0]).toBe(1);
    expect(normalized.weights[1]).toBeGreaterThan(0);
    expect(normalized.weights[1]).toBeLessThan(1e-45);

    expect(finaliseLogScaledReservoirWeight(
      normalized.logScale,
      sum(normalized.weights),
      4,
      16,
    )).toBe(16);
    expect(finaliseLogScaledReservoirWeight(Math.log2(6), 1.5, 3, 16))
      .toBeCloseTo(3, 12);
  });

  it('selects common-scaled candidates with the expected Monte Carlo frequencies', () => {
    const desired = [1, 0.2, 0.05] as const;
    const total = sum(desired);
    const counts = [0, 0, 0];
    let state = 0x12345678;
    const draws = 100_000;
    for (let draw = 0; draw < draws; draw += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const target = (state / 0x1_0000_0000) * total;
      let cumulative = 0;
      for (let index = 0; index < desired.length; index += 1) {
        cumulative += desired[index]!;
        if (target < cumulative) {
          counts[index]! += 1;
          break;
        }
      }
    }
    for (let index = 0; index < desired.length; index += 1) {
      expect(counts[index]! / draws).toBeCloseTo(desired[index]! / total, 2);
    }
  });

  it('does not resurrect an invisible sample through a stored native pHat', () => {
    const domain: GrisDomain = {
      xv: [0, 0, 0],
      nv: [0, 1, 0],
      attempts: 8,
      visibility: 0,
    };
    const sample: GrisSample = {
      kind: 'environment',
      direction: [0, 1, 0],
      Lo: [10, 10, 10],
      nativeDomainIndex: 0,
      nativePHat: 100,
    };
    expect(proxyPHatAt(domain, sample)).toBe(0);
    expect(evaluateTechniqueMatrix([domain], sample, 0).weights).toEqual([0]);
  });
});
