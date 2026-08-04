import { describe, expect, it } from 'vitest';
import {
  REPRESENTED_PROPOSAL_BUCKET_COUNT,
  buildRepresentedDistributionF32,
  quantizeOpenUnitProbabilityF32,
  representBernoulliProbabilityF32,
} from '../representedDistribution.js';

describe('represented categorical distribution', () => {
  it('preserves adversarial positive support and publishes exact 24-bit PMFs', () => {
    const represented = buildRepresentedDistributionF32([2 ** -30, 2 ** -30, 1]);

    expect([...represented.bucketCounts]).toEqual([
      1,
      1,
      REPRESENTED_PROPOSAL_BUCKET_COUNT - 2,
    ]);
    expect([...represented.pmf]).toEqual(
      [...represented.bucketCounts].map((count) => count / REPRESENTED_PROPOSAL_BUCKET_COUNT),
    );
    expect(represented.cdf[0]).toBe(0);
    expect(represented.cdf[represented.cdf.length - 1]).toBe(1);
    expect(represented.pmf[0]).toBeGreaterThan(0);
    expect(represented.pmf[1]).toBeGreaterThan(0);
  });

  it('is deterministic and breaks equal remainders by source order', () => {
    const represented = buildRepresentedDistributionF32([1, 1, 1], { bucketBits: 3 });
    expect([...represented.bucketCounts]).toEqual([3, 3, 2]);
    expect([...represented.cdf]).toEqual([0, 3 / 8, 6 / 8, 1]);

    const realizedCounts = new Uint32Array(3);
    for (let bucket = 0; bucket < represented.totalBucketCount; bucket += 1) {
      const u = bucket / represented.totalBucketCount;
      const selected = [...represented.pmf].findIndex(
        (_, index) => u < represented.cdf[index + 1]!,
      );
      expect(selected).toBeGreaterThanOrEqual(0);
      realizedCounts[selected] = (realizedCounts[selected] ?? 0) + 1;
    }
    expect([...realizedCounts]).toEqual([...represented.bucketCounts]);
  });

  it('supports an exact uniform fallback for an all-zero distribution', () => {
    const empty = buildRepresentedDistributionF32([0, 0, 0], { bucketBits: 3 });
    expect([...empty.bucketCounts]).toEqual([0, 0, 0]);
    expect([...empty.cdf]).toEqual([0, 0, 0, 0]);

    const uniform = buildRepresentedDistributionF32([0, 0, 0], {
      bucketBits: 3,
      zeroWeightFallback: 'uniform',
    });
    expect([...uniform.bucketCounts]).toEqual([3, 3, 2]);
    expect(uniform.cdf[uniform.cdf.length - 1]).toBe(1);
  });

  it('rejects invalid weights and distributions wider than their random domain', () => {
    expect(() => buildRepresentedDistributionF32([1, Number.NaN])).toThrow(/finite and non-negative/);
    expect(() => buildRepresentedDistributionF32([1, -1])).toThrow(/finite and non-negative/);
    expect(() =>
      buildRepresentedDistributionF32([1], { zeroWeightFallback: 'invalid' } as never),
    ).toThrow(/zeroWeightFallback/);
    expect(() => buildRepresentedDistributionF32([1], null as never)).toThrow(/options/);
    expect(() =>
      buildRepresentedDistributionF32([1, 1, 1, 1, 1], { bucketBits: 2 }),
    ).toThrow(/exact positive support is impossible/);
  });

  it('quantizes defensive Bernoulli mixtures to exact interior 24-bit buckets', () => {
    expect(quantizeOpenUnitProbabilityF32(0.5)).toBe(0.5);
    expect(quantizeOpenUnitProbabilityF32(Number.MIN_VALUE)).toBe(
      1 / REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
    expect(quantizeOpenUnitProbabilityF32(1 - Number.EPSILON)).toBe(
      (REPRESENTED_PROPOSAL_BUCKET_COUNT - 1) / REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
    expect(() => quantizeOpenUnitProbabilityF32(0)).toThrow(/strictly between/);
    expect(() => quantizeOpenUnitProbabilityF32(1)).toThrow(/strictly between/);
    expect(() => quantizeOpenUnitProbabilityF32(Number.NaN)).toThrow(/strictly between/);
  });

  it('publishes the realized probability for closed-unit weighted Bernoulli branches', () => {
    expect(representBernoulliProbabilityF32(0)).toBe(0);
    expect(representBernoulliProbabilityF32(1)).toBe(1);
    expect(representBernoulliProbabilityF32(Number.MIN_VALUE)).toBe(
      1 / REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
    expect(representBernoulliProbabilityF32(1 - Number.EPSILON)).toBe(
      (REPRESENTED_PROPOSAL_BUCKET_COUNT - 1) / REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
    for (const ideal of [0.1, 1 / 3, 0.499_999_97, 0.75]) {
      const represented = representBernoulliProbabilityF32(ideal);
      const realizedBuckets = represented * REPRESENTED_PROPOSAL_BUCKET_COUNT;
      expect(Number.isInteger(realizedBuckets)).toBe(true);
      expect(realizedBuckets).toBeGreaterThan(0);
      expect(realizedBuckets).toBeLessThan(REPRESENTED_PROPOSAL_BUCKET_COUNT);
      expect(Math.abs(represented - ideal)).toBeLessThanOrEqual(
        0.5 / REPRESENTED_PROPOSAL_BUCKET_COUNT,
      );
    }
    expect(() => representBernoulliProbabilityF32(-Number.MIN_VALUE)).toThrow(/inside/);
    expect(() => representBernoulliProbabilityF32(1 + Number.EPSILON)).toThrow(/inside/);
    expect(() => representBernoulliProbabilityF32(Number.NaN)).toThrow(/inside/);
  });
});
