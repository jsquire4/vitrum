import { describe, expect, it } from 'vitest';

import {
  EMITTER_CDF_F32_BUCKETS,
  buildRepresentedEmitterCdfF32,
} from '../emitterList.js';

describe('emitter CDF Float32 representation', () => {
  it('retains a positive interval for every emitter across extreme dynamic range', () => {
    const { cdf, representedPmf } = buildRepresentedEmitterCdfF32([
      Math.fround(3e38),
      Math.fround(2 ** -149),
      Math.fround(1),
    ]);

    expect(cdf[0]).toBeGreaterThan(0);
    expect(cdf[1]).toBeGreaterThan(cdf[0]!);
    expect(cdf[2]).toBeGreaterThan(cdf[1]!);
    expect(cdf[2]).toBe(1);
    expect(representedPmf[1]).toBe(1 / EMITTER_CDF_F32_BUCKETS);
    expect(representedPmf[0]! + representedPmf[1]! + representedPmf[2]!)
      .toBeCloseTo(1, 7);
  });

  it('derives each shader-visible PMF from its exact CDF interval', () => {
    const { cdf, representedPmf } = buildRepresentedEmitterCdfF32([7, 2, 1]);
    expect(representedPmf[0]).toBe(cdf[0]);
    expect(representedPmf[1]).toBe(Math.fround(cdf[1]! - cdf[0]!));
    expect(representedPmf[2]).toBe(Math.fround(cdf[2]! - cdf[1]!));
    expect(cdf[2]).toBe(1);
  });
});
