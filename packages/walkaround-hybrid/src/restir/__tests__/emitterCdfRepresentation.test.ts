import { describe, expect, it } from 'vitest';

import {
  EMITTER_CDF_F32_BUCKETS,
  buildEmitterListFromCore,
  buildRepresentedEmitterCdfF32,
} from '../emitterList.js';
import { EMITTER_SAMPLING_WGSL } from '../../shaders/emitterSampling.wgsl.js';

describe('emitter CDF Float32 representation', () => {
  it('retains a positive interval for every emitter across extreme dynamic range', () => {
    const { cdf, representedPmf } = buildRepresentedEmitterCdfF32([3e77, 1e-90, 1]);

    expect(cdf[0]).toBeGreaterThan(0);
    expect(cdf[1]).toBeGreaterThan(cdf[0]!);
    expect(cdf[2]).toBeGreaterThan(cdf[1]!);
    expect(cdf[2]).toBe(1);
    expect(representedPmf[1]).toBe(1 / EMITTER_CDF_F32_BUCKETS);
    expect(representedPmf[0]! + representedPmf[1]! + representedPmf[2]!).toBeCloseTo(1, 7);
  });

  it('rejects invalid host proposal weights without pre-quantizing finite weights', () => {
    expect(() => buildRepresentedEmitterCdfF32([1, Number.POSITIVE_INFINITY])).toThrow(
      /finite and non-negative/,
    );
    expect(() => buildRepresentedEmitterCdfF32([1, -Number.MIN_VALUE])).toThrow(
      /finite and non-negative/,
    );
  });

  it('retains a real emitter whose f32 radiance-times-area heuristic is below f32 range', () => {
    const side = Math.fround(2e-19);
    const built = buildEmitterListFromCore(
      new Uint32Array([0, 1, 2]),
      new Float32Array([0, 0, 0, 0, side, 0, 0, 0, 0, side, 0, 0]),
      new Float32Array(12),
      new Uint32Array([0]),
      [
        {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          emissive: [2 ** -149, 0, 0],
        },
      ],
      {},
    );

    expect(built.emitterFloats[16]).toBe(2 ** -149);
    expect(built.totalEmissivePower).toBeGreaterThan(0);
    expect(built.cdfArray).toEqual(new Float32Array([1]));
    expect(built.treeInput.powers).toEqual([1]);
  });

  it('publishes hemisphere cones only for positive one-sided emitters', () => {
    const built = buildEmitterListFromCore(
      new Uint32Array([0, 1, 2, 3, 4, 5]),
      new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]),
      new Float32Array(24),
      new Uint32Array([0, 1]),
      [
        {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          emissive: [1, 1, 1],
        },
        {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          emissive: [1, 1, 1],
          doubleSided: true,
        },
      ],
      {},
    );

    expect(built.treeInput.cones[0]).toEqual({
      axis: [0, 0, 1],
      thetaO: 0,
      thetaE: Math.PI / 2,
    });
    expect(built.treeInput.cones[1]).toEqual({
      axis: [0, 0, 0],
      thetaO: Math.PI,
      thetaE: Math.PI,
    });
  });

  it('turns a degenerate zero-normal triangle into a full-sphere placeholder', () => {
    const built = buildEmitterListFromCore(
      new Uint32Array([0, 1, 2]),
      new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]),
      new Float32Array(12),
      new Uint32Array([0]),
      [
        {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          emissive: [1, 1, 1],
        },
      ],
      {},
    );

    expect(built.totalEmissivePower).toBe(0);
    expect(built.treeInput.cones).toEqual([
      {
        axis: [0, 0, 0],
        thetaO: Math.PI,
        thetaE: Math.PI,
      },
    ]);
  });

  it('derives each shader-visible PMF from its exact CDF interval', () => {
    const { cdf, representedPmf } = buildRepresentedEmitterCdfF32([7, 2, 1]);
    expect(representedPmf[0]).toBe(cdf[0]);
    expect(representedPmf[1]).toBe(Math.fround(cdf[1]! - cdf[0]!));
    expect(representedPmf[2]).toBe(Math.fround(cdf[2]! - cdf[1]!));
    expect(cdf[2]).toBe(1);
  });

  it('assigns an exact 24-bit CDF boundary to the following interval', () => {
    expect(EMITTER_SAMPLING_WGSL).toContain('if (sceneLoadEmitterCdf(mid) <= xi)');
    expect(EMITTER_SAMPLING_WGSL).not.toContain('if (sceneLoadEmitterCdf(mid) < xi)');

    const { cdf, representedPmf } = buildRepresentedEmitterCdfF32([1, 1]);
    const sample = (xi: number): number => {
      let lo = 0;
      let hi = cdf.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (cdf[mid]! <= xi) lo = mid + 1;
        else hi = mid;
      }
      return Math.min(lo, cdf.length - 1);
    };

    expect(cdf[0]).toBe(0.5);
    expect(sample(cdf[0]!)).toBe(1);
    const realized = new Uint32Array(2);
    for (let bucket = 0; bucket < EMITTER_CDF_F32_BUCKETS; bucket += 1) {
      const index = sample(bucket / EMITTER_CDF_F32_BUCKETS);
      realized[index] = realized[index]! + 1;
    }
    expect(realized[0]! / EMITTER_CDF_F32_BUCKETS).toBe(representedPmf[0]);
    expect(realized[1]! / EMITTER_CDF_F32_BUCKETS).toBe(representedPmf[1]);
  });
});
