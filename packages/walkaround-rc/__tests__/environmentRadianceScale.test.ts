import { describe, expect, it } from 'vitest';
import {
  RC_ENVIRONMENT_RADIANCE_SCALE_WGSL,
  assertRcEnvironmentRadianceF32,
  assertRcEnvironmentScaleF32,
  scaleRcEnvironmentRadianceF32,
} from '../src/environmentRadianceScale.js';

const F32_MAX = 3.4028234663852886e38;
const F32_MIN_SUBNORMAL = 2 ** -149;

describe('walkaround-rc environment radiance binary32 policy', () => {
  it('preserves ordinary and near-maximum finite products exactly', () => {
    expect(scaleRcEnvironmentRadianceF32([1, 2, 4], 0.5))
      .toEqual([0.5, 1, 2]);
    expect(scaleRcEnvironmentRadianceF32(
      [F32_MAX / 2, 0, 0],
      2,
    )[0]).toBe(F32_MAX);
  });

  it('fails an overflowing RGB stage closed as a whole', () => {
    expect(scaleRcEnvironmentRadianceF32(
      [F32_MAX, 1, 1],
      2,
    )).toEqual([0, 0, 0]);
    expect(RC_ENVIRONMENT_RADIANCE_SCALE_WGSL)
      .toContain('let scaled = value * scale;');
  });

  it('allows one channel to underflow when another channel survives', () => {
    expect(scaleRcEnvironmentRadianceF32(
      [F32_MIN_SUBNORMAL, 1, 0],
      0.5,
    )).toEqual([0, 0.5, 0]);
  });

  it('rejects positive scalar and complete RGB publication collapse', () => {
    expect(() => assertRcEnvironmentScaleF32(
      2 ** -150,
      'envIntensity',
    )).toThrow(/Float32 packing/);
    expect(() => assertRcEnvironmentRadianceF32(
      [2 ** -150, 0, 0],
      'scalarSkyRadiance',
    )).toThrow(/underflows entirely/);
  });

  it('accepts per-channel publication underflow when positive RGB survives', () => {
    expect(assertRcEnvironmentRadianceF32(
      [2 ** -150, 1, 0],
      'scalarSkyRadiance',
    )).toEqual([0, 1, 0]);
  });
});
