import { describe, expect, it } from 'vitest';
import {
  WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL,
  assertWalkaroundEnvironmentMapScaleEnvelopeF32,
  assertWalkaroundEnvironmentMaterialEnvelopeF32,
  assertWalkaroundEnvironmentScaleF32,
  packWalkaroundEnvironmentRotationF32,
  scaleWalkaroundEnvironmentRadianceF32,
  stageWalkaroundEnvironmentScaleProductF32,
  stagedWalkaroundEnvironmentRadianceF32,
} from '../environmentRadianceScale.js';

const F32_MAX = 3.4028234663852886e38;
const F32_MIN_SUBNORMAL = 2 ** -149;

describe('walkaround environment radiance binary32 policy', () => {
  it('preserves ordinary and near-maximum finite products exactly', () => {
    expect(scaleWalkaroundEnvironmentRadianceF32([1, 2, 4], 0.5))
      .toEqual([0.5, 1, 2]);
    expect(scaleWalkaroundEnvironmentRadianceF32(
      [F32_MAX / 2, 0, 0],
      2,
    )[0]).toBe(F32_MAX);
  });

  it('fails an overflowing RGB stage closed without changing PDF state', () => {
    expect(scaleWalkaroundEnvironmentRadianceF32(
      [F32_MAX, 1, 1],
      2,
    )).toEqual([0, 0, 0]);
    expect(WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL)
      .toContain('let scaled = value * scale;');
    expect(WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL)
      .not.toContain('pdf');
  });

  it('allows per-channel subnormal underflow in the shader mirror', () => {
    expect(scaleWalkaroundEnvironmentRadianceF32(
      [F32_MIN_SUBNORMAL, 1, 0],
      0.5,
    )).toEqual([0, 0.5, 0]);
  });

  it('stages global and receiver-material products independently', () => {
    expect(stagedWalkaroundEnvironmentRadianceF32(
      [F32_MAX / 4, 1, 0],
      2,
      2,
    )).toEqual([F32_MAX, 4, 0]);
    expect(stagedWalkaroundEnvironmentRadianceF32(
      [F32_MAX / 2, 1, 0],
      2,
      2,
    )).toEqual([0, 0, 0]);
  });

  it('rejects scalar publication overflow and positive scalar underflow', () => {
    expect(() => assertWalkaroundEnvironmentScaleF32(
      F32_MAX * 2,
      'environment intensity',
    )).toThrow(/Float32 packing/);
    expect(() => assertWalkaroundEnvironmentScaleF32(
      2 ** -150,
      'environment intensity',
    )).toThrow(/Float32 packing/);
  });

  it('stages scalar products in binary32 and rejects overflow or collapse', () => {
    expect(stageWalkaroundEnvironmentScaleProductF32(
      0.1,
      0.2,
      'sky radiance',
    )).toBe(Math.fround(Math.fround(0.1) * Math.fround(0.2)));
    expect(() => stageWalkaroundEnvironmentScaleProductF32(
      F32_MAX,
      2,
      'sky radiance',
    )).toThrow(/remain finite/);
    expect(() => stageWalkaroundEnvironmentScaleProductF32(
      F32_MIN_SUBNORMAL,
      0.5,
      'sky radiance',
    )).toThrow(/underflows to zero/);
  });

  it('rejects global map overflow and total positive-map collapse', () => {
    expect(() => assertWalkaroundEnvironmentMapScaleEnvelopeF32(
      new Float32Array([F32_MAX, 0, 0, 1]),
      2,
    )).toThrow(/remain finite/);
    expect(() => assertWalkaroundEnvironmentMapScaleEnvelopeF32(
      new Float32Array([F32_MIN_SUBNORMAL, 0, 0, 1]),
      0.5,
    )).toThrow(/underflows entirely/);
  });

  it('rejects final receiver overflow and complete positive collapse', () => {
    expect(() => assertWalkaroundEnvironmentMaterialEnvelopeF32(
      new Float32Array([F32_MAX / 2, 0, 0, 1]),
      1,
      [3],
    )).toThrow(/exceed Float32 range/);
    expect(() => assertWalkaroundEnvironmentMaterialEnvelopeF32(
      new Float32Array([F32_MIN_SUBNORMAL, 0, 0, 1]),
      1,
      [0.5],
    )).toThrow(/underflow entirely/);
  });

  it('accepts one-channel underflow when positive environment energy survives', () => {
    expect(() => assertWalkaroundEnvironmentMaterialEnvelopeF32(
      new Float32Array([F32_MIN_SUBNORMAL, 1, 0, 1]),
      1,
      [0.5],
    )).not.toThrow();
  });

  it('shares wrap-then-fround environment rotation semantics', () => {
    expect(packWalkaroundEnvironmentRotationF32(Math.PI / 3))
      .toBe(Math.fround(Math.PI / 3));
    expect(packWalkaroundEnvironmentRotationF32(1e300))
      .toBe(Math.fround(1e300 % (2 * Math.PI)));
    expect(packWalkaroundEnvironmentRotationF32(-0)).toBe(0);
    expect(packWalkaroundEnvironmentRotationF32(2 ** -150)).toBe(0);
  });
});
