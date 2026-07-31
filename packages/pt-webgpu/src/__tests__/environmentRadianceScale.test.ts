import { describe, expect, it } from 'vitest';
import {
  CIE_D65_TABLE,
  CIE_LAMBDA_STEP,
  CIE_Y_TABLE,
  HERO_D65_MAX_NORMALISED_F32,
  Y_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';
import {
  PT_WEBGPU_ENVIRONMENT_RADIANCE_SCALE_WGSL,
  assertPtWebgpuEnvironmentMaterialEnvelopeF32,
  assertPtWebgpuEnvironmentScaleF32,
  packPtWebgpuEnvironmentRotationF32,
  PT_WEBGPU_MAX_D65_SPECTRAL_EXPANSION_F32,
  scalePtWebgpuEnvironmentRadianceF32,
  stagedPtWebgpuEnvironmentRadianceF32,
} from '../environmentRadianceScale.js';

const F32_MAX = 3.4028234663852886e38;
const F32_MIN_SUBNORMAL = 2 ** -149;

describe('pt-webgpu environment radiance binary32 policy', () => {
  it('derives its D65 expansion bound from the exact WGSL table staging', () => {
    const wgslF32Literal = (value: number): number =>
      Math.fround(Number(value.toFixed(8)));
    let d65YIntegral = 0;
    for (let index = 0; index < CIE_Y_TABLE.length; index += 1) {
      d65YIntegral +=
        (CIE_D65_TABLE[index] ?? 0) *
        (CIE_Y_TABLE[index] ?? 0) *
        CIE_LAMBDA_STEP;
    }
    const normalisation = Math.fround(
      wgslF32Literal(Y_CMF_INTEGRAL) /
        Math.max(wgslF32Literal(d65YIntegral), Math.fround(1e-9)),
    );
    let derivedMaximum = 0;
    for (const sample of CIE_D65_TABLE) {
      derivedMaximum = Math.max(
        derivedMaximum,
        Math.fround(wgslF32Literal(sample) * normalisation),
      );
    }

    expect(derivedMaximum).toBe(1.1913174390792847);
    expect(HERO_D65_MAX_NORMALISED_F32).toBe(derivedMaximum);
    expect(PT_WEBGPU_MAX_D65_SPECTRAL_EXPANSION_F32).toBe(derivedMaximum);
  });

  it('preserves ordinary and near-maximum finite products exactly', () => {
    expect(scalePtWebgpuEnvironmentRadianceF32([1, 2, 4], 0.5))
      .toEqual([0.5, 1, 2]);
    expect(scalePtWebgpuEnvironmentRadianceF32(
      [F32_MAX / 2, 0, 0],
      2,
    )[0]).toBe(F32_MAX);
  });

  it('fails an overflowing RGB stage closed without changing PDF state', () => {
    expect(scalePtWebgpuEnvironmentRadianceF32(
      [F32_MAX, 1, 1],
      2,
    )).toEqual([0, 0, 0]);
    expect(PT_WEBGPU_ENVIRONMENT_RADIANCE_SCALE_WGSL)
      .toContain('let scaled = value * scale;');
    expect(PT_WEBGPU_ENVIRONMENT_RADIANCE_SCALE_WGSL)
      .not.toContain('pdf');
  });

  it('allows per-channel subnormal underflow in the shader mirror', () => {
    expect(scalePtWebgpuEnvironmentRadianceF32(
      [F32_MIN_SUBNORMAL, 1, 0],
      0.5,
    )).toEqual([0, 0.5, 0]);
  });

  it('stages global and receiver-material products independently', () => {
    expect(stagedPtWebgpuEnvironmentRadianceF32(
      [F32_MAX / 4, 1, 0],
      2,
      2,
    )).toEqual([F32_MAX, 4, 0]);
    expect(stagedPtWebgpuEnvironmentRadianceF32(
      [F32_MAX / 2, 1, 0],
      2,
      2,
    )).toEqual([0, 0, 0]);
  });

  it('rejects scalar publication overflow and positive scalar underflow', () => {
    expect(() => assertPtWebgpuEnvironmentScaleF32(
      F32_MAX * 2,
      'HDRI intensity',
    )).toThrow(/Float32 packing/);
    expect(() => assertPtWebgpuEnvironmentScaleF32(
      2 ** -150,
      'HDRI intensity',
    )).toThrow(/Float32 packing/);
  });

  it('rejects final receiver overflow and complete positive collapse', () => {
    expect(() => assertPtWebgpuEnvironmentMaterialEnvelopeF32(
      new Float32Array([F32_MAX / 2, 0, 0, 1]),
      1,
      [3],
    )).toThrow(/exceed Float32 range/);
    expect(() => assertPtWebgpuEnvironmentMaterialEnvelopeF32(
      new Float32Array([F32_MIN_SUBNORMAL, 0, 0, 1]),
      1,
      [0.5],
    )).toThrow(/underflow entirely/);
  });

  it('accepts one-channel underflow when positive environment energy survives', () => {
    expect(() => assertPtWebgpuEnvironmentMaterialEnvelopeF32(
      new Float32Array([F32_MIN_SUBNORMAL, 1, 0, 1]),
      1,
      [0.5],
    )).not.toThrow();
  });

  it('applies the D65 expansion only when spectral execution is enabled', () => {
    const nextF32 = (value: number): number => {
      const f32 = new Float32Array([value]);
      const bits = new Uint32Array(f32.buffer);
      bits[0] = bits[0]! + 1;
      return f32[0]!;
    };
    let justBelow = Math.fround(
      F32_MAX / PT_WEBGPU_MAX_D65_SPECTRAL_EXPANSION_F32,
    );
    while (!Number.isFinite(Math.fround(
      justBelow * PT_WEBGPU_MAX_D65_SPECTRAL_EXPANSION_F32,
    ))) {
      const f32 = new Float32Array([justBelow]);
      const bits = new Uint32Array(f32.buffer);
      bits[0] = bits[0]! - 1;
      justBelow = f32[0]!;
    }
    const justAbove = nextF32(justBelow);
    const below = new Float32Array([
      justBelow, justBelow, justBelow, 1,
    ]);
    const above = new Float32Array([
      justAbove, justAbove, justAbove, 1,
    ]);

    expect(() => assertPtWebgpuEnvironmentMaterialEnvelopeF32(
      below,
      1,
      [1],
      true,
    )).not.toThrow();
    expect(() => assertPtWebgpuEnvironmentMaterialEnvelopeF32(
      above,
      1,
      [1],
      true,
    )).toThrow(/spectral D65 emission stage/);
    expect(() => assertPtWebgpuEnvironmentMaterialEnvelopeF32(
      above,
      1,
      [1],
      false,
    )).not.toThrow();
  });

  it('shares wrap-then-fround environment rotation semantics', () => {
    expect(packPtWebgpuEnvironmentRotationF32(Math.PI / 3))
      .toBe(Math.fround(Math.PI / 3));
    expect(packPtWebgpuEnvironmentRotationF32(1e300))
      .toBe(Math.fround(1e300 % (2 * Math.PI)));
    expect(packPtWebgpuEnvironmentRotationF32(-0)).toBe(0);
    expect(packPtWebgpuEnvironmentRotationF32(2 ** -150)).toBe(0);
  });
});
