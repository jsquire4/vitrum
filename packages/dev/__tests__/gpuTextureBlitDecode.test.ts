import { describe, expect, it } from 'vitest';
import {
  decodeGpuTexturePixelForDisplay,
  type GpuTextureBlitDecodeMode,
} from '../src/react/gpuTextureBlit.js';

function decode(
  rgba: readonly [number, number, number, number],
  mode: GpuTextureBlitDecodeMode,
): readonly number[] {
  const out = new Float32Array(3);
  decodeGpuTexturePixelForDisplay(new Float32Array(rgba), mode, out);
  return Array.from(out);
}

describe('gpuTextureBlit DDGI atlas decode', () => {
  it('preserves the generic linear RGB behavior', () => {
    expect(decode([2, 3, 4, 115], 'linear-rgb')).toEqual([2, 3, 4]);
  });

  it('decodes legacy and cleared irradiance texels', () => {
    expect(decode([1, 2, 3, 1], 'ddgi-irradiance')).toEqual([1, 2, 3]);
    expect(decode([0, 0, 0, 0], 'ddgi-irradiance')).toEqual([0, 0, 0]);
  });

  it('decodes high-range irradiance with the shared alpha exponent', () => {
    const scale = 2 ** 100;
    expect(decode([2, 3, 4, 101], 'ddgi-irradiance')).toEqual([
      2 * scale,
      3 * scale,
      4 * scale,
    ]);
    const maxFinite = decode([16_376, 0, 0, 115], 'ddgi-irradiance');
    expect(maxFinite[0]).toBeGreaterThan(3.4e38);
    expect(Number.isFinite(maxFinite[0])).toBe(true);
  });

  it('decodes the negative-exponent subnormal range', () => {
    expect(decode([1, 0, 0, -149], 'ddgi-irradiance')).toEqual([
      Math.fround(2 ** -149),
      0,
      0,
    ]);
  });

  it('fails the whole irradiance texel closed for malformed metadata', () => {
    expect(decode([1, 2, 3, 1.5], 'ddgi-irradiance')).toEqual([0, 0, 0]);
    expect(decode([16_384, 1, 1, 115], 'ddgi-irradiance')).toEqual([0, 0, 0]);
    expect(decode([2 ** -24, 0, 0, -149], 'ddgi-irradiance')).toEqual([0, 0, 0]);
  });

  it('decodes visibility mean and second moment from independent exponents', () => {
    expect(decode([2, 10, 3, 4], 'ddgi-visibility')).toEqual([
      8,
      80,
      16,
    ]);
    expect(decode([2, 5, 0, 1], 'ddgi-visibility')).toEqual([
      2,
      5,
      1,
    ]);
  });

  it('preserves legacy and cleared visibility texels', () => {
    expect(decode([2, 4, 0, 1], 'ddgi-visibility')).toEqual([2, 4, 0]);
    expect(decode([0, 0, 0, 0], 'ddgi-visibility')).toEqual([0, 0, 0]);
  });

  it('fails visibility closed for malformed or negative moments', () => {
    expect(decode([2, 4, 0.5, 1], 'ddgi-visibility')).toEqual([0, 0, 0]);
    expect(decode([-2, 4, 0, 1], 'ddgi-visibility')).toEqual([0, 0, 0]);
  });
});
