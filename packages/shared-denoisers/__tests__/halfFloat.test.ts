import { describe, expect, it } from 'vitest';
import { float16BitsToFloat32, float32ToFloat16Bits } from '../src/halfFloat.js';

describe('halfFloat', () => {
  it('round-trips common HDR scalars', () => {
    for (const x of [0, 1, 2, 0.015, 10, 1e-4, -0.25]) {
      const h = float32ToFloat16Bits(x);
      expect(float16BitsToFloat32(h)).toBeCloseTo(x, 3);
    }
  });

  it('maps ±Infinity to half infinity', () => {
    const p = float16BitsToFloat32(float32ToFloat16Bits(Number.POSITIVE_INFINITY));
    const n = float16BitsToFloat32(float32ToFloat16Bits(Number.NEGATIVE_INFINITY));
    expect(p).toBe(Number.POSITIVE_INFINITY);
    expect(n).toBe(Number.NEGATIVE_INFINITY);
  });

  it('maps NaN to a half NaN (decode is NaN)', () => {
    const bits = float32ToFloat16Bits(Number.NaN);
    expect(Number.isNaN(float16BitsToFloat32(bits))).toBe(true);
  });

  it('round-trips a subnormal scalar', () => {
    const x = Math.pow(2, -14) * (1 / 512);
    const h = float32ToFloat16Bits(x);
    expect(float16BitsToFloat32(h)).toBeCloseTo(x, 5);
  });

  it('maps finite half infinity encoding (0x7c00) to float infinity', () => {
    expect(float16BitsToFloat32(0x7c00)).toBe(Number.POSITIVE_INFINITY);
  });
});
