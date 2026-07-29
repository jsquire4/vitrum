import { describe, expect, it } from 'vitest';
import {
  finiteFloat16Bits,
  FLOAT16_HALF_MIN_SUBNORMAL,
  FLOAT16_MAX_FINITE,
  FLOAT16_MIN_SUBNORMAL,
  float16BitsToFloat32,
  quantizeFiniteFloat16,
} from './halfFloat.js';

describe('checked finite binary16 storage', () => {
  it.each([
    ['maximum finite', FLOAT16_MAX_FINITE, 0x7bff],
    ['negative maximum finite', -FLOAT16_MAX_FINITE, 0xfbff],
    ['minimum positive subnormal', FLOAT16_MIN_SUBNORMAL, 0x0001],
    ['minimum negative subnormal', -FLOAT16_MIN_SUBNORMAL, 0x8001],
    ['positive zero', 0, 0x0000],
    ['negative zero', -0, 0x8000],
  ])('preserves the %s boundary', (_label, value, expectedBits) => {
    const bits = finiteFloat16Bits(value);
    expect(bits).toBe(expectedBits);
    expect(Object.is(quantizeFiniteFloat16(value), float16BitsToFloat32(expectedBits)))
      .toBe(true);
  });

  it('rounds the first float32 above the half-minimum tie to the minimum subnormal', () => {
    const aboveTie = Math.fround(FLOAT16_HALF_MIN_SUBNORMAL + 2 ** -48);
    expect(aboveTie).toBeGreaterThan(FLOAT16_HALF_MIN_SUBNORMAL);
    expect(finiteFloat16Bits(aboveTie)).toBe(0x0001);
    expect(finiteFloat16Bits(-aboveTie)).toBe(0x8001);
  });

  it.each([
    ['positive half-minimum tie', FLOAT16_HALF_MIN_SUBNORMAL, /\+0/],
    ['negative half-minimum tie', -FLOAT16_HALF_MIN_SUBNORMAL, /-0/],
    ['positive below-half underflow', FLOAT16_HALF_MIN_SUBNORMAL / 2, /\+0/],
    ['negative below-half underflow', -FLOAT16_HALF_MIN_SUBNORMAL / 2, /-0/],
  ])('rejects the %s instead of storing signed zero', (_label, value, zeroPattern) => {
    expect(() => finiteFloat16Bits(value)).toThrow(zeroPattern);
  });

  it.each([
    ['NaN', Number.NaN, /must be finite/],
    ['positive infinity', Number.POSITIVE_INFINITY, /must be finite/],
    ['negative infinity', Number.NEGATIVE_INFINITY, /must be finite/],
    ['positive overflow', FLOAT16_MAX_FINITE + 1, /exceeds the finite RGBA16F range/],
    ['negative overflow', -FLOAT16_MAX_FINITE - 1, /exceeds the finite RGBA16F range/],
  ])('rejects %s', (_label, value, message) => {
    expect(() => finiteFloat16Bits(value)).toThrow(message);
  });
});
