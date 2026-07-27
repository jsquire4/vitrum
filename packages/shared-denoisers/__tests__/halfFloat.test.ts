import { describe, expect, it } from 'vitest';
import { float16BitsToFloat32, float32ToFloat16Bits } from '../src/halfFloat.js';

const F16_POSITIVE_INFINITY = 0x7c00;
const F16_QUIET_NAN = 0x7e00;
const F16_SIGN = 0x8000;
const F16_MAX_FINITE = 0x7bff;

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

function nextFloat32(value: number, direction: 'up' | 'down'): number {
  FLOAT32_SCRATCH[0] = value;
  const rounded = FLOAT32_SCRATCH[0]!;
  if (Number.isNaN(rounded)) return Number.NaN;
  if (direction === 'up' && rounded === Number.POSITIVE_INFINITY) return rounded;
  if (direction === 'down' && rounded === Number.NEGATIVE_INFINITY) return rounded;

  if (Object.is(rounded, 0) || Object.is(rounded, -0)) {
    UINT32_SCRATCH[0] = direction === 'up' ? 1 : 0x8000_0001;
    return FLOAT32_SCRATCH[0]!;
  }

  let bits = UINT32_SCRATCH[0]!;
  const increaseBits =
    (rounded > 0 && direction === 'up') ||
    (rounded < 0 && direction === 'down');
  bits = increaseBits ? bits + 1 : bits - 1;
  UINT32_SCRATCH[0] = bits;
  return FLOAT32_SCRATCH[0]!;
}

function negativeFloat32NaN(): number {
  UINT32_SCRATCH[0] = 0xffc1_2345;
  return FLOAT32_SCRATCH[0]!;
}

describe('IEEE-754 binary16 conversion', () => {
  it('round-trips every binary16 encoding (with canonical-NaN allowance)', () => {
    const mismatches: string[] = [];
    for (let bits = 0; bits <= 0xffff; bits += 1) {
      const value = float16BitsToFloat32(bits);
      const encoded = float32ToFloat16Bits(value);
      const isNaNEncoding = (bits & 0x7c00) === 0x7c00 && (bits & 0x03ff) !== 0;
      const matches = isNaNEncoding
        ? (encoded & 0x7fff) === F16_QUIET_NAN
        : encoded === bits;
      if (!matches && mismatches.length < 8) {
        mismatches.push(
          `0x${bits.toString(16).padStart(4, '0')} -> ` +
          `0x${encoded.toString(16).padStart(4, '0')}`,
        );
      }
    }
    expect(mismatches).toEqual([]);
  });

  it('selects the even endpoint at every finite positive and negative midpoint', () => {
    const mismatches: string[] = [];
    for (let lowerBits = 0; lowerBits < F16_MAX_FINITE; lowerBits += 1) {
      const upperBits = lowerBits + 1;
      const lower = float16BitsToFloat32(lowerBits);
      const upper = float16BitsToFloat32(upperBits);
      const midpoint = (lower + upper) / 2;
      const expectedMagnitude = (lowerBits & 1) === 0 ? lowerBits : upperBits;
      const positive = float32ToFloat16Bits(midpoint);
      const negative = float32ToFloat16Bits(-midpoint);
      if (positive !== expectedMagnitude && mismatches.length < 8) {
        mismatches.push(
          `+ midpoint 0x${lowerBits.toString(16)}..0x${upperBits.toString(16)} ` +
          `-> 0x${positive.toString(16)}`,
        );
      }
      if (negative !== (F16_SIGN | expectedMagnitude) && mismatches.length < 8) {
        mismatches.push(
          `- midpoint 0x${lowerBits.toString(16)}..0x${upperBits.toString(16)} ` +
          `-> 0x${negative.toString(16)}`,
        );
      }
    }
    expect(mismatches).toEqual([]);
  });

  it('rounds just below and above representative midpoint boundaries', () => {
    for (const lowerBits of [0x0000, 0x0001, 0x03ff, 0x3c00, 0x3c01, 0x3fff]) {
      const lower = float16BitsToFloat32(lowerBits);
      const upper = float16BitsToFloat32(lowerBits + 1);
      const midpoint = (lower + upper) / 2;
      expect(float32ToFloat16Bits(nextFloat32(midpoint, 'down'))).toBe(lowerBits);
      expect(float32ToFloat16Bits(nextFloat32(midpoint, 'up'))).toBe(lowerBits + 1);
    }
  });

  it('handles signed zero and the zero/subnormal tie exactly', () => {
    expect(float32ToFloat16Bits(0)).toBe(0x0000);
    expect(float32ToFloat16Bits(-0)).toBe(0x8000);
    expect(Object.is(float16BitsToFloat32(0x0000), 0)).toBe(true);
    expect(Object.is(float16BitsToFloat32(0x8000), -0)).toBe(true);

    const zeroToMinSubnormalMidpoint = 2 ** -25;
    expect(float32ToFloat16Bits(zeroToMinSubnormalMidpoint)).toBe(0x0000);
    expect(float32ToFloat16Bits(nextFloat32(zeroToMinSubnormalMidpoint, 'up')))
      .toBe(0x0001);
    expect(float32ToFloat16Bits(nextFloat32(zeroToMinSubnormalMidpoint, 'down')))
      .toBe(0x0000);
  });

  it('handles the largest-subnormal/minimum-normal boundary', () => {
    const lower = float16BitsToFloat32(0x03ff);
    const upper = float16BitsToFloat32(0x0400);
    const midpoint = (lower + upper) / 2;
    expect(float32ToFloat16Bits(lower)).toBe(0x03ff);
    expect(float32ToFloat16Bits(midpoint)).toBe(0x0400);
    expect(float32ToFloat16Bits(upper)).toBe(0x0400);
  });

  it('rounds the reproduced DDGI near-zero relocation component as a subnormal', () => {
    const value = 1.6740297621090525e-7;
    expect(float32ToFloat16Bits(value)).toBe(0x0003);
    expect(float16BitsToFloat32(float32ToFloat16Bits(value)))
      .toBe(1.7881393432617188e-7);
  });

  it('handles maximum finite, overflow midpoint, infinities, and NaNs', () => {
    expect(float32ToFloat16Bits(65504)).toBe(F16_MAX_FINITE);
    expect(float32ToFloat16Bits(nextFloat32(65520, 'down'))).toBe(F16_MAX_FINITE);
    expect(float32ToFloat16Bits(65520)).toBe(F16_POSITIVE_INFINITY);
    expect(float32ToFloat16Bits(-65520)).toBe(F16_SIGN | F16_POSITIVE_INFINITY);
    expect(float32ToFloat16Bits(70000)).toBe(F16_POSITIVE_INFINITY);
    expect(float32ToFloat16Bits(-70000)).toBe(F16_SIGN | F16_POSITIVE_INFINITY);
    expect(float32ToFloat16Bits(Number.POSITIVE_INFINITY)).toBe(F16_POSITIVE_INFINITY);
    expect(float32ToFloat16Bits(Number.NEGATIVE_INFINITY))
      .toBe(F16_SIGN | F16_POSITIVE_INFINITY);
    expect(float32ToFloat16Bits(Number.NaN)).toBe(F16_QUIET_NAN);
    expect(float32ToFloat16Bits(negativeFloat32NaN())).toBe(F16_SIGN | F16_QUIET_NAN);
    expect(Number.isNaN(float16BitsToFloat32(F16_QUIET_NAN))).toBe(true);
  });
});
