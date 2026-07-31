import { describe, expect, it } from 'vitest';
import { rgbF32ToRgba16fRowAligned } from '../src/rgba16fConversions.js';

describe('rgbF32ToRgba16fRowAligned input validation', () => {
  it('requires an exact RGB payload before allocating the packed buffer', () => {
    expect(() => rgbF32ToRgba16fRowAligned(new Float32Array(2), 1, 1))
      .toThrow(/src length must equal 3/);
    expect(() => rgbF32ToRgba16fRowAligned(new Float32Array(4), 1, 1))
      .toThrow(/src length must equal 3/);
  });

  it.each([
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ])('rejects non-finite input %s', (value) => {
    expect(() => rgbF32ToRgba16fRowAligned(
      new Float32Array([value, 0, 0]),
      1,
      1,
    )).toThrow(/src\[0\].*finite/);
  });

  it('accepts the finite half limit and rejects either finite overflow boundary', () => {
    expect(() => rgbF32ToRgba16fRowAligned(
      new Float32Array([65504, -65504, 0]),
      1,
      1,
    )).not.toThrow();

    expect(() => rgbF32ToRgba16fRowAligned(
      new Float32Array([65520, 0, 0]),
      1,
      1,
    )).toThrow(/src\[0\].*finite float16/);
    expect(() => rgbF32ToRgba16fRowAligned(
      new Float32Array([-65520, 0, 0]),
      1,
      1,
    )).toThrow(/src\[0\].*finite float16/);
  });
});
