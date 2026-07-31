import { describe, expect, it } from 'vitest';
import { canonicalizeEnvironmentRotationF32 } from '../environmentRotation.js';

describe('canonical environment rotation publication', () => {
  it('preserves the f32 value of ordinary in-range angles', () => {
    expect(canonicalizeEnvironmentRotationF32(Math.PI / 3))
      .toBe(Math.fround(Math.PI / 3));
  });

  it('wraps huge finite angles before f32 publication', () => {
    const expected = Math.fround(1e300 % (2 * Math.PI));
    expect(canonicalizeEnvironmentRotationF32(1e300)).toBe(expected);
    expect(Number.isFinite(expected)).toBe(true);
  });

  it('normalizes either signed zero and allows tiny-angle underflow', () => {
    expect(Object.is(canonicalizeEnvironmentRotationF32(0), -0)).toBe(false);
    expect(Object.is(canonicalizeEnvironmentRotationF32(-0), -0)).toBe(false);
    expect(canonicalizeEnvironmentRotationF32(2 ** -150)).toBe(0);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    'rejects non-finite input %s',
    (value) => {
      expect(() => canonicalizeEnvironmentRotationF32(value))
        .toThrow(/must be finite/);
    },
  );
});
