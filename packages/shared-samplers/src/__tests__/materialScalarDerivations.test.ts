/** Strict, exact scalar derivations shared by every renderer backend. */
import { describe, expect, it } from 'vitest';
import {
  dispersionStrengthFromAbbe,
  resolveEmissiveIntensity,
  sampleSpectralCurve,
  sampleSpectralGrid,
  sigmaAFromAttenuation,
  SPECTRAL_GRID_END_NM,
  SPECTRAL_GRID_SAMPLE_COUNT,
  SPECTRAL_GRID_START_NM,
  type SpectralCurveLike,
} from '../materialScalarDerivations.js';

describe('sigmaAFromAttenuation', () => {
  it('uses the exact Beer-Lambert inverse without a transmittance floor', () => {
    const result = sigmaAFromAttenuation([0.5, 0.25, 1], 2);
    expect(result[0]).toBe(-Math.log(0.5) / 2);
    expect(result[1]).toBe(-Math.log(0.25) / 2);
    expect(result[2]).toBe(0);
  });

  it('represents authored zero transmittance as infinite absorption', () => {
    expect(sigmaAFromAttenuation([0, 0.5, 1], 1)).toEqual([
      Number.POSITIVE_INFINITY,
      -Math.log(0.5),
      0,
    ]);
  });

  it('treats positive-infinite attenuation distance as no attenuation', () => {
    expect(sigmaAFromAttenuation([0, 0.25, 1], Number.POSITIVE_INFINITY)).toEqual([
      0,
      0,
      0,
    ]);
  });

  it.each([
    [[Number.NaN, 0.5, 1] as const, 1],
    [[-0.1, 0.5, 1] as const, 1],
    [[1.1, 0.5, 1] as const, 1],
    [[0.5, 0.5, 1] as const, 0],
    [[0.5, 0.5, 1] as const, -1],
    [[0.5, 0.5, 1] as const, Number.NaN],
    [[0.5, 0.5, 1] as const, Number.NEGATIVE_INFINITY],
  ])('rejects malformed color/distance input %#', (color, distance) => {
    expect(() => sigmaAFromAttenuation(color, distance)).toThrow(RangeError);
  });
});

describe('sampleSpectralCurve', () => {
  const curve: SpectralCurveLike = {
    wavelengthStart: 400,
    wavelengthEnd: 700,
    values: [0.1, 0.5, 0.9, 0.3],
  };

  it('linearly interpolates and clamps wavelengths to exact endpoints', () => {
    expect(sampleSpectralCurve(curve, 350)).toBe(0.1);
    expect(sampleSpectralCurve(curve, 400)).toBe(0.1);
    expect(sampleSpectralCurve(curve, 450)).toBeCloseTo(0.3, 14);
    expect(sampleSpectralCurve(curve, 700)).toBe(0.3);
    expect(sampleSpectralCurve(curve, 900)).toBe(0.3);
  });

  it('returns zero only for an absent curve', () => {
    expect(sampleSpectralCurve(null, 500)).toBe(0);
    expect(sampleSpectralCurve(undefined, 500)).toBe(0);
  });

  it('supports an explicitly allowed constant one-sample curve', () => {
    const constant = { wavelengthStart: 400, wavelengthEnd: 700, values: [0.7] };
    expect(sampleSpectralCurve(constant, 500)).toBe(0.7);
    expect(() => sampleSpectralCurve(constant, 500, { minValueCount: 3 }))
      .toThrow(RangeError);
  });

  it.each([
    [{ wavelengthStart: Number.NaN, wavelengthEnd: 700, values: [0, 0.5, 1] }, 500],
    [{ wavelengthStart: 700, wavelengthEnd: 400, values: [0, 0.5, 1] }, 500],
    [{ wavelengthStart: 400, wavelengthEnd: 700, values: [] }, 500],
    [{ wavelengthStart: 400, wavelengthEnd: 700, values: [0, Number.NaN, 1] }, 500],
    [{ wavelengthStart: 400, wavelengthEnd: 700, values: [0, -0.1, 1] }, 500],
    [curve, Number.NaN],
  ])('rejects malformed present curve/query %#', (candidate, lambda) => {
    expect(() => sampleSpectralCurve(candidate, lambda)).toThrow(RangeError);
  });

  it('rejects invalid validation options even when the curve is absent', () => {
    expect(() => sampleSpectralCurve(null, 500, { minValueCount: 0 }))
      .toThrow(RangeError);
  });
});

describe('sampleSpectralGrid', () => {
  it('uses the canonical 32-sample 380–780 nm grid', () => {
    expect(SPECTRAL_GRID_SAMPLE_COUNT).toBe(32);
    expect(SPECTRAL_GRID_START_NM).toBe(380);
    expect(SPECTRAL_GRID_END_NM).toBe(780);
  });

  it('samples and folds a linear attenuation curve exactly', () => {
    const grid = sampleSpectralGrid({
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: [0, 1, 2],
    }, { minValueCount: 3 });
    const expected = Array.from({ length: 32 }, (_, i) => (2 * i) / 31);
    grid.samples.forEach((value, index) => {
      expect(value).toBeCloseTo(expected[index]!, 14);
    });
    expect(grid.avg).toBeCloseTo(1, 14);
    expect(grid.max).toBe(2);
    expect(grid.sampleCount).toBe(32);
  });

  it('returns an empty signal for an absent curve', () => {
    expect(sampleSpectralGrid(null)).toEqual({
      samples: new Array<number>(32).fill(0),
      avg: 0,
      max: 0,
      sampleCount: 0,
    });
  });

  it('rejects malformed present curves before emitting samples', () => {
    expect(() => sampleSpectralGrid({
      wavelengthStart: 380,
      wavelengthEnd: 780,
      values: [0, Number.POSITIVE_INFINITY, 1],
    })).toThrow(RangeError);
  });
});

describe('dispersionStrengthFromAbbe', () => {
  it('evaluates the exact Fraunhofer C/F formula', () => {
    const denom = 1 / (486.1 ** 2) - 1 / (656.3 ** 2);
    expect(dispersionStrengthFromAbbe(1.5, 64)).toBe((1.5 - 1) / (64 * denom));
  });

  it('uses exact disabled values without accepting malformed input', () => {
    expect(dispersionStrengthFromAbbe(1, 50)).toBe(0);
    expect(dispersionStrengthFromAbbe(1.5, 0)).toBe(0);
    expect(() => dispersionStrengthFromAbbe(0, 50)).toThrow(RangeError);
    expect(() => dispersionStrengthFromAbbe(1.5, -1)).toThrow(RangeError);
    expect(() => dispersionStrengthFromAbbe(Number.NaN, 50)).toThrow(RangeError);
    expect(() => dispersionStrengthFromAbbe(1.5, Number.POSITIVE_INFINITY))
      .toThrow(RangeError);
  });
});

describe('resolveEmissiveIntensity', () => {
  it('defaults only an absent intensity and preserves valid authored values', () => {
    expect(resolveEmissiveIntensity(undefined)).toBe(1);
    expect(resolveEmissiveIntensity(0)).toBe(0);
    expect(resolveEmissiveIntensity(2.5)).toBe(2.5);
  });

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    'rejects malformed intensity %s',
    (value) => expect(() => resolveEmissiveIntensity(value)).toThrow(RangeError),
  );
});
