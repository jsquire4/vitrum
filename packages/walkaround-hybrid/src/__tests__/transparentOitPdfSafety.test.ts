import { describe, expect, it } from 'vitest';

import { TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';

const HALF_MAX = 65_504;
const MIN_COSINE_PDF = 1 / (4096 * Math.PI);

function boundedCosineImportanceDivide(
  numerator: readonly [number, number, number],
  pdf: number,
): readonly [number, number, number] {
  if (
    !numerator.every(Number.isFinite) ||
    !(pdf > 0) ||
    !Number.isFinite(pdf)
  ) {
    return [0, 0, 0];
  }
  const safePdf = Math.max(pdf, MIN_COSINE_PDF);
  return numerator.map((value) =>
    Math.min(Math.max(value, 0), HALF_MAX * safePdf) / safePdf
  ) as [number, number, number];
}

describe('transparent OIT cosine-importance safety', () => {
  it('pins the shader-side finite and rgba16float bounds', () => {
    for (const token of [
      'fn oitBoundedCosineImportanceDivide(numerator: vec3f, pdf: f32) -> vec3f',
      'let safePdf = max(pdf, INV_PI / 4096.0);',
      'let maxOutput = 65504.0;',
      'vec3f(maxOutput * safePdf)',
      'return boundedNumerator / safePdf;',
    ]) {
      expect(TRANSPARENT_OIT_WGSL).toContain(token);
    }
    expect(TRANSPARENT_OIT_WGSL).not.toContain(
      'oitLayerEnvSampleRadiance(payload, normal, wo, wi) / pdf',
    );
  });

  it('keeps the exact cosine/Lambertian cancellation at the 24-bit endpoints', () => {
    for (const xi of [0, 0.25, 0.5, 0.999, 1 - 2 ** -24]) {
      const cosine = Math.sqrt(1 - xi);
      const pdf = cosine / Math.PI;
      const result = boundedCosineImportanceDivide(
        [pdf, pdf, pdf],
        pdf,
      );
      expect(result[0]).toBeCloseTo(1, 12);
      expect(result[1]).toBeCloseTo(1, 12);
      expect(result[2]).toBeCloseTo(1, 12);
    }
  });

  it('rejects invalid densities and bounds every finite division result', () => {
    expect(boundedCosineImportanceDivide([1, 2, 3], 0)).toEqual([0, 0, 0]);
    expect(boundedCosineImportanceDivide([1, 2, 3], Number.NaN)).toEqual([0, 0, 0]);
    expect(boundedCosineImportanceDivide([Number.POSITIVE_INFINITY, 1, 1], 1))
      .toEqual([0, 0, 0]);

    for (const [numerator, pdf] of [
      [[Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE], Number.MIN_VALUE],
      [[1e30, 1e-30, -1], MIN_COSINE_PDF],
      [[1, 2, 3], 1e-30],
    ] as const) {
      const result = boundedCosineImportanceDivide(numerator, pdf);
      expect(result.every(Number.isFinite)).toBe(true);
      expect(result.every((value) => value >= 0 && value <= HALF_MAX)).toBe(true);
    }
  });
});
