/**
 * jakobHanika.test.ts — Unit tests for Sprint 8 spectral upsampling.
 *
 * Note: This is the placeholder implementation (no precomputed table).
 * Tests verify the mathematical properties of the approximation, not
 * exact spectral accuracy against the full Jakob+Hanika table.
 *
 * Covers:
 *  - White/grey: flat spectrum across visible range
 *  - Pure red: spectrum peaks near 700nm, lower near 450nm
 *  - Pure green: spectrum peaks near 550nm
 *  - Pure blue: spectrum peaks near 450nm, lower near 700nm
 *  - Black: spectrum near 0 everywhere
 *  - evaluateSpectrum returns values in [0, 1]
 *  - Coefficients from identical RGB produce identical results (determinism)
 *  - Achromatic: c1 = c2 = 0 (flat spectrum)
 */

import { describe, it, expect } from 'vitest';
import { rgbToSpectralCoefficients, evaluateSpectrum, VISIBLE_LAMBDA_MIN, VISIBLE_LAMBDA_MAX } from '../src/jakobHanika.js';

// ── Helper ────────────────────────────────────────────────────────────────────

/** Sample the spectrum at N evenly spaced wavelengths across [380, 780]. */
function sampleSpectrum(
  coeffs: readonly [number, number, number],
  n = 40,
): number[] {
  const samples: number[] = [];
  for (let i = 0; i < n; i++) {
    const lambda = VISIBLE_LAMBDA_MIN + ((VISIBLE_LAMBDA_MAX - VISIBLE_LAMBDA_MIN) * i) / (n - 1);
    samples.push(evaluateSpectrum(coeffs, lambda));
  }
  return samples;
}

// ── evaluateSpectrum output range ─────────────────────────────────────────────

describe('evaluateSpectrum', () => {
  it('always returns values in [0, 1]', () => {
    const testCases: [number, number, number][] = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
      [0, 0, 0],
      [0.5, 0.5, 0],
      [0.2, 0.8, 0.3],
    ];
    for (const [r, g, b] of testCases) {
      const coeffs = rgbToSpectralCoefficients(r, g, b);
      for (const s of sampleSpectrum(coeffs)) {
        expect(s).toBeGreaterThanOrEqual(0);
        expect(s).toBeLessThanOrEqual(1);
      }
    }
  });
});

// ── White / grey (achromatic) ─────────────────────────────────────────────────

describe('rgbToSpectralCoefficients — achromatic', () => {
  it('white (1,1,1): flat spectrum ≈ constant across visible range', () => {
    const coeffs = rgbToSpectralCoefficients(1, 1, 1);
    const samples = sampleSpectrum(coeffs);
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    // Flat means max - min is small; white should be near 1.
    expect(max).toBeGreaterThan(0.8);
    expect(max - min).toBeLessThan(0.1); // flatness criterion
  });

  it('mid grey (0.5,0.5,0.5): flat spectrum ≈ constant', () => {
    const coeffs = rgbToSpectralCoefficients(0.5, 0.5, 0.5);
    const samples = sampleSpectrum(coeffs);
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    expect(max - min).toBeLessThan(0.1);
  });

  it('achromatic: c1 ≈ 0 and c2 ≈ 0', () => {
    const [, c1, c2] = rgbToSpectralCoefficients(0.7, 0.7, 0.7);
    expect(Math.abs(c1)).toBeLessThan(1e-6);
    expect(Math.abs(c2)).toBeLessThan(1e-6);
  });
});

// ── Chromatic primaries ───────────────────────────────────────────────────────

describe('rgbToSpectralCoefficients — chromatic primaries', () => {
  it('pure red (1,0,0): spectrum peaks near 700nm (higher than near 450nm)', () => {
    const coeffs = rgbToSpectralCoefficients(1, 0, 0);
    const atRed  = evaluateSpectrum(coeffs, 700);
    const atBlue = evaluateSpectrum(coeffs, 450);
    expect(atRed).toBeGreaterThan(atBlue);
  });

  it('pure green (0,1,0): spectrum peaks near 550nm', () => {
    const coeffs = rgbToSpectralCoefficients(0, 1, 0);
    const atGreen = evaluateSpectrum(coeffs, 550);
    const atRed   = evaluateSpectrum(coeffs, 700);
    const atBlue  = evaluateSpectrum(coeffs, 450);
    expect(atGreen).toBeGreaterThan(atRed);
    expect(atGreen).toBeGreaterThan(atBlue);
  });

  it('pure blue (0,0,1): spectrum peaks near 450nm (higher than near 700nm)', () => {
    const coeffs = rgbToSpectralCoefficients(0, 0, 1);
    const atBlue = evaluateSpectrum(coeffs, 450);
    const atRed  = evaluateSpectrum(coeffs, 700);
    expect(atBlue).toBeGreaterThan(atRed);
  });
});

// ── Black ─────────────────────────────────────────────────────────────────────

describe('rgbToSpectralCoefficients — black', () => {
  it('black (0,0,0): spectrum near 0 everywhere', () => {
    const coeffs = rgbToSpectralCoefficients(0, 0, 0);
    const samples = sampleSpectrum(coeffs);
    for (const s of samples) {
      expect(s).toBeLessThan(0.01); // sigmoid(−10) ≈ 5e-5
    }
  });
});

// ── Determinism ───────────────────────────────────────────────────────────────

describe('rgbToSpectralCoefficients — determinism', () => {
  it('same RGB → identical coefficients on repeated calls', () => {
    const r = 0.3, g = 0.7, b = 0.2;
    const c1 = rgbToSpectralCoefficients(r, g, b);
    const c2 = rgbToSpectralCoefficients(r, g, b);
    expect(c1[0]).toBe(c2[0]);
    expect(c1[1]).toBe(c2[1]);
    expect(c1[2]).toBe(c2[2]);
  });
});

// ── Input clamping ────────────────────────────────────────────────────────────

describe('rgbToSpectralCoefficients — input clamping', () => {
  it('does not throw for out-of-range inputs', () => {
    expect(() => rgbToSpectralCoefficients(1.5, -0.3, 2.0)).not.toThrow();
  });

  it('clamps identically to in-range equivalents', () => {
    const clamped = rgbToSpectralCoefficients(1, 0, 0);
    const overRange = rgbToSpectralCoefficients(1.5, -0.1, 0);
    // Both should produce the same result after clamping to [0,1]
    expect(clamped[0]).toBeCloseTo(overRange[0], 8);
    expect(clamped[1]).toBeCloseTo(overRange[1], 8);
    expect(clamped[2]).toBeCloseTo(overRange[2], 8);
  });
});
