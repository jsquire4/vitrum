/**
 * jakobHanika.test.ts — Unit tests for the real Jakob & Hanika 2019 RGB→spectrum
 * upsampling (Gauss–Newton sigmoid-coefficient solve).
 *
 * The headline check is the SPECTRAL ROUND-TRIP, the standard Jakob-Hanika
 * accuracy test: for a representative set of linear-sRGB colours,
 *
 *   RGB  --(rgbToSpectralCoefficients)-->  sigmoid coeffs
 *        --(integrate S(λ) under D65 + CIE CMFs)-->  XYZ  -->  linear sRGB
 *
 * must reconstruct the input colour to a tight tolerance. `spectralCoefficients-
 * ToRGB` is the exact inverse used by the solver and is what the GPU emulates.
 *
 * Plus:
 *  - Bounded reflectance: S(λ) ∈ [0, 1] for all coefficients / wavelengths.
 *  - Determinism and input clamping.
 *  - Known-colour spectral-shape sanity (red high in red band, blue high in
 *    blue band, green peaked mid-band).
 *
 * Reference: Jakob & Hanika 2019, "A Low-Dimensional Function Space for
 * Efficient Spectral Upsampling", CGF 38(2). https://rgl.epfl.ch/publications/Jakob2019Spectral
 */

import { describe, it, expect } from 'vitest';
import {
  rgbToSpectralCoefficients,
  evaluateSpectrum,
  spectralCoefficientsToRGB,
  VISIBLE_LAMBDA_MIN,
  VISIBLE_LAMBDA_MAX,
} from '../src/jakobHanika.js';
import {
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  CIE_D65_TABLE,
  CIE_LAMBDA_STEP,
  CIE_TABLE_LENGTH,
} from '../src/cieCmf.js';

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Sample the spectrum at N evenly spaced wavelengths across [380, 780]. */
function sampleSpectrum(coeffs: readonly [number, number, number], n = 81): number[] {
  const samples: number[] = [];
  for (let i = 0; i < n; i++) {
    const lambda =
      VISIBLE_LAMBDA_MIN + ((VISIBLE_LAMBDA_MAX - VISIBLE_LAMBDA_MIN) * i) / (n - 1);
    samples.push(evaluateSpectrum(coeffs, lambda));
  }
  return samples;
}

/** Max absolute per-channel error between a target colour and its round-trip. */
function roundTripError(r: number, g: number, b: number): number {
  const coeffs = rgbToSpectralCoefficients(r, g, b);
  const [rr, gg, bb] = spectralCoefficientsToRGB(coeffs);
  return Math.max(Math.abs(rr - r), Math.abs(gg - g), Math.abs(bb - b));
}

// ════════════════════════════════════════════════════════════════════════════
// Spectral round-trip accuracy — the canonical Jakob-Hanika check
// ════════════════════════════════════════════════════════════════════════════

describe('spectral round-trip: RGB → spectrum → RGB', () => {
  // Interior colours sit well inside the sRGB gamut and should reconstruct to
  // near machine precision (the solver drives the CIE-Lab residual to ~1e-4).
  const interiorColors: [string, number, number, number][] = [
    ['mid grey', 0.5, 0.5, 0.5],
    ['dark grey', 0.2, 0.2, 0.2],
    ['light grey', 0.8, 0.8, 0.8],
    ['warm tan', 0.6, 0.45, 0.3],
    ['olive', 0.35, 0.4, 0.15],
    ['steel blue', 0.25, 0.4, 0.55],
    ['dusty rose', 0.55, 0.35, 0.4],
    ['sage', 0.45, 0.55, 0.4],
    ['mauve', 0.5, 0.4, 0.5],
    ['skin tone', 0.75, 0.55, 0.45],
  ];

  for (const [name, r, g, b] of interiorColors) {
    it(`reconstructs ${name} (${r}, ${g}, ${b}) within 1e-3`, () => {
      expect(roundTripError(r, g, b)).toBeLessThan(1e-3);
    });
  }

  // Saturated primaries / secondaries lie on the gamut boundary, where the
  // sigmoid model is exact in chroma but can have a small luminance offset.
  // Still reconstructs tightly (Jakob-Hanika report < 1 ΔE; here < 5e-3 RGB).
  const saturatedColors: [string, number, number, number][] = [
    ['pure red', 1, 0, 0],
    ['pure green', 0, 1, 0],
    ['pure blue', 0, 0, 1],
    ['yellow', 1, 1, 0],
    ['cyan', 0, 1, 1],
    ['magenta', 1, 0, 1],
    ['white', 1, 1, 1],
    ['orange', 0.8, 0.4, 0.1],
    ['purple', 0.5, 0.1, 0.7],
  ];

  for (const [name, r, g, b] of saturatedColors) {
    it(`reconstructs ${name} (${r}, ${g}, ${b}) within 5e-3`, () => {
      expect(roundTripError(r, g, b)).toBeLessThan(5e-3);
    });
  }

  it('reconstructs a 6×6×6 sweep of the sRGB cube with mean error < 2e-3', () => {
    let sum = 0;
    let worst = 0;
    let count = 0;
    for (let ri = 0; ri < 6; ri++) {
      for (let gi = 0; gi < 6; gi++) {
        for (let bi = 0; bi < 6; bi++) {
          const r = ri / 5;
          const g = gi / 5;
          const b = bi / 5;
          const e = roundTripError(r, g, b);
          sum += e;
          worst = Math.max(worst, e);
          count++;
        }
      }
    }
    const mean = sum / count;
    expect(mean).toBeLessThan(2e-3);
    // Even the worst gamut-corner cell stays well bounded.
    expect(worst).toBeLessThan(1.5e-2);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Bounded reflectance: S(λ) ∈ [0, 1]
// ════════════════════════════════════════════════════════════════════════════

describe('evaluateSpectrum — bounded reflectance', () => {
  it('always returns values in [0, 1] across the visible range', () => {
    const testCases: [number, number, number][] = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
      [0, 0, 0],
      [0.5, 0.5, 0],
      [0.2, 0.8, 0.3],
      [0.9, 0.1, 0.4],
      [0.05, 0.05, 0.05],
    ];
    for (const [r, g, b] of testCases) {
      const coeffs = rgbToSpectralCoefficients(r, g, b);
      for (const s of sampleSpectrum(coeffs)) {
        expect(s).toBeGreaterThanOrEqual(0);
        expect(s).toBeLessThanOrEqual(1);
      }
    }
  });

  it('stays bounded even for wavelengths outside the fitted [380, 780] range', () => {
    const coeffs = rgbToSpectralCoefficients(0.7, 0.2, 0.5);
    for (const lambda of [200, 300, 360, 800, 900, 1200]) {
      const s = evaluateSpectrum(coeffs, lambda);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Known-colour spectral-shape sanity
// ════════════════════════════════════════════════════════════════════════════

describe('rgbToSpectralCoefficients — spectral shape sanity', () => {
  it('white (1,1,1): near-flat spectrum close to 1 across the band', () => {
    const coeffs = rgbToSpectralCoefficients(1, 1, 1);
    const samples = sampleSpectrum(coeffs);
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    expect(min).toBeGreaterThan(0.9);
    expect(max - min).toBeLessThan(0.1);
  });

  it('mid grey (0.5,0.5,0.5): near-flat spectrum around 0.5', () => {
    const coeffs = rgbToSpectralCoefficients(0.5, 0.5, 0.5);
    const samples = sampleSpectrum(coeffs);
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    expect(max - min).toBeLessThan(0.1);
    const mean = samples.reduce((a, c) => a + c, 0) / samples.length;
    expect(mean).toBeGreaterThan(0.4);
    expect(mean).toBeLessThan(0.6);
  });

  it('pure red (1,0,0): reflectance is high in the red band, low in the blue band', () => {
    const coeffs = rgbToSpectralCoefficients(1, 0, 0);
    const atRed = evaluateSpectrum(coeffs, 680);
    const atBlue = evaluateSpectrum(coeffs, 450);
    expect(atRed).toBeGreaterThan(0.8);
    expect(atBlue).toBeLessThan(0.2);
    expect(atRed).toBeGreaterThan(atBlue);
  });

  it('pure green (0,1,0): reflectance peaks in the mid band over the extremes', () => {
    const coeffs = rgbToSpectralCoefficients(0, 1, 0);
    const atGreen = evaluateSpectrum(coeffs, 540);
    const atRed = evaluateSpectrum(coeffs, 700);
    const atBlue = evaluateSpectrum(coeffs, 420);
    expect(atGreen).toBeGreaterThan(atRed);
    expect(atGreen).toBeGreaterThan(atBlue);
  });

  it('pure blue (0,0,1): reflectance is high in the blue band, low in the red band', () => {
    const coeffs = rgbToSpectralCoefficients(0, 0, 1);
    const atBlue = evaluateSpectrum(coeffs, 440);
    const atRed = evaluateSpectrum(coeffs, 700);
    expect(atBlue).toBeGreaterThan(0.8);
    expect(atBlue).toBeGreaterThan(atRed);
  });

  it('black (0,0,0): spectrum near 0 everywhere', () => {
    const coeffs = rgbToSpectralCoefficients(0, 0, 0);
    for (const s of sampleSpectrum(coeffs)) {
      expect(s).toBeLessThan(0.01);
    }
    // Black round-trips to (0,0,0).
    const [r, g, b] = spectralCoefficientsToRGB(coeffs);
    expect(Math.max(Math.abs(r), Math.abs(g), Math.abs(b))).toBeLessThan(1e-3);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Determinism + input clamping
// ════════════════════════════════════════════════════════════════════════════

describe('rgbToSpectralCoefficients — determinism', () => {
  it('same RGB → identical coefficients on repeated calls', () => {
    const r = 0.3,
      g = 0.7,
      b = 0.2;
    const c1 = rgbToSpectralCoefficients(r, g, b);
    const c2 = rgbToSpectralCoefficients(r, g, b);
    expect(c1[0]).toBe(c2[0]);
    expect(c1[1]).toBe(c2[1]);
    expect(c1[2]).toBe(c2[2]);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Absolute colorimetry guards
// ────────────────────────────────────────────────────────────────────────────
// The RGB→spectrum→RGB round-trip above is SELF-CONSISTENT: the solver and the
// inverse `spectralCoefficientsToRGB` share the exact same discretised D65 SPD,
// CMF tables, white point, and luminance normaliser. A round-trip therefore
// stays tight even if the illuminant, white point, or RGB↔XYZ matrices were all
// silently wrong-but-mutually-consistent. These tests pin the ABSOLUTE values so
// such a regression can't hide behind the round-trip.
// ════════════════════════════════════════════════════════════════════════════

describe('Jakob–Hanika — absolute colorimetry (round-trip cannot catch these)', () => {
  it('the discretised D65 white point matches the standard CIE D65 (Y-normalised)', () => {
    // The module folds Δλ·D65(λ)·CMF(λ)/N into per-sample weights with the
    // luminance normaliser N chosen so a unit reflector has Y = 1. Summing those
    // weights (S ≡ 1) must reproduce the CIE D65 white point (0.95047, 1, 1.08883)
    // up to 5 nm discretisation. A missing illuminant term or wrong normaliser
    // would shift this materially.
    let normY = 0;
    for (let i = 0; i < CIE_TABLE_LENGTH; i++) {
      normY += (CIE_D65_TABLE[i] ?? 0) * (CIE_Y_TABLE[i] ?? 0) * CIE_LAMBDA_STEP;
    }
    let wX = 0;
    let wY = 0;
    let wZ = 0;
    for (let i = 0; i < CIE_TABLE_LENGTH; i++) {
      const d = CIE_D65_TABLE[i] ?? 0;
      wX += (d * (CIE_X_TABLE[i] ?? 0) * CIE_LAMBDA_STEP) / normY;
      wY += (d * (CIE_Y_TABLE[i] ?? 0) * CIE_LAMBDA_STEP) / normY;
      wZ += (d * (CIE_Z_TABLE[i] ?? 0) * CIE_LAMBDA_STEP) / normY;
    }
    expect(wX).toBeCloseTo(0.95047, 2);
    expect(wY).toBeCloseTo(1.0, 5);
    expect(wZ).toBeCloseTo(1.08883, 2);
  });

  it('a flat S ≡ ½ reflectance (coeffs [0,0,0]) integrates to neutral grey (0.5, 0.5, 0.5)', () => {
    // sigmoid(0) = ½ everywhere. Under correct D65 + CMF + sRGB-inverse this must
    // land on neutral grey. A white-point/matrix mismatch tints this grey.
    const [r, g, b] = spectralCoefficientsToRGB([0, 0, 0]);
    expect(r).toBeCloseTo(0.5, 3);
    expect(g).toBeCloseTo(0.5, 3);
    expect(b).toBeCloseTo(0.5, 3);
  });

  it('a flat S ≡ 1 reflectance (saturated sigmoid) integrates to white (1, 1, 1)', () => {
    // A huge positive constant pins sigmoid(x) ≈ 1 across the band. By the
    // Y-normalisation this is the white point and must map to linear-sRGB white.
    const [r, g, b] = spectralCoefficientsToRGB([1e6, 0, 0]);
    expect(r).toBeCloseTo(1.0, 3);
    expect(g).toBeCloseTo(1.0, 3);
    expect(b).toBeCloseTo(1.0, 3);
  });

  it('solved white (1,1,1) produces a near-flat spectrum saturated toward 1', () => {
    // Guards the saturated-tail seed/convergence behaviour: white must drive the
    // sigmoid into its upper flat region across the whole band, not just on average.
    const coeffs = rgbToSpectralCoefficients(1, 1, 1);
    for (const s of sampleSpectrum(coeffs)) {
      expect(s).toBeGreaterThan(0.99);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});

describe('rgbToSpectralCoefficients — input clamping', () => {
  it('does not throw for out-of-range inputs', () => {
    expect(() => rgbToSpectralCoefficients(1.5, -0.3, 2.0)).not.toThrow();
  });

  it('clamps identically to in-range equivalents', () => {
    const clamped = rgbToSpectralCoefficients(1, 0, 0);
    const overRange = rgbToSpectralCoefficients(1.5, -0.1, 0);
    expect(clamped[0]).toBeCloseTo(overRange[0], 8);
    expect(clamped[1]).toBeCloseTo(overRange[1], 8);
    expect(clamped[2]).toBeCloseTo(overRange[2], 8);
  });

  it('produces finite coefficients for all sampled colours', () => {
    for (let ri = 0; ri <= 4; ri++) {
      for (let gi = 0; gi <= 4; gi++) {
        for (let bi = 0; bi <= 4; bi++) {
          const [c0, c1, c2] = rgbToSpectralCoefficients(ri / 4, gi / 4, bi / 4);
          expect(Number.isFinite(c0)).toBe(true);
          expect(Number.isFinite(c1)).toBe(true);
          expect(Number.isFinite(c2)).toBe(true);
        }
      }
    }
  });
});
