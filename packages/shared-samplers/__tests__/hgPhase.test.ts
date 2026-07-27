/**
 * hgPhase.test.ts — Unit tests for Sprint 7 Henyey-Greenstein phase function.
 *
 * Covers:
 *  - g=0 isotropic: evaluateHG = 1/(4π) for any cosTheta
 *  - g→1 forward delta: strongly peaks near cosTheta=1
 *  - g→-1 backward delta: strongly peaks near cosTheta=-1
 *  - pdfHG matches evaluateHG exactly
 *  - sampleHG: g=0 samples hemisphere uniformly (cosTheta distribution ~ uniform)
 *  - sampleHG: g→1 samples primarily in the forward direction
 *  - Numerical PDF normalization: ∫ over sphere = 1 via Hammersley sampling
 */

import { describe, it, expect } from 'vitest';
import { evaluateHG, sampleHG, pdfHG } from '../src/hgPhase.js';

const INV_4PI = 1 / (4 * Math.PI);

// ── evaluateHG ────────────────────────────────────────────────────────────────

describe('evaluateHG', () => {
  it('g=0 (isotropic): returns 1/(4π) for any cosTheta', () => {
    for (const cosTheta of [-1, -0.5, 0, 0.5, 1]) {
      expect(evaluateHG(cosTheta, 0)).toBeCloseTo(INV_4PI, 8);
    }
  });

  it('g=0.9 (forward): peaks near cosTheta=1', () => {
    const forward = evaluateHG(1.0, 0.9);
    const backward = evaluateHG(-1.0, 0.9);
    const side = evaluateHG(0.0, 0.9);
    expect(forward).toBeGreaterThan(side);
    expect(side).toBeGreaterThan(backward);
  });

  it('g=-0.9 (backward): peaks near cosTheta=-1', () => {
    const forward = evaluateHG(1.0, -0.9);
    const backward = evaluateHG(-1.0, -0.9);
    expect(backward).toBeGreaterThan(forward);
  });

  it('returns finite positive value for all valid inputs', () => {
    const gValues = [-0.9, -0.5, 0, 0.5, 0.9];
    const cosValues = [-1, -0.5, 0, 0.5, 1];
    for (const g of gValues) {
      for (const cosTheta of cosValues) {
        const val = evaluateHG(cosTheta, g);
        expect(val).toBeGreaterThan(0);
        expect(isFinite(val)).toBe(true);
      }
    }
  });
});

// ── pdfHG ─────────────────────────────────────────────────────────────────────

describe('pdfHG', () => {
  it('equals evaluateHG exactly (same formula)', () => {
    const gValues = [-0.8, 0, 0.5, 0.8];
    const cosValues = [-0.9, -0.3, 0, 0.7, 0.95];
    for (const g of gValues) {
      for (const cosTheta of cosValues) {
        expect(pdfHG(cosTheta, g)).toBeCloseTo(evaluateHG(cosTheta, g), 10);
      }
    }
  });
});

// ── sampleHG ──────────────────────────────────────────────────────────────────

describe('sampleHG', () => {
  it('returns a unit vector (|d| ≈ 1)', () => {
    const cases: [number, number, number][] = [
      [0.1, 0.5, 0],
      [0.3, 0.7, 0.8],
      [0.9, 0.2, -0.5],
      [0.5, 0.5, 0.9],
    ];
    for (const [u1, u2, g] of cases) {
      const [x, y, z] = sampleHG(u1, u2, g);
      const len = Math.sqrt(x * x + y * y + z * z);
      expect(len).toBeCloseTo(1.0, 5);
    }
  });

  it('g=0 (isotropic): sampled z ≈ 1 - 2*u2 (uniform cosTheta distribution)', () => {
    // For g=0, cosTheta = 1 - 2*u2 exactly.
    const pairs: [number, number][] = [
      [0.5, 0.0],
      [0.5, 0.25],
      [0.5, 0.5],
      [0.5, 0.75],
      [0.5, 1 - Number.EPSILON],
    ];
    for (const [u1, u2] of pairs) {
      const [, , z] = sampleHG(u1, u2, 0);
      const expected = 1 - 2 * u2;
      expect(z).toBeCloseTo(expected, 5);
    }
  });

  it('rejects random variates outside the half-open [0, 1) domain', () => {
    expect(() => sampleHG(1, 0.5, 0)).toThrow(RangeError);
    expect(() => sampleHG(0.5, 1, 0)).toThrow(RangeError);
    expect(() => sampleHG(-Number.EPSILON, 0.5, 0)).toThrow(RangeError);
    expect(() => sampleHG(0.5, Number.NaN, 0)).toThrow(RangeError);
  });
  it('g=0.9 (forward): z > 0 for most samples', () => {
    let positiveCount = 0;
    const N = 100;
    for (let i = 0; i < N; i++) {
      const u1 = (i + 0.5) / N;
      const u2 = ((i * 37) % N + 0.5) / N;
      const [, , z] = sampleHG(u1, u2, 0.9);
      if (z > 0) positiveCount++;
    }
    // Strongly forward: well over 80% should be z > 0
    expect(positiveCount).toBeGreaterThan(80);
  });

  it('g=-0.9 (backward): z < 0 for most samples', () => {
    let negativeCount = 0;
    const N = 100;
    for (let i = 0; i < N; i++) {
      const u1 = (i + 0.5) / N;
      const u2 = ((i * 37) % N + 0.5) / N;
      const [, , z] = sampleHG(u1, u2, -0.9);
      if (z < 0) negativeCount++;
    }
    expect(negativeCount).toBeGreaterThan(80);
  });
});

// ── PDF normalization ─────────────────────────────────────────────────────────

describe('HG PDF normalization (∫ p dω = 1)', () => {
  /**
   * Numerical integration via Hammersley sampling over the sphere.
   * ∫_{S²} p(cosθ, g) dω ≈ (4π / N) × Σ p(cosθ_i, g)
   *
   * Because the HG PDF is a function only of cosθ (rotationally symmetric),
   * we can integrate over [−1, 1]:
   * ∫_{-1}^{1} p(cosθ, g) × 2π dω_z = ∫_{-1}^{1} p(cosθ, g) × 2π dcosθ
   *
   * Using midpoint rule over N steps:
   * ≈ (2π × 2 / N) × Σ_{i=0}^{N-1} p(cosθ_i, g)  where cosθ_i = -1 + (2i+1)/N
   */
  it('∫ over sphere ≈ 1 for g=0 (isotropic)', () => {
    const N = 1000;
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const cosTheta = -1 + (2 * i + 1) / N;
      sum += evaluateHG(cosTheta, 0);
    }
    const integral = sum * (2 * Math.PI) * (2 / N); // 2π solid-angle factor × 2/N step
    expect(integral).toBeCloseTo(1.0, 2);
  });

  it('∫ over sphere ≈ 1 for g=0.8 (forward-scattering)', () => {
    const N = 2000;
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const cosTheta = -1 + (2 * i + 1) / N;
      sum += evaluateHG(cosTheta, 0.8);
    }
    const integral = sum * (2 * Math.PI) * (2 / N);
    expect(integral).toBeCloseTo(1.0, 2);
  });

  it('∫ over sphere ≈ 1 for g=-0.5 (back-scattering)', () => {
    const N = 2000;
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const cosTheta = -1 + (2 * i + 1) / N;
      sum += evaluateHG(cosTheta, -0.5);
    }
    const integral = sum * (2 * Math.PI) * (2 / N);
    expect(integral).toBeCloseTo(1.0, 2);
  });
});
