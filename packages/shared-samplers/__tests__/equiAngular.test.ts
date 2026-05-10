/**
 * equiAngular.test.ts — Unit tests for Sprint 7 equi-angular volume sampling.
 *
 * Covers:
 *  - Returned t ≥ 0 for all valid inputs
 *  - PDF > 0 for all valid inputs
 *  - Closer light → tighter PDF concentration around the closest-point distance
 *  - Degenerate case: light on ray does not throw or return NaN
 *  - Distribution property: uniform random u → t distributed roughly toward
 *    the closest approach point
 */

import { describe, it, expect } from 'vitest';
import { sampleEquiAngular } from '../src/equiAngular.js';

type Vec3 = readonly [number, number, number];

const ORIGIN: Vec3 = [0, 0, 0];
const DIR_Z: Vec3 = [0, 0, 1];

// ── Basic invariants ──────────────────────────────────────────────────────────

describe('sampleEquiAngular — basic invariants', () => {
  it('returns t >= 0 for a light directly above the ray midpoint', () => {
    const lightPos: Vec3 = [0, 5, 10]; // 5 units above the ray, at t=10
    for (let u = 0.05; u < 1.0; u += 0.1) {
      const { t, pdf } = sampleEquiAngular(u, ORIGIN, DIR_Z, lightPos);
      expect(t).toBeGreaterThanOrEqual(0);
      expect(pdf).toBeGreaterThan(0);
      expect(isFinite(t)).toBe(true);
      expect(isFinite(pdf)).toBe(true);
    }
  });

  it('returns t >= 0 and pdf > 0 for light far away', () => {
    const lightPos: Vec3 = [0, 1000, 0]; // far above origin
    const { t, pdf } = sampleEquiAngular(0.5, ORIGIN, DIR_Z, lightPos);
    expect(t).toBeGreaterThanOrEqual(0);
    expect(pdf).toBeGreaterThan(0);
  });

  it('handles light behind the ray origin (negative t_closest)', () => {
    const lightPos: Vec3 = [0, 5, -100]; // behind origin (t_closest < 0)
    for (let u = 0.1; u < 1.0; u += 0.2) {
      const { t, pdf } = sampleEquiAngular(u, ORIGIN, DIR_Z, lightPos);
      expect(t).toBeGreaterThanOrEqual(0);
      expect(pdf).toBeGreaterThan(0);
      expect(isFinite(t)).toBe(true);
    }
  });
});

// ── Degenerate case ───────────────────────────────────────────────────────────

describe('sampleEquiAngular — degenerate (light on ray)', () => {
  it('does not throw or return NaN when light is on the ray', () => {
    // Light exactly on the ray (D ≈ 0): falls back to uniform sampling
    const lightOnRay: Vec3 = [0, 0, 10]; // exactly on the z-axis
    const { t, pdf } = sampleEquiAngular(0.5, ORIGIN, DIR_Z, lightOnRay);
    expect(isFinite(t)).toBe(true);
    expect(isFinite(pdf)).toBe(true);
    expect(t).toBeGreaterThanOrEqual(0);
    expect(pdf).toBeGreaterThan(0);
  });
});

// ── Tighter concentration for closer lights ───────────────────────────────────

describe('sampleEquiAngular — closer light → tighter pdf around t_closest', () => {
  /**
   * Property: when the light is closer to the ray (smaller D), the equi-angular
   * distribution concentrates more probability mass around the closest-approach
   * distance t_closest.  We verify this by comparing variance of sampled t
   * over many samples for a near light vs. a far light.
   */
  it('near light has lower t-variance than far light', () => {
    const N = 200;
    const tClosest = 10; // ray parameter at closest approach

    // Near light: 1 unit perpendicular to the ray
    const lightNear: Vec3 = [0, 1, tClosest];
    // Far light: 50 units perpendicular to the ray
    const lightFar: Vec3 = [0, 50, tClosest];

    let meanNear = 0, meanFar = 0;
    const tsNear: number[] = [];
    const tsFar: number[] = [];

    for (let i = 0; i < N; i++) {
      const u = (i + 0.5) / N;
      const { t: tNear } = sampleEquiAngular(u, ORIGIN, DIR_Z, lightNear);
      const { t: tFar } = sampleEquiAngular(u, ORIGIN, DIR_Z, lightFar);
      tsNear.push(tNear < 1000 ? tNear : 1000); // cap outliers
      tsFar.push(tFar < 1000 ? tFar : 1000);
      meanNear += tsNear[i]!;
      meanFar += tsFar[i]!;
    }

    meanNear /= N;
    meanFar /= N;

    let varNear = 0, varFar = 0;
    for (let i = 0; i < N; i++) {
      varNear += (tsNear[i]! - meanNear) ** 2;
      varFar += (tsFar[i]! - meanFar) ** 2;
    }
    varNear /= N;
    varFar /= N;

    // Near light should produce tighter concentration (lower variance in t)
    expect(varNear).toBeLessThan(varFar);
  });
});

// ── Distribution test ─────────────────────────────────────────────────────────

describe('sampleEquiAngular — distribution around closest approach', () => {
  it('median t is near the closest-approach distance for a nearby light', () => {
    // Light at [0, 2, 10]: closest approach at t=10 (directly below the light)
    const lightPos: Vec3 = [0, 2, 10];
    const N = 100;
    const ts: number[] = [];

    for (let i = 0; i < N; i++) {
      const u = (i + 0.5) / N;
      const { t } = sampleEquiAngular(u, ORIGIN, DIR_Z, lightPos);
      ts.push(Math.min(t, 200)); // cap at 200 to avoid unbounded samples
    }

    ts.sort((a, b) => a - b);
    const median = ts[Math.floor(N / 2)]!;

    // Median should be within a reasonable range of t_closest=10
    // (equi-angular concentrates around closest approach)
    expect(median).toBeGreaterThan(2);
    expect(median).toBeLessThan(30);
  });
});
