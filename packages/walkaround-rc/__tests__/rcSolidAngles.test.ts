/**
 * Octahedral solid-angle weight tests — Item 22 (RC normalization).
 *
 * Verifies that `computeOctahedralSolidAngles` returns physically correct
 * per-bin solid-angle weights for the N×N octahedral direction grid used by
 * the Radiance Cascades receiver irradiance integral.
 *
 * Reference: Cigolle et al. 2014, "A Survey of Efficient Representations for
 * Independent Unit Vectors", JCGT §2 / Appendix A.2.
 */

import { describe, it, expect } from 'vitest';
import { computeOctahedralSolidAngles } from '../src/octahedralSolidAngles.js';

const TWO_PI = 2 * Math.PI;
const FOUR_PI = 4 * Math.PI;

// ─── Helper ──────────────────────────────────────────────────────────────────

function sumArray(arr: Float32Array): number {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i]!;
  return s;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('computeOctahedralSolidAngles', () => {
  // ── Length ──────────────────────────────────────────────────────────────────

  it('gridSize=4 returns exactly 16 values', () => {
    const w = computeOctahedralSolidAngles(4);
    expect(w.length).toBe(16);
  });

  it('gridSize=8 returns exactly 64 values', () => {
    const w = computeOctahedralSolidAngles(8);
    expect(w.length).toBe(64);
  });

  it('gridSize=16 returns exactly 256 values', () => {
    const w = computeOctahedralSolidAngles(16);
    expect(w.length).toBe(256);
  });

  it('gridSize=32 returns exactly 1024 values', () => {
    const w = computeOctahedralSolidAngles(32);
    expect(w.length).toBe(1024);
  });

  // ── Sum ≈ 4π ────────────────────────────────────────────────────────────────
  // The octahedral grid covers the full unit sphere; the total solid angle
  // of all N² bins must equal 4π (the solid angle of the full sphere).
  // Tolerance: 1e-3 relative (per-task requirement).

  it('gridSize=4: sum of weights ≈ 4π within 1e-3', () => {
    const w = computeOctahedralSolidAngles(4);
    const total = sumArray(w);
    expect(Math.abs(total - FOUR_PI) / FOUR_PI).toBeLessThan(1e-3);
  });

  it('gridSize=8: sum of weights ≈ 4π within 1e-3', () => {
    const w = computeOctahedralSolidAngles(8);
    const total = sumArray(w);
    expect(Math.abs(total - FOUR_PI) / FOUR_PI).toBeLessThan(1e-3);
  });

  it('gridSize=16: sum of weights ≈ 4π within 1e-3', () => {
    const w = computeOctahedralSolidAngles(16);
    const total = sumArray(w);
    expect(Math.abs(total - FOUR_PI) / FOUR_PI).toBeLessThan(1e-3);
  });

  it('gridSize=32: sum of weights ≈ 4π within 1e-3', () => {
    const w = computeOctahedralSolidAngles(32);
    const total = sumArray(w);
    expect(Math.abs(total - FOUR_PI) / FOUR_PI).toBeLessThan(1e-3);
  });

  // ── Positive and bounded ────────────────────────────────────────────────────
  // All per-bin solid angles must be strictly positive and less than 2π
  // (the solid angle of a full hemisphere — no single bin can exceed it).

  for (const N of [4, 8, 16, 32]) {
    it(`gridSize=${N}: all weights are positive and bounded by 2π`, () => {
      const w = computeOctahedralSolidAngles(N);
      for (let i = 0; i < w.length; i++) {
        expect(w[i]!).toBeGreaterThan(0);
        expect(w[i]!).toBeLessThan(TWO_PI);
      }
    });
  }

  // ── Octahedral 4-way symmetry ───────────────────────────────────────────────
  // The octahedral grid has 4-fold symmetry under 90° rotation of the UV
  // square.  For an N×N grid, the bin at (col, row) has the same solid angle
  // as the bins at:
  //   - (N-1-row, col)       — 90° CCW rotation
  //   - (N-1-col, N-1-row)   — 180° rotation
  //   - (row, N-1-col)       — 270° CCW rotation
  //
  // Tolerance: 1% relative (the two-triangle area approximation introduces
  // a small asymmetry at corners that resolves with finer subdivision).

  it('gridSize=4: symmetric bins have equal solid angle (1% tolerance)', () => {
    const N = 4;
    const w = computeOctahedralSolidAngles(N);
    const tol = 0.01;

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const omega0 = w[row * N + col]!;
        // 90° CCW: (col, N-1-row) in new (row,col) → bin [col * N + (N-1-row)]
        const omega90 = w[col * N + (N - 1 - row)]!;
        // 180°: (N-1-row, N-1-col) → bin [(N-1-row)*N + (N-1-col)]
        const omega180 = w[(N - 1 - row) * N + (N - 1 - col)]!;
        // 270° CCW: (N-1-col, row) → bin [(N-1-col)*N + row]
        const omega270 = w[(N - 1 - col) * N + row]!;

        const avg = (omega0 + omega90 + omega180 + omega270) / 4;

        expect(Math.abs(omega0   - avg) / avg).toBeLessThan(tol);
        expect(Math.abs(omega90  - avg) / avg).toBeLessThan(tol);
        expect(Math.abs(omega180 - avg) / avg).toBeLessThan(tol);
        expect(Math.abs(omega270 - avg) / avg).toBeLessThan(tol);
      }
    }
  });

  it('gridSize=8: symmetric bins have equal solid angle (1% tolerance)', () => {
    const N = 8;
    const w = computeOctahedralSolidAngles(N);
    const tol = 0.01;

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const omega0   = w[row * N + col]!;
        const omega90  = w[col * N + (N - 1 - row)]!;
        const omega180 = w[(N - 1 - row) * N + (N - 1 - col)]!;
        const omega270 = w[(N - 1 - col) * N + row]!;
        const avg = (omega0 + omega90 + omega180 + omega270) / 4;

        expect(Math.abs(omega0   - avg) / avg).toBeLessThan(tol);
        expect(Math.abs(omega90  - avg) / avg).toBeLessThan(tol);
        expect(Math.abs(omega180 - avg) / avg).toBeLessThan(tol);
        expect(Math.abs(omega270 - avg) / avg).toBeLessThan(tol);
      }
    }
  });
});
