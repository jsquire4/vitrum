/**
 * transformNormal.test.ts — unit tests for the (M⁻¹)ᵀ normal transform.
 *
 * Verifies Foundations Item #17: surface normals under non-uniform scale must
 * remain perpendicular to the transformed tangent after the transform.
 * Using M directly (as transformDirection does) breaks this invariant.
 */

import { describe, expect, it } from 'vitest';
import { asMat4, type Mat4 } from '@vitrum/core';
import { transformDirection, transformNormal } from '../math/mat4.js';

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Column-major 4×4 matrix representing a diagonal (scale-only) matrix. */
function scaleMat4(sx: number, sy: number, sz: number): Mat4 {
  // Column-major: col0 = [sx,0,0,0], col1 = [0,sy,0,0], col2 = [0,0,sz,0], col3 = [0,0,0,1]
  return asMat4(new Float32Array([
    sx, 0, 0, 0,
    0, sy, 0, 0,
    0, 0, sz, 0,
    0, 0, 0, 1,
  ]));
}

/** Column-major 4×4 rotation matrix, 90° around Y-axis. */
function rotY90Mat4(): Mat4 {
  const c = Math.cos(Math.PI / 2);
  const s = Math.sin(Math.PI / 2);
  // col0=[c,0,-s,0], col1=[0,1,0,0], col2=[s,0,c,0], col3=[0,0,0,1]
  return asMat4(new Float32Array([
    c, 0, -s, 0,
    0, 1, 0, 0,
    s, 0, c, 0,
    0, 0, 0, 1,
  ]));
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

describe('transformNormal', () => {
  it('preserves perpendicularity under non-uniform scale S=diag(2,1,3)', () => {
    const S = scaleMat4(2, 1, 3);
    // Face normal of XZ plane → points along Y
    const n: [number, number, number] = [0, 1, 0];
    // Tangent along X
    const t: [number, number, number] = [1, 0, 0];

    const tn = transformNormal(S, n);
    const tt = transformDirection(S, t);

    // Normal should remain (0, 1, 0) — Y axis is the only valid perpendicular
    // to XZ after diagonal-only scale; the cofactor collapses correctly.
    expect(tn[0]).toBeCloseTo(0, 6);
    expect(tn[1]).toBeCloseTo(1, 6);
    expect(tn[2]).toBeCloseTo(0, 6);

    // Tangent along X becomes (1, 0, 0) after normalization (scaled 2×, then normalized).
    expect(tt[0]).toBeCloseTo(1, 6);
    expect(tt[1]).toBeCloseTo(0, 6);
    expect(tt[2]).toBeCloseTo(0, 6);

    // Perpendicularity: dot must be ≈ 0.
    expect(dot(tn, tt)).toBeCloseTo(0, 6);
  });

  it('normal invariant: transformNormal ≈ transformDirection for pure rotation (uniform scale = 1)', () => {
    const M = rotY90Mat4();
    // Arbitrary normal in XZ plane
    const n: [number, number, number] = [1, 0, 0];

    const tn = transformNormal(M, n);
    const td = transformDirection(M, n);

    // For a pure rotation, both should give the same result.
    expect(tn[0]).toBeCloseTo(td[0], 5);
    expect(tn[1]).toBeCloseTo(td[1], 5);
    expect(tn[2]).toBeCloseTo(td[2], 5);
  });

  it('returns a unit vector for arbitrary non-uniform scale + arbitrary normal', () => {
    const S = scaleMat4(5, 0.1, 3);
    const n: [number, number, number] = [0.577, 0.577, 0.577]; // ~(1,1,1)/√3
    const tn = transformNormal(S, n);
    const len = Math.hypot(tn[0], tn[1], tn[2]);
    expect(len).toBeCloseTo(1, 6);
  });
});
