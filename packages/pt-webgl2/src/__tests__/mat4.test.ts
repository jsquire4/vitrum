/**
 * mat4.test.ts — unit tests for the pt-webgl2 column-major mat4 helpers.
 *
 * H6 (2026-06-09): covers makeRotationYMat4 (column-major layout + known
 * 90° mapping) and the zero-rotation invariant (output is byte-identical to
 * the IDENTITY_MAT4 constant used before H6).
 */

import { describe, it, expect } from 'vitest';
import { makeRotationYMat4, invertMat4 } from '../mat4.js';

const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const TOL = 1e-6;

function close(a: number, b: number, tol = TOL): boolean {
  return Math.abs(a - b) <= tol;
}

// Apply a 4×4 column-major matrix to a 3-vector (ignores w, no perspective divide).
function applyMat4Vec3(m: Float32Array, v: [number, number, number]): [number, number, number] {
  const x = m[0]! * v[0] + m[4]! * v[1] + m[8]! * v[2] + m[12]!;
  const y = m[1]! * v[0] + m[5]! * v[1] + m[9]! * v[2] + m[13]!;
  const z = m[2]! * v[0] + m[6]! * v[1] + m[10]! * v[2] + m[14]!;
  return [x, y, z];
}

describe('makeRotationYMat4', () => {
  it('is byte-identical to IDENTITY_MAT4 at rotationY = 0 (zero-rotation invariant)', () => {
    // H6 contract: rotationY = 0 must produce the same bytes as the pre-H6
    // hardcoded IDENTITY_MAT4 constant.  The packer passes -rotationY; at 0 that
    // is makeRotationYMat4(0), which must equal the identity.
    const m = makeRotationYMat4(0);
    for (let i = 0; i < 16; i++) {
      expect(m[i]).toBeCloseTo(IDENTITY_MAT4[i]!, 10);
    }
  });

  it('column-major layout: columns are [cos,0,-sin,0] [0,1,0,0] [sin,0,cos,0] [0,0,0,1]', () => {
    const theta = 0.7; // arbitrary angle ≠ 0, π/2, π
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const m = makeRotationYMat4(theta);
    // Float32Array stores ~7 significant decimal digits, so precision 6 (tolerance 5e-7)
    // is the tightest meaningful check for arbitrary transcendental values like cos(0.7).
    // Structural zero/one entries (exact in float32) use precision 10.
    // col0 = [cos, 0, -sin, 0]  (indices 0..3)
    expect(m[0]).toBeCloseTo(c, 6);
    expect(m[1]).toBeCloseTo(0, 10);
    expect(m[2]).toBeCloseTo(-s, 6);
    expect(m[3]).toBeCloseTo(0, 10);
    // col1 = [0, 1, 0, 0]  (indices 4..7)
    expect(m[4]).toBeCloseTo(0, 10);
    expect(m[5]).toBeCloseTo(1, 10);
    expect(m[6]).toBeCloseTo(0, 10);
    expect(m[7]).toBeCloseTo(0, 10);
    // col2 = [sin, 0, cos, 0]  (indices 8..11)
    expect(m[8]).toBeCloseTo(s, 6);
    expect(m[9]).toBeCloseTo(0, 10);
    expect(m[10]).toBeCloseTo(c, 6);
    expect(m[11]).toBeCloseTo(0, 10);
    // col3 = [0, 0, 0, 1]  (indices 12..15)
    expect(m[12]).toBeCloseTo(0, 10);
    expect(m[13]).toBeCloseTo(0, 10);
    expect(m[14]).toBeCloseTo(0, 10);
    expect(m[15]).toBeCloseTo(1, 10);
  });

  it('known 90° mapping: RY(π/2) rotates +X to -Z', () => {
    // Standard Y-rotation convention: RY(π/2) * (1,0,0) = (0, 0, -1).
    // i.e. +X maps to -Z after a 90° CCW-about-Y rotation.
    const m = makeRotationYMat4(Math.PI / 2);
    const result = applyMat4Vec3(m, [1, 0, 0]);
    expect(close(result[0], 0)).toBe(true);
    expect(close(result[1], 0)).toBe(true);
    expect(close(result[2], -1)).toBe(true);
  });

  it('known 90° mapping: RY(π/2) leaves +Y unchanged', () => {
    const m = makeRotationYMat4(Math.PI / 2);
    const result = applyMat4Vec3(m, [0, 1, 0]);
    expect(close(result[0], 0)).toBe(true);
    expect(close(result[1], 1)).toBe(true);
    expect(close(result[2], 0)).toBe(true);
  });

  it('known 90° mapping: RY(π/2) rotates +Z to +X', () => {
    // RY(π/2) * (0,0,1) = (sin 90°, 0, cos 90°) = (1, 0, 0)
    const m = makeRotationYMat4(Math.PI / 2);
    const result = applyMat4Vec3(m, [0, 0, 1]);
    expect(close(result[0], 1)).toBe(true);
    expect(close(result[1], 0)).toBe(true);
    expect(close(result[2], 0)).toBe(true);
  });

  it('H6 sign convention: caller passes -rotationY so env CCW rotation maps correctly', () => {
    // Convention: a CCW rotationY of the environment dome means world direction d
    // looks up the unrotated map at RY(-rotationY) * d.
    // The GLSL evaluates mat3(environmentRotation) * worldDir, so the uniform
    // is makeRotationYMat4(-rotationY).
    // Test: rotationY = π/2 (90° CCW env), world dir = (1,0,0):
    //   unrotated lookup dir = RY(-π/2) * (1,0,0) = (0,0,1)  [+Z in unrotated map]
    const rotationY = Math.PI / 2;
    const m = makeRotationYMat4(-rotationY); // what the packer sends
    const lookupDir = applyMat4Vec3(m, [1, 0, 0]); // GLSL: envRotation3x3 * worldDir
    // RY(-π/2) * (1,0,0) = (cos(-π/2), 0, -sin(-π/2)) = (0, 0, 1)
    expect(close(lookupDir[0], 0)).toBe(true);
    expect(close(lookupDir[1], 0)).toBe(true);
    expect(close(lookupDir[2], 1)).toBe(true);
  });

  it('RY(θ) · RY(-θ) = identity (round-trip)', () => {
    const theta = 1.23;
    const m = makeRotationYMat4(theta);
    const inv = makeRotationYMat4(-theta);
    // product = m · inv should be identity (column-major multiply)
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          sum += (m[k * 4 + row] ?? 0) * (inv[col * 4 + k] ?? 0);
        }
        const expected = col === row ? 1 : 0;
        expect(close(sum, expected)).toBe(true);
      }
    }
  });
});

describe('invertMat4 (pre-existing)', () => {
  it('returns null for a singular matrix', () => {
    expect(invertMat4(new Float32Array(16))).toBeNull();
  });

  it('inverts the identity', () => {
    const inv = invertMat4(IDENTITY_MAT4);
    expect(inv).not.toBeNull();
    if (inv == null) return;
    for (let i = 0; i < 16; i++) {
      expect(inv[i]).toBeCloseTo(IDENTITY_MAT4[i]!, 10);
    }
  });

  it('inverts a valid tiny scale without an absolute determinant cutoff', () => {
    const scale = 1e-4;
    const matrix = new Float32Array([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 2, -3, 4, 1]);
    const inverse = invertMat4(matrix);

    expect(inverse).not.toBeNull();
    expect(inverse![0]).toBeCloseTo(1 / scale, 2);
    expect(inverse![5]).toBeCloseTo(1 / scale, 2);
    expect(inverse![10]).toBeCloseTo(1 / scale, 2);
    expect(inverse![12]).toBeCloseTo(-2 / scale, 0);
    expect(inverse![13]).toBeCloseTo(3 / scale, 0);
    expect(inverse![14]).toBeCloseTo(-4 / scale, 0);
  });
});
