/**
 * mathUtils.ts — shared column-major mat4 + vec3 utilities for shared-bvh.
 *
 * Extracted from scenePack.ts (D11.1) and consolidated with the private copies
 * in pickPrimitiveCpu.ts (D11.7). All functions use the @vitrum/core column-major
 * convention: element (row, col) = m[col*4 + row].
 */

import type { Mat4 } from '@vitrum/core';

// ── mat4 ──────────────────────────────────────────────────────────────────────

/**
 * Invert a column-major 4×4 matrix (float-array of 16 elements).
 * Returns a new Float32Array of 16 floats, or null when the matrix is singular,
 * non-finite, not representable in Float32, or loses reciprocal accuracy after
 * Float32 packing. Singularity is not classified with an absolute determinant
 * epsilon: determinant magnitude scales with authored units, so a small but
 * well-conditioned transform remains valid.
 */
export function invertMat4(m: Mat4): Float32Array | null {
  const at = (index: number): number => m[index] ?? 0;
  const out = new Float32Array(16);
  const a00 = at(0),
    a01 = at(1),
    a02 = at(2),
    a03 = at(3);
  const a10 = at(4),
    a11 = at(5),
    a12 = at(6),
    a13 = at(7);
  const a20 = at(8),
    a21 = at(9),
    a22 = at(10),
    a23 = at(11);
  const a30 = at(12),
    a31 = at(13),
    a32 = at(14),
    a33 = at(15);
  if (
    ![a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33].every(
      Number.isFinite,
    )
  ) {
    return null;
  }
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(det) || det === 0) return null;
  const invDet = 1.0 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;
  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  if (!Array.from(out).every(Number.isFinite)) return null;

  // Match core scene validation's reciprocal criterion. Checking both orders
  // catches a finite cofactor result that becomes unusable after Float32
  // packing without reintroducing an authored-scale-dependent determinant gate.
  for (const [left, right] of [
    [m, out],
    [out, m],
  ] as const) {
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        let product = 0;
        let absoluteTermSum = 0;
        for (let inner = 0; inner < 4; inner += 1) {
          const term = left[inner * 4 + row]! * right[column * 4 + inner]!;
          product += term;
          absoluteTermSum += Math.abs(term);
        }
        const expected = row === column ? 1 : 0;
        const tolerance = 1e-5 * Math.max(1, absoluteTermSum);
        if (!Number.isFinite(product) || Math.abs(product - expected) > tolerance) {
          return null;
        }
      }
    }
  }
  return out;
}

/**
 * Multiply two column-major 4×4 Float32Array matrices: result = a · b.
 * element (row, col) = m[col*4 + row].
 */
export function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += (a[k * 4 + row] ?? 0) * (b[col * 4 + k] ?? 0);
      o[col * 4 + row] = s;
    }
  }
  return o;
}

/**
 * Multiply a column-major 4×4 Float32Array by a vec4 (x,y,z,w).
 * Returns [x',y',z',w'].
 */
export function mat4MulVec4(
  m: Float32Array,
  x: number,
  y: number,
  z: number,
  w: number,
): [number, number, number, number] {
  const g = (i: number): number => m[i] ?? 0;
  return [
    g(0) * x + g(4) * y + g(8) * z + g(12) * w,
    g(1) * x + g(5) * y + g(9) * z + g(13) * w,
    g(2) * x + g(6) * y + g(10) * z + g(14) * w,
    g(3) * x + g(7) * y + g(11) * z + g(15) * w,
  ];
}

// ── vec3 ──────────────────────────────────────────────────────────────────────

export type V3 = readonly [number, number, number];

export const v3Sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

export const v3Cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export const v3Dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function v3Normalize(a: V3): V3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : a;
}
