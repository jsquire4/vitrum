import type { Mat4, Vec3 } from '@vitrum/core';
export { invertMat4 } from '@vitrum/shared-bvh';

export function multiplyMat4(a: Mat4, b: Mat4): Float32Array {
  const atA = (index: number): number => a[index] ?? 0;
  const atB = (index: number): number => b[index] ?? 0;
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        atA(0 * 4 + row) * atB(col * 4 + 0) +
        atA(1 * 4 + row) * atB(col * 4 + 1) +
        atA(2 * 4 + row) * atB(col * 4 + 2) +
        atA(3 * 4 + row) * atB(col * 4 + 3);
    }
  }
  return out;
}

export function transformPoint(m: Mat4, p: Vec3): [number, number, number] {
  const at = (index: number): number => m[index] ?? 0;
  const x = p[0];
  const y = p[1];
  const z = p[2];
  const tx = at(0) * x + at(4) * y + at(8) * z + at(12);
  const ty = at(1) * x + at(5) * y + at(9) * z + at(13);
  const tz = at(2) * x + at(6) * y + at(10) * z + at(14);
  const tw = at(3) * x + at(7) * y + at(11) * z + at(15);
  if (Math.abs(tw) > 1e-8) {
    return [tx / tw, ty / tw, tz / tw];
  }
  return [tx, ty, tz];
}

export function transformDirection(m: Mat4, v: Vec3): [number, number, number] {
  const at = (index: number): number => m[index] ?? 0;
  const x = at(0) * v[0] + at(4) * v[1] + at(8) * v[2];
  const y = at(1) * v[0] + at(5) * v[1] + at(9) * v[2];
  const z = at(2) * v[0] + at(6) * v[1] + at(10) * v[2];
  const len = Math.hypot(x, y, z);
  if (len < 1e-8) return [0, 1, 0];
  return [x / len, y / len, z / len];
}

/**
 * Transform a surface normal by the inverse-transpose of the upper-left 3×3
 * sub-matrix of `m` (the cofactor matrix divided by det(M)).
 *
 * Under non-uniform scale, applying M directly to normals distorts them so
 * they are no longer perpendicular to the transformed surface.  The correct
 * transform is `(M⁻¹)ᵀ · n`, which preserves the orthogonality invariant
 * `n · t = 0` after the transform of tangent `t` by M.
 *
 * The cofactor expansion avoids an explicit matrix inversion: the cofactor
 * matrix equals `det(M) · (M⁻¹)ᵀ` for the 3×3 submatrix, so the
 * `1/det` factor cancels in the post-normalize step.
 *
 * For a pure rotation (uniform scale = 1), this is equivalent to
 * `transformDirection(m, v)` up to floating-point rounding.
 *
 * References: Shirley & Morley "Realistic Ray Tracing" §2.7; any linear-algebra
 * text on the adjugate / classical adjoint of a 3×3 matrix.
 */
export function transformNormal(m: Mat4, v: Vec3): [number, number, number] {
  const [m00, m10, m20, m01, m11, m21, m02, m12, m22] = [
    m[0] ?? 0, m[1] ?? 0, m[2] ?? 0,
    m[4] ?? 0, m[5] ?? 0, m[6] ?? 0,
    m[8] ?? 0, m[9] ?? 0, m[10] ?? 0,
  ];
  // Cofactor matrix of the 3×3 upper-left submatrix.
  // c_ij = (-1)^(i+j) * M_ij  (minor of row i, col j).
  const c00 = m11 * m22 - m21 * m12;
  const c01 = -(m01 * m22 - m21 * m02);
  const c02 = m01 * m12 - m11 * m02;
  const c10 = -(m10 * m22 - m20 * m12);
  const c11 = m00 * m22 - m20 * m02;
  const c12 = -(m00 * m12 - m10 * m02);
  const c20 = m10 * m21 - m20 * m11;
  const c21 = -(m00 * m21 - m20 * m01);
  const c22 = m00 * m11 - m10 * m01;
  const x = c00 * v[0] + c10 * v[1] + c20 * v[2];
  const y = c01 * v[0] + c11 * v[1] + c21 * v[2];
  const z = c02 * v[0] + c12 * v[1] + c22 * v[2];
  const len = Math.hypot(x, y, z);
  if (len < 1e-8) return [0, 1, 0];
  return [x / len, y / len, z / len];
}
