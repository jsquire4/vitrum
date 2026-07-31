/**
 * worldTransforms.ts — column-major 4×4 / 3×3 transform kernels extracted from
 * worldSpaceMerge.ts (D12-9, pure move). These are bit-for-bit THREE-parity
 * point/normal/direction transforms + the normal matrix + determinant used by
 * the world-space merged tri-stream builder. Re-exported from worldSpaceMerge.ts
 * for backward-compatibility.
 */

/** Column-major identity 4×4. */
export const IDENTITY_MAT4: readonly number[] = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

/**
 * `worldPos = m · localPos` with the full perspective divide, bit-for-bit
 * THREE's `Vector3.applyMatrix4` (`w = 1/(…); xyz *= w`). Keeps the THREE
 * `w`-divide convention so the f32 round-off matches SGG exactly.
 */
export function applyMatrix4(m: ArrayLike<number>, x: number, y: number, z: number): [number, number, number] {
  const e0 = m[0] ?? 0, e4 = m[4] ?? 0, e8 = m[8] ?? 0, e12 = m[12] ?? 0;
  const e1 = m[1] ?? 0, e5 = m[5] ?? 0, e9 = m[9] ?? 0, e13 = m[13] ?? 0;
  const e2 = m[2] ?? 0, e6 = m[6] ?? 0, e10 = m[10] ?? 0, e14 = m[14] ?? 0;
  const e3 = m[3] ?? 0, e7 = m[7] ?? 0, e11 = m[11] ?? 0, e15 = m[15] ?? 0;
  const w = 1 / (e3 * x + e7 * y + e11 * z + e15);
  return [
    (e0 * x + e4 * y + e8 * z + e12) * w,
    (e1 * x + e5 * y + e9 * z + e13) * w,
    (e2 * x + e6 * y + e10 * z + e14) * w,
  ];
}

export function finiteVec3(v: readonly [number, number, number]): boolean {
  return Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

/**
 * Normal matrix (upper-left 3×3 inverse-transpose) of a column-major 4×4, as a
 * length-9 row-major-by-THREE-convention array — bit-for-bit THREE's
 * `Matrix3.getNormalMatrix(m4) = setFromMatrix4(m4).invert().transpose()`.
 *
 * `setFromMatrix4` lays out the upper-left 3×3 of the column-major m4 into the
 * Matrix3 as: n11=m[0], n21=m[1], n31=m[2], n12=m[4], n22=m[5], n32=m[6],
 * n13=m[8], n23=m[9], n33=m[10] (so te = [m0,m1,m2, m4,m5,m6, m8,m9,m10]).
 * `invert()` then writes te' via THREE's exact cofactor expressions, and
 * `transpose()` swaps the off-diagonal pairs. We fold invert+transpose into the
 * final element assignment so there's a single, mirror-able expression set.
 */
export function getNormalMatrix3(m: ArrayLike<number>): Float64Array {
  // setFromMatrix4(m4) → Matrix3 elements (te):
  const n11 = m[0] ?? 0, n21 = m[1] ?? 0, n31 = m[2] ?? 0;
  const n12 = m[4] ?? 0, n22 = m[5] ?? 0, n32 = m[6] ?? 0;
  const n13 = m[8] ?? 0, n23 = m[9] ?? 0, n33 = m[10] ?? 0;

  // invert(): THREE's exact cofactor formula (Matrix3.invert).
  const t11 = n33 * n22 - n32 * n23;
  const t12 = n32 * n13 - n33 * n12;
  const t13 = n23 * n12 - n22 * n13;
  const det = n11 * t11 + n21 * t12 + n31 * t13;

  const inv = new Float64Array(9);
  if (det === 0) {
    // THREE sets all nine elements to 0 on a singular matrix.
    return inv;
  }
  const detInv = 1 / det;
  // inv = te' (the inverted Matrix3), THREE's exact element assignments.
  inv[0] = t11 * detInv;
  inv[1] = (n31 * n23 - n33 * n21) * detInv;
  inv[2] = (n32 * n21 - n31 * n22) * detInv;
  inv[3] = t12 * detInv;
  inv[4] = (n33 * n11 - n31 * n13) * detInv;
  inv[5] = (n31 * n12 - n32 * n11) * detInv;
  inv[6] = t13 * detInv;
  inv[7] = (n21 * n13 - n23 * n11) * detInv;
  inv[8] = (n22 * n11 - n21 * n12) * detInv;

  // transpose() in place: swap (1,3),(2,6),(5,7).
  const out = new Float64Array(9);
  out[0] = inv[0]!; out[4] = inv[4]!; out[8] = inv[8]!;
  out[1] = inv[3]!; out[3] = inv[1]!;
  out[2] = inv[6]!; out[6] = inv[2]!;
  out[5] = inv[7]!; out[7] = inv[5]!;
  return out;
}

/**
 * `worldN = normalize( normalMatrix · localN )` — bit-for-bit THREE's
 * `Vector3.applyNormalMatrix(m3) = applyMatrix3(m3).normalize()`. `nm` is the
 * length-9 Matrix3 from {@link getNormalMatrix3}; `applyMatrix3` reads it as
 * `x' = nm0·x + nm3·y + nm6·z`, etc. (THREE's `Vector3.applyMatrix3`).
 */
export function applyNormalMatrix(
  nm: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const nx = (nm[0] ?? 0) * x + (nm[3] ?? 0) * y + (nm[6] ?? 0) * z;
  const ny = (nm[1] ?? 0) * x + (nm[4] ?? 0) * y + (nm[7] ?? 0) * z;
  const nz = (nm[2] ?? 0) * x + (nm[5] ?? 0) * y + (nm[8] ?? 0) * z;
  // THREE's Vector3.normalize: divide by length, or by 1 when length 0.
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  return [nx / len, ny / len, nz / len];
}

/** Transform a tangent/vector direction by the upper-left 3×3 of a column-major
 * matrix and normalize it. Unlike normals, tangents are ordinary directions, so
 * they use the direct linear transform rather than the inverse-transpose. */
export function applyDirectionMatrix4(
  m: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const tx = (m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z;
  const ty = (m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z;
  const tz = (m[2] ?? 0) * x + (m[6] ?? 0) * y + (m[10] ?? 0) * z;
  const scale = Math.max(Math.abs(tx), Math.abs(ty), Math.abs(tz));
  if (!(scale > 0) || !Number.isFinite(scale)) return [0, 0, 0];
  const sx = tx / scale;
  const sy = ty / scale;
  const sz = tz / scale;
  const scaledLength = Math.hypot(sx, sy, sz);
  if (!(scaledLength > 0) || !Number.isFinite(scaledLength)) return [0, 0, 0];
  return [sx / scaledLength, sy / scaledLength, sz / scaledLength];
}

/** Full 4×4 determinant of a column-major matrix — = THREE's
 *  `Matrix4.determinant()` (used to detect the winding flip). */
export function determinant4(m: ArrayLike<number>): number {
  const n11 = m[0] ?? 0, n12 = m[4] ?? 0, n13 = m[8] ?? 0, n14 = m[12] ?? 0;
  const n21 = m[1] ?? 0, n22 = m[5] ?? 0, n23 = m[9] ?? 0, n24 = m[13] ?? 0;
  const n31 = m[2] ?? 0, n32 = m[6] ?? 0, n33 = m[10] ?? 0, n34 = m[14] ?? 0;
  const n41 = m[3] ?? 0, n42 = m[7] ?? 0, n43 = m[11] ?? 0, n44 = m[15] ?? 0;
  return (
    n41 * (
      +n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 +
      n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34
    ) +
    n42 * (
      +n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 -
      n13 * n21 * n34 + n13 * n24 * n31 - n14 * n23 * n31
    ) +
    n43 * (
      +n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32 +
      n12 * n21 * n34 + n14 * n22 * n31 - n12 * n24 * n31
    ) +
    n44 * (
      -n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 +
      n13 * n21 * n32 - n12 * n21 * n33 + n12 * n23 * n31
    )
  );
}
