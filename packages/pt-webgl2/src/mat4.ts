// mat4 — column-major 4×4 linear algebra for the THREE-free WebGL2 path tracer.
//
// H6 (2026-06-09): added makeRotationYMat4 for the HDRI environment rotation
// convention.  The matrix is used as:
//
//   envRotation3x3 = mat3(environmentRotation)
//   lookupDir = envRotation3x3 * worldDir
//
// So to look up the unrotated map at the direction that corresponds to a CCW
// rotationY of the environment, the matrix must rotate worldDir by −rotationY
// (i.e. CW by rotationY).  makeRotationYMat4(radians) builds a standard
// column-major RY(radians) matrix; the caller passes −rotationY.  See the
// sign-convention proof in the JSDoc below.
//
// @vitrum/core exports only asMat4/isMat4 (no invert), and pt-webgpu's invert
// lives in its own package, so the 4×4 inverse is kept here. Matrices are
// column-major (three.js / WebGL convention), matching `FrameInput.viewMatrix` /
// `projMatrix`.

/**
 * Build a column-major 4×4 rotation matrix around the world +Y axis by
 * `radians` (counter-clockwise when viewed from above, i.e. from +Y).
 *
 * Sign-convention proof for H6 rotationY:
 *   The GLSL shader computes `envRotation3x3 = mat3(environmentRotation)` and
 *   then evaluates `envRotation3x3 * worldDir` before the equirect UV lookup.
 *   A CCWH `rotationY` of the *environment* means a world-space direction `d`
 *   should sample the UNROTATED map at `RY(−rotationY) * d`.  Therefore the
 *   caller supplies `makeRotationYMat4(−rotationY)` so that:
 *
 *     envRotation3x3 * d = RY(−rotationY) * d    ✓ correct unrotated-map UV
 *
 *   Column-major layout (WebGL convention, same as gl-matrix / Three.js):
 *
 *     [ cos θ   0   sin θ   0 ]   stored as columns:
 *     [  0      1    0      0 ]   col0 = [cos,-0, sin, 0]  wait, Y-rot is:
 *     [-sin θ   0   cos θ   0 ]   col0 = [cos, 0,-sin, 0]
 *     [  0      0    0      1 ]   col1 = [0,   1,  0,  0]
 *                                 col2 = [sin, 0, cos, 0]
 *                                 col3 = [0,   0,  0,  1]
 *
 *   Flat column-major index order: [0..3]=col0, [4..7]=col1, [8..11]=col2, [12..15]=col3.
 *
 * Verification (θ = π/2):
 *   RY(π/2) * (1,0,0) = (cos 90°, 0, −sin 90°) = (0, 0, −1)   ✓
 *   i.e. +X maps to −Z after a 90° CCW-Y rotation.
 */
export function makeRotationYMat4(radians: number): Float32Array {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  // Column-major: [col0 | col1 | col2 | col3]
  return new Float32Array([
    c,
    0,
    -s,
    0, // col0
    0,
    1,
    0,
    0, // col1
    s,
    0,
    c,
    0, // col2
    0,
    0,
    0,
    1, // col3
  ]);
}

/**
 * Invert a column-major 4×4 matrix via cofactor expansion. Returns `null` for a
 * singular/non-finite matrix or when its inverse cannot remain reciprocal after
 * Float32 packing. Determinant magnitude is not compared to an absolute
 * epsilon because valid scene/camera matrices may use very small authored
 * units. Column-major in, column-major out.
 *
 * Reference: the standard adjugate/determinant inverse (e.g. gl-matrix
 * `mat4.invert`, MESA `__gluInvertMatrixd`).
 */
export function invertMat4(m: ArrayLike<number>): Float32Array | null {
  const a00 = m[0] ?? 0,
    a01 = m[1] ?? 0,
    a02 = m[2] ?? 0,
    a03 = m[3] ?? 0;
  const a10 = m[4] ?? 0,
    a11 = m[5] ?? 0,
    a12 = m[6] ?? 0,
    a13 = m[7] ?? 0;
  const a20 = m[8] ?? 0,
    a21 = m[9] ?? 0,
    a22 = m[10] ?? 0,
    a23 = m[11] ?? 0;
  const a30 = m[12] ?? 0,
    a31 = m[13] ?? 0,
    a32 = m[14] ?? 0,
    a33 = m[15] ?? 0;
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
  if (!Number.isFinite(det) || det === 0) {
    return null;
  }
  const invDet = 1.0 / det;

  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * invDet;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * invDet;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  if (!Array.from(out).every(Number.isFinite)) return null;
  for (const [left, right] of [
    [m, out],
    [out, m],
  ] as const) {
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        let product = 0;
        let absoluteTermSum = 0;
        for (let inner = 0; inner < 4; inner += 1) {
          const term = (left[inner * 4 + row] ?? 0) * (right[column * 4 + inner] ?? 0);
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
