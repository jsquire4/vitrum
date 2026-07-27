/**
 * bmfrRegression.ts — CPU reference of the BMFR per-block feature regression.
 *
 * BMFR ("Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing
 * Reconstruction", Koskela et al., ACM TOG 38(5), 2019 — the "Ray-Traced
 * Reconstruction" / "BMFR" paper) reconstructs a noisy 1-spp color signal as a
 * least-squares fit against a per-pixel feature matrix, solved independently
 * per overlapping pixel block.
 *
 * For each block the system is:
 *
 *     minimise  || T · α − c ||²     (per color channel)
 *
 * where each row of T is the feature vector of one pixel in the block and c is
 * that pixel's noisy color channel. The fit is solved via a Householder QR
 * factorisation of T (the numerically stable scheme the reference BMFR
 * implementation uses — Koskela 2019 §4.2), then the reconstructed color is
 * the in-block evaluation `T · α`.
 *
 * Feature set (the "multi-order" augmented set, Koskela 2019 §4.1):
 *   [ 1, px, py, pz, nx, ny, nz, px², py², pz² ]   → BMFR_FEATURE_COUNT = 10
 *
 * Albedo is handled by demodulation (color is divided by albedo before the fit
 * and re-multiplied after) rather than as a feature column, matching the
 * engine-wide Schied-2017-style demodulation convention; the paper folds the
 * same effect in by including albedo features, but demodulation is the more
 * direct route here and keeps the feature matrix conditioning tighter.
 *
 * This module is pure CPU math (no GPU, no THREE) so the regression can be
 * unit-tested deterministically; `wgsl/bmfr.wgsl.ts` reimplements the identical
 * per-block fit on the GPU.
 *
 * References:
 *   Koskela, Immonen, Mäkitalo, Foi, Viitanen, Jääskeläinen, Kultala, Takala.
 *   "Blockwise Multi-Order Feature Regression for Real-Time Path-Tracing
 *   Reconstruction." ACM Transactions on Graphics 38(5), 2019.
 */

/** Number of regression feature columns: [1, p.xyz, n.xyz, p².xyz]. */
export const BMFR_FEATURE_COUNT = 10 as const;

/** Default square block edge length in pixels (Koskela 2019 use 32×32). */
export const BMFR_BLOCK_SIZE = 32 as const;

/**
 * Regularisation added to the QR pivot when a feature column is (near-)
 * degenerate — e.g. a flat block where p² columns are collinear with p, or a
 * single-plane block where one normal axis is constant. Keeps the fit from
 * exploding on rank-deficient blocks (the reference impl adds a similar
 * diagonal loading). Tuned small enough not to bias well-conditioned fits.
 */
export const BMFR_QR_REGULARISATION = 1e-3 as const;

/**
 * Build the 10-feature row for one pixel.
 *
 * World position is expected pre-normalised into a block-local frame by the
 * caller (subtract the block centroid, divide by a characteristic scale) so
 * the squared terms stay well-conditioned; this function does NOT renormalise.
 *
 * @param p  world position (block-local), length-3
 * @param n  unit normal, length-3
 * @param out  destination, length >= BMFR_FEATURE_COUNT (written in place)
 */
export function bmfrFeatureRow(
  p: readonly [number, number, number],
  n: readonly [number, number, number],
  out: Float32Array,
): void {
  out[0] = 1;
  out[1] = p[0];
  out[2] = p[1];
  out[3] = p[2];
  out[4] = n[0];
  out[5] = n[1];
  out[6] = n[2];
  out[7] = p[0] * p[0];
  out[8] = p[1] * p[1];
  out[9] = p[2] * p[2];
}

/**
 * Solve the regularised least-squares system directly by Householder QR.
 *
 * Tikhonov regularisation is represented as an augmented system rather than by
 * forming normal equations:
 *
 *     [ A          ] x ≈ [ b ]
 *     [ sqrt(λ) I  ]     [ 0 ]
 *
 * Factoring this rectangular matrix preserves the condition number of `A`;
 * factoring `AᵀA` would square it and is specifically unsuitable for the
 * nearly-collinear polynomial features BMFR encounters on planar surfaces.
 *
 * @param rows   array of feature rows (each length BMFR_FEATURE_COUNT)
 * @param values target value per row (the noisy color channel)
 * @param lambda Tikhonov diagonal loading (BMFR_QR_REGULARISATION default)
 * @returns the BMFR_FEATURE_COUNT-length weight vector α
 */
export function bmfrSolveChannel(
  rows: readonly Float32Array[],
  values: readonly number[],
  lambda: number = BMFR_QR_REGULARISATION,
): Float32Array {
  const f = BMFR_FEATURE_COUNT;
  if (rows.length !== values.length) {
    throw new RangeError('bmfrSolveChannel: rows and values must have equal length');
  }
  if (!Number.isFinite(lambda) || lambda < 0) {
    throw new RangeError('bmfrSolveChannel: lambda must be finite and nonnegative');
  }

  const regularisationRows = lambda > 0 ? f : 0;
  const rowCount = rows.length + regularisationRows;
  const A = new Float64Array(rowCount * f);
  const b = new Float64Array(rowCount);
  for (let k = 0; k < rows.length; k++) {
    const row = rows[k]!;
    if (row.length < f) {
      throw new RangeError(`bmfrSolveChannel: row ${k} has fewer than ${f} features`);
    }
    const v = values[k]!;
    if (!Number.isFinite(v)) {
      throw new RangeError(`bmfrSolveChannel: values[${k}] must be finite`);
    }
    for (let i = 0; i < f; i++) {
      const ri = row[i]!;
      if (!Number.isFinite(ri)) {
        throw new RangeError(`bmfrSolveChannel: row ${k} feature ${i} must be finite`);
      }
      A[k * f + i] = ri;
    }
    b[k] = v;
  }
  if (regularisationRows > 0) {
    const diagonal = Math.sqrt(lambda);
    for (let i = 0; i < f; i++) {
      A[(rows.length + i) * f + i] = diagonal;
    }
  }

  return householderLeastSquares(A, b, rowCount, f);
}

/**
 * Solve a dense rectangular least-squares system via direct Householder QR.
 *
 * `A` is row-major with `rowCount × columnCount` entries. The routine reduces
 * it to upper-triangular `R`, applies the same reflectors to `b`, and solves the
 * leading square system. Inputs are copied and never mutated.
 */
export function householderLeastSquares(
  A: Float64Array,
  b: Float64Array,
  rowCount: number,
  columnCount: number,
): Float32Array {
  if (!Number.isInteger(rowCount) || rowCount < 0) {
    throw new RangeError('householderLeastSquares: rowCount must be a nonnegative integer');
  }
  if (!Number.isInteger(columnCount) || columnCount < 0) {
    throw new RangeError('householderLeastSquares: columnCount must be a nonnegative integer');
  }
  if (A.length < rowCount * columnCount || b.length < rowCount) {
    throw new RangeError('householderLeastSquares: matrix or rhs is too short');
  }

  const R = A.slice(0, rowCount * columnCount);
  const y = b.slice(0, rowCount);
  const pivotCount = Math.min(rowCount, columnCount);

  for (let col = 0; col < pivotCount; col++) {
    let normSq = 0;
    for (let i = col; i < rowCount; i++) {
      const v = R[i * columnCount + col]!;
      normSq += v * v;
    }
    let norm = Math.sqrt(normSq);
    if (norm < 1e-20) continue;
    const x0 = R[col * columnCount + col]!;
    const sign = x0 >= 0 ? 1 : -1;
    norm *= sign;
    const v = new Float64Array(rowCount);
    v[col] = x0 + norm;
    for (let i = col + 1; i < rowCount; i++) {
      v[i] = R[i * columnCount + col]!;
    }
    let vNormSq = 0;
    for (let i = col; i < rowCount; i++) vNormSq += v[i]! * v[i]!;
    if (vNormSq < 1e-30) continue;

    for (let j = col; j < columnCount; j++) {
      let dot = 0;
      for (let i = col; i < rowCount; i++) {
        dot += v[i]! * R[i * columnCount + j]!;
      }
      const factor = (2 * dot) / vNormSq;
      for (let i = col; i < rowCount; i++) {
        R[i * columnCount + j]! -= factor * v[i]!;
      }
    }
    let dotY = 0;
    for (let i = col; i < rowCount; i++) dotY += v[i]! * y[i]!;
    const fy = (2 * dotY) / vNormSq;
    for (let i = col; i < rowCount; i++) y[i]! -= fy * v[i]!;
  }

  const x = new Float32Array(columnCount);
  for (let i = columnCount - 1; i >= 0; i--) {
    let acc = i < rowCount ? y[i]! : 0;
    for (let j = i + 1; j < columnCount; j++) {
      acc -= (i < rowCount ? R[i * columnCount + j]! : 0) * x[j]!;
    }
    const diag = i < rowCount ? R[i * columnCount + i]! : 0;
    x[i] = Math.abs(diag) < 1e-20 ? 0 : acc / diag;
  }
  return x;
}

/** Solve a square dense system via the same direct Householder implementation. */
export function householderSolve(
  A: Float64Array,
  b: Float64Array,
  n: number,
): Float32Array {
  return householderLeastSquares(A, b, n, n);
}

/**
 * Full per-block BMFR fit + reconstruction (CPU reference / oracle).
 *
 * Fits all three color channels against the shared feature matrix and writes
 * the reconstructed color back into each pixel of the block.
 *
 * @param features  flat per-pixel feature rows, length pixelCount*BMFR_FEATURE_COUNT
 * @param colors    flat per-pixel RGB, length pixelCount*3 (already demodulated)
 * @param pixelCount number of pixels in the block
 * @returns flat reconstructed RGB, length pixelCount*3
 */
export function bmfrFitBlock(
  features: Float32Array,
  colors: Float32Array,
  pixelCount: number,
  lambda: number = BMFR_QR_REGULARISATION,
): Float32Array {
  const f = BMFR_FEATURE_COUNT;
  const rows: Float32Array[] = new Array<Float32Array>(pixelCount);
  for (let p = 0; p < pixelCount; p++) {
    rows[p] = features.subarray(p * f, p * f + f);
  }
  const out = new Float32Array(pixelCount * 3);
  for (let ch = 0; ch < 3; ch++) {
    const vals: number[] = new Array<number>(pixelCount);
    for (let p = 0; p < pixelCount; p++) vals[p] = colors[p * 3 + ch]!;
    const alpha = bmfrSolveChannel(rows, vals, lambda);
    for (let p = 0; p < pixelCount; p++) {
      let acc = 0;
      const row = rows[p]!;
      for (let i = 0; i < f; i++) acc += row[i]! * alpha[i]!;
      out[p * 3 + ch] = acc;
    }
  }
  return out;
}
