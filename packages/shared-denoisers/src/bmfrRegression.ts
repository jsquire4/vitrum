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
 * Solve the normal-equations system `(AΑᵀA + λI) x = Aᵀb` for x.
 *
 * BMFR's QR-on-the-feature-matrix is mathematically equivalent to solving the
 * normal equations; doing it on the (10×10) normal matrix is what the GPU
 * kernel does too (one Householder QR of the SYMMETRIC normal matrix), so the
 * CPU oracle mirrors that exact path. We assemble the symmetric positive-
 * (semi)definite normal matrix `M = AᵀA + λI` and right-hand side `r = Aᵀb`,
 * then solve `M x = r` via Householder QR of M (stable for the small dense
 * symmetric system; no pivoting needed once λ-loaded).
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
  // Normal matrix M (f×f) and rhs r (f).
  const M = new Float64Array(f * f);
  const r = new Float64Array(f);
  for (let k = 0; k < rows.length; k++) {
    const row = rows[k]!;
    const v = values[k]!;
    for (let i = 0; i < f; i++) {
      const ri = row[i]!;
      r[i] = (r[i] ?? 0) + ri * v;
      for (let j = 0; j < f; j++) {
        const idx = i * f + j;
        M[idx] = (M[idx] ?? 0) + ri * row[j]!;
      }
    }
  }
  for (let i = 0; i < f; i++) {
    const idx = i * f + i;
    M[idx] = (M[idx] ?? 0) + lambda;
  }

  return householderSolve(M, r, f);
}

/**
 * Solve a dense `A x = b` (A is f×f, row-major) via Householder QR.
 *
 * Reduces A to upper-triangular R by left-multiplying with Householder
 * reflectors, applying the same reflectors to b, then back-substitutes.
 * Returns x as a Float32Array (single-precision to match the GPU result
 * magnitude; the accumulation is done in f64 for stability).
 */
// MUST-MATCH MIRROR: householderSolve (CPU ↔ GPU)
//
// This function is the CPU reference for the WGSL kernel in
// shared-denoisers/src/wgsl/bmfr.wgsl.ts::householderSolve.
// The two implementations MUST stay bit-for-bit equivalent on every
// convergence guard and back-substitution step:
//
//   • norm < 1e-20         — skip near-zero pivot columns       ← BOTH sides
//   • vNormSq < 1e-30      — skip near-degenerate reflectors    ← BOTH sides
//   • abs(diag) < 1e-20    — back-substitution singularity gate ← BOTH sides
//   • back-substitution traversal order: i = n-1 .. 0 (descending)
//
// If you change any of these guards or the back-substitution order here,
// apply the IDENTICAL change in bmfr.wgsl.ts::householderSolve and vice-versa.
export function householderSolve(
  A: Float64Array,
  b: Float64Array,
  n: number,
): Float32Array {
  // Work on copies so the inputs are not mutated.
  const R = A.slice();
  const y = b.slice();

  for (let col = 0; col < n; col++) {
    // Norm of the sub-column R[col..n, col].
    let normSq = 0;
    for (let i = col; i < n; i++) {
      const v = R[i * n + col]!;
      normSq += v * v;
    }
    let norm = Math.sqrt(normSq);
    if (norm < 1e-20) continue; // already zero below the pivot — MUST-MATCH bmfr.wgsl.ts
    // Householder vector v = x - sign(x0)*||x|| e0.
    const x0 = R[col * n + col]!;
    const sign = x0 >= 0 ? 1 : -1;
    norm *= sign;
    const v = new Float64Array(n);
    v[col] = x0 + norm;
    for (let i = col + 1; i < n; i++) v[i] = R[i * n + col]!;
    let vNormSq = 0;
    for (let i = col; i < n; i++) vNormSq += v[i]! * v[i]!;
    if (vNormSq < 1e-30) continue; // MUST-MATCH bmfr.wgsl.ts

    // Apply reflector H = I - 2 v vᵀ / (vᵀv) to remaining columns of R.
    for (let j = col; j < n; j++) {
      let dot = 0;
      for (let i = col; i < n; i++) dot += v[i]! * R[i * n + j]!;
      const factor = (2 * dot) / vNormSq;
      for (let i = col; i < n; i++) R[i * n + j]! -= factor * v[i]!;
    }
    // Apply the same reflector to y.
    let dotY = 0;
    for (let i = col; i < n; i++) dotY += v[i]! * y[i]!;
    const fy = (2 * dotY) / vNormSq;
    for (let i = col; i < n; i++) y[i]! -= fy * v[i]!;
  }

  // Back-substitution on the upper-triangular R.
  const x = new Float32Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let acc = y[i]!;
    for (let j = i + 1; j < n; j++) acc -= R[i * n + j]! * x[j]!;
    const diag = R[i * n + i]!;
    x[i] = Math.abs(diag) < 1e-20 ? 0 : acc / diag; // MUST-MATCH bmfr.wgsl.ts
  }
  return x;
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
