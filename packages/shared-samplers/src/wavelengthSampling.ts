/**
 * @internal
 * Test/spec oracle for the GLSL hero-wavelength sampler in the PT fork.
 * Production consumer (`pt-webgl/forkUniformBridge.ts`) imports the underlying
 * `CIE_X/Y/Z_TABLE` from `./cieCmf.js` directly and computes its own integrals
 * + CDFs at module load. Not exported from the package public index — not for
 * production import. Existing tests reach in via `../src/wavelengthSampling.js`.
 *
 * wavelengthSampling.ts — Hero-wavelength sampling utilities.
 *
 * Implements stochastic wavelength selection by importance-sampling the CIE Y
 * color-matching function (luminous efficiency curve).  The resulting PDF
 * concentrates samples in the green–yellow peak (~500–600 nm) where the eye is
 * most sensitive, reducing Monte Carlo noise in the luminance channel.
 *
 * Technique: hero-wavelength spectral path tracing.
 *   Each path samples one wavelength λ from pdf(λ) ∝ Y(λ), traces the path
 *   in that monochromatic mode, then converts the scalar throughput to an RGB
 *   contribution via the CIE CMF and D65 illuminant at that wavelength.
 *
 * PDF: pdf(λ) = Y(λ) / ∫Y(λ)dλ  (Y normalised to integrate to 1)
 *
 * Reconstruction: the spectral radiance L(λ) at a single hero wavelength is
 * converted to a CIE XYZ colour via:
 *   XYZ += L(λ) × [x̄(λ), ȳ(λ), z̄(λ)] / (pdf(λ) × ∫Y dλ)
 * (Monte Carlo estimator of the spectral integral normalized to Y ≈ 1 for a
 * flat unit spectrum).
 * The XYZ is then converted to linear sRGB for display accumulation.
 *
 * References:
 *   Fascione, L. et al. "Hero Wavelength Spectral Sampling", EGSR 2015.
 *   CIE 015:2018 Colorimetry.
 *   Pharr, M. et al. "Physically Based Rendering", 4th ed., §4.6.2.
 *
 * Sprint 12 (Phase 6) deliverable — hero-wavelength spectral path tracing.
 * See: plan/sprint-12-pt-fork-patch.md for the fork-side implementation spec.
 */

import {
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_MAX,
  CIE_LAMBDA_STEP,
  CIE_TABLE_LENGTH,
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  sampleCMF,
  xyzToLinearSRGB,
} from './cieCmf.js';

// ── Build CDF from Y table at module load ─────────────────────────────────────
// This is a one-time O(N) cost; the resulting CDF is used for all sampling.
// Both Y_INTEGRAL and Y_CDF are derived from the same trapezoidal pass, so
// they are computed together to remove the ordering hazard of two sequential
// dependent IIFEs.

/**
 * Y_INTEGRAL: Integral of Y CMF over [380, 780] nm via trapezoidal rule at 5 nm steps.
 * Y_CDF:     Normalised CDF of the Y CMF. CDF[i] is the probability that a sample
 *            drawn from pdf(λ) ∝ Y(λ) has index < i (wavelength < 380 + 5·i nm).
 *            Length = CIE_TABLE_LENGTH + 1 (CDF[0] = 0, CDF[N] = 1 by construction).
 */
/**
 * buildIntegralAndCdf — derive the trapezoidal-rule integral and the
 * piecewise-linear normalised CDF for a 81-entry CMF table at 5 nm steps.
 *
 * Used to build importance-sampling tables for X, Y, and Z CMFs.  The CDF
 * has length CIE_TABLE_LENGTH + 1 (CDF[0] = 0, CDF[N] = 1 by construction).
 */
function buildIntegralAndCdf(table: Readonly<Float32Array>): { integral: number; cdf: Float64Array } {
  let integral = 0;
  for (let i = 0; i < CIE_TABLE_LENGTH; i++) {
    const w = i === 0 || i === CIE_TABLE_LENGTH - 1 ? 0.5 : 1.0;
    integral += w * (table[i] ?? 0);
  }
  integral *= CIE_LAMBDA_STEP;

  const cdf = new Float64Array(CIE_TABLE_LENGTH + 1);
  cdf[0] = 0;
  for (let i = 1; i <= CIE_TABLE_LENGTH; i++) {
    const vPrev = table[i - 1] ?? 0;
    const vCurr = i < CIE_TABLE_LENGTH ? (table[i] ?? 0) : 0;
    cdf[i] = (cdf[i - 1] ?? 0) + (vPrev + vCurr) * 0.5 * CIE_LAMBDA_STEP;
  }
  const total = cdf[CIE_TABLE_LENGTH] ?? integral;
  for (let i = 0; i <= CIE_TABLE_LENGTH; i++) {
    cdf[i] = (cdf[i] ?? 0) / total;
  }

  return { integral, cdf };
}

const { integral: Y_INTEGRAL, cdf: Y_CDF } = buildIntegralAndCdf(CIE_Y_TABLE);
const { integral: X_INTEGRAL, cdf: X_CDF } = buildIntegralAndCdf(CIE_X_TABLE);
const { integral: Z_INTEGRAL, cdf: Z_CDF } = buildIntegralAndCdf(CIE_Z_TABLE);

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Sample a hero wavelength by importance-sampling the CIE Y CMF.
 *
 * The PDF is proportional to Y(λ) (luminous efficiency), which peaks near
 * 555 nm and concentrates samples in the 500–600 nm range.  This reduces Monte
 * Carlo noise in the luminance channel relative to uniform wavelength sampling.
 *
 * Inversion: piecewise-linear CDF inversion on the 81-entry Y table.
 *
 * @param u - Uniform random variate in [0, 1).
 * @returns An object `{ lambdaNm, pdf }` where `lambdaNm` is the sampled
 *   wavelength in nm and `pdf` is the probability density at that wavelength
 *   (units: nm⁻¹).
 */
/**
 * sampleCmfCdfInverse — generic piecewise-linear CDF inversion for one CMF table.
 *
 * Given a uniform random `u ∈ [0, 1)`, returns the wavelength `λ` whose CDF
 * value equals `u` under the importance distribution `pdf(λ) ∝ table(λ)`,
 * plus the per-strategy `pdf(λ) = table(λ) / integral`.
 *
 * Shared by `sampleHeroWavelength` (legacy Y-only, single strategy) and
 * `sampleHeroWavelengthMIS` (multiple-strategy, balance heuristic).
 */
function sampleCmfCdfInverse(
  u: number,
  table: Readonly<Float32Array>,
  cdf: Float64Array,
  integral: number,
): { lambdaNm: number; pdf: number } {
  const uClamped = Math.max(0, Math.min(1 - 1e-7, u));

  let lo = 0;
  let hi = CIE_TABLE_LENGTH - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((cdf[mid + 1] ?? 0) <= uClamped) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const cdfLo = cdf[lo] ?? 0;
  const cdfHi = cdf[lo + 1] ?? 1;
  const vLo = table[lo] ?? 0;
  const vHi = table[lo + 1] ?? 0;

  const t = cdfHi > cdfLo ? (uClamped - cdfLo) / (cdfHi - cdfLo) : 0;
  const lambdaNm = CIE_LAMBDA_MIN + lo * CIE_LAMBDA_STEP + t * CIE_LAMBDA_STEP;
  const lambdaClamped = Math.max(CIE_LAMBDA_MIN, Math.min(CIE_LAMBDA_MAX, lambdaNm));

  const vAtLambda = vLo + t * (vHi - vLo);
  const pdf = vAtLambda / integral;

  return { lambdaNm: lambdaClamped, pdf };
}

/**
 * misMixturePdf — evaluate the balance-heuristic mixture PDF at λ.
 *
 * Used by the MIS estimator to weight a sampled wavelength regardless of
 * which underlying strategy (X, Y, or Z) drew it. The mixture is uniform
 * over the three strategies (each picked with probability 1/3), so:
 *
 *   pdf_mis(λ) = (pdf_X(λ) + pdf_Y(λ) + pdf_Z(λ)) / 3
 *             = (X(λ)/∫X + Y(λ)/∫Y + Z(λ)/∫Z) / 3
 */
function misMixturePdf(lambdaNm: number): number {
  const [x, y, z] = sampleCMF(lambdaNm);
  return (x / X_INTEGRAL + y / Y_INTEGRAL + z / Z_INTEGRAL) / 3;
}

export function sampleHeroWavelength(u: number): { lambdaNm: number; pdf: number } {
  return sampleCmfCdfInverse(u, CIE_Y_TABLE, Y_CDF, Y_INTEGRAL);
}

/**
 * Sample a hero wavelength using one-sample multiple-importance sampling
 * across the X, Y, and Z CMFs (Wilkie et al. extension to Fascione 2015).
 *
 * Y-only importance sampling — the legacy `sampleHeroWavelength` — clusters
 * samples around 555 nm because Y(λ) peaks there. At low SPP this leaves
 * the blue band (λ ≈ 445 nm, where Z(λ) peaks but Y(λ) ≈ 0.04) almost
 * empty, so blue-rich scenes (or any scene at low SPP) reconstruct with a
 * strong green/yellow bias and slow blue convergence.
 *
 * One-sample MIS distributes the sample budget across all three CMFs. With
 * uniform strategy selection, ~⅓ of samples land near the X peak (~600 nm,
 * red-orange), ~⅓ near Y (~555 nm, green), and ~⅓ near Z (~445 nm, blue) —
 * giving every chromatic region adequate coverage.
 *
 * The estimator weight uses the **mixture pdf** evaluated at the sampled λ,
 * not the per-strategy pdf:
 *
 *   pdf_mis(λ) = (pdf_X(λ) + pdf_Y(λ) + pdf_Z(λ)) / 3
 *
 * This is the balance-heuristic combination — variance-optimal for additive
 * estimators when individual strategy variances are similar.
 *
 * Reference: Wilkie, A. et al. "Hero Wavelength Spectral Sampling", EGSR 2015,
 * §3.3 (Multi-strategy hero wavelength sampling).
 *
 * @param uStrategy - Uniform random in [0, 1) used to pick X / Y / Z.
 * @param uLambda   - Uniform random in [0, 1) used for inverse-CDF on the chosen strategy.
 * @returns `{ lambdaNm, pdf }` where `pdf` is the mixture pdf — i.e. the
 *   denominator the caller should divide their throughput by, not the
 *   per-strategy pdf.
 */
export function sampleHeroWavelengthMIS(
  uStrategy: number,
  uLambda: number,
): { lambdaNm: number; pdf: number } {
  const s = Math.max(0, Math.min(1 - 1e-7, uStrategy));
  let lambdaNm: number;
  if (s < 1 / 3) {
    lambdaNm = sampleCmfCdfInverse(uLambda, CIE_X_TABLE, X_CDF, X_INTEGRAL).lambdaNm;
  } else if (s < 2 / 3) {
    lambdaNm = sampleCmfCdfInverse(uLambda, CIE_Y_TABLE, Y_CDF, Y_INTEGRAL).lambdaNm;
  } else {
    lambdaNm = sampleCmfCdfInverse(uLambda, CIE_Z_TABLE, Z_CDF, Z_INTEGRAL).lambdaNm;
  }
  return { lambdaNm, pdf: misMixturePdf(lambdaNm) };
}

/**
 * Convert a hero-wavelength path result to an RGB radiance contribution.
 *
 * For a path that sampled wavelength `lambdaNm` with scalar throughput
 * `throughput` and wavelength PDF `pdfLambda`, this function returns the RGB
 * contribution to accumulate into the framebuffer.
 *
 * The estimator is:
 *   RGB += [x̄(λ), ȳ(λ), z̄(λ)] · throughput / (pdfLambda × Y_CMF_INTEGRAL)
 * converted from CIE XYZ to linear sRGB.
 *
 * The D65 illuminant is baked into the `throughput` value by convention: the
 * fork shader multiplies the emitter's spectral power by the D65 SPD at λ
 * during emission sampling.  This function does pure CMF-based reconstruction.
 *
 * @param lambdaNm  - Hero wavelength in nm.
 * @param throughput - Scalar path throughput (energy fraction, dimensionless).
 * @param pdfLambda  - Wavelength sample PDF in nm⁻¹ (from `sampleHeroWavelength`).
 * @returns [r, g, b] linear sRGB radiance contribution.  May be negative for
 *   out-of-gamut colours; callers should clamp before display accumulation.
 */
export function wavelengthToRGB(
  lambdaNm: number,
  throughput: number,
  pdfLambda: number,
): readonly [number, number, number] {
  if (pdfLambda <= 0) return [0, 0, 0] as const;

  const [x, y, z] = sampleCMF(lambdaNm);
  const weight = throughput / (pdfLambda * Y_INTEGRAL);
  const [r, g, b] = xyzToLinearSRGB(x * weight, y * weight, z * weight);
  return [r, g, b] as const;
}

// ── Exported constants (for fork-side documentation / unit tests) ──────────────

/** Integral of Y CMF over [380, 780] nm.  Used as the PDF normalisation constant. */
export const Y_CMF_INTEGRAL: number = Y_INTEGRAL;

/** Integral of X CMF over [380, 780] nm. Used by the MIS sampler's mixture pdf. */
export const X_CMF_INTEGRAL: number = X_INTEGRAL;

/** Integral of Z CMF over [380, 780] nm. Used by the MIS sampler's mixture pdf. */
export const Z_CMF_INTEGRAL: number = Z_INTEGRAL;

/** Normalised CDF of X CMF (length 82). Mirrored to GLSL by the fork material upload. */
export const X_CMF_CDF: Readonly<Float64Array> = X_CDF;

/** Normalised CDF of Y CMF (length 82). Mirrored to GLSL by the fork material upload. */
export const Y_CMF_CDF: Readonly<Float64Array> = Y_CDF;

/** Normalised CDF of Z CMF (length 82). Mirrored to GLSL by the fork material upload. */
export const Z_CMF_CDF: Readonly<Float64Array> = Z_CDF;

/** Visible wavelength range minimum, nm. */
export const HERO_LAMBDA_MIN: number = CIE_LAMBDA_MIN;

/** Visible wavelength range maximum, nm. */
export const HERO_LAMBDA_MAX: number = CIE_LAMBDA_MAX;
