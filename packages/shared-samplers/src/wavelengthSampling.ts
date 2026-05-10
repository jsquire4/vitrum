/**
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
 *   XYZ += L(λ) × [x̄(λ), ȳ(λ), z̄(λ)] / pdf(λ)
 * (Monte Carlo estimator of the spectral integral).
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
  CIE_Y_TABLE,
  sampleCMF,
  xyzToLinearSRGB,
} from './cieCmf.js';

// ── Build CDF from Y table at module load ─────────────────────────────────────
// This is a one-time O(N) cost; the resulting CDF is used for all sampling.

/** Integral of Y CMF over [380, 780] nm via trapezoidal rule at 5 nm steps. */
const Y_INTEGRAL: number = (() => {
  let s = 0;
  for (let i = 0; i < CIE_TABLE_LENGTH; i++) {
    const w = i === 0 || i === CIE_TABLE_LENGTH - 1 ? 0.5 : 1.0;
    s += w * (CIE_Y_TABLE[i] ?? 0);
  }
  return s * CIE_LAMBDA_STEP;
})();

/**
 * Normalised CDF of the Y CMF.  CDF[i] is the probability that a sample drawn
 * from pdf(λ) ∝ Y(λ) has index < i (i.e. wavelength < 380 + 5·i nm).
 * Length = CIE_TABLE_LENGTH + 1 (CDF[0] = 0, CDF[N] = 1 by construction).
 */
const Y_CDF: Float64Array = (() => {
  const cdf = new Float64Array(CIE_TABLE_LENGTH + 1);
  cdf[0] = 0;
  for (let i = 1; i <= CIE_TABLE_LENGTH; i++) {
    const yPrev = CIE_Y_TABLE[i - 1] ?? 0;
    const yCurr = i < CIE_TABLE_LENGTH ? (CIE_Y_TABLE[i] ?? 0) : 0;
    cdf[i] = (cdf[i - 1] ?? 0) + (yPrev + yCurr) * 0.5 * CIE_LAMBDA_STEP;
  }
  // Normalise
  const total = cdf[CIE_TABLE_LENGTH] ?? Y_INTEGRAL;
  for (let i = 0; i <= CIE_TABLE_LENGTH; i++) {
    cdf[i] = (cdf[i] ?? 0) / total;
  }
  return cdf;
})();

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
export function sampleHeroWavelength(u: number): { lambdaNm: number; pdf: number } {
  // Clamp u to a valid range
  const uClamped = Math.max(0, Math.min(1 - 1e-7, u));

  // Binary search for the CDF segment containing uClamped.
  // Y_CDF has length CIE_TABLE_LENGTH + 1.
  let lo = 0;
  let hi = CIE_TABLE_LENGTH - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((Y_CDF[mid + 1] ?? 0) <= uClamped) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // lo is the lower table index: lambda range [lo_nm, lo_nm + step].
  const cdfLo = Y_CDF[lo] ?? 0;
  const cdfHi = Y_CDF[lo + 1] ?? 1;
  const yLo = CIE_Y_TABLE[lo] ?? 0;
  const yHi = CIE_Y_TABLE[lo + 1] ?? 0;

  // Linear interpolation within the segment.
  const t = cdfHi > cdfLo ? (uClamped - cdfLo) / (cdfHi - cdfLo) : 0;
  const lambdaNm = (CIE_LAMBDA_MIN + lo * CIE_LAMBDA_STEP) + t * CIE_LAMBDA_STEP;
  const lambdaClamped = Math.max(CIE_LAMBDA_MIN, Math.min(CIE_LAMBDA_MAX, lambdaNm));

  // PDF at the sampled wavelength: pdf(λ) = Y(λ) / ∫Y(λ)dλ
  const yAtLambda = yLo + t * (yHi - yLo);
  const pdf = yAtLambda / Y_INTEGRAL;

  return { lambdaNm: lambdaClamped, pdf };
}

/**
 * Convert a hero-wavelength path result to an RGB radiance contribution.
 *
 * For a path that sampled wavelength `lambdaNm` with scalar throughput
 * `throughput` and wavelength PDF `pdfLambda`, this function returns the RGB
 * contribution to accumulate into the framebuffer.
 *
 * The estimator is:
 *   RGB += [x̄(λ), ȳ(λ), z̄(λ)] · throughput / pdfLambda
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
  const weight = throughput / pdfLambda;
  const [r, g, b] = xyzToLinearSRGB(x * weight, y * weight, z * weight);
  return [r, g, b] as const;
}

// ── Exported constants (for fork-side documentation / unit tests) ──────────────

/** Integral of Y CMF over [380, 780] nm.  Used as the PDF normalisation constant. */
export const Y_CMF_INTEGRAL: number = Y_INTEGRAL;

/** Visible wavelength range minimum, nm. */
export const HERO_LAMBDA_MIN: number = CIE_LAMBDA_MIN;

/** Visible wavelength range maximum, nm. */
export const HERO_LAMBDA_MAX: number = CIE_LAMBDA_MAX;
