/**
 * jakobHanika.ts — Jakob+Hanika 2019 spectral upsampling (compact approximation).
 *
 * Converts an RGB color into a 3-coefficient polynomial that approximates a
 * smooth reflectance spectrum across the visible range [380, 780] nm.
 *
 * The spectrum at wavelength λ (nm) is given by:
 *   s(λ) = sigmoid(c0 + c1·λ + c2·λ²)
 * where sigmoid(x) = 0.5 + x / (2·√(1 + x²))  (smooth approximation of [0,1])
 *
 * ────────────────────────────────────────────────────────────────────────────
 * IMPLEMENTATION NOTE — Placeholder vs. Full Table
 * ────────────────────────────────────────────────────────────────────────────
 *
 * The original Jakob+Hanika 2019 paper provides a precomputed 3D lookup table
 * of polynomial coefficients indexed by quantized RGB triples.  The table
 * covers the full sRGB gamut with high accuracy and is the "correct"
 * implementation of the paper.
 *
 * Source: rgl.epfl.ch/publications/Jakob2019Spectral
 *
 * The precomputed table (coefficient_table.bin) is a multi-MB binary file.
 * Loading and distributing it as part of a browser library is undesirable:
 *   - Bundle size: the full table is ~24 MB; even compressed, it adds
 *     significant startup weight for a feature used only in bevel cells.
 *   - Network dependency: loading at runtime requires async fetch + caching.
 *   - License: the table data is research output; its distribution in a
 *     binary package requires explicit clearance from the authors.
 *
 * This implementation provides a PLACEHOLDER that uses a compact analytic
 * approximation instead of the precomputed table.  The placeholder maps each
 * RGB channel to a simplified polynomial by modeling each channel as a
 * Gaussian-like peak centered at its representative wavelength:
 *   Red   → λ_peak ≈ 700 nm
 *   Green → λ_peak ≈ 550 nm
 *   Blue  → λ_peak ≈ 450 nm
 *
 * The approximation fits a quadratic polynomial to the combined channel
 * spectrum using a linear combination of the three Gaussian peaks.  This
 * gives a smooth, physically plausible spectrum that is NOT as accurate as
 * the full Jakob+Hanika table but is visually correct for the primary use
 * case (bevel rainbow dispersion with 3 discrete spectral bands).
 *
 * Visual accuracy comparison:
 *   - Full table: smooth rainbow, spectrally accurate, matches paper fig 4.
 *   - This placeholder: 3-band linear rainbow.  Visually plausible for bevel
 *     dispersion; may show banding artifacts on extreme chromatic colors.
 *
 * TODO (post-Sprint-12):
 *   Integrate the full precomputed table once the distribution license is
 *   confirmed.  The public mitsuba-renderer repo at
 *   github.com/mitsuba-renderer/mitsuba3 contains a C++ implementation of the
 *   table lookup at `src/render/film.cpp` and the table at
 *   `resources/data/spectral/`.  A TypeScript port of the lookup is
 *   straightforward once the table is available. See plan/phase-6-status.md
 *   Known Issues for tracking.
 *
 * References:
 *   Jakob, Hanika 2019, "A Low-Dimensional Function Space for Efficient
 *   Spectral Upsampling", Computer Graphics Forum 38(2) (Eurographics 2019).
 *   https://rgl.epfl.ch/publications/Jakob2019Spectral
 *
 *   Mitsuba 3 implementation:
 *   github.com/mitsuba-renderer/mitsuba3 src/render/film.cpp
 *
 *   Phase 6 Sprint 8 spec: plan/phase-6-roadmap.md §Sprint 8.
 *   GLSL mirror: plan/sprint-8-pt-fork-patch.md §dielectric BSDF.
 */

// ────────────────────────────────────────────────────────────────────────────
// Visible range
// ────────────────────────────────────────────────────────────────────────────

const LAMBDA_MIN = 380; // nm
const LAMBDA_MAX = 780; // nm

// Representative wavelengths for R, G, B primaries (sRGB Rec.709 primaries)
const LAMBDA_R = 700; // nm — red primary
const LAMBDA_G = 550; // nm — green primary
const LAMBDA_B = 450; // nm — blue primary

// ────────────────────────────────────────────────────────────────────────────
// Smooth sigmoid (Jakob+Hanika §3)
// ────────────────────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  // Algebraically equivalent to 0.5 + x / (2 * sqrt(1 + x²))
  // Safe for large |x|: approaches 0 or 1 without overflow.
  return 0.5 + x / (2 * Math.sqrt(1 + x * x));
}

// ────────────────────────────────────────────────────────────────────────────
// Placeholder coefficient fitting
// ────────────────────────────────────────────────────────────────────────────

/**
 * Fit a quadratic polynomial c0 + c1·λ + c2·λ² to represent the spectral
 * reflectance of a color that is a mixture of three primary Gaussian-like
 * peaks at λ_R, λ_G, λ_B with weights r, g, b.
 *
 * Method: we want sigmoid(c0 + c1·λ + c2·λ²) to approximate:
 *   s(λ) = r·peak_R(λ) + g·peak_G(λ) + b·peak_B(λ)
 *
 * where peak_X(λ) is a unit-height Gaussian centred at λ_X with σ = 30 nm.
 *
 * Rather than inverting the sigmoid analytically (which is complex for a
 * mixed spectrum), we fit the polynomial to the log-odds (logit) of the
 * clamped spectrum via a 3-point interpolation:
 *   - Evaluate s at {λ_B, λ_G, λ_R}
 *   - Invert sigmoid (logit) to get the polynomial value at those 3 points
 *   - Solve the 3×3 Vandermonde system for c0, c1, c2
 *
 * This gives the exact quadratic fit through the three channel peaks.
 */
function fitCoefficients(r: number, g: number, b: number): [number, number, number] {
  // Gaussian peak width (nm)
  const sigma = 30;
  const sigma2 = sigma * sigma;

  /**
   * Spectral value at wavelength lambda, from the three channel contributions.
   * Clamped to [ε, 1−ε] for logit inversion stability.
   */
  function spectrum(lambda: number): number {
    const peakR = r * Math.exp(-((lambda - LAMBDA_R) ** 2) / (2 * sigma2));
    const peakG = g * Math.exp(-((lambda - LAMBDA_G) ** 2) / (2 * sigma2));
    const peakB = b * Math.exp(-((lambda - LAMBDA_B) ** 2) / (2 * sigma2));
    return Math.max(1e-6, Math.min(1 - 1e-6, peakR + peakG + peakB));
  }

  /** Logit (inverse sigmoid). */
  function logit(s: number): number {
    // sigmoid(x) = s → x = s - 0.5 / sqrt(s(1-s)) × ... (algebraic inverse)
    // For sigmoid(x) = 0.5 + x / (2√(1+x²)), the inverse is:
    //   x = (2s - 1) / sqrt(1 - (2s-1)²) = (2s-1) / sqrt(4s(1-s))
    const d = 2 * s - 1;
    const denom = Math.sqrt(4 * s * (1 - s));
    return denom < 1e-12 ? 0 : d / denom;
  }

  // Sample the polynomial at three representative wavelengths
  const lam = [LAMBDA_B, LAMBDA_G, LAMBDA_R] as const;
  const y = lam.map((l) => logit(spectrum(l)));

  // Solve Vandermonde system:
  //   [ 1  λ_0  λ_0² ] [ c0 ]   [ y_0 ]
  //   [ 1  λ_1  λ_1² ] [ c1 ] = [ y_1 ]
  //   [ 1  λ_2  λ_2² ] [ c2 ]   [ y_2 ]
  //
  // For 3×3 Vandermonde with fixed nodes λ_0=450, λ_1=550, λ_2=700 we
  // precompute V⁻¹ numerically.
  //
  // V = [ [1, 450, 202500], [1, 550, 302500], [1, 700, 490000] ]
  // det(V) = 3750000
  // V⁻¹ (row-major, computed via cofactors / det):
  //
  //   Row 0: [ 154/10,   -210/10,   66/10   ]  =  [ 15.4,  -21.0,   6.6  ]
  //   Row 1: [ -5e-2,   23/300,  -4/150    ]  =  [ -0.05, 0.07666, -0.02666 ]
  //   Row 2: [ 4e-5,   -2/30000,  4/150000 ]  =  [ 4e-5, -6.666e-5, 2.666e-5 ]

  // row 0: coefficients for c0 — derived from 3-node Vandermonde inverse
  // with nodes [LAMBDA_B=450, LAMBDA_G=550, LAMBDA_R=700] nm.
  const v00 =  15.4;                              // =  154 / 10
  const v01 = -21.0;                              // = -210 / 10
  const v02 =  6.6;                               // =   66 / 10

  // row 1: coefficients for c1
  const v10 = -5.0e-2;                            // = -5 / 100
  const v11 =  7.666_666_666_666_666_7e-2;        // = 23 / 300
  const v12 = -2.666_666_666_666_666_7e-2;        // = -4 / 150

  // row 2: coefficients for c2
  const v20 =  4.0e-5;                            // = 4 / 100000
  const v21 = -6.666_666_666_666_666_7e-5;        // = -2 / 30000
  const v22 =  2.666_666_666_666_666_7e-5;        // = 4 / 150000

  const c0 = v00 * y[0]! + v01 * y[1]! + v02 * y[2]!;
  const c1 = v10 * y[0]! + v11 * y[1]! + v12 * y[2]!;
  const c2 = v20 * y[0]! + v21 * y[1]! + v22 * y[2]!;

  return [c0, c1, c2];
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/**
 * Compact RGB→spectral coefficient fit (placeholder vs. full Jakob–Hanika table).
 *
 * Converts an RGB color (linear sRGB, components in [0, 1]) into a 3-coefficient
 * polynomial that approximates a smooth reflectance spectrum across [380, 780] nm.
 *
 * The spectrum at wavelength λ (nm) is:
 *   s(λ) = sigmoid(c0 + c1·λ + c2·λ²)
 * where sigmoid(x) = 0.5 + x / (2·√(1 + x²)).
 *
 * ⚠️ Not the paper’s precomputed table — see file-level documentation.
 *
 * @internal Placeholder approximation. Slated for replacement by the
 *           paper's precomputed table in Sprint 12. New code should call the
 *           stable alias `rgbToSpectralCoefficients` defined below.
 *
 * @param r - Red channel, linear sRGB [0, 1].
 * @param g - Green channel, linear sRGB [0, 1].
 * @param b - Blue channel, linear sRGB [0, 1].
 * @returns (c0, c1, c2) polynomial coefficients.
 */
export function rgbToApproxSpectralCoefficients(
  r: number,
  g: number,
  b: number,
): readonly [number, number, number] {
  // Clamp to [0, 1]
  r = Math.max(0, Math.min(1, r));
  g = Math.max(0, Math.min(1, g));
  b = Math.max(0, Math.min(1, b));

  // Achromatic shortcut: white/grey → flat spectrum (constant c0, c1=c2=0)
  // For a flat spectrum s(λ) = v: c0 = logit(v), c1 = c2 = 0.
  const maxCh = Math.max(r, g, b);
  if (maxCh < 1e-6) {
    // Black → spectrum near 0: sigmoid(−10) ≈ 0
    return [-10, 0, 0] as const;
  }

  const rng = Math.max(r, g, b) - Math.min(r, g, b);
  if (rng < 1e-4) {
    // Achromatic (grey): flat spectrum at value = maxCh
    // sigmoid(c0) = maxCh → c0 = logit(maxCh)
    const v = Math.max(1e-6, Math.min(1 - 1e-6, maxCh));
    const d = 2 * v - 1;
    const c0 = d / Math.sqrt(4 * v * (1 - v));
    return [c0, 0, 0] as const;
  }

  return fitCoefficients(r, g, b);
}

/**
 * Evaluate the polynomial spectrum at a given wavelength using coefficients
 * produced by `rgbToSpectralCoefficients`.
 *
 * s(λ) = sigmoid(c0 + c1·λ + c2·λ²)
 *
 * @param coeffs   - (c0, c1, c2) from rgbToApproxSpectralCoefficients.
 * @param lambdaNm - Wavelength in nm.  Values outside [380, 780] are accepted
 *                   but may extrapolate beyond the fitted range.
 * @returns Spectral reflectance in [0, 1].
 */
export function evaluateSpectrum(
  coeffs: readonly [number, number, number],
  lambdaNm: number,
): number {
  const [c0, c1, c2] = coeffs;
  return sigmoid(c0 + c1 * lambdaNm + c2 * lambdaNm * lambdaNm);
}

// ── Exported constants ────────────────────────────────────────────────────────

/** Visible range used by this implementation. */
export const VISIBLE_LAMBDA_MIN = LAMBDA_MIN;
export const VISIBLE_LAMBDA_MAX = LAMBDA_MAX;

/**
 * Stable public alias for the RGB→spectral coefficient fit. Use this name
 * in production code; `rgbToApproxSpectralCoefficients` is the current
 * approximation-only implementation and is marked @internal — the
 * precomputed-table replacement is scheduled to swap in via Sprint 12.
 */
export const rgbToSpectralCoefficients = rgbToApproxSpectralCoefficients;
