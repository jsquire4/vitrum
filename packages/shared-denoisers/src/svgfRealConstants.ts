/**
 * svgfRealConstants.ts — Numeric policy for the real Schied 2017 SVGF pipeline.
 *
 * Implements the default values from the paper (§4):
 *   α_min = 0.05 (Eq. 4 minimum EMA weight — paper recommendation)
 *   σ_z   = 0.10 (Eq. 2 depth tolerance — 10% relative difference)
 *   σ_n   = 0.95 (Eq. 2 normal dot threshold — ≈18° tolerance)
 *
 * GPU memory budget (at 1920×1080):
 *   historyLength  (r16uint,  W×H×2  bytes): ~  4 MB
 *   momentsHistory (rg32float, W×H×8 bytes): ~ 16 MB
 *   prevRadiance   (rgba16float, W×H×8 bytes): ~ 16 MB
 *   motionVec      (rg32float, W×H×8 bytes): ~ 16 MB (shared with atrous-variance path)
 *   Total:                                    ~ 52 MB at 1080p
 *
 * At 3840×2160 (4K): ~208 MB. Monitor GPU memory limit on mobile/iGPU targets.
 * The `'svgf-real'` mode should be advertised as desktop-class only.
 */

/** Default minimum EMA blend weight (Schied Eq. 4, paper default α_min). */
export const SVGF_REAL_DEFAULT_ALPHA_MIN = 0.05 as const;

/** Default depth-deviation tolerance for disocclusion test (Schied Eq. 2, σ_z). */
export const SVGF_REAL_DEFAULT_SIGMA_DEPTH = 0.10 as const;

/** Default normal dot-product threshold for disocclusion test (Schied Eq. 2, σ_n). */
export const SVGF_REAL_DEFAULT_SIGMA_NORMAL = 0.95 as const;

/** Default number of à-trous iterations run after the variance pass in svgf-real mode. */
export const SVGF_REAL_DEFAULT_ATROUS_ITERATIONS = 5 as const;

/** Maximum à-trous iterations in svgf-real mode (hard cap). */
export const SVGF_REAL_MAX_ATROUS_ITERATIONS = 12 as const;
