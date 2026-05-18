/**
 * Canonical Rec.709 luminance helpers (WGSL).
 *
 * The same `0.2126·R + 0.7152·G + 0.0722·B` formula is duplicated across
 * ~14 sites in the workspace (shade.wgsl / atrousVariance.wgsl /
 * spatialFilter.wgsl / svgfReprojection.wgsl / etc.). This module is the
 * single source of truth for the WGSL form; consumers either prepend
 * `LUMINANCE_WGSL` to their shader string directly (shared-denoisers
 * standalone-compile path) or declare `requires: ['luminance']` in their
 * `*_MODULE` registration so `composeWgsl` resolves it (walkaround-hybrid
 * include-graph path).
 *
 * The TS canonical lives at `@vitrum/shared-samplers/luminance.ts`;
 * both call sites share the same weights so CPU-side per-pixel luminance
 * tone mapping (svgf-real albedo demod) agrees with shader-side
 * variance/threshold calculations.
 *
 * Reference: Rec. ITU-R BT.709-6 (2015), §3 "Y'C_B'C_R'", coefficients
 * for luma from gamma-corrected R'G'B'.
 */
export const LUMINANCE_WGSL = /* wgsl */`

// Rec.709 / sRGB luminance weights (vec3f form for explicit dot products).
const LUM_W709: vec3f = vec3f(0.2126, 0.7152, 0.0722);

// Per-pixel luminance from linear RGB. Matches @vitrum/shared-samplers'
// TS \`luminance(r,g,b)\`.
fn luminance(c: vec3f) -> f32 {
  return dot(c, LUM_W709);
}

`;

/**
 * Stable name used by walkaround-hybrid's W1-R6 WGSL include-graph registry
 * (`packages/walkaround-hybrid/src/pipeline/wgslModules.ts`). Consumers
 * declare `requires: ['luminance']` to pull this module in without
 * dragging in the heavier `common` module.
 */
export const LUMINANCE_MODULE_NAME = 'luminance' as const;
