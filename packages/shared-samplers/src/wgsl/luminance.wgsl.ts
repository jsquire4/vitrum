/**
 * Canonical Rec. 709 (ITU-R BT.709) luminance vector + scalar helper.
 *
 * Single source of truth for the RGB→Y conversion weights used by every
 * denoiser, sampler, and post-process pass in vitrum. Pre-C10 the constant
 * `(0.2126, 0.7152, 0.0722)` was open-coded in eight WGSL files across
 * `@vitrum/shared-denoisers` and `@vitrum/walkaround-hybrid` with five
 * different local identifier names (`LUM_W`, `SVGF_LUM_W`, `SVGF7_LUM_W`,
 * `luminanceWeights`, `lumW`, plus an anonymous `vec3f(...)` literal).
 *
 * The arithmetic is identical to the pre-C10 dot products at every replaced
 * site — Rec.709 is the universal sRGB primary weighting and there were no
 * Rec.601 (`0.299, 0.587, 0.114`) sites to preserve separately.
 *
 * References:
 *   ITU-R Recommendation BT.709-6 (2015), Table 3 — luminance coefficients
 *     for the sRGB / Rec.709 RGB primaries.
 *   Poynton, "Digital Video and HD: Algorithms and Interfaces", §24.
 */

export const LUMINANCE_WGSL = /* wgsl */ `
// Canonical Rec. 709 luminance weight vector and scalar helper.
// Single source: @vitrum/shared-samplers/src/wgsl/luminance.wgsl.ts.
const REC709_LUMINANCE: vec3f = vec3f(0.2126, 0.7152, 0.0722);
fn rec709Luminance(c: vec3f) -> f32 { return dot(c, REC709_LUMINANCE); }
`;
