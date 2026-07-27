import { requireFinite, requireInteger } from './numericGuards.js';

/**
 * Canonical Rec.709 luminance helper.
 *
 * The TS canonical lives here so shared-denoisers, pt-webgl2, and pt-webgpu
 * can import without taking external host-adapter deps. The `intensity`-
 * multiplied form used by the stained-glass light-packing path is the
 * caller's responsibility (multiply after calling this function).
 *
 * For the WGSL canonical, see `./wgsl/luminance.wgsl.ts` (`LUMINANCE_WGSL`
 * exported from package index). Most WGSL consumers across walkaround-hybrid
 * (shade, welfordTemporal, ppgUpdate, atrous), shared-denoisers
 * (hdrLuminanceBilateral, atrousVariance, svgfReprojection,
 * svgf7x7SpatialFallback), and pt-webgpu (pathTrace/material) pull the WGSL
 * canonical via the W1-R6 include graph (`LUMINANCE_MODULE`), a direct
 * `${LUMINANCE_WGSL}` prepend, or inheritance from COMMON_WGSL's `fn
 * luminance` re-export. The TS-side helper here is the host-CPU equivalent;
 * callers should import it rather than re-inlining the Rec.709 coefficients.
 */
export function luminance(r: number, g: number, b: number): number {
  requireFinite(r, 'luminance.r');
  requireFinite(g, 'luminance.g');
  requireFinite(b, 'luminance.b');
  const result = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  if (!Number.isFinite(result)) throw new RangeError('luminance result overflowed');
  return result;
}

/** Rec.709 luminance applied to a Float32Array RGB triple at the given index.
 *  `i` is the base index of the R channel; G is at `i+1`, B at `i+2`. */
export function luminanceAt(rgb: Float32Array, i: number): number {
  requireInteger(i, 'luminanceAt.i');
  if (i + 2 >= rgb.length) {
    throw new RangeError(`luminanceAt RGB triple at ${i} exceeds array length ${rgb.length}`);
  }
  return luminance(rgb[i]!, rgb[i + 1]!, rgb[i + 2]!);
}
