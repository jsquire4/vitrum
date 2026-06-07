/**
 * Canonical Rec.709 luminance helper. THREE-independent.
 *
 * The TS canonical lives here so shared-denoisers and pt-webgl can import
 * without taking a THREE peer dep. `three-bindings/src/math.ts:luminance`
 * is the THREE-coupled equivalent with an extra `intensity` multiplier
 * needed by the light-packing path; both forms share this base.
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
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Rec.709 luminance applied to a Float32Array RGB triple at the given index.
 *  `i` is the base index of the R channel; G is at `i+1`, B at `i+2`. */
export function luminanceAt(rgb: Float32Array, i: number): number {
  return 0.2126 * (rgb[i] ?? 0) + 0.7152 * (rgb[i + 1] ?? 0) + 0.0722 * (rgb[i + 2] ?? 0);
}
