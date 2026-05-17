/**
 * albedoModulation — CPU-side albedo demodulate / remodulate helpers.
 *
 * Schied 2017 §4.1 ("Spatiotemporal Variance-Guided Filtering").
 *
 * The à-trous / SVGF spatial filter operates more cleanly on lighting
 * (the diffuse irradiance × NoL term) than on outgoing radiance, because
 * radiance carries albedo discontinuities that the edge-stop weights
 * would otherwise have to learn around. The standard workaround is:
 *
 *   1. demodulate:  L_in  = L_out / albedo            (per pixel)
 *   2. filter:      L_in' = atrous(L_in, ...)
 *   3. remodulate:  L_out'= L_in' × albedo            (per pixel)
 *
 * Edge-stop weights see clean lighting; the final radiance carries the
 * full original albedo. This module exposes the CPU implementations
 * used by atrousVarianceWebGPU's host pipeline (and reusable by any
 * other CPU denoiser path — they know nothing about WebGPU).
 *
 * Reference: Christoph Schied et al. 2017, "Spatiotemporal Variance-Guided
 * Filtering: Real-Time Reconstruction for Path-Traced Global Illumination",
 * HPG 2017, §4.1.
 */

/**
 * Divide `rgb` by `albedo` per channel.
 *
 * Black-surface guard: each albedo channel is clamped to a 1e-3 floor
 * before the division so totally-black surfaces (`albedo == 0`) do not
 * produce Inf / NaN. The floor is unitless and matches the value used
 * by Schied 2017's reference implementation.
 *
 * Returns a fresh Float32Array of the same length as `rgb` (interleaved
 * RGB layout, `pixelCount * 3` floats).
 */
export function demodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    const ar = Math.max(albedo[si]     ?? 0, 1e-3);
    const ag = Math.max(albedo[si + 1] ?? 0, 1e-3);
    const ab = Math.max(albedo[si + 2] ?? 0, 1e-3);
    out[si]     = (rgb[si]     ?? 0) / ar;
    out[si + 1] = (rgb[si + 1] ?? 0) / ag;
    out[si + 2] = (rgb[si + 2] ?? 0) / ab;
  }
  return out;
}

/**
 * Multiply `rgb` by `albedo` per channel, in place.
 *
 * Returns the same `rgb` reference for fluent chaining. No black-surface
 * guard here: multiplication by 0 is the correct outcome (a black surface
 * reflects no light), and this is the canonical inverse of
 * `demodulateAlbedo` for non-zero albedos.
 *
 * For pixels where `albedo[i]` is undefined (truncated buffer), the
 * fallback is identity (multiply by 1) — preserves the legacy semantics
 * of the original private helper inside atrousVarianceWebGPU.ts.
 */
export function remodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    const ar = albedo[si]     !== undefined ? albedo[si]!     : 1;
    const ag = albedo[si + 1] !== undefined ? albedo[si + 1]! : 1;
    const ab = albedo[si + 2] !== undefined ? albedo[si + 2]! : 1;
    rgb[si]     = (rgb[si]     ?? 0) * ar;
    rgb[si + 1] = (rgb[si + 1] ?? 0) * ag;
    rgb[si + 2] = (rgb[si + 2] ?? 0) * ab;
  }
  return rgb;
}
