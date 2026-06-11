/**
 * albedoModulation.ts — shared albedo demodulate / remodulate helpers (Schied 2017 §4.1).
 *
 * Before a denoiser's variance + à-trous chain, divide HDR radiance by the
 * per-pixel albedo so the spatial filter operates on pure lighting (the
 * "lighting estimate" L = c/ρ in Schied 2017 §4.1). After the chain, the
 * filtered lighting is re-multiplied by albedo to restore the physically
 * correct outgoing radiance.
 *
 * The benefit is that high-frequency albedo variation (e.g. a red/green
 * checkerboard) no longer participates in the cross-bilateral weights of the
 * à-trous kernel, so the filter cannot bleed colors across material boundaries
 * that share the same depth + normal.
 *
 * Single source of truth for both the `svgf-real` and `atrous-variance` host
 * paths, so a denoiser switch is invariant under the modulation step.
 *
 * References:
 *   Schied et al. "Spatiotemporal Variance-Guided Filtering" HPG 2017, §4.1.
 */

/** Divide rgb by albedo (per channel, clamped to 1e-3 to avoid divide-by-zero); returns a new Float32Array. */
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

/** Multiply rgb by albedo in-place; returns the same Float32Array. */
export function remodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    // Float32Array indexing returns number|undefined; ?? 1 gives a white albedo
    // fallback (neutral multiply) when the albedo buffer is shorter than the rgb
    // buffer. Safe: Float32Array values are never null, only undefined on OOB.
    const ar = albedo[si]     ?? 1;
    const ag = albedo[si + 1] ?? 1;
    const ab = albedo[si + 2] ?? 1;
    rgb[si]     = (rgb[si]     ?? 0) * ar;
    rgb[si + 1] = (rgb[si + 1] ?? 0) * ag;
    rgb[si + 2] = (rgb[si + 2] ?? 0) * ab;
  }
  return rgb;
}
