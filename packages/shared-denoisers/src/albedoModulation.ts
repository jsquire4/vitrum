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

/**
 * Per-channel albedo used by BOTH directions of the modulation pair.
 *
 * The two directions must resolve the divisor and the multiplier identically or
 * demodulate→filter→remodulate is not an inverse. Two asymmetries previously
 * broke that:
 *   - demodulate clamped to 1e-3 while remodulate multiplied by the RAW value,
 *     so a channel in (0, 1e-3) was attenuated by up to 1000x and a channel at
 *     exactly 0 was annihilated. Any pixel carrying radiance that is not purely
 *     diffuse-reflected — an emitter, or the environment background — came out
 *     of an albedo-aware denoiser black, even though it renders correctly with
 *     the denoiser off.
 *   - out-of-range indices resolved to 0 (→ clamped to 1e-3, a 1000x scale-up)
 *     on demodulate but to 1 (neutral) on remodulate, so a short albedo buffer
 *     amplified radiance by 1000x instead of passing it through.
 *
 * Resolving both through this helper makes the round trip exact for every input:
 * `1` is the neutral fallback for a short buffer, and the shared 1e-3 floor
 * cancels itself. Channels at or above the floor are unaffected, so ordinary
 * SVGF/BMFR behaviour is byte-identical to before.
 */
const ALBEDO_FLOOR = 1e-3;

function albedoChannel(albedo: Float32Array, index: number): number {
  // Float32Array indexing returns number|undefined; values are never null, only
  // undefined past the end. `?? 1` is the neutral (white) fallback.
  return Math.max(albedo[index] ?? 1, ALBEDO_FLOOR);
}

/** Divide rgb by albedo (per channel, floored at 1e-3 to avoid divide-by-zero); returns a new Float32Array. */
export function demodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    out[si]     = (rgb[si]     ?? 0) / albedoChannel(albedo, si);
    out[si + 1] = (rgb[si + 1] ?? 0) / albedoChannel(albedo, si + 1);
    out[si + 2] = (rgb[si + 2] ?? 0) / albedoChannel(albedo, si + 2);
  }
  return out;
}

/** Multiply rgb by the SAME floored albedo demodulate used, in-place; returns the same Float32Array. */
export function remodulateAlbedo(
  rgb: Float32Array,
  albedo: Float32Array,
  pixelCount: number,
): Float32Array {
  for (let i = 0; i < pixelCount; i += 1) {
    const si = i * 3;
    rgb[si]     = (rgb[si]     ?? 0) * albedoChannel(albedo, si);
    rgb[si + 1] = (rgb[si + 1] ?? 0) * albedoChannel(albedo, si + 1);
    rgb[si + 2] = (rgb[si + 2] ?? 0) * albedoChannel(albedo, si + 2);
  }
  return rgb;
}
