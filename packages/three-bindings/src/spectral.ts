/**
 * Tier 1 spectral RFE hooks (`external_requests/01-spectral-rendering.md`).
 *
 * three.js materials have no native μ(λ) samples; hosts may attach negotiated
 * `SpectralCurve` payloads on `@vitrum/core` `Material.extensions` under this
 * key when hand-assembling scenes, or extend bindings later to read custom
 * userData from THREE materials.
 */
export const VITRUM_SPECTRAL_EXTENSION_KEY = 'vitrum.spectral' as const;
