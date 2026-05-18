/**
 * Canonical userData key strings for the Three.js ↔ vitrum scene-binding
 * layer. The forward (vitrum → three) and reverse (three → vitrum)
 * conversions live in separate files and must agree byte-for-byte on the
 * key names; defining them once here removes the chance of a silent typo
 * breaking the round trip.
 */
export const VITRUM_USER_DATA_KEYS = {
  DISPERSION_ABBE: 'vitrumDispersionAbbeNumber',
  SCATTERING_COEFF: 'vitrumScatteringCoefficient',
  SCATTERING_RGB: 'vitrumScatteringCoefficientRGB',
  SCATTERING_ANISO: 'vitrumScatteringAnisotropy',
  SPECTRAL_ATTEN: 'vitrumSpectralAttenuation',
  THIN_FILM_STACK: 'vitrumThinFilmStack',
  FRONT_LAYER: 'vitrumFrontLayer',
  BACK_LAYER: 'vitrumBackLayer',
  // RFE-10 dichroic addendum (PHY.1 — 2026-05-12). Pre-convolved
  // angle-indexed LUTs produced by the stainedGlass dichroic baker via
  // TMM × CIE 1931. The raster fragment shader reads them directly; PT
  // backends may use them as a fast-path alternative to evaluating the
  // TMM in-shader from the (already-stamped) THIN_FILM_STACK.
  DICHROIC_REFLECTANCE_LUT: 'vitrumDichroicReflectanceLUT',
  DICHROIC_TRANSMITTANCE_LUT: 'vitrumDichroicTransmittanceLUT',
} as const;
