/**
 * Canonical userData key strings for the Three.js ↔ vitrum scene-binding
 * layer. The forward (vitrum → three) and reverse (three → vitrum)
 * conversions live in separate files and must agree byte-for-byte on the
 * key names; defining them once here removes the chance of a silent typo
 * breaking the round trip.
 */
export const VITRUM_USER_DATA_KEYS = {
  DISPERSION_ABBE:    'vitrumDispersionAbbeNumber',
  SCATTERING_COEFF:   'vitrumScatteringCoefficient',
  SCATTERING_RGB:     'vitrumScatteringCoefficientRGB',
  SCATTERING_ANISO:   'vitrumScatteringAnisotropy',
  SPECTRAL_ATTEN:     'vitrumSpectralAttenuation',
  THIN_FILM_STACK:    'vitrumThinFilmStack',
  FRONT_LAYER:        'vitrumFrontLayer',
  BACK_LAYER:         'vitrumBackLayer',
} as const;
