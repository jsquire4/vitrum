/**
 * Canonical userData key strings for stained-glass-specific Three.js ↔ vitrum
 * scene binding extensions.
 */
export const VITRUM_USER_DATA_KEYS = {
  DISPERSION_ABBE:            'vitrumDispersionAbbeNumber',
  SCATTERING_COEFF:           'vitrumScatteringCoefficient',
  SCATTERING_RGB:             'vitrumScatteringCoefficientRGB',
  SCATTERING_ANISO:           'vitrumScatteringAnisotropy',
  SPECTRAL_ATTEN:             'vitrumSpectralAttenuation',
  THIN_FILM_STACK:            'vitrumThinFilmStack',
  FRONT_LAYER:                'vitrumFrontLayer',
  BACK_LAYER:                 'vitrumBackLayer',
  DICHROIC_REFLECTANCE_LUT:   'vitrumDichroicReflectanceLUT',
  DICHROIC_TRANSMITTANCE_LUT: 'vitrumDichroicTransmittanceLUT',
} as const;
