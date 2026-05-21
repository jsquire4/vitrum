import { getSunIntensity } from '@vitrum/pt-webgl';

// Legacy host-facing intensity tables preserved for staging compatibility.
// Canonical runtime logic now lives in package-level lighting state.
export const PT_IBL_INTENSITY = {
  sky: 0.6,
  night: 0.15,
  studio: 0.4,
  sunset: 0.5,
  none: 0,
} as const;

export const PT_BACKGROUND_INTENSITY = {
  sky: 1.0,
  night: 1.0,
  studio: 1.0,
  sunset: 1.0,
  none: 0,
} as const;

export { getSunIntensity };
