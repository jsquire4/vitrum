// Library-grade physics moved to @vitrum/pt-webgl/src/lightingIntensityTable.ts; only host-domain BackdropMode-typed constants remain here.

import type { BackdropMode } from '@/store/uiSlice';

/** PT-mode HDR dampening per backdrop — controls scene.environmentIntensity.
 *  Low values serve two purposes:
 *   1. three-gpu-pathtracer's specular lobe gets ~32.5% sampling weight on
 *      glass at normal incidence (per-Disney getLobeWeights flooring Fresnel
 *      sampling at max(0.25, F)) — bright env intensity overwhelms the
 *      Beer-Lambert tinted transmission and cells read as pure-white.
 *   2. NEE picks light-or-env 50/50 (lightsDenom = lights.count + 1) — env
 *      intensity competes with caustic intensity in the floor's noise
 *      budget. Dropping env makes the sun-NEE caustic dominate.
 *
 *  Values dropped a second time (0.10 → 0.04 sky) when the user reported
 *  caustics still invisible despite the cells being chromatic — the floor
 *  ambient was hiding the caustic-to-ambient contrast.
 *
 *  `satisfies Record<BackdropMode, number>` makes adding a new backdrop
 *  enum value a typecheck error here, not an undefined-at-runtime
 *  surprise that would assign NaN to scene.environmentIntensity. */
export const PT_IBL_INTENSITY = {
  sky:    0.04,
  night:  0.02,
  studio: 0.05,
  sunset: 0.05,
  none:   0.0,
} as const satisfies Record<BackdropMode, number>;

/** Background intensity (the visible HDR backdrop sphere). Usually 1.0
 *  unless we want to dim the HDR for visual effect. */
export const PT_BACKGROUND_INTENSITY = {
  sky:    1.0,
  night:  1.0,
  studio: 1.0,
  sunset: 1.0,
  none:   1.0,
} as const satisfies Record<BackdropMode, number>;
