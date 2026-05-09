/**
 * Physically-grounded lighting reference table for the rendering pipeline.
 * Three.js r155+ uses physically-correct light units by default.
 *
 *   - PointLight / SpotLight: candela (cd) = lumens / (4π)
 *   - DirectionalLight: dimensionless multiplier; pre-r155 × π for parity
 *   - RectAreaLight: cd/m² = lumens / (area_m² × π)
 */

/** Color temperatures (Kelvin) → sRGB hex (computed via Tanner Helland's
 *  approximation of the Planckian locus). Use as `color={0x...}` directly
 *  when initialising a Three.js light; r3f will gamma-linearise via
 *  `outputColorSpace`. */
export const COLOR_TEMP_HEX = {
  candle:        0xFF8100, // 1850 K — wax flame, tea light
  incandescent:  0xFFA757, // 2700 K — household 60W A19
  halogen:       0xFFB16E, // 3000 K — halogen / warm LED
  ledWarm:       0xFFB16E, // 3000 K — warm white LED bulb
  sunset:        0xFFB16E, // 3000–4000 K — golden-hour sun
  ledNeutral:    0xFFCEA6, // 4000 K — fluorescent / neutral LED
  ledCool:       0xFFDABB, // 4500 K — neutral-cool LED
  moonlight:     0xFFD3AD, // 4100 K — reflected sunlight
  ledDaylight:   0xFFE4CE, // 5000 K — daylight LED
  noonSun:       0xFFF0E8, // 5500–5800 K — direct noon sun
  overcastSky:   0xF6F7FF, // 6900 K — overcast daytime sky
  blueSkyZenith: 0xE6EBFF, // 7500–8000 K — open zenith
  twilight:      0xCADAFF, // 10000 K — deep twilight blue
} as const;

/** Three.js DirectionalLight intensity values for the sun by time-of-day.
 *  These are "post-r155 physical-lights default" values — i.e. the units
 *  match pre-r155 × π = roughly what the renderer needs to match a real
 *  sun on a tone-mapped scene. */
export const SUN_INTENSITY = {
  noon:           Math.PI,        // pre-r155 1.0 × π — clear-sky noon
  afternoon:      Math.PI * 0.7,  // late-day clear sun
  sunset:         Math.PI * 0.3,  // golden-hour sun
  overcast:       0.0,            // sun hidden — IBL-only illumination
  twilight:       Math.PI * 0.02, // deep twilight directional
  moonlight:      Math.PI * 0.001,// full moon
} as const;

/** Map a normalized time-of-day t ∈ [0,1] to DirectionalLight intensity for
 *  the exterior sun. Discrete buckets are used (rather than a smooth ramp)
 *  so PT sample-budget characteristics stay roughly constant within each
 *  phase; skyParams.ts uses a continuous sin/cos arc, so the directional
 *  light's brightness will appear to step relative to the sun-disc visual,
 *  but only at the bucket boundaries listed below.
 *
 *    t < 0.05 or t > 0.95  → SUN_INTENSITY.twilight
 *    t < 0.15 or t > 0.85  → SUN_INTENSITY.sunset
 *    t < 0.35 or t > 0.65  → SUN_INTENSITY.afternoon
 *    else (midday)         → SUN_INTENSITY.noon
 */
export function getSunIntensity(timeOfDay: number): number {
  const t = Math.max(0, Math.min(1, timeOfDay));
  if (t < 0.05 || t > 0.95) return SUN_INTENSITY.twilight;
  if (t < 0.15 || t > 0.85) return SUN_INTENSITY.sunset;
  if (t < 0.35 || t > 0.65) return SUN_INTENSITY.afternoon;
  return SUN_INTENSITY.noon;
}

/** Convert a real-world light source's lumen rating to a three.js
 *  PointLight `intensity` value (candela). */
export function pointIntensityFromLumens(lumens: number): number {
  return lumens / (4 * Math.PI);
}

/** Convert a real-world light source's lumen rating + emitter area to a
 *  three.js RectAreaLight `intensity` value (cd/m²). */
export function rectAreaIntensityFromLumens(lumens: number, areaSqMeters: number): number {
  return lumens / (areaSqMeters * Math.PI);
}

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
