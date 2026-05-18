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
  candle: 0xff8100, // 1850 K — wax flame, tea light
  incandescent: 0xffa757, // 2700 K — household 60W A19
  halogen: 0xffb16e, // 3000 K — halogen / warm LED
  ledWarm: 0xffb16e, // 3000 K — warm white LED bulb
  sunset: 0xffb16e, // 3000–4000 K — golden-hour sun
  ledNeutral: 0xffcea6, // 4000 K — fluorescent / neutral LED
  ledCool: 0xffdabb, // 4500 K — neutral-cool LED
  moonlight: 0xffd3ad, // 4100 K — reflected sunlight
  ledDaylight: 0xffe4ce, // 5000 K — daylight LED
  noonSun: 0xfff0e8, // 5500–5800 K — direct noon sun
  overcastSky: 0xf6f7ff, // 6900 K — overcast daytime sky
  blueSkyZenith: 0xe6ebff, // 7500–8000 K — open zenith
  twilight: 0xcadaff, // 10000 K — deep twilight blue
} as const;

/** Three.js DirectionalLight intensity values for the sun by time-of-day.
 *  Post-r155 physical-lights default values — units match pre-r155 × π. */
export const SUN_INTENSITY = {
  noon: Math.PI, // pre-r155 1.0 × π — clear-sky noon
  afternoon: Math.PI * 0.7, // late-day clear sun
  sunset: Math.PI * 0.3, // golden-hour sun
  overcast: 0.0, // sun hidden — IBL-only illumination
  twilight: Math.PI * 0.02, // deep twilight directional
  moonlight: Math.PI * 0.001, // full moon
} as const;

/** Map a normalized time-of-day t ∈ [0,1] to DirectionalLight intensity for
 *  the exterior sun. Discrete buckets keep PT sample-budget characteristics
 *  roughly constant within each phase; skyParams.ts uses a continuous arc so
 *  the directional light's brightness will appear to step at bucket boundaries.
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
