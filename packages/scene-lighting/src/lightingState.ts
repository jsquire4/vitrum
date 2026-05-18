/**
 * Shared lighting source-of-truth for all render modes (raster, walkaround, PT).
 * Reads time-of-day + skyParams + a per-mode intensity multiplier; produces a
 * small struct that each renderer consumes identically.
 *
 * Every mode derives sun direction, sun intensity, and sky-dome tint from this
 * single function. Per-mode multipliers stay explicit so the (still real)
 * physics-model gap between modes is visible, not papered over.
 */

import * as THREE from 'three';
import type { SkyParams } from './skyParams.js';
import { getSunIntensity, SUN_INTENSITY } from './lightingIntensityTable.js';

export interface LightingState {
  /** Unit vector pointing FROM origin TOWARD sun. Both PT (via three.js
   *  DirectionalLight.position-derived direction) and walkaround (via
   *  WGSL UBO.sunDirection) use this same orientation. */
  sunDirection: THREE.Vector3;
  /** Sun radiance multiplier, dimensionless. PT's three.js DirectionalLight.intensity
   *  is set to this; walkaround's WGSL UBO reads it as `sunIntensity`. */
  sunIntensity: number;
  /** Approximate diffuse-sky-dome RGB tint derived from turbidity (rises near
   *  horizons → warmer/redder sky). */
  skyTint: [number, number, number];
  /** Sky-dome irradiance scalar, dimensionless. ~0.5×sunIntensity at noon
   *  (clear-sky hemisphere integration), much lower at night. */
  skyIrradiance: number;
}

export interface LightingStateInputs {
  /** Live time-of-day in [0,1]. 0 = dawn, 0.5 = noon, 1 = dusk. */
  timeOfDay: number;
  /** Active skyParams (computed from timeOfDay or nightTimeOfDay externally). */
  skyParams: SkyParams;
  /** True when the current backdrop is the night sky — applies a moonlight
   *  intensity floor independent of timeOfDay's day-bucket mapping. */
  isNight: boolean;
  /** Per-light user-tunable multiplier. Defaults to 1.0 for parity with PT's
   *  `getSunIntensity(t) × 1.0`. */
  intensityMultiplier?: number;
}

/** Pure compute. Both PT (three.js scene) and walkaround (WebGPU UBO) read
 *  this so visual divergence isn't caused by hardcoded magic numbers in either
 *  path. */
export function computeLightingState(opts: LightingStateInputs): LightingState {
  const { timeOfDay, skyParams, isNight, intensityMultiplier = 1.0 } = opts;

  // ── Sun direction ──────────────────────────────────────────────────
  // skyParams.sunPosition is non-unit (~1.12–1.41 across the arc).
  // Normalize so WGSL dot-products against surface normals are bounded in [-1, 1].
  const [sx, sy, sz] = skyParams.sunPosition;
  const sunDirection = new THREE.Vector3(sx, sy, sz).normalize();

  // ── Sun intensity ──────────────────────────────────────────────────
  // getSunIntensity(t) × intensityMultiplier — same value PT/raster pass to
  // three.js DirectionalLight.intensity.
  const baseIntensity = isNight
    ? SUN_INTENSITY.moonlight
    : getSunIntensity(timeOfDay);
  const sunIntensity = baseIntensity * intensityMultiplier;

  // ── Sky tint ──────────────────────────────────────────────────────
  // Approximate clear-sky-dome hemisphere color from turbidity.
  // turbidity 2 (noon) → cool blue (0.55, 0.75, 1.00).
  // turbidity 8 (horizons) → warm red (1.00, 0.80, 0.35).
  let skyTint: [number, number, number];
  if (isNight) {
    skyTint = [0.05, 0.08, 0.15];
  } else {
    const t = Math.max(0, Math.min(1, (skyParams.turbidity - 2) / 6));
    skyTint = [
      0.55 + 0.45 * t,
      0.75 + 0.05 * t,
      1.00 - 0.65 * t,
    ];
  }

  // ── Sky irradiance ────────────────────────────────────────────────
  // Rough integrated-hemisphere brightness (~0.5×sun at noon via Rayleigh
  // scattering). Night: tiny constant moonlit dome irradiance.
  const skyIrradiance = isNight ? 0.02 : 0.5 * sunIntensity;

  return { sunDirection, sunIntensity, skyTint, skyIrradiance };
}
