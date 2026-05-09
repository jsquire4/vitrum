/**
 * Compute drei <Sky> parameters from a normalized time-of-day value.
 *
 * `t` is in [0, 1]: 0 = dawn, 0.5 = noon, 1 = dusk. We map this to
 *   - sunPosition: an arc from horizon-east through zenith back to horizon-west
 *   - turbidity:   higher near horizons (warmer/hazier dawn-dusk look)
 *   - rayleigh:    higher near horizons for redder scattering
 */
export interface SkyParams {
  sunPosition: [number, number, number];
  turbidity: number;
  rayleigh: number;
  mieCoefficient: number;
  mieDirectionalG: number;
}

export function skyParamsFor(timeOfDay: number): SkyParams {
  const t = Math.max(0, Math.min(1, timeOfDay));
  // Solar arc: theta swings from -PI/2 (east) through 0 (zenith) to +PI/2 (west)
  const theta = (t - 0.5) * Math.PI;
  const x = Math.sin(theta);
  const y = Math.cos(theta); // 1 at noon, 0 at horizons
  // Sun altitude scaled 0.4× (was 1.0×). User-flagged 2026-05-08 that
  // a Y=1.0 noon-zenith placed the sun 45° above the panel, so light
  // entered at a steep down-angle, predominantly lighting the floor
  // near the panel base while the WALLS adjacent to the panel got
  // blown out by very-close Lo_direct emitter samples — perceived
  // as 'sunlight entering through the top of the building.'
  // Y×0.4 caps the noon altitude at ~22° above horizon, putting
  // the sun mostly BEHIND the panel along Z. Light direction is
  // now ~(0, -0.37, +0.93) at noon — predominantly +Z (through the
  // panel, into the room) instead of -Y (down from above).
  const sunY = Math.max(0.05, y * 0.4);
  // Push the sun slightly behind the panel (-Z) so glass reads as backlit.
  const sunZ = -Math.cos(theta) * 0.5 - 0.5;

  // Turbidity & rayleigh ramp up near horizons for warmer/redder atmosphere.
  const horizonProx = 1 - y; // 0 at noon, ~1 at horizons
  const turbidity = 2 + horizonProx * 6; // 2..8
  const rayleigh = 1 + horizonProx * 2; // 1..3

  return {
    sunPosition: [x, sunY, sunZ],
    turbidity,
    rayleigh,
    mieCoefficient: 0.005,
    mieDirectionalG: 0.8,
  };
}

/** World-space distance to place the directional sun light. The
 *  DirectionalLight only uses position to derive a direction (rays
 *  parallel from the sun toward the origin), but the position also
 *  anchors the orthographic shadow-camera frustum — too far and the
 *  near plane clips the scene, too close and grazing-angle shadows
 *  pop out of the frustum. 12 inches matches the previous hardcoded
 *  EXTERIOR_SUN_POSITION magnitude (sqrt(5²+5²+10²) ≈ 12.25). */
export const SUN_LIGHT_DISTANCE = 12;

/** Convert a SkyParams sun direction into a world-space position
 *  suitable for `<directionalLight position>`. Each component of
 *  `params.sunPosition` is multiplied by SUN_LIGHT_DISTANCE; the
 *  result is NOT a unit vector scaled to that magnitude (skyParams's
 *  sun vector is itself non-unit, ~1.12–1.41 across the time-of-day
 *  arc). For the existing shadow-camera frustum (near=0.5, far=50,
 *  ±15 in xy) the resulting magnitudes (~13.4 at horizons,
 *  ~17 at noon) sit comfortably inside the frustum. */
export function worldSunPosition(params: SkyParams): [number, number, number] {
  const [x, y, z] = params.sunPosition;
  return [x * SUN_LIGHT_DISTANCE, y * SUN_LIGHT_DISTANCE, z * SUN_LIGHT_DISTANCE];
}
