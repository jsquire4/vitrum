/**
 * Compute Preetham sky parameters from a normalized time-of-day value.
 *
 * `t` is in [0, 1]: 0 = dawn, 0.5 = noon, 1 = dusk.
 *
 * Why SkyParams lives in @vitrum/scene-lighting rather than being collapsed into
 * core's ProceduralSkyEnvironment: SkyParams.sunPosition is a RAW non-unit
 * Preetham position vector (~1.08–1.16 magnitude across the day arc — minimum
 * 1.077 at noon, maximum 1.164 at t≈0.14), while
 * ProceduralSkyEnvironment.sunDirection is a unit vector. The IBL baker and
 * lightingState both read the non-unit components directly (to normalize or to
 * pass to THREE.Sky's uniforms). Collapsing would require storing the magnitude
 * separately or reconstructing it — unnecessary complexity.
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
  // Y×0.4 caps the noon altitude at ~22° above horizon, putting the sun
  // mostly behind the panel along Z. Eliminates the "sunlight entering through
  // the top of the building" artifact reported 2026-05-08.
  const sunY = Math.max(0.05, y * 0.4);
  // Push the sun slightly behind the panel (-Z) so glass reads as backlit.
  const sunZ = -Math.cos(theta) * 0.5 - 0.5;

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

/** World-space distance to place the directional sun light. The DirectionalLight
 *  only uses position to derive a direction, but the position anchors the
 *  orthographic shadow-camera frustum — too far and the near plane clips the
 *  scene, too close and grazing-angle shadows pop. 12 inches matches the
 *  previous hardcoded EXTERIOR_SUN_POSITION magnitude (sqrt(5²+5²+10²) ≈ 12.25). */
export const SUN_LIGHT_DISTANCE = 12;

/** Convert a SkyParams sun direction into a world-space position suitable for
 *  `<directionalLight position>`. Each component of `params.sunPosition` is
 *  multiplied by SUN_LIGHT_DISTANCE. The result is NOT a unit vector scaled to
 *  that magnitude — skyParams's sun vector is itself non-unit (~1.08–1.16
 *  across the arc). The resulting magnitudes (~13.4 at horizons, ~12.9 at noon)
 *  sit comfortably inside the default shadow-camera frustum (near=0.5, far=50,
 *  ±15 in xy). */
export function worldSunPosition(params: SkyParams): [number, number, number] {
  const [x, y, z] = params.sunPosition;
  return [x * SUN_LIGHT_DISTANCE, y * SUN_LIGHT_DISTANCE, z * SUN_LIGHT_DISTANCE];
}
