/**
 * Sun area-light geometry constants for the PT-mode sun disc emitter.
 * Extracted from the host-only `sunPathTraced.tsx` component; the host
 * imports from here rather than redeclaring them locally.
 *
 * Geometry contract:
 *   - Place the area light at PT_SUN_DISTANCE along the sun-direction unit vector.
 *     Far enough that rays from any room point are approximately parallel.
 *   - Disc diameter = 2 × distance × tan(SUN_ANGULAR_RADIUS) → subtends ~0.5°
 *     from the room, matching Earth's view of the sun.
 *   - PT_SUN_AREA_INTENSITY calibrates total irradiance to match a DirectionalLight
 *     at intensity=π (the prior noon baseline).
 */

/** How far to push the area-light sun (inches). At this distance the angular
 *  spread across a ~200-inch room is ~1.1°, well under the sun-disc angular size. */
export const PT_SUN_DISTANCE = 10000;

/** Sun's angular radius as seen from Earth. ~0.25° = 0.00436 rad. */
export const SUN_ANGULAR_RADIUS = 0.00436;

/** Disc diameter at PT_SUN_DISTANCE that subtends 2 × SUN_ANGULAR_RADIUS (the
 *  full sun-disc angular size). With distance=10000 this is ~87 inches in
 *  diameter — small angular size from the room. */
export const PT_SUN_DISC_DIAMETER = 2 * PT_SUN_DISTANCE * Math.tan(SUN_ANGULAR_RADIUS);

/** ShapedAreaLight intensity (cd/m²). Calibrated so total room irradiance matches
 *  a DirectionalLight at intensity=π: intensity × π × (D/2)² / d² ≈ π, solving
 *  with our values gives ~52,600; rounded to 50,000. Adjust here if the scene
 *  reads too bright or too dim after area-light sun integration. */
export const PT_SUN_AREA_INTENSITY = 50_000;
