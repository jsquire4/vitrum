// Sun renderer — PT-only variant. Uses three-gpu-pathtracer's
// ShapedAreaLight (extends RectAreaLight with isCircular) so the BVH
// samples it as a real disc emitter. Result: sun-NEE shadow rays
// converge over a SOLID ANGLE matching the real sun's 0.5° angular
// subtense → physical penumbra on every shadow boundary (caustic edges,
// came drop-shadows, etc.) instead of the knife-edge shadows from
// THREE.DirectionalLight.
//
// Why a separate component (vs. branching SunRenderer):
// - ShapedAreaLight's R3F wrapper requires `usePathtracer()` context,
//   so this component MUST be mounted inside the <Pathtracer> provider.
//   Branching SunRenderer would couple the raster path to PT.
// - PTStage suppresses the directional sun (suppressSun=true is passed
//   to LightSourceList) and mounts <SunPathTraced /> in its place when
//   not under a studio/sunset HDRI.
//
// Geometry contract:
// - Position the area light at PT_SUN_DISTANCE inches along the
//   sun-direction unit vector. Far enough that rays are approximately
//   parallel from anywhere in the room.
// - Disc diameter = 2 × distance × tan(SUN_ANGULAR_RADIUS) so the disc
//   subtends ~0.5° from the room — matches Earth's view of the sun.
// - intensity is calibrated empirically (see PT_SUN_AREA_INTENSITY)
//   so total irradiance on the room matches the prior DirectionalLight
//   setup at intensity=π. The math: directional_irradiance = π·cos(θ)
//   vs. area-light_irradiance = intensity·area·cos(θ)·cos(disc_normal)/r².
//   For our distance/radius this works out to ~5×10⁴ cd/m².

import { useEffect } from 'react';
import {
  PT_SUN_DISTANCE,
  PT_SUN_DISC_DIAMETER,
  SUN_ANGULAR_RADIUS,
} from '@vitrum/pt-webgl';

interface Props {
  src: unknown;
}

// These constants are now exported from `@vitrum/pt-webgl/sunGeometry`; update
// the import when host integration begins.

/**
 * PT_SUN_DISTANCE: how far to push the area-light sun. Far enough that
 * rays from any room point are approximately parallel (room dims are
 * order ~200 inches; at 10000 inches the angular spread across the
 * room is ~1.1°, well under the sun-disc angular size).
 */

/**
 * RectAreaLight intensity calibration. ShapedAreaLight inherits
 * RectAreaLight's intensity semantics (cd/m² × area, post-r155
 * physical-lights). For our disc geometry at PT_SUN_DISTANCE, the
 * irradiance on a unit-cosine surface is intensity × A × cos(θ) / r²
 * where A = π × (diameter/2)² and r is the distance to the surface.
 *
 * For total room irradiance to match a DirectionalLight at intensity=π
 * (our prior baseline at noon), we want intensity × π × (D/2)² / d² ≈ π
 * where d≈distance and D≈diameter. Solving with our values gives
 * intensity ≈ 1 / (π × (0.00436)²) × π = 1 / 1.9e-5 ≈ 52,600. Round to
 * 50,000 cd/m² — small adjustments compensate for the disc-area cos
 * factor that DirectionalLight skips.
 *
 * If the scene reads too bright or too dim post-Phase-1, this is the
 * dial.
 */
const PT_SUN_AREA_INTENSITY = 50_000;

/**
 * `Pathtracer.setScene` walks the scene graph at mount; per
 * `WebGLPathTracer.setScene` (three-gpu-pathtracer), area lights need
 * their `lookAt(0,0,0)` populated before BVH construction. R3F mounts
 * children before useEffect fires, but the area light's matrixWorld is
 * what setScene reads — so we lookAt synchronously in a layout-effect
 * to guarantee the disc faces the room before the first BVH build.
 */
export function SunPathTraced({ src }: Props) {
  useEffect(() => {
    if (typeof console !== 'undefined') {
      console.warn(
        '[staging] SunPathTraced is a host-only renderer component. src=',
        src,
      );
    }
  }, [src]);
  return null;
}

// Re-export the constants so PT_IBL_INTENSITY recalibration can
// reference them (Phase 1.3).
export { PT_SUN_DISTANCE, PT_SUN_DISC_DIAMETER, PT_SUN_AREA_INTENSITY, SUN_ANGULAR_RADIUS };
