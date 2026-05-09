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

import { useSelector } from 'react-redux';
import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { ShapedAreaLight, ShapedAreaLightImpl } from '@react-three/gpu-pathtracer';
import { selectActiveTimeOfDay } from '@/store/selectors';
import type { SunLight } from '../lightSourceTypes';
import { resolveColorHex } from '../lightSourceTypes';
import { getSunIntensity } from '../../lightingIntensityTable';
import { worldSunPosition, skyParamsFor, SUN_LIGHT_DISTANCE } from '../../skyParams';

interface Props {
  src: SunLight;
}

/**
 * PT_SUN_DISTANCE: how far to push the area-light sun. Far enough that
 * rays from any room point are approximately parallel (room dims are
 * order ~200 inches; at 10000 inches the angular spread across the
 * room is ~1.1°, well under the sun-disc angular size).
 */
const PT_SUN_DISTANCE = 10000;

/** Sun's angular radius as seen from Earth. ~0.25° = 0.00436 rad. */
const SUN_ANGULAR_RADIUS = 0.00436;

/**
 * Disc diameter at PT_SUN_DISTANCE that subtends 2 × SUN_ANGULAR_RADIUS
 * (the full sun-disc angular size). With distance=10000, this yields
 * a disc ~87 inches in diameter — large in absolute terms but small
 * angular size from the room.
 */
const PT_SUN_DISC_DIAMETER = 2 * PT_SUN_DISTANCE * Math.tan(SUN_ANGULAR_RADIUS);

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
  // Sweep finding (correctness 2026-05-08 Bug 18): active selector picks
  // viewport vs roomDoc timeOfDay based on the active document.
  const timeOfDay = useSelector(selectActiveTimeOfDay);
  const ref = useRef<ShapedAreaLightImpl | null>(null);

  // Sun direction from skyParams; distance scaled out to PT_SUN_DISTANCE.
  // skyParams already carries the sun direction in `sunPosition` (post
  // Y×0.4 scaling — keeps the backlit-panel character at noon).
  const baseDir = worldSunPosition(skyParamsFor(timeOfDay));
  const dirLen = Math.hypot(baseDir[0], baseDir[1], baseDir[2]) || 1;
  const ux = baseDir[0] / dirLen;
  const uy = baseDir[1] / dirLen;
  const uz = baseDir[2] / dirLen;
  const px = ux * PT_SUN_DISTANCE;
  const py = uy * PT_SUN_DISTANCE;
  const pz = uz * PT_SUN_DISTANCE;

  const colorHex = resolveColorHex(src.color);

  // Sun intensity multiplier from props × time-of-day bucket.
  // Bucket 0 (overcast/twilight) gives Math.PI × 0.001-ish (very dim);
  // we still want SOME PT sun in those cases since the env IBL also
  // scales down. Final intensity = PT_SUN_AREA_INTENSITY × bucket
  // factor × user-multiplier, keeping the existing diurnal contract.
  const baseIntensity = src.followsTimeOfDay ? getSunIntensity(timeOfDay) : Math.PI;
  // Convert directional bucket to area-light scalar by dividing by
  // Math.PI (the noon baseline) so noon=1.0.
  const diurnalFactor = baseIntensity / Math.PI;
  const intensity = PT_SUN_AREA_INTENSITY * diurnalFactor * src.intensity;

  // Aim the disc toward the room origin so its emissive face points at
  // the scene. lookAt mutates matrixWorld; do it whenever position changes.
  useEffect(() => {
    const light = ref.current;
    if (!light) return;
    light.lookAt(0, 0, 0);
    light.updateMatrixWorld();
  }, [px, py, pz]);

  return (
    <ShapedAreaLight
      ref={ref as unknown as React.Ref<typeof ShapedAreaLightImpl>}
      color={colorHex}
      intensity={intensity}
      width={PT_SUN_DISC_DIAMETER}
      height={PT_SUN_DISC_DIAMETER}
      isCircular
      position={[px, py, pz]}
    />
  );
}

// Re-export the constants so PT_IBL_INTENSITY recalibration can
// reference them (Phase 1.3).
export { PT_SUN_DISTANCE, PT_SUN_DISC_DIAMETER, PT_SUN_AREA_INTENSITY, SUN_ANGULAR_RADIUS };

// Reserved for future use. Currently SUN_LIGHT_DISTANCE is consumed
// only by SunRenderer (raster directional). PT path uses PT_SUN_DISTANCE.
void SUN_LIGHT_DISTANCE;
void THREE;
