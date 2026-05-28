/**
 * coreEmittersToDDGILights — radiometrically-faithful `@vitrum/core`
 * `SceneEmitter` → `DDGILight` projection for the DDGI probe-update pass.
 *
 * Background — the lossy round-trip this replaces (Theme T16):
 * -----------------------------------------------------------
 * Before this mapper, `HybridEngineLifecycle` fed DDGI by walking the
 * THREE root produced by `vitrumSceneToThree` and RE-DERIVING each light's
 * intensity from the THREE light object
 * (`collectDDGILightsFromThreeRoot`). That round-trip was lossy on two
 * radiometric axes for rect-area emitters:
 *
 *   1. Chroma dropped. `collectDDGILightsFromRectAreaLights` built a
 *      `DDGILight` with NO `color`, so the GPU packer defaulted the
 *      fixture colour to white (1,1,1). A red rect emitter bled white
 *      light into the probe atlas.
 *   2. Wrong area metric. It used `light.width * light.height`
 *      (= `4·|uAxis|·|vAxis|`), which only equals the true emissive area
 *      `4·|uAxis × vAxis|` when the half-axes are orthogonal. Sheared /
 *      non-orthogonal rect emitters got the wrong flux. The old code's own
 *      comment admitted "factor-of-π errors ... negligible against the
 *      multiple-of-10 dynamic range" — i.e. it knowingly approximated.
 *
 * It also discarded the emitter id (DDGILight carried no id), so host-side
 * code could not correlate a DDGI light back to its core emitter.
 *
 * This mapper consumes the core `SceneEmitter` union DIRECTLY — no THREE
 * objects, no `userData.cellPower` round-trip — and produces the same
 * fixture-light projection the DDGI probe pass already consumes
 * (`kind: 'fixture'` point-light approximations), but with:
 *
 *   - chroma preserved (`color = emitter.color`);
 *   - the true cross-product area `4·|uAxis × vAxis|` for rect emitters
 *     (`π·r²` for disc emitters, matching `vitrumSceneToThree`'s
 *     area-preserving disc→rect conversion);
 *   - the source emitter id preserved on `DDGILight.id`.
 *
 * Radiometric convention: the DDGI probe shader (`probeUpdateRays.wgsl`
 * `evalPointLight`) evaluates a fixture as `color · intensity / (dist² + 1)
 * · nDotL`. So `color` carries the chroma and `intensity` carries the
 * scalar radiant magnitude. For an area emitter the flux-equivalent point
 * intensity is `emitter.intensity · area` (NO π factor — matching the
 * codebase's existing area-emitter power convention in
 * `restir/emitterList.ts`, where emitter power is `luminance(color)·area`).
 *
 * Taxonomy mapping (core → DDGI):
 *   - directional → EXCLUDED. The current pipeline routes the primary
 *     directional light to DDGI via `ProbeUpdatePass.setSunIntensityMultiplier`
 *     + the BVH emitter buffers, NOT through a `sun` DDGILight, and the GPU
 *     packer's `sun` path hardcodes the sun direction `(0,-1,0)` and colour
 *     `(1,0.95,0.85)` (it cannot carry a per-emitter direction). Emitting a
 *     `sun` DDGILight here would (a) be mis-oriented and (b) double-count
 *     against the multiplier path. So directional emitters are omitted —
 *     preserving the prior behaviour exactly (the old THREE walk produced a
 *     THREE.DirectionalLight, which `collectDDGILightsFromThreeRoot` did not
 *     collect either).
 *   - rect-area  → fixture at `position`, area `4·|uAxis × vAxis|`.
 *   - disc-area  → fixture at `position`, area `π·r²`.
 *   - point      → fixture at `position`, scalar intensity (no area).
 *   - spot       → fixture at `position` (point-like approximation; the
 *     probe shader has no cone handling, so the cone is dropped — this
 *     matches the low-frequency indirect-only role DDGI plays).
 *   - mesh-area  → EXCLUDED. Folded into the referenced mesh's emissive
 *     material by `vitrumSceneToThree`; it reaches DDGI as emissive
 *     geometry probe rays hit, not as an analytic light.
 */

import type { Scene, SceneEmitter, Vec3 } from '@vitrum/core';
import type { DDGILight } from './ddgi/types.js';

/** True emissive area of a rect-area emitter from its two HALF-axis vectors:
 *  `4·|uAxis × vAxis|`. (`uAxis`/`vAxis` are half-width/half-height, so the
 *  full rectangle is `2·uAxis` by `2·vAxis`; its area is the magnitude of
 *  `(2u) × (2v) = 4·(u × v)`.) This is the same metric `vitrumSceneToThree`'s
 *  `buildRectAreaLight` uses for its `cellPower` helper — the lossy THREE
 *  walk instead used `width·height = 4·|u|·|v|`, which under-/over-states the
 *  area whenever the half-axes are not orthogonal. */
function rectAreaFromHalfAxes(uAxis: Vec3, vAxis: Vec3): number {
  // cross = uAxis × vAxis
  const cx = uAxis[1] * vAxis[2] - uAxis[2] * vAxis[1];
  const cy = uAxis[2] * vAxis[0] - uAxis[0] * vAxis[2];
  const cz = uAxis[0] * vAxis[1] - uAxis[1] * vAxis[0];
  const crossLen = Math.sqrt(cx * cx + cy * cy + cz * cz);
  return 4 * crossLen;
}

/** Project a single core emitter onto a DDGILight, or null if the emitter
 *  kind is not represented as an analytic DDGI light (directional, mesh-area)
 *  or is degenerate. */
export function coreEmitterToDDGILight(e: SceneEmitter): DDGILight | null {
  switch (e.kind) {
    case 'rect-area': {
      const area = rectAreaFromHalfAxes(e.uAxis, e.vAxis);
      if (area < 1e-12) return null; // degenerate (parallel/zero half-axes)
      return {
        kind: 'fixture',
        id: String(e.id),
        on: true,
        intensity: e.intensity * area,
        position: { x: e.position[0], y: e.position[1], z: e.position[2] },
        color: { r: e.color[0], g: e.color[1], b: e.color[2] },
      };
    }
    case 'disc-area': {
      // Area-preserving footprint: a disc of radius r has area π·r², matching
      // vitrumSceneToThree's disc→rect conversion (√π·r/2 half-spans → π·r²).
      const area = Math.PI * e.radius * e.radius;
      if (area < 1e-12) return null;
      return {
        kind: 'fixture',
        id: String(e.id),
        on: true,
        intensity: e.intensity * area,
        position: { x: e.position[0], y: e.position[1], z: e.position[2] },
        color: { r: e.color[0], g: e.color[1], b: e.color[2] },
      };
    }
    case 'point':
    case 'spot': {
      return {
        kind: 'fixture',
        id: String(e.id),
        on: true,
        intensity: e.intensity,
        position: { x: e.position[0], y: e.position[1], z: e.position[2] },
        color: { r: e.color[0], g: e.color[1], b: e.color[2] },
      };
    }
    case 'directional':
    case 'mesh-area':
      // See module header: directional is routed via setSunIntensityMultiplier
      // + the BVH emitter buffers (the packer's sun path can't carry a
      // per-emitter direction); mesh-area is folded into the mesh's emissive
      // material and reaches DDGI as emissive geometry, not an analytic light.
      return null;
    default: {
      // Exhaustiveness guard — a new emitter kind added to the core union
      // should fail loudly here rather than silently vanish from DDGI.
      const _exhaustive: never = e;
      void _exhaustive;
      return null;
    }
  }
}

/**
 * Map a core `Scene`'s emitter list directly to the DDGILight projection
 * consumed by the DDGI probe-update pass — preserving chroma, using the true
 * emissive area for area emitters, and carrying the source emitter id. This
 * is the authoritative, non-lossy replacement for the THREE-round-trip
 * `collectDDGILightsFromThreeRoot` whenever a core scene is available.
 */
export function coreEmittersToDDGILights(scene: Scene): DDGILight[] {
  const out: DDGILight[] = [];
  for (const e of scene.emitters) {
    const light = coreEmitterToDDGILight(e);
    if (light != null) out.push(light);
  }
  return out;
}
