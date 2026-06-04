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
 *   - directional → `sun` DDGILight carrying the emitter's REAL direction,
 *     intensity, and colour. The `@vitrum/core` `direction` field points AT
 *     the light (toward-light), but the GPU packer's `sun` slot + the WGSL
 *     probe shader expect a TRAVEL direction (the shader does `lightDir =
 *     normalize(-light.direction)`), so the mapper NEGATES it. This is the
 *     authoritative scene-directional → DDGI-sun path: the DDGI sun direction
 *     is no longer the packer's hardcoded `(0,-1,0)`, and the colour is no
 *     longer the packer's hardcoded `(1,0.95,0.85)`.
 *
 *     Single-count: a `sun` DDGILight carries `intensity = emitter.intensity`
 *     directly. The host (`HybridEngineLifecycle` / `_syncDdgiLightsFromThreeRoot`)
 *     sets `ProbeUpdatePass.setSunIntensityMultiplier(1)` whenever a scene
 *     directional is present, so the packed sun intensity is exactly
 *     `emitter.intensity` (NOT `emitter.intensity · primaryLightIntensity` —
 *     the multiplier and the emitter intensity would otherwise double-apply).
 *     When NO scene directional is present the host keeps the legacy
 *     `setSunIntensityMultiplier(primaryLightIntensity)` config path (which is
 *     a no-op for the probe pass today since nothing injects a `sun` light in
 *     that case — the directional's render contribution flows through
 *     `shade.wgsl`'s `Lo_emit` instead). See `directionalSunMultiplier`.
 *   - rect-area  → fixture at `position`, area `4·|uAxis × vAxis|`.
 *   - disc-area  → fixture at `position`, area `π·r²`.
 *   - point      → fixture at `position`, scalar intensity (no area).
 *   - spot       → fixture at `position` WITH cone. `spotAxis` (toward-light
 *     unit vector), `spotCosInner`, and `spotCosOuter` are packed alongside the
 *     position. `evalPointLight` in the probe shader applies the cone falloff
 *     `smoothstep(outerCone, innerCone, dot(toLightDir, spotAxis))` when the
 *     axis length is non-trivial (axisLen² > 0.25), which correctly confines the
 *     spot GI contribution to the cone (KHR_lights_punctual convention).
 *     Point fixtures have a zero axis, so they stay omnidirectional.
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
 *  kind is not represented as an analytic DDGI light (mesh-area) or is
 *  degenerate. A `directional` emitter maps to a `sun` DDGILight (see header). */
export function coreEmitterToDDGILight(e: SceneEmitter): DDGILight | null {
  switch (e.kind) {
    case 'directional': {
      // Scene directional → DDGI sun. `e.direction` points AT the light
      // (toward-light); the packer + WGSL want a TRAVEL direction (the shader
      // negates it back), so negate here. Intensity carried directly (the
      // host sets sunIntensityMul=1 for the single-count — see header).
      const len = Math.hypot(e.direction[0], e.direction[1], e.direction[2]);
      if (len < 1e-12) return null; // degenerate (zero direction)
      const inv = 1 / len;
      return {
        kind: 'sun',
        id: String(e.id),
        on: true,
        intensity: e.intensity,
        direction: {
          x: -e.direction[0] * inv,
          y: -e.direction[1] * inv,
          z: -e.direction[2] * inv,
        },
        color: { r: e.color[0], g: e.color[1], b: e.color[2] },
      };
    }
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
    case 'point': {
      return {
        kind: 'fixture',
        id: String(e.id),
        on: true,
        intensity: e.intensity,
        position: { x: e.position[0], y: e.position[1], z: e.position[2] },
        color: { r: e.color[0], g: e.color[1], b: e.color[2] },
      };
    }
    case 'spot': {
      // Spot → fixture WITH its cone. The probe shader now confines the spot's
      // GI contribution to the cone (was a point-like omnidirectional flood — the
      // cone used to be dropped). `e.direction` is `normalize(position - target)`
      // (toward-light axis); the shader uses it un-negated. Inner = full-intensity
      // cone (angle·(1−penumbra)); outer = the full half-angle. glTF
      // KHR_lights_punctual spot falloff.
      const len = Math.hypot(e.direction[0], e.direction[1], e.direction[2]);
      const inv = len > 1e-12 ? 1 / len : 0;
      const penumbra = Math.min(Math.max(e.penumbra ?? 0, 0), 1);
      return {
        kind: 'fixture',
        id: String(e.id),
        on: true,
        intensity: e.intensity,
        position: { x: e.position[0], y: e.position[1], z: e.position[2] },
        color: { r: e.color[0], g: e.color[1], b: e.color[2] },
        spotAxis: { x: e.direction[0] * inv, y: e.direction[1] * inv, z: e.direction[2] * inv },
        spotCosOuter: Math.cos(e.angle),
        spotCosInner: Math.cos(e.angle * (1 - penumbra)),
      };
    }
    case 'mesh-area':
      // See module header: mesh-area is folded into the referenced mesh's
      // emissive material by vitrumSceneToThree and reaches DDGI as emissive
      // geometry probe rays hit, not as an analytic light.
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

/** True when the scene supplies at least one `directional` emitter — i.e. the
 *  DDGI sun is driven by a scene-emitter `sun` DDGILight (carrying its own
 *  intensity) rather than the host's config-only `primaryLightIntensity`. */
export function sceneHasDirectionalEmitter(scene: Scene): boolean {
  return scene.emitters.some((e) => e.kind === 'directional');
}

/**
 * Resolve the sun-intensity multiplier the host should pass to
 * `ProbeUpdatePass.setSunIntensityMultiplier` — the single-count knob.
 *
 *   - Scene directional present → `1`. The `sun` DDGILight already carries
 *     `intensity = emitter.intensity`; a multiplier > 1 would double-apply it.
 *   - No scene directional       → `primaryLightIntensity` (legacy config
 *     path). This is a no-op for the probe pass today (nothing injects a `sun`
 *     light absent a scene directional), but kept so a host that manually
 *     injects a directionless `sun` light still scales by the config intensity
 *     as before.
 *
 * Centralising this keeps the "pick one path" single-count decision in one
 * place rather than duplicated across the init coordinator and the
 * incremental `_syncDdgiLightsFromThreeRoot` path.
 */
export function directionalSunMultiplier(
  scene: Scene | null,
  primaryLightIntensity: number,
): number {
  if (scene != null && sceneHasDirectionalEmitter(scene)) return 1;
  return primaryLightIntensity;
}
