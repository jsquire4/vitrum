/**
 * coreEmittersToDDGILights — `@vitrum/core` `SceneEmitter` → `DDGILight`
 * projection for the DDGI probe-update pass's ANALYTIC light list (suns +
 * point + spot fixtures).
 *
 * Background — the lossy host-adapter round-trip this replaces (Theme T16):
 * -----------------------------------------------------------------------
 * Before this mapper, `HybridEngineLifecycle` fed DDGI by walking a
 * host-renderer light graph and re-deriving each light's intensity from that
 * renderer object. That round-trip was lossy (chroma dropped; wrong area metric
 * for sheared rect emitters). This mapper consumes the core `SceneEmitter`
 * union DIRECTLY, preserving chroma and the source emitter id.
 *
 * Single-count invariant for area emitters (2026-06-10):
 * -------------------------------------------------------
 * rect-area and disc-area emitters are NOT mapped to fixture point-proxies
 * here. They reach DDGI probes via the H18 NEE path instead:
 * `collectRectAreaEmitterTrisFromCore` + `setEmitterTris` → `ddgiEmitterNEE`.
 * Adding a fixture for the same emitter here would double-count it, because
 * `probeUpdateRays.wgsl` sums `direct_analytic + direct_emitter` (line 587).
 * The NEE triangle path is physically correct (solid-angle–correct, accounts
 * for emitter orientation and occlusion); the point-proxy is an approximation
 * that is now superseded.
 * (rect/disc fixture-proxy removed: was double-counted with H18 NEE, 2026-06-10)
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
 *     directly. The host (`HybridEngineLifecycle`)
 *     sets `ProbeUpdatePass.setSunIntensityMultiplier(1)` whenever a scene
 *     directional is present, so the packed sun intensity is exactly
 *     `emitter.intensity` (NOT `emitter.intensity · primaryLightIntensity` —
 *     the multiplier and the emitter intensity would otherwise double-apply).
 *     When NO scene directional is present the host keeps the legacy
 *     `setSunIntensityMultiplier(primaryLightIntensity)` config path (which is
 *     a no-op for the probe pass today since nothing injects a `sun` light in
 *     that case — the directional's render contribution flows through
 *     `shade.wgsl`'s `Lo_emit` instead). See `directionalSunMultiplier`.
 *   - rect-area  → EXCLUDED from the fixture list. The H18 NEE path supplies
 *     the same emitter as two tessellated triangles via `setEmitterTris` /
 *     `ddgiEmitterNEE`. Adding a fixture-proxy here would double-count it:
 *     `probeUpdateRays.wgsl` sums `direct_analytic + direct_emitter` on line
 *     587, so a fixture AND an emitter-tri for the same rect-area source would
 *     contribute twice. The NEE triangle path is the physically-correct one
 *     (solid-angle–correct, accounts for emitter orientation). (2026-06-10)
 *   - disc-area  → EXCLUDED for the same reason. Its triangulated fan reaches
 *     probes via `ddgiEmitterNEE`; a point-proxy fixture here would double it.
 *   - point      → fixture at `position`, scalar intensity (no area).
 *   - spot       → fixture at `position` WITH cone. `spotAxis` (forward beam
 *     unit vector), `spotCosInner`, and `spotCosOuter` are packed alongside the
 *     position. `evalPointLight` in the probe shader applies the cone falloff
 *     `smoothstep(outerCone, innerCone, dot(-spotAxis, toLightDir))` when the
 *     axis length is non-trivial (axisLen² > 0.25), which correctly confines the
 *     spot GI contribution to the cone (KHR_lights_punctual convention).
 *     Point fixtures have a zero axis, so they stay omnidirectional.
 *   - mesh-area  → EXCLUDED from the analytic light list. Emissive mesh triangles
 *     reach DDGI probes via the H18 NEE path: `collectMeshAreaEmitterTrisFromCore`
 *     includes them in the emitter-tri buffer → `ddgiEmitterNEE` in the probe
 *     shader samples them directly. probeUpdateRays.wgsl does NOT read
 *     mat.emissive on BVH hits, so the prior claim that they "reach DDGI as
 *     emissive geometry probe rays hit" was incorrect. (2026-06-10)
 *
 * Radiometric convention for point/spot: the DDGI probe shader (`evalPointLight`)
 * evaluates a fixture as `color · intensity / (dist² + 1) · nDotL`. So `color`
 * carries the chroma and `intensity` carries the scalar radiant magnitude.
 */

import type { Scene, SceneEmitter } from '@vitrum/core';
import type { DDGILight } from './ddgi/types.js';

/** Convert the core/lighting toward-light vector into DDGI's sun travel vector. */
function primaryLightDirToDdgiSunDirection(
  dir: readonly [number, number, number],
): DDGILight['direction'] | null {
  const len = Math.hypot(dir[0], dir[1], dir[2]);
  if (len < 1e-12) return null;
  const inv = 1 / len;
  return { x: -dir[0] * inv, y: -dir[1] * inv, z: -dir[2] * inv };
}

/**
 * Re-orient all DDGI sun lights to the runtime primary-light direction.
 *
 * Scene/host lights still supply sun identity, color, and intensity; the
 * mutable `HybridEngine.updateLighting({ primaryLightDir })` field is the
 * direction source of truth so shade-side direct lighting and DDGI bounce stay
 * aligned during time-of-day scrubs.
 */
export function orientDdgiSunLights(
  lights: readonly DDGILight[],
  primaryLightDir: readonly [number, number, number],
): DDGILight[] {
  const direction = primaryLightDirToDdgiSunDirection(primaryLightDir);
  if (direction == null) return [...lights];
  return lights.map((light) => (
    light.kind === 'sun'
      ? { ...light, direction }
      : light
  ));
}

/**
 * Resolve the first authored scene directional into the normalized
 * toward-light vector consumed by shade/RC/BVH primary-light paths.
 *
 * DDGI keeps every scene sun's individual direction; this helper is only for
 * consumers whose realtime contract has one primary directional slot.
 */
export function scenePrimaryLightDirection(
  scene: Scene | null,
): [number, number, number] | null {
  const emitter = scene?.emitters.find((candidate) => candidate.kind === 'directional');
  if (emitter == null) return null;
  const len = Math.hypot(
    emitter.direction[0],
    emitter.direction[1],
    emitter.direction[2],
  );
  if (len < 1e-12) return null;
  const inv = 1 / len;
  return [
    emitter.direction[0] * inv,
    emitter.direction[1] * inv,
    emitter.direction[2] * inv,
  ];
}

/** Project a single core emitter onto a DDGILight, or null if the emitter
 *  kind is not represented as an analytic DDGI light (rect-area, disc-area,
 *  mesh-area) or is degenerate. A `directional` emitter maps to a `sun`
 *  DDGILight (see header). */
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
        ...(typeof e.angularDiameter === 'number' && Number.isFinite(e.angularDiameter)
          ? { angularRadius: Math.max(0, e.angularDiameter) * 0.5 }
          : {}),
        ...(e.castShadow !== undefined ? { castShadow: e.castShadow } : {}),
      };
    }
    case 'rect-area':
      // Excluded: rect-area emitters reach DDGI probes as NEE triangles via
      // `setEmitterTris` / `ddgiEmitterNEE` — a fixture here would double-count
      // the contribution (`direct_analytic + direct_emitter` in the probe shader).
      // (rect/disc fixture-proxy removed: was double-counted with H18 NEE, 2026-06-10)
      return null;
    case 'disc-area':
      // Same reason as rect-area: tessellated fan via ddgiEmitterNEE.
      // (rect/disc fixture-proxy removed: was double-counted with H18 NEE, 2026-06-10)
      return null;
    case 'point': {
      return {
        kind: 'fixture',
        id: String(e.id),
        on: true,
        intensity: e.intensity,
        position: { x: e.position[0], y: e.position[1], z: e.position[2] },
        color: { r: e.color[0], g: e.color[1], b: e.color[2] },
        distance: typeof e.distance === 'number' && e.distance > 0 ? e.distance : 0,
        decay: typeof e.decay === 'number' ? e.decay : 2,
        ...(e.castShadow !== undefined ? { castShadow: e.castShadow } : {}),
      };
    }
    case 'spot': {
      // Spot → fixture WITH its cone. The probe shader now confines the spot's
      // GI contribution to the cone (was a point-like omnidirectional flood — the
      // cone used to be dropped). `e.direction` is the forward beam/travel axis;
      // receiver-to-light directions are tested against `-axis`. Inner = full-intensity
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
        distance: typeof e.distance === 'number' && e.distance > 0 ? e.distance : 0,
        decay: typeof e.decay === 'number' ? e.decay : 2,
        ...(e.castShadow !== undefined ? { castShadow: e.castShadow } : {}),
      };
    }
    case 'mesh-area':
      // Excluded from the analytic light list. Emissive mesh triangles reach
      // DDGI probes via the H18 NEE path: `collectMeshAreaEmitterTrisFromCore`
      // (in bvhSceneHelpers.ts) includes them in the emitter-tri buffer →
      // `ddgiEmitterNEE` in the probe shader samples them directly.
      // probeUpdateRays.wgsl does NOT read mat.emissive on BVH hits.
      // (2026-06-10)
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
 * consumed by the DDGI probe-update pass — point and spot fixtures + sun.
 * rect-area, disc-area, and mesh-area emitters are excluded (they reach DDGI
 * through the H18 NEE triangle path instead).
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
function sceneHasDirectionalEmitter(scene: Scene): boolean {
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
 * incremental DDGI light-sync path.
 */
export function directionalSunMultiplier(
  scene: Scene | null,
  primaryLightIntensity: number,
): number {
  if (scene != null && sceneHasDirectionalEmitter(scene)) return 1;
  return primaryLightIntensity;
}
