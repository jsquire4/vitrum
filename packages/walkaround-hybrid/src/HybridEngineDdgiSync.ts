/**
 * Shared DDGI-sync helper — the DDGI light-update sequence that both
 * `HybridEngine._syncDdgiLightsFromCoreScene` (incremental fast-update path)
 * and the `HybridEngineLifecycle` init-publish phase share.
 *
 * Extracted from HybridEngine.ts (R3 B-chain decomposition sweep, step 4).
 *
 * INTENTIONAL DIFFERENCES between the two call-sites (preserved, not merged):
 *
 * 1. `setLightsConditional` (default false):
 *    - Engine path (false): always calls `ddgi.setLights(merged)` — harmless
 *      when scene has no emitters (merged = ctorLights only).
 *    - Lifecycle path (true): only calls `ddgi.setLights` when the scene
 *      contributes at least one light, preserving whatever was set previously
 *      for a no-emitter scene. This keeps the lifecycle's pre-existing
 *      conditional guard intact.
 *
 * 2. After returning, the engine path additionally calls:
 *    - `applyDirectionalEnvironment(scene.environment ?? { kind: 'none' })` (B3)
 *    - `ddgi.invalidateProbeCache()`
 *    These are NOT called by this function — each caller appends them as needed.
 *
 * 3. The lifecycle path calls `pipeline?.updateAnalyticLights(scene)` BEFORE
 *    `publishPipeline`; the engine path calls it on `this._pipeline` after it
 *    exists. Both flow through this function's `pipeline?.updateAnalyticLights`.
 *
 * H41/H18/B3 comment annotations: preserved at each call-site.
 */

import type { EngineWarning, Scene } from '@vitrum/core';
import type { PrimitiveTlasBinding } from '@vitrum/shared-bvh';
import type { DDGI } from './ddgi/DDGI.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import type { DDGILight } from './ddgi/types.js';
import {
  coreEmittersToDDGILights,
  directionalSunMultiplier,
  orientDdgiSunLights,
} from './coreEmittersToDDGILights.js';
import { mergeDDGILightsDedupSun } from './HybridEngineLifecycle.js';
import {
  collectRectAreaEmitterTrisFromCore,
  collectMeshAreaEmitterTrisFromCore,
  packEmitterTrisForDDGI,
} from './restir/bvhSceneHelpers.js';

export interface SyncDdgiFromCoreSceneDeps {
  /** DDGI subsystem to update. */
  ddgi: DDGI;
  /**
   * Live pipeline (for H41 analytic lights upload). May be null before init
   * completes — `updateAnalyticLights` is a no-op when null.
   */
  pipeline: WalkaroundGPUPipeline | null;
  /** Constructor-time lights (opts.lights). */
  ctorLights: readonly DDGILight[];
  /** Engine config primary-light intensity (used by directionalSunMultiplier). */
  primaryLightIntensity: number;
  /**
   * Engine config primary-light direction (used by orientDdgiSunLights).
   * When provided, the merged lights are passed through `orientDdgiSunLights`.
   * When absent, scene directional emitters keep their own direction.
   */
  primaryLightDir?: readonly [number, number, number];
  /** TLAS primitive bindings from the active shared-BVH pack, when available. */
  tlasPrimitiveBindings?: readonly PrimitiveTlasBinding[];
  /** Structured warning sink for light-sync fallbacks. */
  onWarning?: (warning: EngineWarning) => void;
  /**
   * When true, only call `ddgi.setLights` if the scene contributes at least
   * one emitter (lifecycle init path). When false (default), always merge and
   * call setLights (engine fast-update path).
   */
  setLightsConditional?: boolean;
}

/**
 * Run the DDGI light-sync sequence for a resolved core scene:
 *   1. setSunIntensityMultiplier (single-count directional sun)
 *   2. setLights (merged ctorLights + scene lights, dedup sun)
 *   3. setEmitterTris (H18 Stage 2 — rect/disc/mesh area emitter NEE)
 *   4. updateAnalyticLights (H41 — point/spot shade NEE)
 *
 * `scene` must be non-null (callers handle their own null guards).
 * See the module docstring for the intentional differences between call-sites.
 */
export function syncDdgiFromCoreScene(
  deps: SyncDdgiFromCoreSceneDeps,
  scene: Scene,
): void {
  // Step 1 — H18/sun single-count: when the scene has a directional emitter,
  // `coreEmittersToDDGILights` emits a `sun` DDGILight with the real
  // intensity, so the multiplier must be 1. Absent a scene directional, keep
  // the legacy config multiplier. See directionalSunMultiplier.
  deps.ddgi.setSunIntensityMultiplier(
    directionalSunMultiplier(scene, deps.primaryLightIntensity),
  );

  // Step 2 — collect scene lights + merge with ctorLights.
  const ddgiSceneLights = coreEmittersToDDGILights(scene);
  if (!deps.setLightsConditional || ddgiSceneLights.length > 0) {
    // De-dup the sun: if the scene contributes a directional→sun AND the host
    // passed an `opts.lights` sun, drop the host sun (scene wins) so DDGI
    // doesn't double-count the sun. See `mergeDDGILightsDedupSun`.
    const mergedLights = mergeDDGILightsDedupSun(
      deps.ctorLights,
      ddgiSceneLights,
      deps.onWarning != null ? { onWarning: deps.onWarning } : {},
    );
    deps.ddgi.setLights(
      deps.primaryLightDir != null
        ? orientDdgiSunLights(mergedLights, deps.primaryLightDir)
        : mergedLights,
    );
  }

  // Step 3 — H18 Stage 2: supply area-emitter NEE triangles to the probe-ray
  // kernel. rect-area/disc-area: same geometry as ReSTIR-DI. mesh-area:
  // DDGI-only expansion — the geometry stream carries them for ReSTIR; DDGI's
  // probe-NEE path has no geometry stream, so they must be added explicitly.
  // Count=0 (sun+point-only scenes) → no-op guard.
  // (mesh-area tris added to probe NEE, 2026-06-10)
  const emitterTris = [
    ...collectRectAreaEmitterTrisFromCore(scene),
    ...collectMeshAreaEmitterTrisFromCore(scene, {
      ...(deps.tlasPrimitiveBindings != null ? { tlasPrimitiveBindings: deps.tlasPrimitiveBindings } : {}),
      ...(deps.onWarning != null
        ? { onWarning: deps.onWarning, warningPhase: 'lifecycle', warningMethod: 'syncDdgiFromCoreScene' }
        : {}),
    }),
  ];
  const packed = packEmitterTrisForDDGI(emitterTris);
  deps.ddgi.setEmitterTris(packed.data, packed.count);

  // Step 4 — H41: re-upload the analytic point/spot lights buffer for shade
  // NEE. No-op when pipeline is null (init in-flight — lifecycle path calls
  // with the local pipeline before publishPipeline; engine fast-update path
  // calls on this._pipeline after init).
  deps.pipeline?.updateAnalyticLights(scene);
}
