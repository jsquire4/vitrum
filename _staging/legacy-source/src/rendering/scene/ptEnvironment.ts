import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { useThree, useFrame } from '@react-three/fiber';
import { usePathtracer } from '@react-three/gpu-pathtracer';
import { bakeSkyEquirect } from './ptIblBaker';
import { PT_IBL_INTENSITY, PT_BACKGROUND_INTENSITY } from './lightingIntensityTable';
import { outdoorHdriForPreset } from './outdoorScenePresets';
import type { OutdoorScenePreset } from '@/store/viewportSlice';
import type { SkyParams } from './skyParams';
import type { BackdropMode } from '@/store/uiSlice';
import type { Space } from '@/types/scene';

/**
 * Drive `scene.environment`, `scene.background`, and the pathtracer's
 * IBL state from the active backdrop mode. Mounted inside the
 * `<Pathtracer>` subtree (via PTStage) so it has access to the
 * pathtracer instance via usePathtracer().
 *
 * Per-backdrop wiring:
 *
 *   sky    → bakeSkyEquirect(skyParams)       intensity 0.6   bg = same texture
 *   night  → bakeSkyEquirect(nightSkyParams)  intensity 0.15  bg = same texture
 *   studio → drei <Environment preset="studio" background />  intensity 0.4
 *   sunset → drei <Environment preset="sunset" background />  intensity 0.5
 *   none   → null env / null bg               intensity 0
 *
 * For studio/sunset, this hook does NOT assign scene.environment —
 * drei's <Environment> in PTStage owns that asynchronously. The
 * useFrame ref-watcher catches drei's load by detecting the
 * scene.environment ref change and calls pathtracer.updateEnvironment()
 * once. After that, the PT material is in sync with the IBL.
 *
 * Outdoor HDRI (space.kind === 'room'):
 *   When `space.kind === 'room'` AND `outdoorHdriForTimeOfDay(timeOfDay)`
 *   returns a defined URL, the .hdr file is loaded via THREE.RGBELoader
 *   and assigned as `scene.environment` + `scene.background` instead of
 *   the procedural skyEquirect path.
 */
export function usePTEnvironment(
  backdropMode: BackdropMode,
  skyParams: SkyParams,
  nightSkyParams: SkyParams,
  ptActive: boolean,
  space: Space = { kind: 'void' },
  timeOfDay = 0.5,
  outdoorScenePreset: OutdoorScenePreset = 'auto',
): void {
  const { gl: renderer, scene } = useThree();
  const { pathtracer } = usePathtracer();

  // Procedural sky/night → bake + assign.
  // None → clear.
  // Studio/sunset → leave to drei <Environment>.
  // Room + outdoor HDRI → load .hdr via RGBELoader when URL is defined.
  useEffect(() => {
    if (!ptActive || !pathtracer) return;

    // Outdoor HDRI path RE-ENABLED 2026-05-09 (Phase 1.2): the previous
    // disable rationale was the "two suns" problem — DirectionalLight
    // sun + HDRI baked sun = double-counting. Phase 1.1 replaced the
    // DirectionalLight with a ShapedAreaLight in PTStage, AND PTStage
    // now suppresses the area-light sun whenever an outdoor HDRI is
    // active (see ptSunSuppressedByHdri there). Single sun comes from
    // the HDRI's IBL sampling — the area-light path provides physical
    // penumbra only when the HDRI doesn't carry a sun (procedural
    // bakeSkyEquirect path).
    const outdoorUrl = space.kind === 'room'
      ? outdoorHdriForPreset(outdoorScenePreset, timeOfDay)
      : undefined;

    let cancelled = false;
    if (outdoorUrl !== undefined) {
      // Load the outdoor HDRI and assign when ready. RGBELoader is the
      // standard path for .hdr equirectangular maps in Three.js r152+.
      // The cancelled guard prevents stale writes when the effect re-runs
      // (e.g. timeOfDay/space change) before the async load completes.
      const loader = new RGBELoader();
      loader.load(outdoorUrl, (tex) => {
        if (cancelled) {
          tex.dispose();
          return;
        }
        tex.mapping = THREE.EquirectangularReflectionMapping;
        scene.environment = tex;
        scene.background = tex;
        // Phase 1.3 IBL recalibration: HDRI is the SOLE light source
        // when an outdoor preset is active (PTStage suppresses the
        // area-light sun via ptSunSuppressedByHdri). Polyhaven HDRs are
        // pre-calibrated for typical scene scales — env=1.0 means
        // "full HDR radiance, including the baked sun-disc," which is
        // what we want when the HDR carries the sun. The 0.04 floor
        // applied earlier was specifically for the procedural-sky-with-
        // explicit-area-light case; that doesn't apply here.
        scene.environmentIntensity = 1.0;
        scene.backgroundIntensity = PT_BACKGROUND_INTENSITY['sky'];
        pathtracer.updateEnvironment();
      });
    } else if (backdropMode === 'sky' || backdropMode === 'night') {
      const params = backdropMode === 'sky' ? skyParams : nightSkyParams;
      const envTex = bakeSkyEquirect(renderer, params);
      // scene.environment is the three.js API for env IBL; useThree()
      // exposes scene as a stable reference and three.js provides no
      // setter alternative — direct assignment is the contract.
      // eslint-disable-next-line react-hooks/immutability
      scene.environment = envTex;
      scene.background = envTex;
    } else if (backdropMode === 'none') {
      scene.environment = null;
      scene.background = null;
    }
    // studio / sunset: drei's <Environment> sets these asynchronously.
    // The useFrame below picks up that assignment.

    if (outdoorUrl === undefined) {
      scene.environmentIntensity = PT_IBL_INTENSITY[backdropMode];
      scene.backgroundIntensity = PT_BACKGROUND_INTENSITY[backdropMode];
    }

    // Fire updateEnvironment immediately for the synchronous cases
    // (sky/night/none). For studio/sunset, scene.environment is still
    // null here (drei assigns asynchronously), so guard the call — the
    // useFrame watcher below handles the deferred case. For outdoor HDRI,
    // the loader callback fires it after async load completes.
    if (outdoorUrl === undefined && scene.environment) {
      pathtracer.updateEnvironment();
    }

    // Cleanup on unmount or before the next effect run: drop the
    // scene's reference to the baked envmap. The ptIblBaker cache owns
    // texture lifetime; we just stop holding the pointer here.
    return () => {
      cancelled = true;
      scene.environment = null;
      scene.background = null;
      scene.environmentIntensity = 1;
      scene.backgroundIntensity = 1;
    };
  }, [ptActive, pathtracer, scene, renderer, backdropMode, skyParams, nightSkyParams, space, timeOfDay, outdoorScenePreset]);

  // Catch drei <Environment>'s async HDR assignment (studio/sunset)
  // and any other downstream change to scene.environment that this
  // hook didn't initiate. Cheap: one ref equality check per frame.
  const lastEnvRef = useRef<THREE.Texture | null>(null);
  useFrame(() => {
    if (!ptActive || !pathtracer) return;
    if (scene.environment !== lastEnvRef.current) {
      lastEnvRef.current = scene.environment;
      if (scene.environment) pathtracer.updateEnvironment();
    }
  });
}
