import { useEffect } from 'react';

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
export function usePTEnvironment(..._args: unknown[]): void {
  useEffect(() => {
    if (typeof console !== 'undefined') {
      console.warn(
        '[staging] usePTEnvironment is legacy host-only code. Use @vitrum/pt-webgl environment wiring in live hosts.',
      );
    }
  }, []);
}
