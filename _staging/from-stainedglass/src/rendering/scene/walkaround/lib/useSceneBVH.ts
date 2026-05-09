/**
 * useSceneBVH — debounced BVH rebuild hook.
 *
 * Tier 2 shared GI primitive — generalised over `SceneBVHCommonOpts` so
 * any GI engine (DDGI / RC / ReSTIR) can configure its own
 * `positionStride` / `proxyMeshNames` / `filter`.
 *
 * Rebuild policy:
 *  - Geometry topology changes (scene mount swap, room swap) → full rebuild.
 *  - Material parameter edits (glass color, IOR) → re-trigger via `trigger`.
 *  - Caller debounces via `DEBOUNCE_MS` (default 100) to avoid thrashing
 *    on rapid edits.
 *
 * Cost: ~50 ms for the canonical post-roommesh honeycomb scene
 * (~30K triangles). Synchronous on the main thread; worker-based rebuild
 * is a planned follow-up.
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  buildSceneBVH,
  type SceneBVHCommonOpts,
  type SceneBVHCommonResult,
} from './bvhCommon';

const DEBOUNCE_MS = 100;

/**
 * Returns a fresh `SceneBVHCommonResult` whenever the scene's topology is
 * considered "changed". For the first render the hook returns `null`; once
 * the debounce timer fires and the build completes, the state updates.
 *
 * @param scene   - The three.js scene object (typically from useThree()).
 * @param opts    - Build options (filter / positionStride / proxyMeshNames /
 *                  skyHideRadiusThreshold).
 * @param trigger - Optional additional dependency to force a rebuild
 *                  (e.g. a Redux selector that changes on room swap).
 */
export function useSceneBVH(
  scene: THREE.Scene,
  opts: SceneBVHCommonOpts = {},
  trigger?: unknown,
): SceneBVHCommonResult | null {
  const [bvh, setBvh] = useState<SceneBVHCommonResult | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const buildCountRef = useRef(0);

  // Capture opts in a ref so callers passing inline objects don't re-trigger
  // every render. The build picks up the latest opts when the debounce
  // timer fires. Updating the ref inside the effect (not during render)
  // keeps the React refs lint rule happy.
  const optsRef = useRef<SceneBVHCommonOpts>(opts);

  useEffect(() => {
    optsRef.current = opts;
    if (timerRef.current !== null) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      const myCount = ++buildCountRef.current;
      const built = buildSceneBVH(scene, optsRef.current);
      // Abort if another rebuild was queued while we were building.
      if (myCount === buildCountRef.current) {
        setBvh(built);
      }
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [scene, trigger, opts]);

  return bvh;
}
