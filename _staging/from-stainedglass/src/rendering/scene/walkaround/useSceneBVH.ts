/**
 * useSceneBVH — RC's debounced BVH rebuild hook.
 *
 * Wraps RC's `bvhCompute.buildSceneBVH` (StorageBufferAttribute-typed
 * adapter) with a debounced rebuild on scene topology change. For the
 * generalised lib version that returns SceneBVHCommonResult and accepts
 * full SceneBVHCommonOpts, see `lib/useSceneBVH.ts` — this file is
 * scheduled for consolidation (D1 in plan/path-tracer-library-readiness.md).
 *
 * Material-only edits rebuild the whole scene today; future work could
 * patch only the materials SSBO.
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { buildSceneBVH, type SceneBVH } from './bvhCompute';

const DEBOUNCE_MS = 100;

/**
 * Returns a fresh SceneBVH whenever the scene's topology is considered
 * "changed." For M0-M5 we rebuild on every scene mount — the debounce
 * prevents thrashing during rapid edits.
 *
 * @param scene  - The three.js scene object from useThree().
 * @param trigger - Optional additional dependency to force a rebuild
 *   (e.g. a Redux selector that changes on room swap).
 */
export function useSceneBVH(
  scene: THREE.Scene,
  trigger?: unknown,
): SceneBVH | null {
  const [bvh, setBvh] = useState<SceneBVH | null>(null);
  const timerRef      = useRef<ReturnType<typeof setTimeout> | null>(null);
  const buildCountRef = useRef(0);

  useEffect(() => {
    // Debounce: cancel any pending rebuild.
    if (timerRef.current !== null) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      const myCount = ++buildCountRef.current;
      // BVH build is synchronous but potentially slow (~50 ms). Main-thread
      // for now; a worker-based rebuild is a planned follow-up.
      const built = buildSceneBVH(scene);
      // Abort if another rebuild was queued while we were building.
      if (myCount === buildCountRef.current) {
        setBvh(built);
      }
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [scene, trigger]);

  return bvh;
}
