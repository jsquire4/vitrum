/**
 * useCascadeBuffers — cascade storage allocation hook.
 *
 * Allocates cascade CPU + GPU storage whenever the scene bounds change.
 * Bounds typically come from the SceneBVH (set by useSceneBVH after a
 * rebuild). The hook returns null until bounds is supplied.
 *
 * Caller contract: pass `bounds = undefined` while you're still
 * computing them (e.g. before BVH first-build); the hook returns null
 * and waits. Do NOT pass a placeholder / zero bounds — the cascade
 * allocator computes probe spacing from `bounds.getSize()` and a
 * degenerate box produces zero-probe cascades.
 *
 * Library-extraction note: this hook used to fall back to a hardcoded
 * stained-glass-room AABB (192×108×168 inches). That default was
 * removed by the C2 sweep pass — callers must now pass real bounds.
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { allocateCascades, disposeCascades, type CascadeBuffers } from './cascadePyramid';

export function useCascadeBuffers(bounds: THREE.Box3 | undefined): CascadeBuffers | null {
  const [cascadeBuffers, setCascadeBuffers] = useState<CascadeBuffers | null>(null);
  const prevBoundsRef = useRef<THREE.Box3 | null>(null);

  useEffect(() => {
    if (!bounds) return;
    const b = bounds;

    // Re-allocate only if bounds changed meaningfully (> 1 inch).
    if (prevBoundsRef.current) {
      const prev = prevBoundsRef.current;
      const sameLo = prev.min.distanceTo(b.min) < 1;
      const sameHi = prev.max.distanceTo(b.max) < 1;
      if (sameLo && sameHi && cascadeBuffers) return;
    }

    // Dispose old buffers.
    if (cascadeBuffers) disposeCascades(cascadeBuffers);

    const newBuffers = allocateCascades(b);
    prevBoundsRef.current = b.clone();
    setCascadeBuffers(newBuffers);

    return () => {
      disposeCascades(newBuffers);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds]);

  return cascadeBuffers;
}
