/**
 * CascadeBuffers — cascade storage allocation manager.
 *
 * Extracted from `_staging/legacy-source/src/rendering/scene/walkaround/useCascadeBuffers.ts`.
 * De-React-ified: the React hook `useCascadeBuffers` (useState, useEffect, useRef)
 * is replaced with a class exposing `initialize()`, `getBuffers()`, `dispose()`.
 *
 * Allocates cascade CPU + GPU storage whenever the scene bounds change.
 * Bounds typically come from the SceneBVH (set by the BVH builder after a
 * rebuild). Returns null until `initialize(bounds)` is called.
 *
 * Caller contract: pass real bounds — do NOT pass a placeholder / zero bounds.
 * The cascade allocator computes probe spacing from `bounds.getSize()` and a
 * degenerate box produces zero-probe cascades.
 */

// W8 Phase 1A (2026-05-18) — THREE.Box3 swapped for the plain
// `CascadeAABB` so this module is THREE-free for HybridEngine integration.
import {
  allocateCascades,
  disposeCascades,
  type CascadeAABB,
  type CascadeBuffers,
} from './cascadePyramid.js';

/** Squared distance between two `[x,y,z]` tuples (used to detect bounds drift
 *  without pulling in `THREE.Vector3.distanceTo`). */
function distance3Sq(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

export class CascadeBufferManager {
  private _buffers: CascadeBuffers | null = null;
  private _prevBounds: { min: [number, number, number]; max: [number, number, number] } | null = null;

  /**
   * Allocate (or re-allocate) cascade storage for the given scene bounds.
   * Re-allocates only if bounds changed meaningfully (> 1 world-unit).
   *
   * Returns `true` if new buffers were allocated, `false` if bounds unchanged.
   */
  initialize(bounds: CascadeAABB): boolean {
    if (this._prevBounds) {
      const prev = this._prevBounds;
      // Pre-W8: `THREE.Vector3.distanceTo < 1` per axis. Equivalent in the
      // plain-tuple form: squared distance < 1.
      const sameLo = distance3Sq(prev.min, bounds.min) < 1;
      const sameHi = distance3Sq(prev.max, bounds.max) < 1;
      if (sameLo && sameHi && this._buffers) return false;
    }

    // Dispose old buffers before allocating new ones.
    if (this._buffers) {
      disposeCascades(this._buffers);
    }

    this._buffers = allocateCascades(bounds);
    this._prevBounds = {
      min: [bounds.min[0], bounds.min[1], bounds.min[2]],
      max: [bounds.max[0], bounds.max[1], bounds.max[2]],
    };
    return true;
  }

  /**
   * Returns the currently allocated cascade buffers, or null if not yet
   * initialized or already disposed.
   */
  getBuffers(): CascadeBuffers | null {
    return this._buffers;
  }

  /**
   * Dispose all cascade storage. After this call, `getBuffers()` returns null.
   * Call this when the RC pipeline is torn down.
   */
  dispose(): void {
    if (this._buffers) {
      disposeCascades(this._buffers);
      this._buffers = null;
    }
    this._prevBounds = null;
  }
}
