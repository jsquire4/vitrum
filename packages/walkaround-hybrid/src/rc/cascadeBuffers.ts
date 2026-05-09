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

import * as THREE from 'three';
import { allocateCascades, disposeCascades, type CascadeBuffers } from './cascadePyramid.js';

export class CascadeBufferManager {
  private _buffers: CascadeBuffers | null = null;
  private _prevBounds: THREE.Box3 | null = null;

  /**
   * Allocate (or re-allocate) cascade storage for the given scene bounds.
   * Re-allocates only if bounds changed meaningfully (> 1 world-unit).
   *
   * Returns `true` if new buffers were allocated, `false` if bounds unchanged.
   */
  initialize(bounds: THREE.Box3): boolean {
    if (this._prevBounds) {
      const prev = this._prevBounds;
      const sameLo = prev.min.distanceTo(bounds.min) < 1;
      const sameHi = prev.max.distanceTo(bounds.max) < 1;
      if (sameLo && sameHi && this._buffers) return false;
    }

    // Dispose old buffers before allocating new ones.
    if (this._buffers) {
      disposeCascades(this._buffers);
    }

    this._buffers = allocateCascades(bounds);
    this._prevBounds = bounds.clone();
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
