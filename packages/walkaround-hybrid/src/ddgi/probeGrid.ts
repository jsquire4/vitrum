/**
 * ProbeGrid — owns the DDGI probe grid dimensions, atlas textures, and
 * GPU uniform data.
 *
 * Atlas layout:
 *  irradianceAtlas: (dimsX * 10) × (dimsY * dimsZ * 10) rgba16float
 *    each probe cell = 8×8 + 2px border = 10×10
 *  visibilityAtlas: (dimsX * 18) × (dimsY * dimsZ * 18) rg16float
 *    each probe cell = 16×16 + 2px border = 18×18
 *
 * Atlas slots are plain `{width, height}` records; ProbeGrid does not hold
 * any GPU or three/webgpu handles. probeUpdatePass.ts maintains the actual
 * `GPUTexture` per slot via a WeakMap keyed on the slot instance.
 * applyDDGIShading.ts (the TSL consumer) wraps each slot in its own
 * three/webgpu `StorageTexture` if needed. This isolates the three/webgpu
 * coupling to the TSL site and lets the compute path be pure raw WebGPU.
 */

import * as THREE from 'three';
// Atlas-layout constants imported from the canonical source so producer
// and consumers (ddgiSampleWgsl.ts + shade.wgsl.ts) stay in lockstep.
import { IRR_CELL, VIS_CELL, BORDER } from './ddgiAtlasLayout.js';

export interface ProbeGridDims {
  x: number;
  y: number;
  z: number;
}

export interface ProbeGridParams {
  origin: THREE.Vector3;
  spacing: number;
  dims: ProbeGridDims;
  irradianceAtlasW: number;
  irradianceAtlasH: number;
  visibilityAtlasW: number;
  visibilityAtlasH: number;
}

/**
 * Backend-agnostic atlas slot. Two slots per role (A/B) form the ping-pong
 * pair. Instance identity is the cache key — both probeUpdatePass.ts and
 * applyDDGIShading.ts WeakMap-cache their backend-specific handles per slot.
 */
export interface AtlasTextureSlot {
  readonly width: number;
  readonly height: number;
}

export class ProbeGrid {
  dims: ProbeGridDims = { x: 2, y: 2, z: 2 };
  worldOrigin: THREE.Vector3 = new THREE.Vector3();
  worldSpacing: number = 24;

  /** Ping-pong irradiance atlases (A read / B write, swap each frame). */
  irradianceA: AtlasTextureSlot | null = null;
  irradianceB: AtlasTextureSlot | null = null;
  /** Ping-pong visibility atlases. */
  visibilityA: AtlasTextureSlot | null = null;
  visibilityB: AtlasTextureSlot | null = null;

  /** Which pair (A=true / B=false) is currently the "write" target. */
  writeIsA = false;

  private _params: ProbeGridParams | null = null;
  /** Dirty flag — set when dims/origin change and atlases need reallocating. */
  dirty = true;

  get params(): ProbeGridParams {
    if (!this._params) throw new Error('ProbeGrid not initialized');
    return this._params;
  }

  get irradianceReadTex(): AtlasTextureSlot {
    return (this.writeIsA ? this.irradianceB : this.irradianceA)!;
  }
  get irradianceWriteTex(): AtlasTextureSlot {
    return (this.writeIsA ? this.irradianceA : this.irradianceB)!;
  }
  get visibilityReadTex(): AtlasTextureSlot {
    return (this.writeIsA ? this.visibilityB : this.visibilityA)!;
  }
  get visibilityWriteTex(): AtlasTextureSlot {
    return (this.writeIsA ? this.visibilityA : this.visibilityB)!;
  }

  swap(): void {
    this.writeIsA = !this.writeIsA;
  }

  /**
   * (Re-)compute probe grid dimensions from the BVH bounds.
   * Returns true if the grid changed and atlases need rebuilding.
   *
   * @param boundingBox     Scene AABB from the BVH.
   * @param spacingInches   Probe spacing in scene units. Pass `undefined` to
   *                        use the auto-derived value (`maxSize / 12`).
   * @param maxProbesPerAxis Hard cap per axis. Defaults to 16. Raise for large
   *                        scenes; lower for performance-constrained devices.
   *                        M11 audit remediation — was previously hardcoded.
   */
  computeFromBounds(
    boundingBox: THREE.Box3,
    spacingInches?: number,
    maxProbesPerAxis = 16,
  ): boolean {
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    // Target ~13 probes along the longest axis (`/ 12`). The denser the
    // grid, the smaller the trilinear-interp blocks visible at shadow
    // boundaries. At 5×5×5 the 0.5-unit cells produce screen-visible
    // block stair-step shadows on a 2-unit Cornell box; at 13³ the cells
    // are ~0.17 units which sit below the resolution-limited blur of the
    // 8-probe stencil, hiding the grid.
    const autoSpacing = Math.max(size.x, size.y, size.z) / 12;
    const PROBE_SPACING = spacingInches ?? autoSpacing;

    const nx = Math.max(3, Math.ceil(size.x / PROBE_SPACING) + 1);
    const ny = Math.max(3, Math.ceil(size.y / PROBE_SPACING) + 1);
    const nz = Math.max(3, Math.ceil(size.z / PROBE_SPACING) + 1);

    const cap = Math.max(3, maxProbesPerAxis);
    const cx = Math.min(nx, cap);
    const cy = Math.min(ny, cap);
    const cz = Math.min(nz, cap);

    const changed =
      cx !== this.dims.x || cy !== this.dims.y || cz !== this.dims.z ||
      !this.worldOrigin.equals(boundingBox.min) ||
      this.worldSpacing !== PROBE_SPACING;

    this.dims = { x: cx, y: cy, z: cz };
    this.worldOrigin.copy(boundingBox.min);
    this.worldSpacing = PROBE_SPACING;
    this.dirty = changed;
    return changed;
  }

  /**
   * Allocate (or reallocate) the atlas pair-of-pairs.
   * Must be called after computeFromBounds returns true.
   */
  allocateAtlases(): void {
    this._disposeAtlases();

    const { x, y, z } = this.dims;
    const irrW = x * (IRR_CELL + BORDER);
    const irrH = y * z * (IRR_CELL + BORDER);
    const visW = x * (VIS_CELL + BORDER);
    const visH = y * z * (VIS_CELL + BORDER);

    const makeSlot = (w: number, h: number): AtlasTextureSlot => ({ width: w, height: h });

    this.irradianceA = makeSlot(irrW, irrH);
    this.irradianceB = makeSlot(irrW, irrH);
    // We store (meanDist, meanDistSq) in .rg of the visibility atlas and
    // leave .ba as zero padding — rgba16float is the writable storage format
    // required by WebGPU (rg16float is not in the required set).
    this.visibilityA = makeSlot(visW, visH);
    this.visibilityB = makeSlot(visW, visH);

    this._params = {
      origin:   this.worldOrigin.clone(),
      spacing:  this.worldSpacing,
      dims:     { ...this.dims },
      irradianceAtlasW: irrW,
      irradianceAtlasH: irrH,
      visibilityAtlasW: visW,
      visibilityAtlasH: visH,
    };

    this.dirty = false;
  }

  // ProbeGrid UBO packing now lives in pipeline/resourceManager.ts as
  // `packDDGIGridParams(grid.params)` — single source for the 64-byte
  // layout shared by ProbeUpdatePass (this package) and HybridEngine
  // (which feeds the same layout into shade.wgsl). Callers that need the
  // raw bytes should import packDDGIGridParams directly.

  get probeCount(): number {
    return this.dims.x * this.dims.y * this.dims.z;
  }

  private _disposeAtlases(): void {
    // Slots are plain records — no GPU disposal here. The consumers
    // (probeUpdatePass for compute, applyDDGIShading for TSL) own their
    // backend handles and clean them up when their WeakMap entries
    // get collected (slots dropped to null below release the weak refs).
    this.irradianceA = null;
    this.irradianceB = null;
    this.visibilityA = null;
    this.visibilityB = null;
  }

  dispose(): void {
    this._disposeAtlases();
  }
}
