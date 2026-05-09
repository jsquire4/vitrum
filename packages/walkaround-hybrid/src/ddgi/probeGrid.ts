/**
 * ProbeGrid — owns the DDGI probe grid dimensions, atlas textures, and
 * GPU uniform data.
 *
 * Atlas layout:
 *  irradianceAtlas: (dimsX * 10) × (dimsY * dimsZ * 10) rgba16float
 *    each probe cell = 8×8 + 2px border = 10×10
 *  visibilityAtlas: (dimsX * 18) × (dimsY * dimsZ * 18) rg16float
 *    each probe cell = 16×16 + 2px border = 18×18
 */

import * as THREE from 'three';
import { StorageTexture } from 'three/webgpu';
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

export class ProbeGrid {
  dims: ProbeGridDims = { x: 2, y: 2, z: 2 };
  worldOrigin: THREE.Vector3 = new THREE.Vector3();
  worldSpacing: number = 24;

  /** Ping-pong irradiance atlases (A read / B write, swap each frame). */
  irradianceA: StorageTexture | null = null;
  irradianceB: StorageTexture | null = null;
  /** Ping-pong visibility atlases. */
  visibilityA: StorageTexture | null = null;
  visibilityB: StorageTexture | null = null;

  /** Which pair (A=true / B=false) is currently the "write" target. */
  writeIsA = false;

  private _params: ProbeGridParams | null = null;
  /** Dirty flag — set when dims/origin change and atlases need reallocating. */
  dirty = true;

  get params(): ProbeGridParams {
    if (!this._params) throw new Error('ProbeGrid not initialized');
    return this._params;
  }

  get irradianceReadTex(): StorageTexture {
    return (this.writeIsA ? this.irradianceB : this.irradianceA)!;
  }
  get irradianceWriteTex(): StorageTexture {
    return (this.writeIsA ? this.irradianceA : this.irradianceB)!;
  }
  get visibilityReadTex(): StorageTexture {
    return (this.writeIsA ? this.visibilityB : this.visibilityA)!;
  }
  get visibilityWriteTex(): StorageTexture {
    return (this.writeIsA ? this.visibilityA : this.visibilityB)!;
  }

  swap(): void {
    this.writeIsA = !this.writeIsA;
  }

  /**
   * (Re-)compute probe grid dimensions from the BVH bounds.
   * Returns true if the grid changed and atlases need rebuilding.
   */
  computeFromBounds(
    boundingBox: THREE.Box3,
    // Default 16" (was 24"). Probe spacing drives diffuse-indirect
    // resolution — at 24" a 12'×9' room had ~6×4×6 = 144 probes; at
    // 16" same room has ~9×6×9 = 486 probes (3.4×). Light propagation
    // detail and shadow boundary sharpness scale with probe density.
    // Hard cap (16, 10, 16 = 2560 max) prevents pathological growth.
    spacingInches = 16,
  ): boolean {
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    const PROBE_SPACING = spacingInches;

    const nx = Math.max(3, Math.ceil(size.x / PROBE_SPACING) + 1);
    const ny = Math.max(3, Math.ceil(size.y / PROBE_SPACING) + 1);
    const nz = Math.max(3, Math.ceil(size.z / PROBE_SPACING) + 1);

    // Hard cap to keep atlas sizes reasonable.
    const cx = Math.min(nx, 16);
    const cy = Math.min(ny, 10);
    const cz = Math.min(nz, 16);

    const changed =
      cx !== this.dims.x || cy !== this.dims.y || cz !== this.dims.z ||
      !this.worldOrigin.equals(boundingBox.min) ||
      this.worldSpacing !== spacingInches;

    this.dims = { x: cx, y: cy, z: cz };
    this.worldOrigin.copy(boundingBox.min);
    this.worldSpacing = spacingInches;
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

    const makeStorageTex = (w: number, h: number, format: THREE.PixelFormat) => {
      const tex = new StorageTexture(w, h);
      tex.format = format;
      tex.type = THREE.HalfFloatType;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.needsUpdate = true;
      return tex;
    };

    this.irradianceA = makeStorageTex(irrW, irrH, THREE.RGBAFormat);
    this.irradianceB = makeStorageTex(irrW, irrH, THREE.RGBAFormat);
    // Use RGBAFormat — rg16float is not writable as a storage texture in WebGPU
    // (only rgba16float is in the required set). We store (meanDist, meanDistSq)
    // in .rg and leave .ba as zero padding.
    this.visibilityA = makeStorageTex(visW, visH, THREE.RGBAFormat);
    this.visibilityB = makeStorageTex(visW, visH, THREE.RGBAFormat);

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

  /**
   * Build the raw Float32Array for the ProbeGridParams uniform.
   * Layout (std140 compatible, 32 bytes):
   *   vec3f origin (12 bytes + 4 pad = 16)
   *   f32  spacing (4)
   *   vec3u dims   (12 + 4 pad = 16 — stored as 3 floats)
   *   f32  _pad
   *   f32  irrW, irrH, visW, visH
   */
  buildUniformData(): Float32Array {
    if (!this._params) this.allocateAtlases();
    const p = this._params!;
    const buf = new Float32Array(16);
    buf[0] = p.origin.x;
    buf[1] = p.origin.y;
    buf[2] = p.origin.z;
    buf[3] = p.spacing;
    buf[4] = p.dims.x;
    buf[5] = p.dims.y;
    buf[6] = p.dims.z;
    buf[7] = 0;
    buf[8]  = p.irradianceAtlasW;
    buf[9]  = p.irradianceAtlasH;
    buf[10] = p.visibilityAtlasW;
    buf[11] = p.visibilityAtlasH;
    return buf;
  }

  get probeCount(): number {
    return this.dims.x * this.dims.y * this.dims.z;
  }

  private _disposeAtlases(): void {
    this.irradianceA?.dispose();
    this.irradianceB?.dispose();
    this.visibilityA?.dispose();
    this.visibilityB?.dispose();
    this.irradianceA = null;
    this.irradianceB = null;
    this.visibilityA = null;
    this.visibilityB = null;
  }

  dispose(): void {
    this._disposeAtlases();
  }
}
