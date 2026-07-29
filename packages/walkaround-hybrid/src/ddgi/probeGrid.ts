/**
 * ProbeGrid — owns the DDGI probe grid dimensions, atlas textures, and
 * GPU uniform data.
 *
 * Atlas layout:
 *  irradianceAtlas: (dimsX * 5) × (dimsY * dimsZ * 5) rgba16float
 *    each probe cell = 3×3 (L2 SH, 9 RGB coefficients) + 2px border = 5×5
 *  visibilityAtlas: (dimsX * 18) × (dimsY * dimsZ * 18) rg16float
 *    each probe cell = 16×16 + 2px border = 18×18
 *
 * The irradiance cell migrated from octahedral 8×8 to L2 SH 3×3 (ddgiSH.wgsl.ts,
 * 2026-06-07). SH has no octahedral seam so the irradiance border is unused
 * (no border pass); the 2px ring is kept only for stride-uniformity with the
 * visibility atlas. See ddgiAtlasLayout.ts for the single-source constants.
 *
 * Atlas slots are plain `{width, height}` records; ProbeGrid does not hold
 * any GPU handles. probeUpdatePass.ts maintains the GPUTexture per slot via
 * a WeakMap keyed on the slot instance, and frame shading consumes the atlas
 * through raw WebGPU bind groups.
 */

// Atlas-layout constants imported from the canonical source so producer
// and consumers (ddgiSampleWgsl.ts + shade.wgsl.ts) stay in lockstep.
import { IRR_CELL, VIS_CELL, BORDER } from './ddgiAtlasLayout.js';
import {
  assertFiniteDdgiNumber,
  assertPositiveDdgiInteger,
} from './inputValidation.js';

/**
 * D6.10 — PlainAabbArrayLike covers @vitrum/shared-bvh PlainAabb where
 * min/max are array-tuples [x,y,z] rather than { x, y, z } objects.
 */
interface PlainAabbArrayLike {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

export interface ProbeGridVector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export class ProbeGridVector3Value implements ProbeGridVector3 {
  constructor(
    public x = 0,
    public y = 0,
    public z = 0,
  ) {}

  set(x: number, y: number, z: number): this {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }

  copy(v: ProbeGridVector3): this {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }

  equals(v: ProbeGridVector3): boolean {
    return this.x === v.x && this.y === v.y && this.z === v.z;
  }

  clone(): ProbeGridVector3Value {
    return new ProbeGridVector3Value(this.x, this.y, this.z);
  }
}

interface ProbeGridBoxLike {
  readonly min: ProbeGridVector3;
  readonly max: ProbeGridVector3;
}

/**
 * ProbeGridBounds accepts both forms:
 *  1. { min: {x,y,z}, max: {x,y,z} }  — RestirBvhAabb, THREE.Box3, ProbeGridBoxLike
 *  2. { min: [x,y,z], max: [x,y,z] }  — PlainAabb from @vitrum/shared-bvh SceneBvh
 *
 * D6.10: computeFromBounds normalises both via _normaliseBounds(); callers
 * pass whichever form they have without conversion.
 */
export type ProbeGridBounds = ProbeGridBoxLike | PlainAabbArrayLike;

/** @internal Normalise either bounds form to a ProbeGridBoxLike { x,y,z } pair. */
function _normaliseBounds(b: ProbeGridBounds): ProbeGridBoxLike {
  // Discriminate on whether .min has numeric properties 0/1/2 (array-tuple)
  // vs x/y/z (box-like).
  const minIsArray = !('x' in (b.min as object));
  if (minIsArray) {
    const a = b as PlainAabbArrayLike;
    return {
      min: { x: a.min[0], y: a.min[1], z: a.min[2] },
      max: { x: a.max[0], y: a.max[1], z: a.max[2] },
    };
  }
  return b as ProbeGridBoxLike;
}

export interface ProbeGridDims {
  x: number;
  y: number;
  z: number;
}

export interface ProbeGridParams {
  origin: ProbeGridVector3;
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
  worldOrigin: ProbeGridVector3Value = new ProbeGridVector3Value();
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
    boundingBox: ProbeGridBounds,
    spacingInches?: number,
    maxProbesPerAxis = 16,
  ): boolean {
    const box = _normaliseBounds(boundingBox);
    const axes = ['x', 'y', 'z'] as const;
    for (const axis of axes) {
      assertFiniteDdgiNumber(box.min[axis], `DDGI probe bounds min.${axis}`);
      assertFiniteDdgiNumber(box.max[axis], `DDGI probe bounds max.${axis}`);
      if (box.min[axis] > box.max[axis]) {
        throw new RangeError(
          `DDGI probe bounds min.${axis} must be <= max.${axis}.`,
        );
      }
    }
    if (spacingInches !== undefined) {
      assertFiniteDdgiNumber(spacingInches, 'DDGI probe spacing');
      if (spacingInches <= 0) {
        throw new RangeError('DDGI probe spacing must be > 0.');
      }
    }
    assertPositiveDdgiInteger(maxProbesPerAxis, 'DDGI max probes per axis');
    const size = new ProbeGridVector3Value();
    const min = new ProbeGridVector3Value();
    size.set(
      box.max.x - box.min.x,
      box.max.y - box.min.y,
      box.max.z - box.min.z,
    );
    min.copy(box.min);
    // Target ~13 probes along the longest axis (`/ 12`). The denser the
    // grid, the smaller the trilinear-interp blocks visible at shadow
    // boundaries. At 5×5×5 the 0.5-unit cells produce screen-visible
    // block stair-step shadows on a 2-unit Cornell box; at 13³ the cells
    // are ~0.17 units which sit below the resolution-limited blur of the
    // 8-probe stencil, hiding the grid.
    const maxExtent = Math.max(size.x, size.y, size.z);
    // A point/fully-degenerate AABB still needs a finite grid. One scene unit is
    // the least-surprising fallback when no positive extent exists to derive it.
    const autoSpacing = maxExtent > 0 ? maxExtent / 12 : 1;
    const requestedSpacing = spacingInches ?? autoSpacing;
    const cap = Math.max(3, maxProbesPerAxis);
    // A dimension cap must coarsen the physical lattice as well as truncate
    // its integer dimensions. Keeping the requested spacing while clipping
    // `dims` leaves the far side of a large scene outside the probe volume.
    // Use one isotropic spacing so the capped lattice still encloses every
    // axis: origin + (dims - 1) * spacing >= bounds.max.
    const PROBE_SPACING = Math.max(
      requestedSpacing,
      size.x / (cap - 1),
      size.y / (cap - 1),
      size.z / (cap - 1),
    );

    const nx = Math.max(3, Math.ceil(size.x / PROBE_SPACING) + 1);
    const ny = Math.max(3, Math.ceil(size.y / PROBE_SPACING) + 1);
    const nz = Math.max(3, Math.ceil(size.z / PROBE_SPACING) + 1);

    const cx = Math.min(nx, cap);
    const cy = Math.min(ny, cap);
    const cz = Math.min(nz, cap);

    const changed =
      cx !== this.dims.x || cy !== this.dims.y || cz !== this.dims.z ||
      !this.worldOrigin.equals(min) ||
      this.worldSpacing !== PROBE_SPACING;

    this.dims = { x: cx, y: cy, z: cz };
    this.worldOrigin.copy(min);
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

  // ProbeGrid UBO packing lives in ddgi/ddgiGridUbo.ts (packDDGIGridParams) —
  // the single source for the 64-byte layout shared by ProbeUpdatePass and
  // HybridEngine (shade.wgsl). Callers that need the raw bytes should import
  // packDDGIGridParams directly from ddgiGridUbo.ts.

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
