/**
 * ProbeGrid — owns the DDGI probe grid dimensions, atlas textures, and
 * GPU uniform data.
 *
 * Atlas layout:
 *  irradianceAtlas: (dimsX * 5) × (dimsY * dimsZ * 5) rgba16float
 *    each probe cell = 3×3 (L2 SH, 9 RGB coefficients) + 2px border = 5×5
 *  visibilityAtlas: (dimsX * 18) × (dimsY * dimsZ * 18) rgba16float
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
import { RAYS_PER_PROBE } from './ddgiConstants.js';
import { DDGI_PROBE_MAX_OFFSET_NORMALIZED } from './probeState.js';
import {
  DDGI_DIAGONAL_COMPONENT_F32,
  DDGI_NORMAL_BIAS_FACTOR_F32,
  DDGI_VISIBILITY_DISTANCE_MAX,
} from './ddgiNumericLimits.js';
import {
  DDGI_F32_MAX,
  assertDdgiU32,
  assertFiniteDdgiNumber,
  packDdgiProbeSpacingFloat32,
  packFiniteDdgiFloat32,
  packPositiveDdgiFloat32,
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
  private _x: number;
  private _y: number;
  private _z: number;
  private readonly _onChange?: () => void;
  private readonly _validateCandidate?: (candidate: ProbeGridVector3) => void;

  constructor(
    x = 0,
    y = 0,
    z = 0,
    onChange?: () => void,
    validateCandidate?: (candidate: ProbeGridVector3) => void,
  ) {
    Object.defineProperty(this, '_onChange', {
      value: onChange,
      enumerable: false,
    });
    Object.defineProperty(this, '_validateCandidate', {
      value: validateCandidate,
      enumerable: false,
    });
    this._x = packFiniteDdgiFloat32(x, 'DDGI probe vector x');
    this._y = packFiniteDdgiFloat32(y, 'DDGI probe vector y');
    this._z = packFiniteDdgiFloat32(z, 'DDGI probe vector z');
  }

  get x(): number { return this._x; }
  set x(value: number) {
    const nextX = packFiniteDdgiFloat32(value, 'DDGI probe vector x');
    this._validateCandidate?.({ x: nextX, y: this._y, z: this._z });
    this._x = nextX;
    this._onChange?.();
  }
  get y(): number { return this._y; }
  set y(value: number) {
    const nextY = packFiniteDdgiFloat32(value, 'DDGI probe vector y');
    this._validateCandidate?.({ x: this._x, y: nextY, z: this._z });
    this._y = nextY;
    this._onChange?.();
  }
  get z(): number { return this._z; }
  set z(value: number) {
    const nextZ = packFiniteDdgiFloat32(value, 'DDGI probe vector z');
    this._validateCandidate?.({ x: this._x, y: this._y, z: nextZ });
    this._z = nextZ;
    this._onChange?.();
  }

  set(x: number, y: number, z: number): this {
    const nextX = packFiniteDdgiFloat32(x, 'DDGI probe vector x');
    const nextY = packFiniteDdgiFloat32(y, 'DDGI probe vector y');
    const nextZ = packFiniteDdgiFloat32(z, 'DDGI probe vector z');
    this._validateCandidate?.({ x: nextX, y: nextY, z: nextZ });
    this._x = nextX;
    this._y = nextY;
    this._z = nextZ;
    this._onChange?.();
    return this;
  }

  copy(v: ProbeGridVector3): this {
    return this.set(v.x, v.y, v.z);
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
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export interface ProbeGridParams {
  readonly origin: ProbeGridVector3;
  readonly spacing: number;
  readonly dims: ProbeGridDims;
  readonly irradianceAtlasW: number;
  readonly irradianceAtlasH: number;
  readonly visibilityAtlasW: number;
  readonly visibilityAtlasH: number;
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

const DDGI_MAX_EXACT_F32_INTEGER = 0x1_000000;

export interface ValidatedGridPublication {
  readonly origin: Readonly<ProbeGridVector3>;
  readonly spacing: number;
  readonly dims: Readonly<ProbeGridDims>;
  readonly irradianceAtlasW: number;
  readonly irradianceAtlasH: number;
  readonly visibilityAtlasW: number;
  readonly visibilityAtlasH: number;
  readonly probeCount: number;
}

function nextPositiveFloat32(value: number, label: string): number {
  let packed = packPositiveDdgiFloat32(value, label);
  if (packed >= value) return packed;
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, packed, true);
  const bits = view.getUint32(0, true);
  if (bits >= 0x7f7f_ffff) {
    throw new RangeError(`${label} cannot be rounded upward to a finite float32.`);
  }
  view.setUint32(0, bits + 1, true);
  packed = view.getFloat32(0, true);
  return packPositiveDdgiFloat32(packed, label);
}

function assertExactPositiveF32Integer(value: number, label: string): void {
  assertDdgiU32(value, label);
  if (value < 1 || value > DDGI_MAX_EXACT_F32_INTEGER || Math.fround(value) !== value) {
    throw new RangeError(`${label} must be a positive integer exactly representable as float32.`);
  }
}

function finiteF32Product(a: number, b: number, label: string): number {
  const product = a * b;
  if (!Number.isFinite(product) || Math.abs(product) > DDGI_F32_MAX) {
    throw new RangeError(`${label} exceeds the finite float32 range.`);
  }
  const packed = Math.fround(product);
  if (!Number.isFinite(packed)) {
    throw new RangeError(`${label} must remain finite when published as float32.`);
  }
  return packed;
}

function stableLength3(x: number, y: number, z: number): number {
  const scale = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
  if (scale === 0) return 0;
  return scale * Math.hypot(x / scale, y / scale, z / scale);
}

export function validateGridPublication(
  originInput: ProbeGridVector3,
  spacingInput: number,
  dimsInput: ProbeGridDims,
): ValidatedGridPublication {
  const origin = Object.freeze({
    x: packFiniteDdgiFloat32(originInput.x, 'DDGI probe origin.x'),
    y: packFiniteDdgiFloat32(originInput.y, 'DDGI probe origin.y'),
    z: packFiniteDdgiFloat32(originInput.z, 'DDGI probe origin.z'),
  });
  const spacing = packDdgiProbeSpacingFloat32(spacingInput, 'DDGI probe spacing');
  const dims = Object.freeze({ x: dimsInput.x, y: dimsInput.y, z: dimsInput.z });
  for (const axis of ['x', 'y', 'z'] as const) {
    assertExactPositiveF32Integer(dims[axis], `DDGI probe dims.${axis}`);
  }

  const probeCount = dims.x * dims.y * dims.z;
  assertDdgiU32(probeCount, 'DDGI probe count');
  assertDdgiU32(
    probeCount * RAYS_PER_PROBE,
    'DDGI probe-ray result count',
  );
  const yz = dims.y * dims.z;
  assertDdgiU32(yz, 'DDGI probe YZ product');
  const irradianceAtlasW = dims.x * (IRR_CELL + BORDER);
  const irradianceAtlasH = yz * (IRR_CELL + BORDER);
  const visibilityAtlasW = dims.x * (VIS_CELL + BORDER);
  const visibilityAtlasH = yz * (VIS_CELL + BORDER);
  assertExactPositiveF32Integer(irradianceAtlasW, 'DDGI irradiance atlas width');
  assertExactPositiveF32Integer(irradianceAtlasH, 'DDGI irradiance atlas height');
  assertExactPositiveF32Integer(visibilityAtlasW, 'DDGI visibility atlas width');
  assertExactPositiveF32Integer(visibilityAtlasH, 'DDGI visibility atlas height');

  const spacingTimes16 = finiteF32Product(spacing, 16, 'DDGI visibility open distance');
  if (spacingTimes16 > DDGI_VISIBILITY_DISTANCE_MAX) {
    throw new RangeError('DDGI probe spacing exceeds the finite visibility-moment distance envelope.');
  }

  const spans = {
    x: finiteF32Product(dims.x - 1, spacing, 'DDGI lattice span.x'),
    y: finiteF32Product(dims.y - 1, spacing, 'DDGI lattice span.y'),
    z: finiteF32Product(dims.z - 1, spacing, 'DDGI lattice span.z'),
  };
  for (const axis of ['x', 'y', 'z'] as const) {
    const endpoint = Math.fround(origin[axis] + spans[axis]);
    if (!Number.isFinite(endpoint)) {
      throw new RangeError(`DDGI lattice endpoint.${axis} must remain finite as float32.`);
    }
    if (dims[axis] > 1 && spans[axis] > 0) {
      const first = Math.fround(origin[axis] + spacing);
      const previous = Math.fround(
        origin[axis] + Math.fround((dims[axis] - 2) * spacing),
      );
      if (first === origin[axis] || endpoint === previous) {
        throw new RangeError(`DDGI lattice adjacency is not resolvable on axis ${axis} in float32.`);
      }
    }
    // Mirror WGSL's two separately-rounded multiplications exactly. A single
    // pre-combined JS factor can admit a lattice whose shader-side bias rounds
    // back onto a large world-space endpoint.
    const normalBias = Math.fround(spacing * DDGI_NORMAL_BIAS_FACTOR_F32);
    const observableBias = Math.fround(
      normalBias * DDGI_DIAGONAL_COMPONENT_F32,
    );
    for (const position of [origin[axis], endpoint]) {
      const plus = Math.fround(position + observableBias);
      const minus = Math.fround(position - observableBias);
      if (
        !Number.isFinite(plus) || !Number.isFinite(minus) ||
        plus === position || minus === position
      ) {
        throw new RangeError(`DDGI world-space bias is not observable at the ${axis}-axis lattice boundary in float32.`);
      }
    }
  }
  const latticeDiagonal = stableLength3(spans.x, spans.y, spans.z);
  const relocationReach = spacing * DDGI_PROBE_MAX_OFFSET_NORMALIZED;
  if (
    !Number.isFinite(latticeDiagonal) ||
    !Number.isFinite(relocationReach) ||
    latticeDiagonal + relocationReach > DDGI_VISIBILITY_DISTANCE_MAX
  ) {
    throw new RangeError('DDGI lattice diagonal exceeds the finite visibility-moment distance envelope.');
  }

  return Object.freeze({
    origin,
    spacing,
    dims,
    irradianceAtlasW,
    irradianceAtlasH,
    visibilityAtlasW,
    visibilityAtlasH,
    probeCount,
  });
}

export class ProbeGrid {
  private _dims: Readonly<ProbeGridDims> = Object.freeze({ x: 2, y: 2, z: 2 });
  readonly worldOrigin = new ProbeGridVector3Value(0, 0, 0, () => {
    this.dirty = true;
  }, (candidate) => {
    if (this._applyingPublication === null) {
      validateGridPublication(candidate, this._worldSpacing, this._dims);
    }
  });
  private _worldSpacing = 24;
  /** Suppresses intermediate validation while an already-validated tuple is committed atomically. */
  private _applyingPublication: ValidatedGridPublication | null = null;

  get dims(): Readonly<ProbeGridDims> { return this._dims; }
  set dims(value: ProbeGridDims) {
    const next = Object.freeze({ x: value.x, y: value.y, z: value.z });
    const publication = validateGridPublication(this.worldOrigin, this._worldSpacing, next);
    this._dims = publication.dims;
    this.dirty = true;
  }

  get worldSpacing(): number { return this._worldSpacing; }
  set worldSpacing(value: number) {
    const publication = validateGridPublication(this.worldOrigin, value, this._dims);
    this._worldSpacing = publication.spacing;
    this.dirty = true;
  }

  private _commitPublication(publication: ValidatedGridPublication): void {
    this._applyingPublication = publication;
    try {
      this._dims = publication.dims;
      this._worldSpacing = publication.spacing;
      this.worldOrigin.copy(publication.origin);
    } finally {
      this._applyingPublication = null;
    }
  }

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

  get irradianceReadTex(): AtlasTextureSlot | null {
    return this.writeIsA ? this.irradianceB : this.irradianceA;
  }
  get irradianceWriteTex(): AtlasTextureSlot | null {
    return this.writeIsA ? this.irradianceA : this.irradianceB;
  }
  get visibilityReadTex(): AtlasTextureSlot | null {
    return this.writeIsA ? this.visibilityB : this.visibilityA;
  }
  get visibilityWriteTex(): AtlasTextureSlot | null {
    return this.writeIsA ? this.visibilityA : this.visibilityB;
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
    const rawMin = { x: box.min.x, y: box.min.y, z: box.min.z };
    const rawMax = { x: box.max.x, y: box.max.y, z: box.max.z };
    for (const axis of axes) {
      assertFiniteDdgiNumber(rawMin[axis], `DDGI probe bounds min.${axis}`);
      assertFiniteDdgiNumber(rawMax[axis], `DDGI probe bounds max.${axis}`);
      if (rawMin[axis] > rawMax[axis]) {
        throw new RangeError(
          `DDGI probe bounds min.${axis} must be <= max.${axis}.`,
        );
      }
    }
    const min = {
      x: packFiniteDdgiFloat32(rawMin.x, 'DDGI probe bounds min.x'),
      y: packFiniteDdgiFloat32(rawMin.y, 'DDGI probe bounds min.y'),
      z: packFiniteDdgiFloat32(rawMin.z, 'DDGI probe bounds min.z'),
    };
    const max = {
      x: packFiniteDdgiFloat32(rawMax.x, 'DDGI probe bounds max.x'),
      y: packFiniteDdgiFloat32(rawMax.y, 'DDGI probe bounds max.y'),
      z: packFiniteDdgiFloat32(rawMax.z, 'DDGI probe bounds max.z'),
    };
    const size = { x: 0, y: 0, z: 0 };
    for (const axis of axes) {
      const rawExtent = rawMax[axis] - rawMin[axis];
      if (!Number.isFinite(rawExtent) || rawExtent > DDGI_F32_MAX) {
        throw new RangeError(`DDGI probe bounds extent.${axis} exceeds the finite float32 range.`);
      }
      const packedExtent = Math.fround(max[axis] - min[axis]);
      if (!Number.isFinite(packedExtent) || (rawExtent > 0 && !(packedExtent > 0))) {
        throw new RangeError(`DDGI probe bounds extent.${axis} is not positive and finite in float32.`);
      }
      size[axis] = packedExtent;
    }
    const packedRequestedSpacing = spacingInches === undefined
      ? undefined
      : packDdgiProbeSpacingFloat32(spacingInches, 'DDGI probe spacing');
    assertPositiveDdgiInteger(maxProbesPerAxis, 'DDGI max probes per axis');
    assertDdgiU32(maxProbesPerAxis, 'DDGI max probes per axis');
    if (maxProbesPerAxis < 3) {
      throw new RangeError('DDGI max probes per axis must be >= 3.');
    }
    // Target ~13 probes along the longest axis (`/ 12`). The denser the
    // grid, the smaller the trilinear-interp blocks visible at shadow
    // boundaries. At 5×5×5 the 0.5-unit cells produce screen-visible
    // block stair-step shadows on a 2-unit Cornell box; at 13³ the cells
    // are ~0.17 units which sit below the resolution-limited blur of the
    // 8-probe stencil, hiding the grid.
    const maxExtent = Math.max(size.x, size.y, size.z);
    // A point/fully-degenerate AABB still needs a finite grid. One scene unit is
    // the least-surprising fallback when no positive extent exists to derive it.
    const autoSpacing = maxExtent > 0
      ? nextPositiveFloat32(maxExtent / 12, 'DDGI auto probe spacing')
      : 1;
    const requestedSpacing = packedRequestedSpacing ?? autoSpacing;
    const cap = maxProbesPerAxis;
    // A dimension cap must coarsen the physical lattice as well as truncate
    // its integer dimensions. Keeping the requested spacing while clipping
    // `dims` leaves the far side of a large scene outside the probe volume.
    // Use one isotropic spacing so the capped lattice still encloses every
    // axis: origin + (dims - 1) * spacing >= bounds.max.
    const probeSpacing = nextPositiveFloat32(Math.max(
      requestedSpacing,
      size.x / (cap - 1),
      size.y / (cap - 1),
      size.z / (cap - 1),
    ), 'DDGI derived probe spacing');

    const nx = Math.max(3, Math.ceil(size.x / probeSpacing) + 1);
    const ny = Math.max(3, Math.ceil(size.y / probeSpacing) + 1);
    const nz = Math.max(3, Math.ceil(size.z / probeSpacing) + 1);

    const cx = Math.min(nx, cap);
    const cy = Math.min(ny, cap);
    const cz = Math.min(nz, cap);

    const publication = validateGridPublication(
      min,
      probeSpacing,
      { x: cx, y: cy, z: cz },
    );
    for (const axis of axes) {
      const span = Math.fround((publication.dims[axis] - 1) * publication.spacing);
      const endpoint = Math.fround(publication.origin[axis] + span);
      if (endpoint < max[axis]) {
        throw new RangeError(`DDGI lattice endpoint.${axis} does not enclose the published scene bound.`);
      }
    }

    const changed =
      publication.dims.x !== this.dims.x ||
      publication.dims.y !== this.dims.y ||
      publication.dims.z !== this.dims.z ||
      !this.worldOrigin.equals(publication.origin) ||
      this.worldSpacing !== publication.spacing;

    this._commitPublication(publication);
    this.dirty = changed;
    return changed;
  }

  /**
   * Allocate (or reallocate) the atlas pair-of-pairs.
   * Must be called after computeFromBounds returns true.
   */
  allocateAtlases(): void {
    const publication = validateGridPublication(
      this.worldOrigin,
      this.worldSpacing,
      this.dims,
    );
    const makeSlot = (w: number, h: number): AtlasTextureSlot =>
      Object.freeze({ width: w, height: h });
    const irradianceA = makeSlot(publication.irradianceAtlasW, publication.irradianceAtlasH);
    const irradianceB = makeSlot(publication.irradianceAtlasW, publication.irradianceAtlasH);
    const visibilityA = makeSlot(publication.visibilityAtlasW, publication.visibilityAtlasH);
    const visibilityB = makeSlot(publication.visibilityAtlasW, publication.visibilityAtlasH);
    const params: ProbeGridParams = Object.freeze({
      origin: publication.origin,
      spacing: publication.spacing,
      dims: publication.dims,
      irradianceAtlasW: publication.irradianceAtlasW,
      irradianceAtlasH: publication.irradianceAtlasH,
      visibilityAtlasW: publication.visibilityAtlasW,
      visibilityAtlasH: publication.visibilityAtlasH,
    });

    // Publish only after the complete CPU/f32/u32 candidate is proven valid.
    this._disposeAtlases();
    this.irradianceA = irradianceA;
    this.irradianceB = irradianceB;
    // We store (meanDist, meanDistSq) in .rg of the visibility atlas and
    // use .ba for range-preserving moment exponents — rgba16float is the writable storage format
    // required by WebGPU (rg16float is not in the required set).
    this.visibilityA = visibilityA;
    this.visibilityB = visibilityB;
    this._params = params;

    this.dirty = false;
  }

  // ProbeGrid UBO packing lives in ddgi/ddgiGridUbo.ts (packDDGIGridParams) —
  // the single source for the 64-byte layout shared by ProbeUpdatePass and
  // HybridEngine (shade.wgsl). Callers that need the raw bytes should import
  // packDDGIGridParams directly from ddgiGridUbo.ts.

  get probeCount(): number {
    const count = this.dims.x * this.dims.y * this.dims.z;
    assertDdgiU32(count, 'DDGI probe count');
    return count;
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
