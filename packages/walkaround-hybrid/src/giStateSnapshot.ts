/**
 * giStateSnapshot.ts — a serializable snapshot of the baked real-time
 * global-illumination state (the "cached light field"), so a host can persist
 * the converged GI (e.g. to IndexedDB) and restore it on a later session
 * WITHOUT re-converging from scratch.
 *
 * The snapshot holds two kinds of converged GI history:
 *
 *   1. The two DDGI probe atlases — irradiance (rgba16float) and visibility
 *      (rgba16float; .rg = meanDist/meanDistSq) — as raw float16 (Uint16Array)
 *      exactly as they sit on the GPU, plus the probe-grid metadata needed to
 *      validate a restore targets a matching grid. (v1+.)
 *
 *   2. (Optional, v2+) The ReSTIR-GI temporal reservoir buffers — the half-res
 *      reconnection-sample reservoirs (`RESERVOIR_GI_STRIDE` u32 per pixel) that
 *      carry the Sprint-16/17 temporal+spatial GI history across frames. Without
 *      these, a "restore" seeds the DDGI probes but discards the ReSTIR-GI
 *      reservoir history, so the high-frequency indirect re-converges from
 *      scratch over the next ~N frames. (See `restirGI` below.)
 *
 * NOTE on the Radiance-Cascade (RC) subsystem: RC is intentionally NOT persisted
 * because it carries NO cross-frame temporal state — `RCDispatcher.dispatchFrameRaw`
 * fully recomputes every cascade from the BVH + sun each frame (the cast passes
 * overwrite the cascade buffers; there is no previous-cascade/accumulation buffer).
 * A restored frame regenerates the cascades in one dispatch from geometry, so
 * there is nothing to round-trip. If RC ever grows a temporal/accumulated cascade
 * buffer, add it as a new section here (the container format is section-extensible).
 *
 *   3. (Optional, v4+) The PPG (Müller 2017 Practical Path Guiding) adaptive
 *      sTree + per-cell dTree CPU state. Without this, an importGIState call
 *      seeds the DDGI probes and ReSTIR-GI reservoirs but discards the learned
 *      guiding distribution — guided sampling restarts cold against warm GI
 *      history (an inconsistent estimator state, acknowledged but previously
 *      undocumented). With this section the coordinator replaces its live sTree
 *      immediately after import so guided sampling resumes from the snapshot's
 *      converged distribution with no cold-start window. The section is gated
 *      on `ppgEnabled` at export time and is OPTIONAL at import time: a v3
 *      snapshot (ppg section absent) imports cleanly with a cold-start PPG, not
 *      an error.
 *
 * `HybridEngine.exportGIState()` / `importGIState()` produce/consume the snapshot;
 * `serializeGIState` / `deserializeGIState` round-trip it through a single
 * transferable `ArrayBuffer` for storage.
 */

import { PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL } from './ppg/ppgConstants.js';
import {
  buildInitialProbeStateData,
  isValidProbeStateData,
} from './ddgi/probeState.js';
import { IRR_STRIDE, VIS_STRIDE } from './ddgi/ddgiAtlasLayout.js';
import { validateSerialisedSTree } from './ppg/validateSerialisedSTree.js';
import {
  RESERVOIR_GI_BASE_STRIDE_U32,
  RESERVOIR_GI_GRIS_STRIDE_U32,
} from './gi/giLayout.js';

/** Magic + version for the binary container (bump VERSION on any layout change). */
const GI_SNAPSHOT_MAGIC = 0x47495353; // "GISS"
/**
 * v1 — DDGI atlases only.
 * v2 — adds the optional ReSTIR-GI reservoir section (gated by SECTION_RESTIR_GI
 *      in the header section-flags word). v2 readers still accept v1 buffers
 *      (which simply carry no reservoir section).
 * v3 — the irradiance atlas now stores L2 SH coefficients (9 RGB E_lm in the
 *      first 3x3 interior texels of each probe cell) instead of an octahedral
 *      cosine-mean map. Same byte container, INCOMPATIBLE contents — a v2 (or
 *      earlier) octahedral irradiance atlas decoded by SH-era code is garbage,
 *      so the version gate must reject it (no backward accept for irradiance).
 * v4 — adds the optional PPG sTree / dTree section (gated by SECTION_PPG in the
 *      header section-flags word). v3 readers reject v4 (version mismatch); v4
 *      readers accept v3 buffers (ppg section absent ⇒ cold PPG start, not an
 *      error). The PPG section records the CPU sTree topology + per-cell dTree
 *      directional distributions as flat Float32Array buffers (exactly the wire
 *      format produced by serialiseSTree), plus the scene-bounds AABB used to
 *      validate that the snapshot's guiding distribution covers the same scene.
 *      `maxSpatialCells` is stored so the importer can reject a snapshot trained
 *      with a different buffer capacity (a mismatched cap means the dTreeIndex
 *      values inside the sTree might be out-of-bounds for the live GPU buffers).
 * v5 — reuses PPG sub-header field [3] for `maxDTreeNodesPerCell`, the per-cell
 *      dTree stride baked into the PPG update shader and flux buffer. v4 PPG
 *      snapshots are still accepted and interpreted with the historical default
 *      cap of 341 nodes/cell.
 * v6 — adds the required explicit DDGI probe relocation/classification block
 *      (four Float32 lanes: offset.xyz + active.w). Header words 56/60 carry
 *      its width and height; SECTION_DDGI_PROBE_STATE gates the data block
 *      immediately after the visibility atlas. v3-v5 snapshots synthesize
 *      offset=0/active=1 state so their converged light fields remain importable.
 */
const GI_SNAPSHOT_VERSION = 6;
const HEADER_BYTES = 64; // fixed header, data blocks follow

/** Header section-flags bitfield (header offset 52, u32). Bit 0 = ReSTIR-GI present. */
const SECTION_RESTIR_GI = 1 << 0;
/** Bit 1 = PPG sTree/dTree present (v4+). */
const SECTION_PPG = 1 << 1;
/** Bit 2 = explicit Float32 DDGI relocation/classification state present (v6+). */
const SECTION_DDGI_PROBE_STATE = 1 << 2;
/** Sub-header preceding the three packed reservoir buffers when SECTION_RESTIR_GI is set. */
const RESTIR_GI_SUBHEADER_BYTES = 20; // halfW(u32) halfH(u32) strideU32(u32) bufU32Len(u32) reserved(u32)
/**
 * Sub-header preceding the PPG blob when SECTION_PPG is set (v4+).
 *
 * Layout (12 × 4-byte fields):
 *   [0] maxSpatialCells   — GPU buffer capacity cap (for compatibility guard)
 *   [1] sTreeNodeCount    — number of sTree nodes serialised into sTreeBuf
 *   [2] dTreeCount        — number of per-cell dTrees (= leaf count)
 *   [3] maxDTreeNodesPerCell — per-cell dTree stride (v5+; v4 repeated [1])
 *   [4] dTreeBufF32Len    — length of the dTreeBuf Float32Array in f32 elements
 *   [5] dTreeOffsetsByteLen — byte length of the dTreeOffsets Uint32Array
 *   [6] sceneBoundsMinX   — (f32) scene AABB min.x
 *   [7] sceneBoundsMinY   — (f32) scene AABB min.y
 *   [8] sceneBoundsMinZ   — (f32) scene AABB min.z
 *   [9] sceneBoundsMaxX   — (f32) scene AABB max.x
 *   [10] sceneBoundsMaxY  — (f32) scene AABB max.y
 *   [11] sceneBoundsMaxZ  — (f32) scene AABB max.z
 *
 * Total: 12 × 4 = 48 bytes.
 */
const PPG_SUBHEADER_BYTES = 48; // 12 × u32/f32 fields × 4 bytes each
const MAX_U32 = 0xffff_ffff;
const KNOWN_SECTION_FLAGS =
  SECTION_RESTIR_GI |
  SECTION_PPG |
  SECTION_DDGI_PROBE_STATE;

/**
 * The ReSTIR-GI temporal reservoir state (Sprint 16/17). Persisting these lets a
 * restore continue the temporal+spatial GI reuse instead of re-converging the
 * high-frequency indirect from scratch.
 *
 * Three half-res reservoir buffers, each the FULL GPU-buffer u32 contents:
 *   - `current`  — `gi-ris` write target / shade read (also the source of the
 *                  end-of-frame copy into `previous`).
 *   - `previous` — the cross-frame temporal-reuse input (`gi-temporal` reads it).
 *   - `spatial`  — within-frame spatial-reuse scratch (ping-ponged in the spatial
 *                  passes). Persisted too so the export→import→export is a complete
 *                  byte-identity round-trip over the reservoir set.
 *
 * `halfW`/`halfH`/`strideU32` are restore-guard metadata (= floor(W/2), floor(H/2),
 * `RESERVOIR_GI_STRIDE`); a restore is rejected if they don't match the live grid.
 * The arrays themselves are the verbatim GPU-buffer u32 contents (which the engine
 * floors to a 256-byte minimum, so the array length is the true buffer length, NOT
 * necessarily `halfW*halfH*strideU32` — that exact-fit holds only above the floor).
 */
export interface RestirGISnapshot {
  /** Half-resolution reservoir grid (= floor(W/2) × floor(H/2)). */
  readonly halfW: number;
  readonly halfH: number;
  /** Per-pixel reservoir stride, in u32 values (RESERVOIR_GI_STRIDE). */
  readonly strideU32: number;
  /** `gi-ris` write target / shade read reservoir, raw u32 (full GPU-buffer contents). */
  readonly current: Uint32Array;
  /** Previous-frame temporal-reuse reservoir, raw u32 (full GPU-buffer contents). */
  readonly previous: Uint32Array;
  /** Spatial-reuse scratch reservoir, raw u32 (full GPU-buffer contents). */
  readonly spatial: Uint32Array;
}

/**
 * PPG (Practical Path Guiding, Müller 2017) snapshot — the CPU-side sTree + per-cell
 * dTree directional distributions, serialised to flat Float32Array buffers.
 *
 * These are exactly the buffers produced by `serialiseSTree` and consumed by
 * `PPGCoordinator._uploadTree()` — so import is a straight pass-through to the
 * existing upload path.
 *
 * Compatibility guard fields (`maxSpatialCells`, `maxDTreeNodesPerCell`, and
 * `sceneBounds`) are recorded so the importer can reject a snapshot trained with
 * a different buffer capacity, shader/resource stride, or scene geometry,
 * rather than silently poisoning the live guiding distribution.
 */
export interface PpgSnapshot {
  /**
   * GPU buffer capacity used when this snapshot was trained. A restore is
   * rejected when the live engine was constructed with a different
   * `ppgMaxSpatialCells` (or the default when that opt-in was not used).
   * Mismatch means the dTreeIndex values inside sTree nodes would be
   * out-of-bounds for the live GPU buffers.
   */
  readonly maxSpatialCells: number;
  /**
   * Per-cell dTree node cap used when this snapshot was trained. A restore is
   * rejected when the live engine was constructed with a different
   * `ppgMaxDTreeNodesPerCell` (or the default when omitted). Mismatch means the
   * serialised dTree offsets and the PPG update shader's flux stride disagree.
   */
  readonly maxDTreeNodesPerCell: number;
  /** Flat sTree node buffer (header + N nodes × 16 f32). */
  readonly sTreeBuf: Float32Array;
  /** Concatenated per-cell dTree buffers. */
  readonly dTreeBuf: Float32Array;
  /**
   * Per-cell dTree start offsets (f32 indices into dTreeBuf).
   * Length == number of leaf cells (== dTree count).
   */
  readonly dTreeOffsets: Uint32Array;
  /**
   * World-space scene AABB used to build the sTree root cell.
   * Validated against the current engine's scene AABB at restore time so a
   * snapshot from a different scene is rejected before its guiding distribution
   * is installed.
   */
  readonly sceneBoundsMin: readonly [number, number, number];
  readonly sceneBoundsMax: readonly [number, number, number];
}

export interface GIStateSnapshot {
  /** Probe grid dimensions (probes per axis). */
  readonly dims: { readonly x: number; readonly y: number; readonly z: number };
  /** World-space origin of the probe grid (the bounding-box min). */
  readonly origin: readonly [number, number, number];
  /** World-space probe spacing. */
  readonly spacing: number;
  /** Irradiance atlas size (texels). */
  readonly irrW: number;
  readonly irrH: number;
  /** Visibility atlas size (texels). */
  readonly visW: number;
  readonly visH: number;
  /** Irradiance atlas as raw rgba16float (4 × u16 per texel, row-major). */
  readonly irrData: Uint16Array;
  /** Visibility atlas as raw rgba16float. */
  readonly visData: Uint16Array;
  /** Probe-state image size. Canonical layout is dims.x × (dims.y*dims.z). */
  readonly probeStateW: number;
  readonly probeStateH: number;
  /** Raw four-Float32 relocation.xyz + active.w records, row-major. */
  readonly probeStateData: Float32Array;
  /**
   * Optional (v2+) ReSTIR-GI temporal reservoir state. Absent when the engine
   * had no ReSTIR-GI reservoirs allocated, or when restoring a v1 snapshot.
   */
  readonly restirGI?: RestirGISnapshot;
  /**
   * Optional (v4+) PPG (Müller 2017) sTree / dTree guiding distribution.
   * Absent when the engine was constructed without `ppgEnabled:true`, or when
   * restoring a v3 (or earlier) snapshot. When absent, PPG starts cold — the
   * coordinator builds a fresh single-cell sTree at boot and trains from scratch.
   */
  readonly ppg?: PpgSnapshot;
}

/** True when every raw binary16 lane represents a finite value. */
export function isFiniteRgba16Data(data: Uint16Array): boolean {
  for (const bits of data) {
    if ((bits & 0x7c00) === 0x7c00) return false;
  }
  return true;
}

/**
 * Compare metadata that crossed the snapshot's float32 wire boundary with its
 * live double-precision source. Canonicalizing both sides to f32 admits every
 * valid f64 → f32 → f64 round-trip while rejecting a distinct adjacent f32
 * coordinate. Values inside one f32 rounding bin are necessarily
 * indistinguishable on the wire.
 */
export function f32SnapshotMetadataMatches(
  snapshotValue: number,
  liveValue: number,
): boolean {
  const snapshotF32 = Math.fround(snapshotValue);
  const liveF32 = Math.fround(liveValue);
  if (
    !Number.isFinite(snapshotValue) ||
    !Number.isFinite(liveValue) ||
    !Number.isFinite(snapshotF32) ||
    !Number.isFinite(liveF32)
  ) {
    return false;
  }
  return snapshotF32 === liveF32;
}

function assertPositiveU32(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_U32) {
    throw new RangeError(`${label} must be a positive uint32 integer.`);
  }
}

function assertFiniteF32(
  value: number,
  label: string,
  positive = false,
): void {
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(Math.fround(value)) ||
    (positive && !(Math.fround(value) > 0))
  ) {
    throw new RangeError(
      `${label} must be ${positive ? 'positive and ' : ''}finite and representable as float32.`,
    );
  }
}

function checkedProduct(label: string, ...factors: number[]): number {
  let result = 1;
  for (const factor of factors) {
    if (
      !Number.isSafeInteger(factor) ||
      factor < 0 ||
      (factor !== 0 && result > Number.MAX_SAFE_INTEGER / factor)
    ) {
      throw new RangeError(`${label} exceeds safe integer arithmetic.`);
    }
    result *= factor;
  }
  return result;
}

function checkedSum(label: string, ...terms: number[]): number {
  let result = 0;
  for (const term of terms) {
    if (
      !Number.isSafeInteger(term) ||
      term < 0 ||
      result > Number.MAX_SAFE_INTEGER - term
    ) {
      throw new RangeError(`${label} exceeds safe integer arithmetic.`);
    }
    result += term;
  }
  return result;
}

function assertFiniteF32Vec3(value: unknown, label: string): void {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new TypeError(`${label} must be an exact 3-vector.`);
  }
  const components: readonly unknown[] = value;
  for (let component = 0; component < 3; component += 1) {
    const componentValue = components[component];
    if (typeof componentValue !== 'number') {
      throw new TypeError(`${label}[${component}] must be a number.`);
    }
    assertFiniteF32(componentValue, `${label}[${component}]`);
  }
}

function assertAtlasMetadata(
  dims: GIStateSnapshot['dims'],
  irrW: number,
  irrH: number,
  visW: number,
  visH: number,
): void {
  assertPositiveU32(dims.x, 'GI snapshot dims.x');
  assertPositiveU32(dims.y, 'GI snapshot dims.y');
  assertPositiveU32(dims.z, 'GI snapshot dims.z');
  assertPositiveU32(irrW, 'GI snapshot irrW');
  assertPositiveU32(irrH, 'GI snapshot irrH');
  assertPositiveU32(visW, 'GI snapshot visW');
  assertPositiveU32(visH, 'GI snapshot visH');

  const yz = checkedProduct('GI snapshot probe-grid yz product', dims.y, dims.z);
  const expectedIrrW = checkedProduct(
    'GI snapshot irradiance width',
    dims.x,
    IRR_STRIDE,
  );
  const expectedIrrH = checkedProduct(
    'GI snapshot irradiance height',
    yz,
    IRR_STRIDE,
  );
  const expectedVisW = checkedProduct(
    'GI snapshot visibility width',
    dims.x,
    VIS_STRIDE,
  );
  const expectedVisH = checkedProduct(
    'GI snapshot visibility height',
    yz,
    VIS_STRIDE,
  );
  if (
    expectedIrrW > MAX_U32 ||
    expectedIrrH > MAX_U32 ||
    expectedVisW > MAX_U32 ||
    expectedVisH > MAX_U32 ||
    irrW !== expectedIrrW ||
    irrH !== expectedIrrH ||
    visW !== expectedVisW ||
    visH !== expectedVisH
  ) {
    throw new RangeError(
      'GI snapshot atlas dimensions do not match the declared probe grid.',
    );
  }
}

function assertRgba16Atlas(
  data: Uint16Array,
  width: number,
  height: number,
  label: string,
): void {
  if (!(data instanceof Uint16Array)) {
    throw new TypeError(`${label} must be a Uint16Array.`);
  }
  const expectedLength = checkedProduct(
    `${label} element count`,
    width,
    height,
    4,
  );
  if (data.length !== expectedLength) {
    throw new RangeError(
      `${label} length ${data.length} does not match ${width}x${height} RGBA data.`,
    );
  }
  if (!isFiniteRgba16Data(data)) {
    throw new RangeError(`${label} must contain only finite float16 values.`);
  }
}

const RESTIR_GI_BASE_FLOAT_LANES = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
] as const;
const RESTIR_GI_GRIS_FLOAT_LANES = [20, 21, 22, 23, 26] as const;

function restirReservoirPayloadIsValid(
  data: Uint32Array,
  recordCount: number,
  strideU32: number,
): boolean {
  const floats = new Float32Array(
    data.buffer,
    data.byteOffset,
    data.length,
  );
  for (let record = 0; record < recordCount; record += 1) {
    const base = record * strideU32;
    for (const lane of RESTIR_GI_BASE_FLOAT_LANES) {
      if (!Number.isFinite(floats[base + lane])) return false;
    }
    if (strideU32 === RESERVOIR_GI_GRIS_STRIDE_U32) {
      for (const lane of RESTIR_GI_GRIS_FLOAT_LANES) {
        if (!Number.isFinite(floats[base + lane])) return false;
      }
      // sampleKind is the only enum-valued u32 lane. M, lightId,
      // prefixVertexCount, and historyEpoch legitimately consume the full u32
      // domain, so they are intentionally not range-restricted.
      const sampleKind = data[base + 25];
      if (sampleKind !== 0 && sampleKind !== 1) return false;
    }
  }
  const logicalWordCount = recordCount * strideU32;
  for (let index = logicalWordCount; index < data.length; index += 1) {
    if (data[index] !== 0) return false;
  }
  return true;
}

function assertSerializableRestirSnapshot(
  value: unknown,
): asserts value is RestirGISnapshot {
  if (value == null || typeof value !== 'object') {
    throw new TypeError('GI snapshot ReSTIR section must be an object.');
  }
  const snapshot = value as RestirGISnapshot;
  assertPositiveU32(snapshot.halfW, 'GI snapshot ReSTIR halfW');
  assertPositiveU32(snapshot.halfH, 'GI snapshot ReSTIR halfH');
  assertPositiveU32(snapshot.strideU32, 'GI snapshot ReSTIR strideU32');
  if (
    snapshot.strideU32 !== RESERVOIR_GI_BASE_STRIDE_U32 &&
    snapshot.strideU32 !== RESERVOIR_GI_GRIS_STRIDE_U32
  ) {
    throw new RangeError(
      'GI snapshot ReSTIR stride must match the compact or GRIS reservoir ABI.',
    );
  }
  for (const [name, data] of [
    ['current', snapshot.current],
    ['previous', snapshot.previous],
    ['spatial', snapshot.spatial],
  ] as const) {
    if (!(data instanceof Uint32Array)) {
      throw new TypeError(`GI snapshot ReSTIR ${name} must be a Uint32Array.`);
    }
  }
  const expectedLength = Math.max(
    64,
    checkedProduct(
      'GI snapshot ReSTIR reservoir element count',
      snapshot.halfW,
      snapshot.halfH,
      snapshot.strideU32,
    ),
  );
  if (
    expectedLength > MAX_U32 ||
    snapshot.current.length !== expectedLength ||
    snapshot.previous.length !== expectedLength ||
    snapshot.spatial.length !== expectedLength
  ) {
    throw new RangeError(
      'GI snapshot ReSTIR reservoir buffers do not match their declared layout.',
    );
  }
  const recordCount = checkedProduct(
    'GI snapshot ReSTIR logical record count',
    snapshot.halfW,
    snapshot.halfH,
  );
  for (const [name, data] of [
    ['current', snapshot.current],
    ['previous', snapshot.previous],
    ['spatial', snapshot.spatial],
  ] as const) {
    if (
      !restirReservoirPayloadIsValid(
        data,
        recordCount,
        snapshot.strideU32,
      )
    ) {
      throw new RangeError(
        `GI snapshot ReSTIR ${name} contains a non-finite or invalid logical reservoir record.`,
      );
    }
  }
}

/** Non-throwing semantic validator used at every ReSTIR restore boundary. */
export function isValidRestirGISnapshot(
  value: unknown,
): value is RestirGISnapshot {
  try {
    assertSerializableRestirSnapshot(value);
    return true;
  } catch {
    return false;
  }
}

function assertSerializablePpgSnapshot(snapshot: PpgSnapshot): void {
  assertPositiveU32(snapshot.maxSpatialCells, 'GI snapshot PPG maxSpatialCells');
  assertPositiveU32(
    snapshot.maxDTreeNodesPerCell,
    'GI snapshot PPG maxDTreeNodesPerCell',
  );
  if (
    !(snapshot.sTreeBuf instanceof Float32Array) ||
    !(snapshot.dTreeBuf instanceof Float32Array) ||
    !(snapshot.dTreeOffsets instanceof Uint32Array)
  ) {
    throw new TypeError(
      'GI snapshot PPG buffers must be Float32Array/Uint32Array instances.',
    );
  }
  if (
    snapshot.sTreeBuf.length > MAX_U32 ||
    snapshot.dTreeBuf.length > MAX_U32 ||
    snapshot.dTreeOffsets.length > MAX_U32
  ) {
    throw new RangeError('GI snapshot PPG buffer lengths must fit in uint32.');
  }
  for (const value of snapshot.sTreeBuf) {
    if (!Number.isFinite(value)) {
      throw new RangeError('GI snapshot PPG sTreeBuf must be finite.');
    }
  }
  for (const value of snapshot.dTreeBuf) {
    if (!Number.isFinite(value)) {
      throw new RangeError('GI snapshot PPG dTreeBuf must be finite.');
    }
  }
  assertFiniteF32Vec3(snapshot.sceneBoundsMin, 'GI snapshot PPG sceneBoundsMin');
  assertFiniteF32Vec3(snapshot.sceneBoundsMax, 'GI snapshot PPG sceneBoundsMax');
  for (let axis = 0; axis < 3; axis += 1) {
    if (snapshot.sceneBoundsMin[axis]! >= snapshot.sceneBoundsMax[axis]!) {
      throw new RangeError(
        `GI snapshot PPG scene bounds min[${axis}] must be < max[${axis}].`,
      );
    }
  }
  validateSerialisedSTree(snapshot, {
    maxSpatialCells: snapshot.maxSpatialCells,
    maxDTreeNodesPerCell: snapshot.maxDTreeNodesPerCell,
    sceneBounds: {
      min: snapshot.sceneBoundsMin,
      max: snapshot.sceneBoundsMax,
    },
  });
}

function assertSerializableSnapshot(
  value: unknown,
  validatePpg = true,
): asserts value is GIStateSnapshot {
  if (value == null || typeof value !== 'object') {
    throw new TypeError('GI snapshot must be an object.');
  }
  const snapshot = value as GIStateSnapshot;
  if (
    snapshot.dims == null ||
    typeof snapshot.dims !== 'object'
  ) {
    throw new TypeError('GI snapshot dims must be an object.');
  }
  assertAtlasMetadata(
    snapshot.dims,
    snapshot.irrW,
    snapshot.irrH,
    snapshot.visW,
    snapshot.visH,
  );
  assertFiniteF32Vec3(snapshot.origin, 'GI snapshot origin');
  assertFiniteF32(snapshot.spacing, 'GI snapshot spacing', true);
  assertRgba16Atlas(
    snapshot.irrData,
    snapshot.irrW,
    snapshot.irrH,
    'GI snapshot irradiance atlas',
  );
  assertRgba16Atlas(
    snapshot.visData,
    snapshot.visW,
    snapshot.visH,
    'GI snapshot visibility atlas',
  );

  assertPositiveU32(snapshot.probeStateW, 'GI snapshot probeStateW');
  assertPositiveU32(snapshot.probeStateH, 'GI snapshot probeStateH');
  const expectedProbeStateH = checkedProduct(
    'GI snapshot probe-state height',
    snapshot.dims.y,
    snapshot.dims.z,
  );
  if (
    snapshot.probeStateW !== snapshot.dims.x ||
    snapshot.probeStateH !== expectedProbeStateH ||
    !(snapshot.probeStateData instanceof Float32Array) ||
    snapshot.probeStateData.length !==
      checkedProduct(
        'GI snapshot probe-state element count',
        snapshot.probeStateW,
        snapshot.probeStateH,
        4,
      ) ||
    !isValidProbeStateData(snapshot.probeStateData, snapshot.spacing)
  ) {
    throw new RangeError(
      'GI snapshot DDGI probe-state dimensions/data do not match the probe grid.',
    );
  }
  if (snapshot.restirGI != null) {
    assertSerializableRestirSnapshot(snapshot.restirGI);
  }
  if (validatePpg && snapshot.ppg != null) {
    assertSerializablePpgSnapshot(snapshot.ppg);
  }

  const restirBytes = snapshot.restirGI == null
    ? 0
    : checkedSum(
        'GI snapshot ReSTIR section size',
        RESTIR_GI_SUBHEADER_BYTES,
        snapshot.restirGI.current.byteLength,
        snapshot.restirGI.previous.byteLength,
        snapshot.restirGI.spatial.byteLength,
      );
  const ppgBytes = !validatePpg || snapshot.ppg == null
    ? 0
    : checkedSum(
        'GI snapshot PPG section size',
        PPG_SUBHEADER_BYTES,
        snapshot.ppg.sTreeBuf.byteLength,
        snapshot.ppg.dTreeBuf.byteLength,
        snapshot.ppg.dTreeOffsets.byteLength,
      );
  checkedSum(
    'GI snapshot container size',
    HEADER_BYTES,
    snapshot.irrData.byteLength,
    snapshot.visData.byteLength,
    snapshot.probeStateData.byteLength,
    restirBytes,
    ppgBytes,
  );
}

/**
 * Non-throwing validation for the sections required by an engine restore.
 * PPG is deliberately excluded: it is an optional best-effort guide whose
 * rejection must not invalidate an otherwise complete DDGI + ReSTIR restore.
 */
export function isValidRequiredGIStateSnapshot(
  value: unknown,
): value is GIStateSnapshot {
  try {
    assertSerializableSnapshot(value, false);
    return true;
  } catch {
    return false;
  }
}

/** Serialize a snapshot to a single ArrayBuffer (IndexedDB / file ready). */
export function serializeGIState(s: GIStateSnapshot): ArrayBuffer {
  assertSerializableSnapshot(s);
  const irrBytes = s.irrData.byteLength;
  const visBytes = s.visData.byteLength;
  const probeStateBytes = s.probeStateData.byteLength;

  const r = s.restirGI;
  const hasRestir = r != null;
  // The three reservoir arrays are the verbatim GPU-buffer u32 contents (equal
  // length, guaranteed equal-sized by the engine); the sub-header records their
  // length explicitly so the floor-padded case round-trips.
  const restirBytes = hasRestir
    ? RESTIR_GI_SUBHEADER_BYTES + r.current.byteLength + r.previous.byteLength + r.spatial.byteLength
    : 0;

  const p = s.ppg;
  const hasPpg = p != null;
  const ppgBytes = hasPpg
    ? PPG_SUBHEADER_BYTES
      + p.sTreeBuf.byteLength
      + p.dTreeBuf.byteLength
      + p.dTreeOffsets.byteLength
    : 0;

  const buf = new ArrayBuffer(
    HEADER_BYTES + irrBytes + visBytes + probeStateBytes + restirBytes + ppgBytes,
  );
  const dv = new DataView(buf);
  let o = 0;
  dv.setUint32(o, GI_SNAPSHOT_MAGIC, true); o += 4;
  dv.setUint32(o, GI_SNAPSHOT_VERSION, true); o += 4;
  dv.setUint32(o, s.dims.x, true); o += 4;
  dv.setUint32(o, s.dims.y, true); o += 4;
  dv.setUint32(o, s.dims.z, true); o += 4;
  dv.setFloat32(o, s.origin[0], true); o += 4;
  dv.setFloat32(o, s.origin[1], true); o += 4;
  dv.setFloat32(o, s.origin[2], true); o += 4;
  dv.setFloat32(o, s.spacing, true); o += 4;
  dv.setUint32(o, s.irrW, true); o += 4;
  dv.setUint32(o, s.irrH, true); o += 4;
  dv.setUint32(o, s.visW, true); o += 4;
  dv.setUint32(o, s.visH, true); o += 4;
  // o = 52 here: section-flags word, then header padded to 64.
  const sectionFlags =
    (hasRestir ? SECTION_RESTIR_GI : 0) |
    (hasPpg ? SECTION_PPG : 0) |
    SECTION_DDGI_PROBE_STATE;
  dv.setUint32(o, sectionFlags, true); o += 4;
  dv.setUint32(o, s.probeStateW, true); o += 4;
  dv.setUint32(o, s.probeStateH, true); o += 4;

  new Uint8Array(buf, HEADER_BYTES, irrBytes).set(new Uint8Array(s.irrData.buffer, s.irrData.byteOffset, irrBytes));
  new Uint8Array(buf, HEADER_BYTES + irrBytes, visBytes).set(new Uint8Array(s.visData.buffer, s.visData.byteOffset, visBytes));
  new Uint8Array(
    buf,
    HEADER_BYTES + irrBytes + visBytes,
    probeStateBytes,
  ).set(
    new Uint8Array(
      s.probeStateData.buffer,
      s.probeStateData.byteOffset,
      probeStateBytes,
    ),
  );

  if (hasRestir) {
    // All three reservoir buffers are allocated with identical size by the
    // engine, so a single bufU32Len describes each. (Guarded below.)
    if (r.current.length !== r.previous.length || r.current.length !== r.spatial.length) {
      throw new Error('serializeGIState: ReSTIR-GI reservoir buffers must be equal-length.');
    }
    let ro = HEADER_BYTES + irrBytes + visBytes + probeStateBytes;
    dv.setUint32(ro, r.halfW, true); ro += 4;
    dv.setUint32(ro, r.halfH, true); ro += 4;
    dv.setUint32(ro, r.strideU32, true); ro += 4;
    dv.setUint32(ro, r.current.length, true); ro += 4; // bufU32Len (per-buffer)
    dv.setUint32(ro, 0, true); ro += 4; // reserved
    const cur = r.current, prev = r.previous, sp = r.spatial;
    new Uint8Array(buf, ro, cur.byteLength).set(new Uint8Array(cur.buffer, cur.byteOffset, cur.byteLength));
    ro += cur.byteLength;
    new Uint8Array(buf, ro, prev.byteLength).set(new Uint8Array(prev.buffer, prev.byteOffset, prev.byteLength));
    ro += prev.byteLength;
    new Uint8Array(buf, ro, sp.byteLength).set(new Uint8Array(sp.buffer, sp.byteOffset, sp.byteLength));
  }

  if (hasPpg) {
    let po = HEADER_BYTES + irrBytes + visBytes + probeStateBytes + restirBytes;
    // Sub-header (PPG_SUBHEADER_BYTES = 48 bytes, 12 × 4-byte fields).
    dv.setUint32(po, p.maxSpatialCells, true); po += 4;           // [0]
    dv.setUint32(po, p.sTreeBuf.length, true); po += 4;            // [1] sTreeBufF32Len
    dv.setUint32(po, p.dTreeOffsets.length, true); po += 4;        // [2] dTreeCount
    dv.setUint32(po, p.maxDTreeNodesPerCell, true); po += 4;       // [3] maxDTreeNodesPerCell (v5)
    dv.setUint32(po, p.dTreeBuf.length, true); po += 4;            // [4] dTreeBufF32Len
    dv.setUint32(po, p.dTreeOffsets.byteLength, true); po += 4;    // [5] dTreeOffsetsByteLen
    dv.setFloat32(po, p.sceneBoundsMin[0], true); po += 4;         // [6]
    dv.setFloat32(po, p.sceneBoundsMin[1], true); po += 4;         // [7]
    dv.setFloat32(po, p.sceneBoundsMin[2], true); po += 4;         // [8]
    dv.setFloat32(po, p.sceneBoundsMax[0], true); po += 4;         // [9]
    dv.setFloat32(po, p.sceneBoundsMax[1], true); po += 4;         // [10]
    dv.setFloat32(po, p.sceneBoundsMax[2], true); po += 4;         // [11]
    // Data blobs follow.
    new Uint8Array(buf, po, p.sTreeBuf.byteLength).set(
      new Uint8Array(p.sTreeBuf.buffer, p.sTreeBuf.byteOffset, p.sTreeBuf.byteLength),
    ); po += p.sTreeBuf.byteLength;
    new Uint8Array(buf, po, p.dTreeBuf.byteLength).set(
      new Uint8Array(p.dTreeBuf.buffer, p.dTreeBuf.byteOffset, p.dTreeBuf.byteLength),
    ); po += p.dTreeBuf.byteLength;
    new Uint8Array(buf, po, p.dTreeOffsets.byteLength).set(
      new Uint8Array(p.dTreeOffsets.buffer, p.dTreeOffsets.byteOffset, p.dTreeOffsets.byteLength),
    );
  }

  return buf;
}

/** Deserialize an ArrayBuffer produced by {@link serializeGIState}. */
export function deserializeGIState(buf: ArrayBuffer): GIStateSnapshot {
  if (!(buf instanceof ArrayBuffer)) {
    throw new TypeError('deserializeGIState: input must be an ArrayBuffer.');
  }
  if (buf.byteLength < HEADER_BYTES) {
    throw new Error('deserializeGIState: buffer is smaller than the fixed header.');
  }
  const dv = new DataView(buf);
  let o = 0;
  const magic = dv.getUint32(o, true); o += 4;
  if (magic !== GI_SNAPSHOT_MAGIC) {
    throw new Error(`deserializeGIState: bad magic 0x${magic.toString(16)} (not a GI snapshot).`);
  }
  const version = dv.getUint32(o, true); o += 4;
  // v3 (SH irradiance, no PPG), v4 (adds optional PPG), v5 (adds PPG
  // maxDTreeNodesPerCell), and v6 (probe relocation/classification state) are
  // accepted. Older SH snapshots synthesize zero-offset active probe state.
  // v3 readers see no PPG
  // section flag and return ppg:undefined (cold PPG start — not an error).
  // v4 PPG sections default the new dTree cap to 341. v1/v2 carried an OCTAHEDRAL irradiance
  // atlas whose bytes are meaningless to the SH-era sampler, so backward-accept
  // was intentionally dropped at the v2->v3 break and remains dropped here.
  if (
    version !== 3 &&
    version !== 4 &&
    version !== 5 &&
    version !== GI_SNAPSHOT_VERSION
  ) {
    throw new Error(`deserializeGIState: unsupported version ${version} (expected ${GI_SNAPSHOT_VERSION}, 5, 4, or 3; v1/v2 octahedral irradiance is incompatible with SH).`);
  }
  const dx = dv.getUint32(o, true); o += 4;
  const dy = dv.getUint32(o, true); o += 4;
  const dz = dv.getUint32(o, true); o += 4;
  const ox = dv.getFloat32(o, true); o += 4;
  const oy = dv.getFloat32(o, true); o += 4;
  const oz = dv.getFloat32(o, true); o += 4;
  const spacing = dv.getFloat32(o, true); o += 4;
  const irrW = dv.getUint32(o, true); o += 4;
  const irrH = dv.getUint32(o, true); o += 4;
  const visW = dv.getUint32(o, true); o += 4;
  const visH = dv.getUint32(o, true); o += 4;
  // Section flags (offset 52). v1 buffers wrote 0 here as header padding, so a
  // v1 buffer reads SECTION_RESTIR_GI=0 and skips the reservoir section.
  const sectionFlags = dv.getUint32(o, true); o += 4;
  const declaredProbeStateW = dv.getUint32(o, true); o += 4;
  const declaredProbeStateH = dv.getUint32(o, true); o += 4;
  if ((sectionFlags & ~KNOWN_SECTION_FLAGS) !== 0) {
    throw new Error('deserializeGIState: snapshot contains unknown section flags.');
  }
  const allowedSectionFlags =
    version === 3
      ? SECTION_RESTIR_GI
      : version === 4 || version === 5
        ? SECTION_RESTIR_GI | SECTION_PPG
        : KNOWN_SECTION_FLAGS;
  if ((sectionFlags & ~allowedSectionFlags) !== 0) {
    throw new Error(
      `deserializeGIState: section flags are incompatible with snapshot version ${version}.`,
    );
  }
  assertAtlasMetadata({ x: dx, y: dy, z: dz }, irrW, irrH, visW, visH);
  assertFiniteF32Vec3([ox, oy, oz], 'GI snapshot origin');
  assertFiniteF32(spacing, 'GI snapshot spacing', true);
  const irrBytes = checkedProduct(
    'GI snapshot irradiance byte length',
    irrW,
    irrH,
    8,
  ); // rgba16float = 8 bytes/texel
  const visBytes = checkedProduct(
    'GI snapshot visibility byte length',
    visW,
    visH,
    8,
  );
  const atlasEnd = checkedSum(
    'GI snapshot atlas payload end',
    HEADER_BYTES,
    irrBytes,
    visBytes,
  );
  if (buf.byteLength < atlasEnd) {
    throw new Error('deserializeGIState: buffer too small for the declared atlas dimensions.');
  }
  // Copy out (slice) so the returned arrays own their memory.
  const irrData = new Uint16Array(buf.slice(HEADER_BYTES, HEADER_BYTES + irrBytes));
  const visData = new Uint16Array(buf.slice(HEADER_BYTES + irrBytes, HEADER_BYTES + irrBytes + visBytes));
  if (!isFiniteRgba16Data(irrData) || !isFiniteRgba16Data(visData)) {
    throw new Error('deserializeGIState: atlas payload contains non-finite float16 values.');
  }

  const hasProbeState = (sectionFlags & SECTION_DDGI_PROBE_STATE) !== 0;
  if (version >= 6 && !hasProbeState) {
    throw new Error('deserializeGIState: v6 snapshot is missing DDGI probe state.');
  }
  const probeStateW = hasProbeState ? declaredProbeStateW : dx;
  const expectedProbeStateH = checkedProduct(
    'GI snapshot probe-state height',
    dy,
    dz,
  );
  const probeStateH = hasProbeState
    ? declaredProbeStateH
    : expectedProbeStateH;
  if (
    probeStateW !== dx ||
    probeStateH !== expectedProbeStateH ||
    !Number.isSafeInteger(probeStateW) ||
    probeStateW <= 0 ||
    !Number.isSafeInteger(probeStateH) ||
    probeStateH <= 0
  ) {
    throw new Error(
      'deserializeGIState: DDGI probe-state dimensions do not match the probe grid.',
    );
  }
  const probeStateBytes = hasProbeState
    ? checkedProduct(
        'GI snapshot probe-state byte length',
        probeStateW,
        probeStateH,
        16,
      )
    : 0;
  const probeStateOffset = atlasEnd;
  const probeStateEnd = checkedSum(
    'GI snapshot probe-state payload end',
    probeStateOffset,
    probeStateBytes,
  );
  if (buf.byteLength < probeStateEnd) {
    throw new Error('deserializeGIState: buffer too small for the declared DDGI probe state.');
  }
  const probeStateData = hasProbeState
    ? new Float32Array(
        buf.slice(probeStateOffset, probeStateOffset + probeStateBytes),
      )
    : buildInitialProbeStateData(
        checkedProduct('GI snapshot probe count', dx, dy, dz),
        true,
      );
  if (!isValidProbeStateData(probeStateData, spacing)) {
    throw new Error('deserializeGIState: malformed DDGI probe-state texels.');
  }

  const base: GIStateSnapshot = {
    dims: { x: dx, y: dy, z: dz },
    origin: [ox, oy, oz],
    spacing,
    irrW,
    irrH,
    visW,
    visH,
    irrData,
    visData,
    probeStateW,
    probeStateH,
    probeStateData,
  };

  // Cursor used by optional sections. Start it right after the atlas data and
  // advance as each optional section is consumed.
  let cursor = probeStateEnd;

  // ── ReSTIR-GI reservoir section (v2+) ──────────────────────────────────────
  let restirGI: RestirGISnapshot | undefined;
  if ((sectionFlags & SECTION_RESTIR_GI) !== 0) {
    let ro = cursor;
    if (buf.byteLength < ro + RESTIR_GI_SUBHEADER_BYTES) {
      throw new Error('deserializeGIState: buffer too small for the ReSTIR-GI sub-header.');
    }
    const halfW = dv.getUint32(ro, true); ro += 4;
    const halfH = dv.getUint32(ro, true); ro += 4;
    const strideU32 = dv.getUint32(ro, true); ro += 4;
    const bufU32Len = dv.getUint32(ro, true); ro += 4;
    const reserved = dv.getUint32(ro, true); ro += 4;
    if (reserved !== 0) {
      throw new Error(
        'deserializeGIState: ReSTIR-GI reserved sub-header word must be zero.',
      );
    }
    assertPositiveU32(halfW, 'GI snapshot ReSTIR halfW');
    assertPositiveU32(halfH, 'GI snapshot ReSTIR halfH');
    assertPositiveU32(strideU32, 'GI snapshot ReSTIR strideU32');
    assertPositiveU32(bufU32Len, 'GI snapshot ReSTIR buffer length');
    const expectedReservoirLength = Math.max(
      64,
      checkedProduct(
        'GI snapshot ReSTIR reservoir element count',
        halfW,
        halfH,
        strideU32,
      ),
    );
    if (bufU32Len !== expectedReservoirLength) {
      throw new Error(
        'deserializeGIState: ReSTIR-GI buffer length does not match its declared layout.',
      );
    }
    const bufBytes = checkedProduct(
      'GI snapshot ReSTIR buffer byte length',
      bufU32Len,
      4,
    );
    const reservoirEnd = checkedSum(
      'GI snapshot ReSTIR payload end',
      ro,
      bufBytes,
      bufBytes,
      bufBytes,
    );
    if (buf.byteLength < reservoirEnd) {
      throw new Error('deserializeGIState: buffer too small for the declared ReSTIR-GI reservoir dimensions.');
    }
    const current  = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
    const previous = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
    const spatial  = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
    restirGI = { halfW, halfH, strideU32, current, previous, spatial };
    if (!isValidRestirGISnapshot(restirGI)) {
      throw new Error(
        'deserializeGIState: ReSTIR-GI payload contains an invalid logical reservoir record.',
      );
    }
    cursor = ro;
  }

  if ((sectionFlags & SECTION_PPG) === 0) {
    if (cursor !== buf.byteLength) {
      throw new Error('deserializeGIState: unexpected trailing snapshot bytes.');
    }
    return restirGI != null ? { ...base, restirGI } : base;
  }

  // ── PPG sTree / dTree section (v4+) ────────────────────────────────────────
  let po = cursor;
  if (buf.byteLength < po + PPG_SUBHEADER_BYTES) {
    throw new Error('deserializeGIState: buffer too small for the PPG sub-header.');
  }
  const maxSpatialCells = dv.getUint32(po, true); po += 4;     // [0]
  const sTreeBufF32Len  = dv.getUint32(po, true); po += 4;     // [1] / [3]
  const dTreeCount      = dv.getUint32(po, true); po += 4;     // [2]
  const ppgField3       = dv.getUint32(po, true); po += 4;     // [3] v5 maxDTreeNodesPerCell / v4 repeat of [1]
  const maxDTreeNodesPerCell = version >= 5
    ? ppgField3
    : PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
  const dTreeBufF32Len      = dv.getUint32(po, true); po += 4; // [4]
  const dTreeOffsetsByteLen = dv.getUint32(po, true); po += 4; // [5]
  const sbMinX = dv.getFloat32(po, true); po += 4;              // [6]
  const sbMinY = dv.getFloat32(po, true); po += 4;              // [7]
  const sbMinZ = dv.getFloat32(po, true); po += 4;              // [8]
  const sbMaxX = dv.getFloat32(po, true); po += 4;              // [9]
  const sbMaxY = dv.getFloat32(po, true); po += 4;              // [10]
  const sbMaxZ = dv.getFloat32(po, true); po += 4;              // [11]
  assertPositiveU32(maxSpatialCells, 'GI snapshot PPG maxSpatialCells');
  assertPositiveU32(
    maxDTreeNodesPerCell,
    'GI snapshot PPG maxDTreeNodesPerCell',
  );
  if (dTreeOffsetsByteLen % Uint32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(
      'deserializeGIState: PPG dTreeOffsets byte length must be u32-aligned.',
    );
  }
  if (
    dTreeOffsetsByteLen !==
    checkedProduct(
      'GI snapshot PPG dTreeOffsets byte length',
      dTreeCount,
      Uint32Array.BYTES_PER_ELEMENT,
    )
  ) {
    throw new Error(
      'deserializeGIState: PPG dTreeOffsets byte length does not match dTreeCount.',
    );
  }
  const sTreeByteLen = checkedProduct(
    'GI snapshot PPG sTree byte length',
    sTreeBufF32Len,
    4,
  );
  const dTreeByteLen = checkedProduct(
    'GI snapshot PPG dTree byte length',
    dTreeBufF32Len,
    4,
  );
  const ppgEnd = checkedSum(
    'GI snapshot PPG payload end',
    po,
    sTreeByteLen,
    dTreeByteLen,
    dTreeOffsetsByteLen,
  );
  if (buf.byteLength < ppgEnd) {
    throw new Error('deserializeGIState: buffer too small for the declared PPG tree data.');
  }
  const sTreeBuf    = new Float32Array(buf.slice(po, po + sTreeByteLen)); po += sTreeByteLen;
  const dTreeBuf    = new Float32Array(buf.slice(po, po + dTreeByteLen)); po += dTreeByteLen;
  const dTreeOffsets = new Uint32Array(buf.slice(po, po + dTreeOffsetsByteLen));
  po += dTreeOffsetsByteLen;
  // Validate dTreeOffsets length matches declared dTreeCount (guard against
  // corrupted blobs that pass the size check above but carry wrong metadata).
  if (dTreeOffsets.length !== dTreeCount) {
    throw new Error(
      `deserializeGIState: PPG dTreeOffsets length ${dTreeOffsets.length} does not match ` +
      `declared dTreeCount ${dTreeCount}.`,
    );
  }
  const ppg: PpgSnapshot = {
    maxSpatialCells,
    maxDTreeNodesPerCell,
    sTreeBuf,
    dTreeBuf,
    dTreeOffsets,
    sceneBoundsMin: [sbMinX, sbMinY, sbMinZ],
    sceneBoundsMax: [sbMaxX, sbMaxY, sbMaxZ],
  };
  assertSerializablePpgSnapshot(ppg);
  if (po !== buf.byteLength) {
    throw new Error('deserializeGIState: unexpected trailing snapshot bytes.');
  }
  return restirGI != null
    ? { ...base, restirGI, ppg }
    : { ...base, ppg };
}
