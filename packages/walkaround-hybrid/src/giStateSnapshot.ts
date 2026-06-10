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
 * NOTE on the Radiance-Cascade (RC) subsystem: RC is intentionally NOT persisted
 * because it carries NO cross-frame temporal state — `RCDispatcher.dispatchFrameRaw`
 * fully recomputes every cascade from the BVH + sun each frame (the cast passes
 * overwrite the cascade buffers; there is no previous-cascade/accumulation buffer).
 * A restored frame regenerates the cascades in one dispatch from geometry, so
 * there is nothing to round-trip. If RC ever grows a temporal/accumulated cascade
 * buffer, add it as a new section here (the container format is section-extensible).
 *
 * `HybridEngine.exportGIState()` / `importGIState()` produce/consume the snapshot;
 * `serializeGIState` / `deserializeGIState` round-trip it through a single
 * transferable `ArrayBuffer` for storage.
 */

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
 */
const GI_SNAPSHOT_VERSION = 4;
const HEADER_BYTES = 64; // fixed header, data blocks follow

/** Header section-flags bitfield (header offset 52, u32). Bit 0 = ReSTIR-GI present. */
const SECTION_RESTIR_GI = 1 << 0;
/** Bit 1 = PPG sTree/dTree present (v4+). */
const SECTION_PPG = 1 << 1;
/** Sub-header preceding the three packed reservoir buffers when SECTION_RESTIR_GI is set. */
const RESTIR_GI_SUBHEADER_BYTES = 20; // halfW(u32) halfH(u32) strideU32(u32) bufU32Len(u32) reserved(u32)
/**
 * Sub-header preceding the PPG blob when SECTION_PPG is set (v4+).
 *
 * Layout (6 × u32 = 24 bytes):
 *   [0] maxSpatialCells   — GPU buffer capacity cap (for compatibility guard)
 *   [1] sTreeNodeCount    — number of sTree nodes serialised into sTreeBuf
 *   [2] dTreeCount        — number of per-cell dTrees (= leaf count)
 *   [3] sTreeBufF32Len    — length of the sTreeBuf Float32Array in f32 elements
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
 * Compatibility guard fields (`maxSpatialCells` and `sceneBounds`) are recorded
 * so the importer can reject a snapshot trained with a different buffer capacity
 * or scene geometry, rather than silently poisoning the live guiding distribution.
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

/** Serialize a snapshot to a single ArrayBuffer (IndexedDB / file ready). */
export function serializeGIState(s: GIStateSnapshot): ArrayBuffer {
  const irrBytes = s.irrData.byteLength;
  const visBytes = s.visData.byteLength;

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

  const buf = new ArrayBuffer(HEADER_BYTES + irrBytes + visBytes + restirBytes + ppgBytes);
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
  const sectionFlags = (hasRestir ? SECTION_RESTIR_GI : 0) | (hasPpg ? SECTION_PPG : 0);
  dv.setUint32(o, sectionFlags, true); o += 4;
  // Remaining 8 bytes of the 64-byte header are zero-padded (already zero from
  // ArrayBuffer allocation). o is now 56; skip to HEADER_BYTES = 64.

  new Uint8Array(buf, HEADER_BYTES, irrBytes).set(new Uint8Array(s.irrData.buffer, s.irrData.byteOffset, irrBytes));
  new Uint8Array(buf, HEADER_BYTES + irrBytes, visBytes).set(new Uint8Array(s.visData.buffer, s.visData.byteOffset, visBytes));

  if (hasRestir) {
    // All three reservoir buffers are allocated with identical size by the
    // engine, so a single bufU32Len describes each. (Guarded below.)
    if (r.current.length !== r.previous.length || r.current.length !== r.spatial.length) {
      throw new Error('serializeGIState: ReSTIR-GI reservoir buffers must be equal-length.');
    }
    let ro = HEADER_BYTES + irrBytes + visBytes;
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
    let po = HEADER_BYTES + irrBytes + visBytes + restirBytes;
    // Sub-header (PPG_SUBHEADER_BYTES = 48 bytes, 12 × 4-byte fields).
    dv.setUint32(po, p.maxSpatialCells, true); po += 4;           // [0]
    dv.setUint32(po, p.sTreeBuf.length, true); po += 4;            // [1] sTreeBufF32Len
    dv.setUint32(po, p.dTreeOffsets.length, true); po += 4;        // [2] dTreeCount
    dv.setUint32(po, p.sTreeBuf.length, true); po += 4;            // [3] sTreeBufF32Len (repeated for clarity)
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
  const dv = new DataView(buf);
  let o = 0;
  const magic = dv.getUint32(o, true); o += 4;
  if (magic !== GI_SNAPSHOT_MAGIC) {
    throw new Error(`deserializeGIState: bad magic 0x${magic.toString(16)} (not a GI snapshot).`);
  }
  const version = dv.getUint32(o, true); o += 4;
  // v3 (SH irradiance, no PPG) and v4 (adds optional PPG section) are both
  // accepted. v3 readers see no PPG section flag and return ppg:undefined
  // (cold PPG start — not an error). v1/v2 carried an OCTAHEDRAL irradiance
  // atlas whose bytes are meaningless to the SH-era sampler, so backward-accept
  // was intentionally dropped at the v2->v3 break and remains dropped here.
  if (version !== 3 && version !== GI_SNAPSHOT_VERSION) {
    throw new Error(`deserializeGIState: unsupported version ${version} (expected ${GI_SNAPSHOT_VERSION} or 3; v1/v2 octahedral irradiance is incompatible with SH).`);
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
  const irrBytes = irrW * irrH * 8; // rgba16float = 8 bytes/texel
  const visBytes = visW * visH * 8;
  if (buf.byteLength < HEADER_BYTES + irrBytes + visBytes) {
    throw new Error('deserializeGIState: buffer too small for the declared atlas dimensions.');
  }
  // Copy out (slice) so the returned arrays own their memory.
  const irrData = new Uint16Array(buf.slice(HEADER_BYTES, HEADER_BYTES + irrBytes));
  const visData = new Uint16Array(buf.slice(HEADER_BYTES + irrBytes, HEADER_BYTES + irrBytes + visBytes));

  const base: GIStateSnapshot = { dims: { x: dx, y: dy, z: dz }, origin: [ox, oy, oz], spacing, irrW, irrH, visW, visH, irrData, visData };

  // Cursor used by optional sections. Start it right after the atlas data and
  // advance as each optional section is consumed.
  let cursor = HEADER_BYTES + irrBytes + visBytes;

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
    ro += 4; // reserved
    const bufBytes = bufU32Len * 4;
    if (buf.byteLength < ro + bufBytes * 3) {
      throw new Error('deserializeGIState: buffer too small for the declared ReSTIR-GI reservoir dimensions.');
    }
    const current  = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
    const previous = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
    const spatial  = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
    restirGI = { halfW, halfH, strideU32, current, previous, spatial };
    cursor = ro;
  }

  if ((sectionFlags & SECTION_PPG) === 0) {
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
  po += 4;                                                       // [3] repeat of [1], skip
  const dTreeBufF32Len      = dv.getUint32(po, true); po += 4; // [4]
  const dTreeOffsetsByteLen = dv.getUint32(po, true); po += 4; // [5]
  const sbMinX = dv.getFloat32(po, true); po += 4;              // [6]
  const sbMinY = dv.getFloat32(po, true); po += 4;              // [7]
  const sbMinZ = dv.getFloat32(po, true); po += 4;              // [8]
  const sbMaxX = dv.getFloat32(po, true); po += 4;              // [9]
  const sbMaxY = dv.getFloat32(po, true); po += 4;              // [10]
  const sbMaxZ = dv.getFloat32(po, true); po += 4;              // [11]
  const sTreeByteLen = sTreeBufF32Len * 4;
  const dTreeByteLen = dTreeBufF32Len * 4;
  if (buf.byteLength < po + sTreeByteLen + dTreeByteLen + dTreeOffsetsByteLen) {
    throw new Error('deserializeGIState: buffer too small for the declared PPG tree data.');
  }
  const sTreeBuf    = new Float32Array(buf.slice(po, po + sTreeByteLen)); po += sTreeByteLen;
  const dTreeBuf    = new Float32Array(buf.slice(po, po + dTreeByteLen)); po += dTreeByteLen;
  const dTreeOffsets = new Uint32Array(buf.slice(po, po + dTreeOffsetsByteLen));
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
    sTreeBuf,
    dTreeBuf,
    dTreeOffsets,
    sceneBoundsMin: [sbMinX, sbMinY, sbMinZ],
    sceneBoundsMax: [sbMaxX, sbMaxY, sbMaxZ],
  };
  return restirGI != null
    ? { ...base, restirGI, ppg }
    : { ...base, ppg };
}
