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
 */
const GI_SNAPSHOT_VERSION = 2;
const HEADER_BYTES = 64; // fixed header, data blocks follow

/** Header section-flags bitfield (header offset 52, u32). Bit 0 = ReSTIR-GI present. */
const SECTION_RESTIR_GI = 1 << 0;
/** Sub-header preceding the three packed reservoir buffers when SECTION_RESTIR_GI is set. */
const RESTIR_GI_SUBHEADER_BYTES = 20; // halfW(u32) halfH(u32) strideU32(u32) bufU32Len(u32) reserved(u32)

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

  const buf = new ArrayBuffer(HEADER_BYTES + irrBytes + visBytes + restirBytes);
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
  dv.setUint32(o, hasRestir ? SECTION_RESTIR_GI : 0, true); o += 4;

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
  // v1 (DDGI-only) and v2 (DDGI + optional ReSTIR-GI) are both accepted; v2
  // is a strict superset (v1 buffers carry no reservoir section / flag).
  if (version !== 1 && version !== GI_SNAPSHOT_VERSION) {
    throw new Error(`deserializeGIState: unsupported version ${version} (expected 1 or ${GI_SNAPSHOT_VERSION}).`);
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

  if ((sectionFlags & SECTION_RESTIR_GI) === 0) return base;

  // ── ReSTIR-GI reservoir section (v2+) ──────────────────────────────────────
  let ro = HEADER_BYTES + irrBytes + visBytes;
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
  const current = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
  const previous = new Uint32Array(buf.slice(ro, ro + bufBytes)); ro += bufBytes;
  const spatial = new Uint32Array(buf.slice(ro, ro + bufBytes));
  return { ...base, restirGI: { halfW, halfH, strideU32, current, previous, spatial } };
}
