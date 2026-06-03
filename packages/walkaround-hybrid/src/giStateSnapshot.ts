/**
 * giStateSnapshot.ts — a serializable snapshot of the baked DDGI global-illumination
 * state (the "cached light field"), so a host can persist the converged probe
 * atlases (e.g. to IndexedDB) and restore them on a later session WITHOUT
 * re-converging from scratch.
 *
 * The snapshot holds the two DDGI probe atlases — irradiance (rgba16float) and
 * visibility (rgba16float; .rg = meanDist/meanDistSq) — as raw float16 (Uint16Array)
 * exactly as they sit on the GPU, plus the probe-grid metadata needed to validate
 * a restore targets a matching grid. `HybridEngine.exportGIState()` /
 * `importGIState()` produce/consume it; `serializeGIState` / `deserializeGIState`
 * round-trip it through a single transferable `ArrayBuffer` for storage.
 */

/** Magic + version for the binary container (bump VERSION on any layout change). */
const GI_SNAPSHOT_MAGIC = 0x47495353; // "GISS"
const GI_SNAPSHOT_VERSION = 1;
const HEADER_BYTES = 64; // fixed header, data blocks follow

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
}

/** Serialize a snapshot to a single ArrayBuffer (IndexedDB / file ready). */
export function serializeGIState(s: GIStateSnapshot): ArrayBuffer {
  const irrBytes = s.irrData.byteLength;
  const visBytes = s.visData.byteLength;
  const buf = new ArrayBuffer(HEADER_BYTES + irrBytes + visBytes);
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
  // (o = 52 here; header is padded to 64.)
  new Uint8Array(buf, HEADER_BYTES, irrBytes).set(new Uint8Array(s.irrData.buffer, s.irrData.byteOffset, irrBytes));
  new Uint8Array(buf, HEADER_BYTES + irrBytes, visBytes).set(new Uint8Array(s.visData.buffer, s.visData.byteOffset, visBytes));
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
  if (version !== GI_SNAPSHOT_VERSION) {
    throw new Error(`deserializeGIState: unsupported version ${version} (expected ${GI_SNAPSHOT_VERSION}).`);
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
  const irrBytes = irrW * irrH * 8; // rgba16float = 8 bytes/texel
  const visBytes = visW * visH * 8;
  if (buf.byteLength < HEADER_BYTES + irrBytes + visBytes) {
    throw new Error('deserializeGIState: buffer too small for the declared atlas dimensions.');
  }
  // Copy out (slice) so the returned arrays own their memory.
  const irrData = new Uint16Array(buf.slice(HEADER_BYTES, HEADER_BYTES + irrBytes));
  const visData = new Uint16Array(buf.slice(HEADER_BYTES + irrBytes, HEADER_BYTES + irrBytes + visBytes));
  return { dims: { x: dx, y: dy, z: dz }, origin: [ox, oy, oz], spacing, irrW, irrH, visW, visH, irrData, visData };
}
