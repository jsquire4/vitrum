/**
 * Fast content fingerprint for GPU buffer CPU mirrors (TLAS refit versioning).
 * Samples up to 64 KiB per buffer so transform-only refits bump the version
 * even when byte lengths are unchanged.
 *
 * **Sampling guarantee (large buffers):** for buffers larger than 64 KiB, the
 * sampler uses a fixed stride = `max(1, floor(len / 65536))` bytes, which means
 * exactly `floor(len / stride)` evenly-spaced bytes are hashed, ALWAYS including
 * the first byte (index 0) and the last byte (index `len - 1`). A single-byte
 * change at a sampled offset will always be detected. A single-byte change at an
 * UNsampled interior offset (gap between two stride-aligned samples) will be
 * missed with probability `(stride - 1) / stride ≈ 1 - 65536/len`. For a 1 MiB
 * buffer (stride = 16) this is ~93.75% miss probability for a single interior
 * byte. Use this fingerprint only where a missed interior byte cannot skip
 * required correctness work. For rebuild-skip gates, use
 * {@link fingerprintBufferExact}.
 */

const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;
const MAX_SAMPLE_POINTS = 65536;

function mixHash(h: number, value: number): number {
  return Math.imul(h ^ value, FNV_PRIME) >>> 0;
}

function byteView(data: ArrayBuffer | ArrayBufferView): Uint8Array {
  return data instanceof ArrayBuffer || data instanceof SharedArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

/** FNV-1a over buffer bytes (evenly-spaced sampled path for large buffers).
 *
 * Small buffers (≤ {@link MAX_SAMPLE_POINTS} bytes): exact — every byte is hashed.
 * Large buffers: fixed stride = `max(1, floor(len / MAX_SAMPLE_POINTS))`.
 * First and last bytes are always included.
 */
export function fingerprintBuffer(data: ArrayBuffer | ArrayBufferView): number {
  const view = byteView(data);
  const len = view.byteLength;
  let h = mixHash(FNV_OFFSET, len);
  if (len === 0) {
    return h;
  }
  if (len <= MAX_SAMPLE_POINTS) {
    // Exact path: hash every byte.
    for (let i = 0; i < len; i += 1) {
      h = mixHash(h, view[i]!);
    }
    return h;
  }
  // Sampled path: evenly-spaced stride, always including first and last bytes.
  const stride = Math.max(1, Math.floor(len / MAX_SAMPLE_POINTS));
  for (let i = 0; i < len; i += stride) {
    h = mixHash(h, view[i]!);
  }
  // Always include the last byte (may already be included if len-1 is stride-aligned).
  h = mixHash(h, view[len - 1]!);
  return h;
}

/** Exact FNV-1a over every byte.
 *
 * This is the correct helper for rebuild-skip / stale-geometry correctness
 * gates. It is intentionally separate from {@link fingerprintBuffer}, whose
 * sampled large-buffer path is retained for versioning and upload heuristics.
 */
export function fingerprintBufferExact(data: ArrayBuffer | ArrayBufferView): number {
  const view = byteView(data);
  let h = mixHash(FNV_OFFSET, view.byteLength);
  for (let i = 0; i < view.byteLength; i += 1) {
    h = mixHash(h, view[i]!);
  }
  return h;
}

/** Combine per-buffer fingerprints into one version tag. */
export function fingerprintBuffers(...parts: Array<ArrayBuffer | ArrayBufferView>): number {
  let h = FNV_OFFSET;
  for (const part of parts) {
    h = mixHash(h, fingerprintBuffer(part));
  }
  return h >>> 0;
}

/**
 * Combine full-byte per-buffer fingerprints into one compact version tag.
 *
 * Every input byte participates, but the 32-bit result is still a hash and can
 * collide. A correctness-sensitive skip gate must retain and compare the
 * underlying state after a hash match (see {@link packedSceneBvhStateEqual}).
 */
export function fingerprintBuffersExact(...parts: Array<ArrayBuffer | ArrayBufferView>): number {
  let h = FNV_OFFSET;
  for (const part of parts) {
    h = mixHash(h, fingerprintBufferExact(part));
  }
  return h >>> 0;
}

/**
 * Exact byte state published by {@link SceneBvh}.
 *
 * `materialEntries` is the canonical 64-byte `MaterialEntry` payload produced
 * by `coreMaterialToMaterialEntry` + `packMaterials`; its final word is a true
 * u32 containing every representation flag. Keeping that already-packed byte
 * stream in the fingerprint prevents a second, manually maintained material
 * field list from drifting away from the GPU ABI.
 */
export interface PackedSceneBvhFingerprintState {
  readonly bvhNodes: ArrayBuffer | ArrayBufferView;
  readonly positions: ArrayBuffer | ArrayBufferView;
  readonly indices: ArrayBuffer | ArrayBufferView;
  readonly normals: ArrayBuffer | ArrayBufferView;
  readonly triMaterialId: ArrayBuffer | ArrayBufferView;
  readonly materialEntries: ArrayBuffer | ArrayBufferView;
  /**
   * Ordered canonical signatures for the raw `materials/coreMaterials` list
   * published alongside the byte buffers. Some behavior-affecting state (for
   * example texture-handle identity, sampler metadata, layered normal maps,
   * and emitter-classification extensions) is intentionally not represented
   * by the compact DDGI `MaterialEntry` payload.
   */
  readonly materialSignatures?: readonly string[];
}

function fingerprintStringExact(value: string): number {
  let h = mixHash(FNV_OFFSET, value.length);
  // Hash UTF-16 code units as two explicit bytes. This is deterministic across
  // hosts and distinguishes every JavaScript string without a TextEncoder
  // allocation on the rebuild hot path.
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    h = mixHash(h, code & 0xff);
    h = mixHash(h, code >>> 8);
  }
  return h;
}

function fingerprintStringsExact(values: readonly string[]): number {
  let h = mixHash(FNV_OFFSET, values.length);
  for (const value of values) {
    h = mixHash(h, fingerprintStringExact(value));
  }
  return h >>> 0;
}

/**
 * Full-input fingerprint of every published SceneBvh buffer, canonical packed
 * material bytes, and raw-material behavior signatures. Byte-identical fresh
 * and incrementally updated CPU mirrors intentionally produce the same value.
 *
 * This is a prefilter, not an equality proof; SceneBvh follows a match with
 * {@link packedSceneBvhStateEqual}.
 */
export function fingerprintPackedSceneBvhState(
  state: PackedSceneBvhFingerprintState,
): number {
  let h = fingerprintBuffersExact(
    state.bvhNodes,
    state.positions,
    state.indices,
    state.normals,
    state.triMaterialId,
    state.materialEntries,
  );
  h = mixHash(h, fingerprintStringsExact(state.materialSignatures ?? []));
  return h >>> 0;
}

function byteEqual(
  a: ArrayBuffer | ArrayBufferView,
  b: ArrayBuffer | ArrayBufferView,
): boolean {
  const aBytes = byteView(a);
  const bBytes = byteView(b);
  if (aBytes.byteLength !== bBytes.byteLength) return false;
  for (let i = 0; i < aBytes.byteLength; i += 1) {
    if (aBytes[i] !== bBytes[i]) return false;
  }
  return true;
}

function stringArrayEqual(
  a: readonly string[],
  b: readonly string[],
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Collision-safe equality for the complete state used by SceneBvh's rebuild
 * skip gate. The 32-bit fingerprint is only a prefilter; publication is skipped
 * only after every retained byte and ordered material signature compares equal.
 */
export function packedSceneBvhStateEqual(
  a: PackedSceneBvhFingerprintState,
  b: PackedSceneBvhFingerprintState,
): boolean {
  return (
    byteEqual(a.bvhNodes, b.bvhNodes) &&
    byteEqual(a.positions, b.positions) &&
    byteEqual(a.indices, b.indices) &&
    byteEqual(a.normals, b.normals) &&
    byteEqual(a.triMaterialId, b.triMaterialId) &&
    byteEqual(a.materialEntries, b.materialEntries) &&
    stringArrayEqual(a.materialSignatures ?? [], b.materialSignatures ?? [])
  );
}

/** True when only TLAS payload changed (transform-only refit; BLAS concat stable). */
export function isTlasOnlyVersionBump(
  blasVersion: number,
  tlasVersion: number,
  prev: { readonly blasContentVersion: number; readonly tlasContentVersion: number },
): boolean {
  return (
    blasVersion === prev.blasContentVersion &&
    tlasVersion !== prev.tlasContentVersion
  );
}

/** Fingerprint the five TLAS CPU mirrors (pt-webgpu / hybrid versioning). */
export function fingerprintTlasBuffers(tlas: {
  readonly tlasNodes: ArrayBuffer | ArrayBufferView;
  readonly tlasInstanceIndices: ArrayBuffer | ArrayBufferView;
  readonly tlasBlasRoots: ArrayBuffer | ArrayBufferView;
  readonly tlasInstanceWorldToLocal: ArrayBuffer | ArrayBufferView;
  readonly tlasInstanceLocalToWorld: ArrayBuffer | ArrayBufferView;
}): number {
  return fingerprintBuffers(
    tlas.tlasNodes,
    tlas.tlasInstanceIndices,
    tlas.tlasBlasRoots,
    tlas.tlasInstanceWorldToLocal,
    tlas.tlasInstanceLocalToWorld,
  );
}
