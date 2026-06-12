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

/** Combine exact per-buffer fingerprints into one correctness-safe version tag. */
export function fingerprintBuffersExact(...parts: Array<ArrayBuffer | ArrayBufferView>): number {
  let h = FNV_OFFSET;
  for (const part of parts) {
    h = mixHash(h, fingerprintBufferExact(part));
  }
  return h >>> 0;
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
