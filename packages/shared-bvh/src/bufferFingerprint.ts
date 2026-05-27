/**
 * Fast content fingerprint for GPU buffer CPU mirrors (TLAS refit versioning).
 * Samples up to 64 KiB per buffer so transform-only refits bump the version
 * even when byte lengths are unchanged.
 */

const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;
const MAX_BYTES_PER_BUFFER = 65536;

function mixHash(h: number, value: number): number {
  return Math.imul(h ^ value, FNV_PRIME) >>> 0;
}

/** FNV-1a over buffer bytes (capped sample for large meshes). */
export function fingerprintBuffer(data: ArrayBuffer | ArrayBufferView): number {
  const view = data instanceof ArrayBuffer || data instanceof SharedArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const len = view.byteLength;
  let h = mixHash(FNV_OFFSET, len);
  if (len === 0) {
    return h;
  }
  if (len <= MAX_BYTES_PER_BUFFER) {
    for (let i = 0; i < len; i += 1) {
      h = mixHash(h, view[i]!);
    }
    return h;
  }
  const step = Math.max(1, Math.floor(len / MAX_BYTES_PER_BUFFER));
  for (let i = 0; i < len; i += step) {
    h = mixHash(h, view[i]!);
  }
  h = mixHash(h, view[len - 1]!);
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
