/**
 * weights.ts — ModelWeights interface + binary weight loader for the vitrum neural denoiser.
 *
 * Binary format (.vitrum-model):
 * ─────────────────────────────
 * Header:
 *   [u32 magic=0xDEAF1984, u32 version=1, u32 layerCount]
 * Per layer:
 *   [u32 nameLen, char[nameLen] name (UTF-8),
 *    u32 weightCount, f32[weightCount] weights,
 *    u32 biasCount,   f32[biasCount]   biases]
 *
 * Weight layout per layer kind:
 *   Conv2D:          OIKW  (outputC × inputC × kH × kW)  — matches PyTorch Conv2d
 *   ConvTranspose2D: IOKW  (inputC × outputC × kH × kW) — matches PyTorch ConvTranspose2d
 *   All others:      No weights (empty arrays).
 *
 * This format is round-tripped by tools/neural-denoiser-training/export_weights.py.
 */

export const VITRUM_MODEL_MAGIC   = 0xDEAF1984 >>> 0;
export const VITRUM_MODEL_VERSION = 1;

// ── ModelWeights ──────────────────────────────────────────────────────────────

/** Per-layer weight payload. */
export interface LayerWeights {
  /** Layer name matching the LayerSpec.name in unetArchitecture.ts. */
  readonly name: string;
  /** Weight tensor as Float32Array. Length = 0 for non-parameterized layers. */
  readonly weights: Float32Array;
  /** Bias tensor as Float32Array. Length = 0 for layers with no bias. */
  readonly biases: Float32Array;
}

/** Full model weights for one U-Net checkpoint. */
export interface ModelWeights {
  /** Layer weights in execution order (matches UNetSpec.layers order). */
  readonly layers: readonly LayerWeights[];
}

// ── Binary loader ─────────────────────────────────────────────────────────────

/**
 * Load model weights from an ArrayBuffer in the vitrum-model binary format.
 *
 * @throws {Error} on magic mismatch, unsupported version, or truncated data.
 *
 * Byte-level layout:
 *   Offset 0: u32 magic (little-endian)
 *   Offset 4: u32 version
 *   Offset 8: u32 layerCount
 *   Offset 12: layer records (variable-length)
 *
 * Each layer record:
 *   u32 nameLen
 *   u8[nameLen] name (UTF-8, not null-terminated)
 *   u32 weightCount
 *   f32[weightCount] weights
 *   u32 biasCount
 *   f32[biasCount] biases
 */
export function loadWeightsFromArrayBuffer(bytes: ArrayBuffer): ModelWeights {
  const view = new DataView(bytes);
  let offset = 0;

  function readU32(): number {
    if (offset + 4 > bytes.byteLength) {
      throw new Error(`[loadWeightsFromArrayBuffer] truncated at offset ${offset}`);
    }
    const v = view.getUint32(offset, /* littleEndian */ true);
    offset += 4;
    return v;
  }

  function readF32Array(count: number): Float32Array {
    const byteLen = count * 4;
    if (offset + byteLen > bytes.byteLength) {
      throw new Error(
        `[loadWeightsFromArrayBuffer] truncated reading ${count} f32s at offset ${offset}`,
      );
    }
    // Slice to own the memory (avoids holding a reference to the full ArrayBuffer).
    const arr = new Float32Array(bytes.slice(offset, offset + byteLen));
    offset += byteLen;
    return arr;
  }

  function readString(len: number): string {
    if (offset + len > bytes.byteLength) {
      throw new Error(
        `[loadWeightsFromArrayBuffer] truncated reading string of length ${len} at offset ${offset}`,
      );
    }
    const arr = new Uint8Array(bytes, offset, len);
    offset += len;
    return new TextDecoder().decode(arr);
  }

  // ── Header ────────────────────────────────────────────────────────────────
  const magic = readU32();
  if (magic !== VITRUM_MODEL_MAGIC) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] invalid magic: expected 0x${VITRUM_MODEL_MAGIC.toString(16).toUpperCase()}, ` +
      `got 0x${magic.toString(16).toUpperCase()}`,
    );
  }

  const version = readU32();
  if (version !== VITRUM_MODEL_VERSION) {
    throw new Error(
      `[loadWeightsFromArrayBuffer] unsupported version ${version} (expected ${VITRUM_MODEL_VERSION})`,
    );
  }

  const layerCount = readU32();
  const layers: LayerWeights[] = [];

  // ── Layer records ─────────────────────────────────────────────────────────
  for (let i = 0; i < layerCount; i++) {
    const nameLen     = readU32();
    const name        = readString(nameLen);
    const weightCount = readU32();
    const weights     = readF32Array(weightCount);
    const biasCount   = readU32();
    const biases      = readF32Array(biasCount);
    layers.push({ name, weights, biases });
  }

  return { layers };
}

// ── Binary serialiser (CPU-side mirror of export_weights.py) ──────────────────

/**
 * Serialize ModelWeights to the vitrum-model binary format.
 *
 * This is a TypeScript mirror of `tools/neural-denoiser-training/export_weights.py`'s
 * output stage, used in tests for round-trip validation.
 */
export function serializeWeightsToArrayBuffer(weights: ModelWeights): ArrayBuffer {
  // Calculate total size.
  let totalBytes = 12; // magic + version + layerCount
  for (const layer of weights.layers) {
    const nameBytes = new TextEncoder().encode(layer.name);
    totalBytes += 4;                          // nameLen
    totalBytes += nameBytes.byteLength;       // name
    totalBytes += 4;                          // weightCount
    totalBytes += layer.weights.length * 4;   // weights (f32)
    totalBytes += 4;                          // biasCount
    totalBytes += layer.biases.length * 4;    // biases (f32)
  }

  const buf  = new ArrayBuffer(totalBytes);
  const view = new DataView(buf);
  const u8   = new Uint8Array(buf);
  let offset = 0;

  function writeU32(v: number): void {
    view.setUint32(offset, v >>> 0, /* littleEndian */ true);
    offset += 4;
  }

  function writeF32Array(arr: Float32Array): void {
    u8.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), offset);
    offset += arr.byteLength;
  }

  function writeString(s: string): void {
    const encoded = new TextEncoder().encode(s);
    writeU32(encoded.byteLength);
    u8.set(encoded, offset);
    offset += encoded.byteLength;
  }

  // ── Header ────────────────────────────────────────────────────────────────
  writeU32(VITRUM_MODEL_MAGIC);
  writeU32(VITRUM_MODEL_VERSION);
  writeU32(weights.layers.length);

  // ── Layer records ─────────────────────────────────────────────────────────
  for (const layer of weights.layers) {
    writeString(layer.name);
    writeU32(layer.weights.length);
    writeF32Array(layer.weights);
    writeU32(layer.biases.length);
    writeF32Array(layer.biases);
  }

  return buf;
}
