// glbParser.ts — Hand-rolled GLB binary container parser (glTF 2.0 §3.6.3).
//
// GLB layout:
//   Header  (12 bytes): magic 0x46546C67, version 2, total length
//   Chunk 0 (JSON):     chunkLength, chunkType 0x4E4F534A, chunkData
//   Chunk 1 (BIN, opt): chunkLength, chunkType 0x004E4942, chunkData
//
// Reference: glTF 2.0 spec §3.6.3 (Binary glTF Layout)

import type { GltfJson } from './gltfTypes.js';
import { GltfParseFailed, type GltfParseFailedInit, type GltfParseFailureReason } from './errors.js';
import { gltfArrayBufferByteLength } from './importResourceBudget.js';

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const CHUNK_TYPE_JSON = 0x4e4f534a;
const CHUNK_TYPE_BIN = 0x004e4942;
// eslint-disable-next-line @typescript-eslint/unbound-method -- Invoked with an explicit, brand-checked ArrayBuffer receiver.
const ARRAY_BUFFER_SLICE = ArrayBuffer.prototype.slice;
const UTF8_FATAL_DECODER = new TextDecoder('utf-8', { fatal: true });
// eslint-disable-next-line @typescript-eslint/unbound-method -- Invoked with the captured decoder receiver.
const TEXT_DECODER_DECODE = TextDecoder.prototype.decode;

export interface GlbParseResult {
  json: GltfJson;
  /** Binary chunk data, or undefined if the GLB has no binary chunk. */
  binChunk: ArrayBuffer | undefined;
}

export interface GlbBinChunkCopyInfo {
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface GlbParseOptions {
  /**
   * Called after the BIN declaration is structurally validated, immediately
   * before the parser allocates its detached copy.
   */
  readonly beforeBinChunkCopy?: (info: GlbBinChunkCopyInfo) => void;
}

/** Strict UTF-8 decode shared by GLB JSON chunks and raw `.gltf` byte inputs. */
export function decodeGltfUtf8(input: ArrayBuffer | Uint8Array): string {
  return Reflect.apply(
    TEXT_DECODER_DECODE,
    UTF8_FATAL_DECODER,
    [input],
  );
}

/**
 * Parse a GLB binary container. Throws typed `GltfParseFailed` on malformed data.
 */
export function parseGlb(
  buffer: ArrayBuffer,
  options: GlbParseOptions = {},
): GlbParseResult {
  const actualLength = gltfArrayBufferByteLength(buffer);
  if (actualLength === undefined) {
    throw glbError(
      'glb-header-too-small',
      '[vitrum/gltf-adapter] GLB: input is not a genuine ArrayBuffer',
    );
  }
  if (actualLength < 12) {
    throw glbError('glb-header-too-small', '[vitrum/gltf-adapter] GLB: buffer too small for header', {
      actualLength,
    });
  }
  const view = new DataView(buffer);

  const magic = view.getUint32(0, true);
  if (magic !== GLB_MAGIC) {
    throw glbError(
      'glb-invalid-magic',
      `[vitrum/gltf-adapter] GLB: invalid magic 0x${magic.toString(16)} (expected 0x46546c67)`,
      { byteOffset: 0 },
    );
  }

  const version = view.getUint32(4, true);
  if (version !== GLB_VERSION) {
    throw glbError(
      'glb-unsupported-version',
      `[vitrum/gltf-adapter] GLB: unsupported version ${version} (only version 2 is supported)`,
      { byteOffset: 4, version },
    );
  }

  const totalLength = view.getUint32(8, true);
  if (totalLength > actualLength) {
    throw glbError(
      'glb-declared-length-exceeds-buffer',
      `[vitrum/gltf-adapter] GLB: declared length ${totalLength} exceeds buffer length ${actualLength}`,
      { byteOffset: 8, declaredLength: totalLength, actualLength },
    );
  }
  if (totalLength !== actualLength) {
    throw glbError(
      'glb-declared-length-mismatch',
      `[vitrum/gltf-adapter] GLB: declared length ${totalLength} does not equal buffer length ${actualLength}`,
      { byteOffset: 8, declaredLength: totalLength, actualLength },
    );
  }
  if (totalLength % 4 !== 0) {
    throw glbError(
      'glb-invalid-chunk-alignment',
      `[vitrum/gltf-adapter] GLB: total length ${totalLength} is not aligned to 4 bytes.`,
      { byteOffset: 8, declaredLength: totalLength, actualLength },
    );
  }

  let offset = 12;
  let json: GltfJson | undefined;
  let binChunk: ArrayBuffer | undefined;
  let chunkIndex = 0;

  while (offset < totalLength) {
    if (totalLength - offset < 8) {
      throw glbError(
        'glb-trailing-bytes',
        `[vitrum/gltf-adapter] GLB: ${totalLength - offset} trailing byte(s) remain without a complete chunk header.`,
        { byteOffset: offset, declaredLength: totalLength, actualLength },
      );
    }
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const chunkHeaderOffset = offset;
    offset += 8;

    if (chunkLength % 4 !== 0) {
      throw glbError(
        'glb-invalid-chunk-alignment',
        `[vitrum/gltf-adapter] GLB: chunk at offset ${chunkHeaderOffset} has non-aligned length ${chunkLength}.`,
        { byteOffset: chunkHeaderOffset, chunkLength, declaredLength: totalLength, actualLength },
      );
    }
    if (offset + chunkLength > totalLength) {
      throw glbError(
        'glb-chunk-out-of-bounds',
        `[vitrum/gltf-adapter] GLB: chunk at offset ${offset - 8} declares length ${chunkLength} but buffer ends at ${totalLength}`,
        { byteOffset: offset - 8, chunkLength, declaredLength: totalLength, actualLength },
      );
    }

    if (chunkType === CHUNK_TYPE_JSON) {
      if (json !== undefined) {
        throw glbError(
          'glb-duplicate-chunk',
          `[vitrum/gltf-adapter] GLB: duplicate JSON chunk at offset ${chunkHeaderOffset}.`,
          { byteOffset: chunkHeaderOffset, chunkLength },
        );
      }
      if (chunkIndex !== 0) {
        throw glbError(
          'glb-invalid-chunk-order',
          `[vitrum/gltf-adapter] GLB: JSON must be the first chunk.`,
          { byteOffset: chunkHeaderOffset, chunkLength },
        );
      }
      const jsonBytes = new Uint8Array(buffer, offset, chunkLength);
      try {
        const jsonString = decodeGltfUtf8(jsonBytes);
        json = JSON.parse(jsonString) as GltfJson;
      } catch (cause) {
        throw glbError(
          'glb-json-parse-failed',
          `[vitrum/gltf-adapter] GLB: JSON chunk at offset ${offset} is not valid JSON.`,
          { byteOffset: offset, chunkLength, cause },
        );
      }
    } else if (chunkType === CHUNK_TYPE_BIN) {
      if (binChunk !== undefined) {
        throw glbError(
          'glb-duplicate-chunk',
          `[vitrum/gltf-adapter] GLB: duplicate BIN chunk at offset ${chunkHeaderOffset}.`,
          { byteOffset: chunkHeaderOffset, chunkLength },
        );
      }
      if (chunkIndex !== 1 || json === undefined) {
        throw glbError(
          'glb-invalid-chunk-order',
          `[vitrum/gltf-adapter] GLB: BIN must be the second chunk, after JSON.`,
          { byteOffset: chunkHeaderOffset, chunkLength },
        );
      }
      options.beforeBinChunkCopy?.({
        byteOffset: offset,
        byteLength: chunkLength,
      });
      binChunk = Reflect.apply(
        ARRAY_BUFFER_SLICE,
        buffer,
        [offset, offset + chunkLength],
      );
    }
    // Other chunk types are ignored per the spec.

    offset += chunkLength;
    chunkIndex += 1;
  }

  if (!json) {
    throw glbError('glb-json-missing', '[vitrum/gltf-adapter] GLB: no JSON chunk found');
  }

  return { json, binChunk };
}

function glbError(
  reason: Exclude<GltfParseFailureReason, 'json-parse-failed'>,
  message: string,
  details: Omit<GltfParseFailedInit, 'format' | 'reason' | 'message'> = {},
): GltfParseFailed {
  return new GltfParseFailed({
    format: 'glb',
    reason,
    message,
    ...details,
  });
}
