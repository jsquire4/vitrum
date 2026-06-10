// glbParser.ts — Hand-rolled GLB binary container parser (glTF 2.0 §3.6.3).
//
// GLB layout:
//   Header  (12 bytes): magic 0x46546C67, version 2, total length
//   Chunk 0 (JSON):     chunkLength, chunkType 0x4E4F534A, chunkData
//   Chunk 1 (BIN, opt): chunkLength, chunkType 0x004E4942, chunkData
//
// Reference: glTF 2.0 spec §3.6.3 (Binary glTF Layout)

import type { GltfJson } from './gltfTypes.js';

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const CHUNK_TYPE_JSON = 0x4e4f534a;
const CHUNK_TYPE_BIN = 0x004e4942;

export interface GlbParseResult {
  json: GltfJson;
  /** Binary chunk data, or undefined if the GLB has no binary chunk. */
  binChunk: ArrayBuffer | undefined;
}

/**
 * Parse a GLB binary container. Throws on magic/version mismatch or truncated data.
 */
export function parseGlb(buffer: ArrayBuffer): GlbParseResult {
  const view = new DataView(buffer);

  if (buffer.byteLength < 12) {
    throw new Error('[vitrum/gltf-adapter] GLB: buffer too small for header');
  }

  const magic = view.getUint32(0, true);
  if (magic !== GLB_MAGIC) {
    throw new Error(
      `[vitrum/gltf-adapter] GLB: invalid magic 0x${magic.toString(16)} (expected 0x46546c67)`,
    );
  }

  const version = view.getUint32(4, true);
  if (version !== GLB_VERSION) {
    throw new Error(
      `[vitrum/gltf-adapter] GLB: unsupported version ${version} (only version 2 is supported)`,
    );
  }

  const totalLength = view.getUint32(8, true);
  if (totalLength > buffer.byteLength) {
    throw new Error(
      `[vitrum/gltf-adapter] GLB: declared length ${totalLength} exceeds buffer length ${buffer.byteLength}`,
    );
  }

  let offset = 12;
  let json: GltfJson | undefined;
  let binChunk: ArrayBuffer | undefined;

  while (offset + 8 <= totalLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    offset += 8;

    if (offset + chunkLength > totalLength) {
      throw new Error(
        `[vitrum/gltf-adapter] GLB: chunk at offset ${offset - 8} declares length ${chunkLength} but buffer ends at ${totalLength}`,
      );
    }

    if (chunkType === CHUNK_TYPE_JSON) {
      const jsonBytes = new Uint8Array(buffer, offset, chunkLength);
      const jsonString = new TextDecoder().decode(jsonBytes);
      json = JSON.parse(jsonString) as GltfJson;
    } else if (chunkType === CHUNK_TYPE_BIN) {
      binChunk = buffer.slice(offset, offset + chunkLength);
    }
    // Other chunk types are ignored per the spec.

    offset += chunkLength;
  }

  if (!json) {
    throw new Error('[vitrum/gltf-adapter] GLB: no JSON chunk found');
  }

  return { json, binChunk };
}
