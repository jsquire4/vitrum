import type { GltfJson } from './gltfTypes.js';

export interface DeclaredBufferRange {
  readonly bufferIndex: number;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly end: number;
  readonly declaredByteLength: number;
}

function safeInteger(value: unknown, path: string, minimum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} must be a safe integer >= ${minimum}; ` +
        `received ${String(value)}.`,
    );
  }
  return value as number;
}

/** Validate a byte range against the glTF descriptor, not merely loaded bytes. */
export function validateDeclaredBufferRange(
  gltf: GltfJson,
  bufferIndexValue: unknown,
  byteOffsetValue: unknown,
  byteLengthValue: unknown,
  path: string,
): DeclaredBufferRange {
  const bufferIndex = safeInteger(bufferIndexValue, `${path}.buffer`, 0);
  const byteOffset = safeInteger(byteOffsetValue, `${path}.byteOffset`, 0);
  const byteLength = safeInteger(byteLengthValue, `${path}.byteLength`, 0);
  const descriptor = gltf.buffers?.[bufferIndex];
  if (descriptor == null) {
    throw new Error(
      `[vitrum/gltf-adapter] ${path} references missing buffer descriptor ` + `${bufferIndex}.`,
    );
  }
  const declaredByteLength = safeInteger(
    descriptor.byteLength,
    `buffers[${bufferIndex}].byteLength`,
    0,
  );
  const end = byteOffset + byteLength;
  if (!Number.isSafeInteger(end)) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path} byte range end is not a safe integer.`);
  }
  if (end > declaredByteLength) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} range [${byteOffset}, ${end}) exceeds ` +
        `declared buffers[${bufferIndex}].byteLength ${declaredByteLength}.`,
    );
  }
  return {
    bufferIndex,
    byteOffset,
    byteLength,
    end,
    declaredByteLength,
  };
}
