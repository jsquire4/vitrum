import type { RawImageHandle } from './textures.js';

export type RawImageFormat = 'png' | 'jpeg' | 'webp';

export interface RawImageDimensions {
  readonly format: RawImageFormat;
  readonly width: number;
  readonly height: number;
}

/**
 * A recognized raw-image container was malformed or did not expose dimensions
 * that can be validated before invoking a decoder.
 */
export class RawImageDimensionsError extends Error {
  readonly format: RawImageFormat;

  constructor(format: RawImageFormat, message: string) {
    super(`[vitrum/gltf-adapter] Invalid ${format.toUpperCase()} image header: ${message}`);
    this.name = 'RawImageDimensionsError';
    this.format = format;
  }
}

/**
 * Read dimensions from PNG IHDR, JPEG SOF, or WebP VP8/VP8L/VP8X headers.
 *
 * Returns `null` for an unsupported image format. A recognized but malformed
 * container throws so callers cannot silently fall through to an allocating
 * decoder without a trustworthy resource preflight.
 *
 * @internal
 */
export function readEncodedImageDimensions(
  data: Uint8Array,
  mimeType?: string,
): RawImageDimensions | null {
  const format = sniffRawImageFormat(data, mimeType);
  if (format === null) return null;
  if (format === 'png') return readPngDimensions(data);
  if (format === 'jpeg') return readJpegDimensions(data);
  return readWebpDimensions(data);
}

/** @internal */
export function readRawImageDimensions(handle: RawImageHandle): RawImageDimensions | null {
  return readEncodedImageDimensions(handle.data, handle.mimeType);
}

/** @internal */
export function sniffRawImageFormat(
  data: Uint8Array,
  mimeType?: string,
): RawImageFormat | null {
  if (hasPngSignature(data)) return 'png';
  if (hasJpegSignature(data)) return 'jpeg';
  if (hasWebpSignature(data)) return 'webp';

  switch (mimeType?.trim().toLowerCase()) {
    case 'image/png':
      return 'png';
    case 'image/jpeg':
    case 'image/jpg':
      return 'jpeg';
    case 'image/webp':
      return 'webp';
    default:
      return null;
  }
}

function readPngDimensions(data: Uint8Array): RawImageDimensions {
  if (!hasPngSignature(data)) {
    throw new RawImageDimensionsError('png', 'the 8-byte PNG signature is missing.');
  }
  if (data.length < 24) {
    throw new RawImageDimensionsError('png', 'the IHDR chunk is truncated.');
  }
  const ihdrLength = readUint32Be(data, 8);
  if (
    ihdrLength !== 13 ||
    data[12] !== 0x49 ||
    data[13] !== 0x48 ||
    data[14] !== 0x44 ||
    data[15] !== 0x52
  ) {
    throw new RawImageDimensionsError('png', 'IHDR must be the first chunk and have length 13.');
  }
  return checkedDimensions('png', readUint32Be(data, 16), readUint32Be(data, 20));
}

function readJpegDimensions(data: Uint8Array): RawImageDimensions {
  if (!hasJpegSignature(data)) {
    throw new RawImageDimensionsError('jpeg', 'the SOI marker is missing.');
  }

  let offset = 2;
  while (offset < data.length) {
    if (data[offset] !== 0xff) {
      throw new RawImageDimensionsError(
        'jpeg',
        `expected a marker prefix at byte ${offset}.`,
      );
    }
    while (offset < data.length && data[offset] === 0xff) offset += 1;
    if (offset >= data.length) {
      throw new RawImageDimensionsError('jpeg', 'a marker is truncated.');
    }

    const marker = data[offset]!;
    offset += 1;
    if (marker === 0x00) {
      throw new RawImageDimensionsError('jpeg', 'encountered an escaped data byte before SOS.');
    }
    if (marker === 0xd9 || marker === 0xda) {
      throw new RawImageDimensionsError('jpeg', 'no SOF dimensions were present before image data.');
    }
    if (
      marker === 0xd8 ||
      marker === 0x01 ||
      (marker >= 0xd0 && marker <= 0xd7)
    ) {
      continue;
    }
    if (offset + 2 > data.length) {
      throw new RawImageDimensionsError('jpeg', 'a segment length is truncated.');
    }

    const segmentLength = readUint16Be(data, offset);
    if (segmentLength < 2) {
      throw new RawImageDimensionsError('jpeg', `segment 0x${marker.toString(16)} has invalid length.`);
    }
    const segmentEnd = offset + segmentLength;
    if (segmentEnd > data.length) {
      throw new RawImageDimensionsError('jpeg', `segment 0x${marker.toString(16)} is truncated.`);
    }

    if (isJpegStartOfFrame(marker)) {
      if (segmentLength < 8) {
        throw new RawImageDimensionsError('jpeg', 'the SOF segment is too short.');
      }
      const height = readUint16Be(data, offset + 3);
      const width = readUint16Be(data, offset + 5);
      return checkedDimensions('jpeg', width, height);
    }
    offset = segmentEnd;
  }

  throw new RawImageDimensionsError('jpeg', 'no SOF dimensions were found.');
}

function readWebpDimensions(data: Uint8Array): RawImageDimensions {
  if (!hasWebpSignature(data)) {
    throw new RawImageDimensionsError('webp', 'the RIFF/WEBP signature is missing.');
  }
  if (data.length < 20) {
    throw new RawImageDimensionsError('webp', 'the first RIFF chunk header is truncated.');
  }

  const declaredRiffLength = readUint32Le(data, 4) + 8;
  if (declaredRiffLength < 20) {
    throw new RawImageDimensionsError('webp', 'the RIFF size is invalid.');
  }
  if (declaredRiffLength > data.length) {
    throw new RawImageDimensionsError('webp', 'the RIFF container is truncated.');
  }
  const containerEnd = declaredRiffLength;
  let offset = 12;
  while (offset + 8 <= containerEnd) {
    const chunkType = ascii4(data, offset);
    const chunkLength = readUint32Le(data, offset + 4);
    const payloadOffset = offset + 8;
    const payloadEnd = payloadOffset + chunkLength;
    if (!Number.isSafeInteger(payloadEnd) || payloadEnd > containerEnd) {
      throw new RawImageDimensionsError('webp', `${chunkType || 'unknown'} chunk is truncated.`);
    }

    if (chunkType === 'VP8X') {
      if (chunkLength < 10) {
        throw new RawImageDimensionsError('webp', 'the VP8X chunk is too short.');
      }
      const width = readUint24Le(data, payloadOffset + 4) + 1;
      const height = readUint24Le(data, payloadOffset + 7) + 1;
      return checkedDimensions('webp', width, height);
    }
    if (chunkType === 'VP8L') {
      if (chunkLength < 5 || data[payloadOffset] !== 0x2f) {
        throw new RawImageDimensionsError('webp', 'the VP8L signature or dimensions are truncated.');
      }
      const b0 = data[payloadOffset + 1]!;
      const b1 = data[payloadOffset + 2]!;
      const b2 = data[payloadOffset + 3]!;
      const b3 = data[payloadOffset + 4]!;
      const width = 1 + b0 + ((b1 & 0x3f) << 8);
      const height = 1 + (b1 >>> 6) + (b2 << 2) + ((b3 & 0x0f) << 10);
      return checkedDimensions('webp', width, height);
    }
    if (chunkType === 'VP8 ') {
      if (
        chunkLength < 10 ||
        data[payloadOffset + 3] !== 0x9d ||
        data[payloadOffset + 4] !== 0x01 ||
        data[payloadOffset + 5] !== 0x2a
      ) {
        throw new RawImageDimensionsError('webp', 'the VP8 key-frame header is invalid or truncated.');
      }
      const width =
        (data[payloadOffset + 6]! | (data[payloadOffset + 7]! << 8)) & 0x3fff;
      const height =
        (data[payloadOffset + 8]! | (data[payloadOffset + 9]! << 8)) & 0x3fff;
      return checkedDimensions('webp', width, height);
    }

    const paddedLength = chunkLength + (chunkLength & 1);
    const next = payloadOffset + paddedLength;
    if (!Number.isSafeInteger(next) || next <= offset) {
      throw new RawImageDimensionsError('webp', 'a RIFF chunk length overflowed.');
    }
    offset = next;
  }

  throw new RawImageDimensionsError('webp', 'no VP8, VP8L, or VP8X dimensions were found.');
}

function checkedDimensions(
  format: RawImageFormat,
  width: number,
  height: number,
): RawImageDimensions {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new RawImageDimensionsError(format, `dimensions must be positive integers, got ${width}x${height}.`);
  }
  return { format, width, height };
}

function isJpegStartOfFrame(marker: number): boolean {
  return marker >= 0xc0 &&
    marker <= 0xcf &&
    marker !== 0xc4 &&
    marker !== 0xc8 &&
    marker !== 0xcc;
}

function hasPngSignature(data: Uint8Array): boolean {
  return data.length >= 8 &&
    data[0] === 0x89 &&
    data[1] === 0x50 &&
    data[2] === 0x4e &&
    data[3] === 0x47 &&
    data[4] === 0x0d &&
    data[5] === 0x0a &&
    data[6] === 0x1a &&
    data[7] === 0x0a;
}

function hasJpegSignature(data: Uint8Array): boolean {
  return data.length >= 3 &&
    data[0] === 0xff &&
    data[1] === 0xd8 &&
    data[2] === 0xff;
}

function hasWebpSignature(data: Uint8Array): boolean {
  return data.length >= 12 &&
    data[0] === 0x52 &&
    data[1] === 0x49 &&
    data[2] === 0x46 &&
    data[3] === 0x46 &&
    data[8] === 0x57 &&
    data[9] === 0x45 &&
    data[10] === 0x42 &&
    data[11] === 0x50;
}

function readUint16Be(data: Uint8Array, offset: number): number {
  return (data[offset]! << 8) | data[offset + 1]!;
}

function readUint24Le(data: Uint8Array, offset: number): number {
  return data[offset]! | (data[offset + 1]! << 8) | (data[offset + 2]! << 16);
}

function readUint32Be(data: Uint8Array, offset: number): number {
  return (
    data[offset]! * 0x1000000 +
    (data[offset + 1]! << 16) +
    (data[offset + 2]! << 8) +
    data[offset + 3]!
  );
}

function readUint32Le(data: Uint8Array, offset: number): number {
  return (
    data[offset]! +
    (data[offset + 1]! << 8) +
    (data[offset + 2]! << 16) +
    data[offset + 3]! * 0x1000000
  );
}

function ascii4(data: Uint8Array, offset: number): string {
  return String.fromCharCode(
    data[offset]!,
    data[offset + 1]!,
    data[offset + 2]!,
    data[offset + 3]!,
  );
}
