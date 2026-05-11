/**
 * Golden HDR accumulation strip for filesystem-backed regression tests.
 * Bytes match `fixtures/hdrAccumGolden.bin`: 2×2 RGBA float32 texels, row-major.
 */

/** Same bytes as `fixtures/hdrAccumGolden.bin` (little-endian float32 × 16). */
export const HDR_ACCUM_GOLDEN_BASE64 =
  'AADAQAAAQEEAAJBBAABAQAAAxkIAAMZCAADGQgAAAAAAAMA/AAAgQAAAYEAAAMhCAAAgQQAAoEEAAPBBAACgQA==' as const;

/** Expected float32 RGB after `accumulationFloatRgbaToRgb(..., divideByAlpha: true)` for the golden texels. */
export const HDR_ACCUM_GOLDEN_EXPECTED_RGB_DIVIDE = new Float32Array([
  2, 4, 6,
  0, 0, 0,
  0.015, 0.025, 0.035,
  2, 4, 6,
]);

export const HDR_ACCUM_GOLDEN_PIXEL_COUNT = 4 as const;
export const HDR_ACCUM_GOLDEN_BYTE_LENGTH = 64 as const;

export function decodeHdrAccumGoldenBin(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength !== HDR_ACCUM_GOLDEN_BYTE_LENGTH) {
    throw new Error(
      `decodeHdrAccumGoldenBin: expected ${HDR_ACCUM_GOLDEN_BYTE_LENGTH} bytes, got ${buffer.byteLength}`,
    );
  }
  return new Float32Array(buffer.slice(0));
}

export function hdrAccumGoldenBinFromBase64(b64: string = HDR_ACCUM_GOLDEN_BASE64): Float32Array {
  const binary = globalThis.atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i) & 0xff;
  }
  return decodeHdrAccumGoldenBin(bytes.buffer);
}
