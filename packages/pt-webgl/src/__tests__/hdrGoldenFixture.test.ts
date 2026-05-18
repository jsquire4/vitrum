import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { accumulationFloatRgbaToRgb } from '../readbackHdr.js';
import {
  HDR_ACCUM_GOLDEN_BASE64,
  HDR_ACCUM_GOLDEN_BYTE_LENGTH,
  HDR_ACCUM_GOLDEN_EXPECTED_RGB_DIVIDE,
  HDR_ACCUM_GOLDEN_PIXEL_COUNT,
  decodeHdrAccumGoldenBin,
  hdrAccumGoldenBinFromBase64,
} from '../hdrGoldenFixture.js';

const fixtureDir = dirname(fileURLToPath(import.meta.url));

describe('HDR accumulation golden fixture', () => {
  it('decoded .bin matches base64 strip', () => {
    const path = join(fixtureDir, '..', '..', 'fixtures', 'hdrAccumGolden.bin');
    const disk = readFileSync(path);
    expect(disk.byteLength).toBe(HDR_ACCUM_GOLDEN_BYTE_LENGTH);
    const fromDisk = decodeHdrAccumGoldenBin(
      disk.buffer.slice(disk.byteOffset, disk.byteOffset + disk.byteLength),
    );
    const fromB64 = hdrAccumGoldenBinFromBase64(HDR_ACCUM_GOLDEN_BASE64);
    expect(Array.from(fromDisk)).toEqual(Array.from(fromB64));
  });

  it('base64 decodes to four RGBA texels with expected divide-by-alpha RGB', () => {
    const rgba = hdrAccumGoldenBinFromBase64();
    expect(rgba.length).toBe(HDR_ACCUM_GOLDEN_PIXEL_COUNT * 4);
    const rgb = accumulationFloatRgbaToRgb(rgba, HDR_ACCUM_GOLDEN_PIXEL_COUNT, true);
    expect(rgb.length).toBe(HDR_ACCUM_GOLDEN_PIXEL_COUNT * 3);
    for (let i = 0; i < rgb.length; i += 1) {
      expect(rgb[i]).toBeCloseTo(HDR_ACCUM_GOLDEN_EXPECTED_RGB_DIVIDE[i]!, 6);
    }
  });
});
