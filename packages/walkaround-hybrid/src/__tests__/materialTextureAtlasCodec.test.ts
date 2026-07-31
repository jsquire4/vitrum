import { describe, expect, it } from 'vitest';
import {
  MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT,
  MATERIAL_ATLAS_ENCODING_RGBA16_SNORM,
  MATERIAL_ATLAS_ENCODING_RGBA16_UNORM,
  MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
  MATERIAL_ATLAS_ENCODING_RGBA8_SNORM,
  MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
  generateMaterialTextureAtlasMip,
  materialTextureAtlasFloatToHalf,
  materialTextureAtlasHalfToFloat,
  packMaterialTextureAtlasPixels,
  unpackMaterialTextureAtlasPixels,
  type MaterialTextureAtlasEncoding,
} from '../bvh/materialTextureAtlasCodec.js';

const ENCODINGS: readonly {
  readonly encoding: MaterialTextureAtlasEncoding;
  readonly input: readonly number[];
  readonly tolerance: number;
}[] = [
  {
    encoding: MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
    input: [0, 0.25, 0.75, 1],
    tolerance: 1 / 255,
  },
  {
    encoding: MATERIAL_ATLAS_ENCODING_RGBA8_SNORM,
    input: [-1, -0.25, 0.75, 1],
    tolerance: 1 / 127,
  },
  {
    encoding: MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT,
    input: [-10, -0.25, 0.75, 1024],
    tolerance: 1 / 1024,
  },
  {
    encoding: MATERIAL_ATLAS_ENCODING_RGBA16_UNORM,
    input: [0, 0.25, 0.75, 1],
    tolerance: 1 / 65535,
  },
  {
    encoding: MATERIAL_ATLAS_ENCODING_RGBA16_SNORM,
    input: [-1, -0.25, 0.75, 1],
    tolerance: 1 / 32767,
  },
  {
    encoding: MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
    input: [-10, -0.25, 0.75, 1024],
    tolerance: 0,
  },
];

describe('material texture atlas codecs', () => {
  for (const { encoding, input, tolerance } of ENCODINGS) {
    it(`round-trips encoding ${encoding} within its declared quantisation`, () => {
      const decoded = unpackMaterialTextureAtlasPixels(
        packMaterialTextureAtlasPixels(Float32Array.from(input), encoding),
        encoding,
      );
      input.forEach((expected, index) => {
        expect(Math.abs(decoded[index]! - expected)).toBeLessThanOrEqual(tolerance);
      });
    });
  }

  it('covers finite, subnormal, overflow, infinity, and NaN half-float classes', () => {
    expect(materialTextureAtlasHalfToFloat(materialTextureAtlasFloatToHalf(1))).toBe(1);
    expect(materialTextureAtlasHalfToFloat(materialTextureAtlasFloatToHalf(2 ** -24)))
      .toBe(2 ** -24);
    expect(materialTextureAtlasHalfToFloat(materialTextureAtlasFloatToHalf(65520)))
      .toBe(Number.POSITIVE_INFINITY);
    expect(materialTextureAtlasHalfToFloat(materialTextureAtlasFloatToHalf(-65520)))
      .toBe(Number.NEGATIVE_INFINITY);
    expect(materialTextureAtlasHalfToFloat(materialTextureAtlasFloatToHalf(Number.POSITIVE_INFINITY)))
      .toBe(Number.POSITIVE_INFINITY);
    expect(materialTextureAtlasHalfToFloat(materialTextureAtlasFloatToHalf(Number.NaN)))
      .toBeNaN();
  });

  it('rounds the minimum-subnormal halfway interval to nearest-even', () => {
    const halfway = 2 ** -25;
    const nextF32 = halfway + 2 ** -48;
    expect(materialTextureAtlasFloatToHalf(halfway)).toBe(0);
    expect(materialTextureAtlasFloatToHalf(nextF32)).toBe(1);
    expect(materialTextureAtlasFloatToHalf(-nextF32)).toBe(0x8001);
  });

  it('preserves a minimum half subnormal produced by odd-footprint mip averaging', () => {
    const minimumHalf = 2 ** -24;
    const source = packMaterialTextureAtlasPixels(
      new Float32Array([
        minimumHalf, 0, 0, 1,
        minimumHalf, 0, 0, 1,
        0, 0, 0, 1,
      ]),
      MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT,
    );
    const mip = generateMaterialTextureAtlasMip(
      source,
      3,
      1,
      MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT,
      false,
    );
    expect(mip.data[0]! & 0xffff).toBe(1);
  });

  it('averages odd-sized mip footprints without dropping the final source texel', () => {
    const source = packMaterialTextureAtlasPixels(
      new Float32Array([
        0, 0, 0, 1,
        0.5, 0.5, 0.5, 1,
        1, 1, 1, 1,
      ]),
      MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
    );
    const mip = generateMaterialTextureAtlasMip(
      source,
      3,
      1,
      MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
      false,
    );
    expect(mip.width).toBe(1);
    expect(mip.height).toBe(1);
    expect(Array.from(unpackMaterialTextureAtlasPixels(
      mip.data,
      MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
    ))).toEqual([0.5, 0.5, 0.5, 1]);
  });

  it('filters encoded-sRGB RGB in linear light while leaving alpha linear', () => {
    const source = packMaterialTextureAtlasPixels(
      new Float32Array([
        0, 0, 0, 0.25,
        1, 1, 1, 0.75,
      ]),
      MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
    );
    const mip = generateMaterialTextureAtlasMip(
      source,
      2,
      1,
      MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
      true,
    );
    const encoded = unpackMaterialTextureAtlasPixels(
      mip.data,
      MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
    );
    const linear = encoded[0]! <= 0.04045
      ? encoded[0]! / 12.92
      : ((encoded[0]! + 0.055) / 1.055) ** 2.4;
    expect(linear).toBeCloseTo(0.5, 2);
    expect(encoded[3]).toBeCloseTo(0.5, 2);
  });

  it('rejects incomplete pixel and plane payloads', () => {
    expect(() => packMaterialTextureAtlasPixels(
      new Float32Array(3),
      MATERIAL_ATLAS_ENCODING_RGBA8_UNORM,
    )).toThrow(/divisible by four/);
    expect(() => unpackMaterialTextureAtlasPixels(
      new Uint32Array(3),
      MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT,
    )).toThrow(/complete codec planes/);
  });
});
