import { describe, expect, it } from 'vitest';
import {
  SOBOL_TEXTURE_CHANNELS,
  SOBOL_TEXTURE_POINTS,
  SOBOL_TEXTURE_SIZE,
  SOBOL_BLUE_NOISE_RANK_8X8,
  SOBOL_BLUE_NOISE_TILE_SIZE,
  generateSobolTextureData,
  initOwenScrambledSobolState,
  laineKarrasPermutation,
  maskedSobol,
  nestedUniformScrambleBase2,
  owenScrambledSobolFloat,
  owenScrambledSobolU32,
  reverseBits32,
  sobolBlueNoiseRotationBits,
  sobolHash,
  sobolHashCombine,
  sobolTextureComponentBits,
  sobolTexturePoint,
} from '../sobol.js';

const INV_24 = 1 / 16_777_216;

function expectPointClose(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]).toBeCloseTo(expected[i]!, 12);
  }
}

describe('Sobol texture table', () => {
  it('matches the GLSL reverse-bit primitive', () => {
    expect(reverseBits32(0x00000001)).toBe(0x80000000);
    expect(reverseBits32(0x80000000)).toBe(0x00000001);
    expect(reverseBits32(0x0f0f0f0f)).toBe(0xf0f0f0f0);
  });

  it('xors masked direction numbers by index bits', () => {
    expect(maskedSobol(0b00, [0x80000000, 0xc0000000])).toBe(0x00000000);
    expect(maskedSobol(0b01, [0x80000000, 0xc0000000])).toBe(0x80000000);
    expect(maskedSobol(0b10, [0x80000000, 0xc0000000])).toBe(0xc0000000);
    expect(maskedSobol(0b11, [0x80000000, 0xc0000000])).toBe(0x40000000);
  });

  it('pins the first texture points against the shader direction tables', () => {
    expectPointClose(sobolTexturePoint(0), [0, 0, 0, 0]);
    expectPointClose(sobolTexturePoint(1), [1 * INV_24, 1 * INV_24, 1 * INV_24, 1 * INV_24]);
    expectPointClose(sobolTexturePoint(2), [3 * INV_24, 3 * INV_24, 3 * INV_24, 2 * INV_24]);
    expectPointClose(sobolTexturePoint(3), [2 * INV_24, 2 * INV_24, 2 * INV_24, 3 * INV_24]);
    expectPointClose(sobolTexturePoint(4), [5 * INV_24, 6 * INV_24, 4 * INV_24, 4 * INV_24]);
  });

  it('pins the hash-based Owen scramble primitives used by shader Sobol streams', () => {
    expect(sobolHash(1)).toBe(0x514e28b7);
    expect(sobolHashCombine(0x12345678, 0x9abcdef0)).toBe(0x3e6bc4f6);
    expect(laineKarrasPermutation(0x13579bdf, 0x2468ace0)).toBe(0x0a4633fb);
    expect(nestedUniformScrambleBase2(0x00abcdef, 0x31415926)).toBe(0x81c5a4ea);
  });

  it('pins the ranked 8x8 Sobol rotation tile', () => {
    expect(SOBOL_BLUE_NOISE_TILE_SIZE).toBe(8);
    expect(SOBOL_BLUE_NOISE_RANK_8X8).toHaveLength(64);
    expect(new Set(SOBOL_BLUE_NOISE_RANK_8X8).size).toBe(64);
    expect(Math.min(...SOBOL_BLUE_NOISE_RANK_8X8)).toBe(0);
    expect(Math.max(...SOBOL_BLUE_NOISE_RANK_8X8)).toBe(63);
    expect(SOBOL_BLUE_NOISE_RANK_8X8.slice(0, 8)).toEqual([0, 63, 12, 60, 3, 55, 15, 62]);
    expect(sobolBlueNoiseRotationBits(0, 9)).toBe(0x000000);
    expect(sobolBlueNoiseRotationBits(1, 9)).toBe(0x89a213);
    expect(sobolBlueNoiseRotationBits(4, 9)).toBe(0xbcda23);
    expect(sobolBlueNoiseRotationBits(63, 9)).toBe(0xe40131);
  });

  it('mirrors the pt-webgpu binding-free Owen-scrambled and rotated stream', () => {
    expect(sobolTextureComponentBits(7, 0)).toBe(0x00000007);
    expect(sobolTextureComponentBits(7, 5)).toBe(0x00000004);
    expect(owenScrambledSobolU32(0, 0)).toBe(0xa66de000);
    expect(owenScrambledSobolU32(0, 1)).toBe(0x8fa66800);
    expect(owenScrambledSobolU32(1, 0)).toBe(0x543dae00);
    expect(owenScrambledSobolU32(12345, 0)).toBe(0x96bc2600);
    expect(owenScrambledSobolU32(12345, 9)).toBe(0x801a7d00);
    expect(owenScrambledSobolU32(12345, 9, 1)).toBe(0x09bc9000);
    expect(owenScrambledSobolU32(12345, 9, 4)).toBe(0x3cf4a000);
    expect(owenScrambledSobolU32(12345, 9, 63)).toBe(0x641bae00);
    expect(owenScrambledSobolFloat(12345, 9, 63)).toBeCloseTo(0.3910473585128784, 12);
    expect(initOwenScrambledSobolState(9, 10, 123)).toBe(0xc0761100);
  });

  it('documents high-dimension behavior: four direction tables, hash-decorrelated dimensions', () => {
    expect(sobolTextureComponentBits(7, 0)).toBe(sobolTextureComponentBits(7, 4));
    expect(owenScrambledSobolU32(12345, 0)).not.toBe(owenScrambledSobolU32(12345, 4));
    const first16 = Array.from({ length: 16 }, (_, dim) => owenScrambledSobolU32(12345, dim));
    expect(new Set(first16).size).toBe(16);
  });

  it('builds the full RGBA table at the expected dimensions and range', () => {
    const data = generateSobolTextureData();
    expect(data).toBeInstanceOf(Float32Array);
    expect(data).toHaveLength(SOBOL_TEXTURE_POINTS * SOBOL_TEXTURE_CHANNELS);
    expect(SOBOL_TEXTURE_SIZE * SOBOL_TEXTURE_SIZE).toBe(SOBOL_TEXTURE_POINTS);
    for (const value of data) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('rejects impossible table sizes instead of silently truncating', () => {
    expect(() => generateSobolTextureData(SOBOL_TEXTURE_POINTS + 1)).toThrow(RangeError);
    expect(() => generateSobolTextureData(-1)).toThrow(RangeError);
    expect(() => generateSobolTextureData(1.5)).toThrow(RangeError);
  });
});
