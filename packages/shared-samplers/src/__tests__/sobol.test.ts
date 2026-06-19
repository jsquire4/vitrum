import { describe, expect, it } from 'vitest';
import {
  SOBOL_TEXTURE_CHANNELS,
  SOBOL_TEXTURE_POINTS,
  SOBOL_TEXTURE_SIZE,
  generateSobolTextureData,
  maskedSobol,
  reverseBits32,
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
