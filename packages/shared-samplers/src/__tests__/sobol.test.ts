import { describe, expect, it } from 'vitest';
import {
  SOBOL_DIMENSION_COUNT,
  SOBOL_SAMPLE_BLOCK_SIZE,
  SOBOL_TEXTURE_CHANNELS,
  SOBOL_TEXTURE_POINTS,
  SOBOL_TEXTURE_SIZE,
  SOBOL_BLUE_NOISE_RANK_8X8,
  SOBOL_BLUE_NOISE_TILE_SIZE,
  generateSobolTextureData,
  initOwenScrambledSobolStream,
  laineKarrasPermutation,
  maskedSobol,
  nestedUniformScrambleBase2,
  nextOwenScrambledSobolFloat,
  nextOwenScrambledSobolU32,
  owenScrambledSobolFloat,
  owenScrambledSobolU32,
  reverseBits32,
  sobolBlueNoiseRotationBits,
  sobolFrameKey,
  sobolHash,
  sobolHashCombine,
  sobolTextureComponentBits,
  sobolTexturePoint,
} from '../sobol.js';
import {
  SOBOL_DIRECTION_BITS,
  SOBOL_DIRECTION_DIMENSION_COUNT,
  SOBOL_DIRECTION_NUMBERS_WGSL,
} from '../sobolDirectionNumbers.js';

const INV_24 = 1 / 16_777_216;

function expectPointClose(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i]).toBeCloseTo(expected[i]!, 12);
  }
}

function starDiscrepancy1D(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  let discrepancy = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    const value = sorted[i] ?? 0;
    discrepancy = Math.max(
      discrepancy,
      (i + 1) / sorted.length - value,
      value - i / sorted.length,
    );
  }
  return discrepancy;
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

  it('keeps rank zero as a zero rotation without aliasing full pixel streams', () => {
    const frameKey = sobolFrameKey(0x12345678, 99);
    const pixelA = initOwenScrambledSobolStream(0, 0, frameKey);
    const pixelB = initOwenScrambledSobolStream(8, 0, frameKey);
    expect(pixelA.rotationTile).toBe(0);
    expect(pixelB.rotationTile).toBe(0);
    expect(sobolBlueNoiseRotationBits(0, 323)).toBe(0);
    expect(Array.from({ length: 16 }, () => nextOwenScrambledSobolU32(pixelA)))
      .not.toEqual(Array.from({ length: 16 }, () => nextOwenScrambledSobolU32(pixelB)));
  });

  it('pins the pt-webgpu CPU oracle and full non-aliasing stream state', () => {
    expect(sobolTextureComponentBits(7, 0)).toBe(0x00000007);
    expect(sobolTextureComponentBits(7, 5)).toBe(0x00000007);
    expect(sobolTextureComponentBits(7, 255)).toBe(0x00000006);
    expect(sobolTextureComponentBits(7, 511)).toBe(0x00000006);
    expect(owenScrambledSobolU32(0, 0)).toBe(0xa66de000);
    expect(owenScrambledSobolU32(0, 1)).toBe(0x1fc82700);
    expect(owenScrambledSobolU32(1, 0)).toBe(0x5336f000);
    expect(owenScrambledSobolU32(12345, 3, 63)).toBe(0xba554d00);
    expect(owenScrambledSobolFloat(12345, 3, 63)).toBeCloseTo(0.7278640866279602, 12);

    const frameKey = sobolFrameKey(123, 0);
    expect(frameKey).toBe(0xc00c0000);
    const stream = initOwenScrambledSobolStream(9, 10, frameKey);
    expect(stream).toEqual({
      sampleIndex: 0,
      dimension: 0,
      pixelX: 9,
      pixelY: 10,
      sequenceKey: 49164,
      rotationTile: 17,
      fallbackState: 3105986336,
    });
    expect(Array.from({ length: 8 }, () => nextOwenScrambledSobolU32(stream))).toEqual([
      0x8aeca400,
      0x419e9400,
      0xb524e900,
      0x23e98800,
      0x7759a400,
      0xe7614000,
      0xf4738a00,
      0xb225f000,
    ]);
  });

  it('keeps dimensions decorrelated and rejects dimensions outside the Sobol prefix', () => {
    expect(SOBOL_DIMENSION_COUNT).toBe(512);
    expect(SOBOL_DIMENSION_COUNT).toBe(SOBOL_DIRECTION_DIMENSION_COUNT);
    expect(SOBOL_DIRECTION_BITS).toBe(16);
    expect(SOBOL_DIRECTION_NUMBERS_WGSL).toContain(
      'const SOBOL_DIRECTION_NUMBERS_PACKED = array<u32, 4096>',
    );
    expect(sobolTextureComponentBits(7, 0)).not.toBe(sobolTextureComponentBits(7, 4));
    expect(owenScrambledSobolU32(12345, 0)).not.toBe(owenScrambledSobolU32(12345, 3));
    const prefix = Array.from(
      { length: SOBOL_DIMENSION_COUNT },
      (_, dim) => owenScrambledSobolU32(12345, dim),
    );
    expect(new Set(prefix).size).toBe(SOBOL_DIMENSION_COUNT);
    expect(() => owenScrambledSobolU32(0, -1)).toThrow(RangeError);
    expect(() => owenScrambledSobolU32(0, 1.5)).toThrow(RangeError);
    expect(() => owenScrambledSobolU32(0, SOBOL_DIMENSION_COUNT)).toThrow(RangeError);
    expect(() => sobolTextureComponentBits(0, SOBOL_DIMENSION_COUNT)).toThrow(RangeError);
  });

  it('maps all 2^32 frame indices to unique block and sample identities', () => {
    const seed = 0x12345678;
    const blockKeys = new Set<number>();
    for (let block = 0; block < SOBOL_SAMPLE_BLOCK_SIZE; block += 1) {
      blockKeys.add(sobolFrameKey(seed, block * SOBOL_SAMPLE_BLOCK_SIZE) >>> 16);
    }
    expect(blockKeys.size).toBe(SOBOL_SAMPLE_BLOCK_SIZE);

    const boundaries = [0, 0xffff, 0x10000, 0x10001, 0xfffffffe, 0xffffffff];
    expect(boundaries.map((frame) => sobolFrameKey(seed, frame))).toEqual([
      0xd1bc0000,
      0xd1bcffff,
      0x6ff30000,
      0x6ff30001,
      0x3385fffe,
      0x3385ffff,
    ]);
    for (const frame of boundaries) {
      expect(sobolFrameKey(seed, frame) & 0xffff).toBe(frame & 0xffff);
    }
  });

  it('retains full distant-pixel identity and reproduces exactly after reset', () => {
    const key = sobolFrameKey(0x76543210, 0x10000);
    const near = initOwenScrambledSobolStream(9, 10, key);
    const far = initOwenScrambledSobolStream(65545, 131082, key);
    expect(near.rotationTile).toBe(far.rotationTile);
    expect(near.pixelX).toBe(9);
    expect(far.pixelX).toBe(65545);
    expect(near.fallbackState).not.toBe(far.fallbackState);

    const take = (pixelX: number, pixelY: number, frameIndex: number): number[] => {
      const stream = initOwenScrambledSobolStream(
        pixelX,
        pixelY,
        sobolFrameKey(0x76543210, frameIndex),
      );
      return Array.from({ length: 32 }, () => nextOwenScrambledSobolU32(stream));
    };
    const firstRun = take(9, 10, 0);
    expect(take(9, 10, 0)).toEqual(firstRun);
    expect(take(9, 10, 1)).not.toEqual(firstRun);
    expect(take(65545, 131082, 0)).not.toEqual(firstRun);
    expect(take(9, 10, 0)).toEqual(firstRun);
  });

  it('has no repeated values within the complete 65,536-sample Sobol block', () => {
    for (const dimension of [0, 1, 2, 3, 255, 511]) {
      const values = new Set<number>();
      for (let sample = 0; sample < SOBOL_SAMPLE_BLOCK_SIZE; sample += 1) {
        values.add(owenScrambledSobolU32(sample, dimension, 17, 0x12345678));
      }
      expect(values.size).toBe(SOBOL_SAMPLE_BLOCK_SIZE);
    }
  });

  it('pins one-dimensional discrepancy and two-dimensional stratification', () => {
    const sampleCount = 4096;
    for (const dimension of [0, 1, 2, 3]) {
      const values = Array.from(
        { length: sampleCount },
        (_, sample) => owenScrambledSobolU32(sample, dimension, 17, 0x12345678) / 2 ** 32,
      );
      expect(new Set(values).size).toBe(sampleCount);
      expect(starDiscrepancy1D(values)).toBeLessThan(0.0005);
      const mean = values.reduce((sum, value) => sum + value, 0) / sampleCount;
      expect(mean).toBeGreaterThan(0.499);
      expect(mean).toBeLessThan(0.501);
    }

    for (const [dimensionX, dimensionY] of [[0, 1], [2, 3]] as const) {
      const bins = new Uint32Array(16 * 16);
      for (let sample = 0; sample < sampleCount; sample += 1) {
        const x = owenScrambledSobolU32(sample, dimensionX, 17, 0x12345678) >>> 28;
        const y = owenScrambledSobolU32(sample, dimensionY, 17, 0x12345678) >>> 28;
        const binIndex = x * 16 + y; bins[binIndex] = (bins[binIndex] ?? 0) + 1;
      }
      expect(Math.min(...bins)).toBeGreaterThanOrEqual(10);
      expect(Math.max(...bins)).toBeLessThanOrEqual(22);
    }
  });

  it('covers the audited renderer budget and pins the deterministic continuation', () => {
    const key = sobolFrameKey(123, 0);
    const rendererBudget = initOwenScrambledSobolStream(9, 10, key);
    const fallbackBeforeBudget = rendererBudget.fallbackState;
    for (let draw = 0; draw < 324; draw += 1) {
      nextOwenScrambledSobolU32(rendererBudget);
    }
    expect(rendererBudget.dimension).toBe(324);
    expect(rendererBudget.fallbackState).toBe(fallbackBeforeBudget);

    const stream = initOwenScrambledSobolStream(9, 10, key);
    stream.dimension = SOBOL_DIMENSION_COUNT - 2;
    expect(Array.from({ length: 7 }, () => nextOwenScrambledSobolU32(stream))).toEqual([
      0x08785900,
      0xe8b6ec00,
      0x9d8c7ac8,
      0xed2474d0,
      0xb1a14c1e,
      0x96716f84,
      0x74ede162,
    ]);
    expect(stream.dimension).toBe(SOBOL_DIMENSION_COUNT + 5);

    const continuationA = initOwenScrambledSobolStream(9, 10, key);
    const continuationB = initOwenScrambledSobolStream(9, 10, key);
    continuationA.dimension = SOBOL_DIMENSION_COUNT;
    continuationB.dimension = SOBOL_DIMENSION_COUNT;
    const valuesA = Array.from({ length: 4096 }, () => nextOwenScrambledSobolU32(continuationA));
    const valuesB = Array.from({ length: 4096 }, () => nextOwenScrambledSobolU32(continuationB));
    expect(valuesB).toEqual(valuesA);
    expect(new Set(valuesA).size).toBe(valuesA.length);

    const saturated = initOwenScrambledSobolStream(9, 10, key);
    saturated.dimension = 0xffffffff;
    const first = nextOwenScrambledSobolU32(saturated);
    const second = nextOwenScrambledSobolU32(saturated);
    expect(saturated.dimension).toBe(0xffffffff);
    expect(second).not.toBe(first);

    const floatStream = initOwenScrambledSobolStream(9, 10, key);
    floatStream.dimension = SOBOL_DIMENSION_COUNT - 2;
    for (let i = 0; i < 16; i += 1) {
      const value = nextOwenScrambledSobolFloat(floatStream);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
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
