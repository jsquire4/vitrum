import {
  BMFR_FEATURE_COUNT,
  bmfrFeatureRow,
  bmfrSolveChannel,
} from '../src/bmfrRegression.js';

interface CpuBmfrOptions {
  readonly rgb: Float32Array;
  readonly worldPosRgb: Float32Array;
  readonly validityW?: Float32Array;
  readonly normalsRgb?: Float32Array;
  readonly width: number;
  readonly height: number;
  readonly blockSize: number;
  readonly blockStride: number;
  readonly positionScale: number;
  readonly regularisation: number;
}

interface CpuBlockFit {
  readonly originX: number;
  readonly originY: number;
  readonly mean: readonly [number, number, number];
  readonly validCount: number;
  readonly alpha: readonly [Float32Array, Float32Array, Float32Array];
}

function positionAt(
  input: CpuBmfrOptions,
  pixel: number,
): [number, number, number] {
  const offset = pixel * 3;
  return [
    input.worldPosRgb[offset]!,
    input.worldPosRgb[offset + 1]!,
    input.worldPosRgb[offset + 2]!,
  ];
}

function normalAt(
  input: CpuBmfrOptions,
  pixel: number,
): [number, number, number] {
  if (input.normalsRgb == null) return [0, 0, 1];
  const offset = pixel * 3;
  return [
    input.normalsRgb[offset]! * 2 - 1,
    input.normalsRgb[offset + 1]! * 2 - 1,
    input.normalsRgb[offset + 2]! * 2 - 1,
  ];
}

function isValid(input: CpuBmfrOptions, pixel: number): boolean {
  return input.validityW == null || input.validityW[pixel]! > 0;
}

function featureAt(
  input: CpuBmfrOptions,
  pixel: number,
  mean: readonly [number, number, number],
): Float32Array {
  const p = positionAt(input, pixel);
  const local: [number, number, number] = [
    (p[0] - mean[0]) / input.positionScale,
    (p[1] - mean[1]) / input.positionScale,
    (p[2] - mean[2]) / input.positionScale,
  ];
  const row = new Float32Array(BMFR_FEATURE_COUNT);
  bmfrFeatureRow(local, normalAt(input, pixel), row);
  return row;
}

/**
 * Independent CPU frame oracle for the GPU fit/resolve pair. It deliberately
 * enumerates every block footprint and, during resolve, brute-force tests every
 * block for coverage instead of sharing the shader's first/last-block formula.
 */
export function bmfrCpuOverlapOracle(input: CpuBmfrOptions): Float32Array {
  const fits: CpuBlockFit[] = [];
  const blocksX = Math.ceil(input.width / input.blockStride);
  const blocksY = Math.ceil(input.height / input.blockStride);
  for (let blockY = 0; blockY < blocksY; blockY += 1) {
    for (let blockX = 0; blockX < blocksX; blockX += 1) {
      const originX = blockX * input.blockStride;
      const originY = blockY * input.blockStride;
      const pixels: number[] = [];
      const sum: [number, number, number] = [0, 0, 0];
      for (let localY = 0; localY < input.blockSize; localY += 1) {
        const y = originY + localY;
        if (y >= input.height) continue;
        for (let localX = 0; localX < input.blockSize; localX += 1) {
          const x = originX + localX;
          if (x >= input.width) continue;
          const pixel = y * input.width + x;
          if (!isValid(input, pixel)) continue;
          pixels.push(pixel);
          const p = positionAt(input, pixel);
          sum[0] += p[0];
          sum[1] += p[1];
          sum[2] += p[2];
        }
      }
      const validCount = pixels.length;
      const mean: [number, number, number] = validCount === 0
        ? [0, 0, 0]
        : [sum[0]! / validCount, sum[1]! / validCount, sum[2]! / validCount];
      const rows = pixels.map((pixel) => featureAt(input, pixel, mean));
      const values = [0, 1, 2].map((channel) =>
        pixels.map((pixel) => input.rgb[pixel * 3 + channel]!));
      fits.push({
        originX,
        originY,
        mean,
        validCount,
        alpha: [
          bmfrSolveChannel(rows, values[0]!, input.regularisation),
          bmfrSolveChannel(rows, values[1]!, input.regularisation),
          bmfrSolveChannel(rows, values[2]!, input.regularisation),
        ],
      });
    }
  }

  const output = input.rgb.slice();
  for (let y = 0; y < input.height; y += 1) {
    for (let x = 0; x < input.width; x += 1) {
      const pixel = y * input.width + x;
      if (!isValid(input, pixel)) continue;
      const sums: [number, number, number] = [0, 0, 0];
      let contributions = 0;
        for (const fit of fits) {
        if (
          fit.validCount === 0 ||
          x < fit.originX || x >= fit.originX + input.blockSize ||
          y < fit.originY || y >= fit.originY + input.blockSize
        ) continue;
        const row = featureAt(input, pixel, fit.mean);
          for (const channel of [0, 1, 2] as const) {
          let reconstructed = 0;
          for (let feature = 0; feature < BMFR_FEATURE_COUNT; feature += 1) {
            reconstructed += row[feature]! * fit.alpha[channel]![feature]!;
          }
          sums[channel] += Math.max(0, reconstructed);
        }
        contributions += 1;
      }
      if (contributions > 0) {
        output[pixel * 3] = sums[0]! / contributions;
        output[pixel * 3 + 1] = sums[1]! / contributions;
        output[pixel * 3 + 2] = sums[2]! / contributions;
      }
    }
  }
  return output;
}
