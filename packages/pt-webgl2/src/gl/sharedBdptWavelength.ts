import {
  heroWavelengthMisMixturePdf,
  sampleHeroWavelengthMIS,
} from '@vitrum/shared-samplers';

export interface SharedBdptWavelengthSample {
  readonly wavelengthNm: number;
  readonly pdf: number;
}

function reverseBits32(value: number): number {
  let x = value >>> 0;
  x = ((x >>> 16) | (x << 16)) >>> 0;
  x = (((x & 0x55555555) << 1) | ((x >>> 1) & 0x55555555)) >>> 0;
  x = (((x & 0x33333333) << 2) | ((x >>> 2) & 0x33333333)) >>> 0;
  x = (((x & 0x0f0f0f0f) << 4) | ((x >>> 4) & 0x0f0f0f0f)) >>> 0;
  return (((x & 0x00ff00ff) << 8) | ((x >>> 8) & 0x00ff00ff)) >>> 0;
}

function radicalInverseBase3(value: number): number {
  let n = value >>> 0;
  let inverseBase = 1 / 3;
  let result = 0;
  while (n > 0) {
    result += (n % 3) * inverseBase;
    n = Math.floor(n / 3);
    inverseBase /= 3;
  }
  return result;
}

function mixUint32(value: number): number {
  let x = value >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * One frame/sample-wide hero wavelength for the global WebGL2 BDPT light path
 * and every eye path that connects to it. A single wavelength is required because
 * the 2x5 light-path texture is shared by all pixels and stores no spectral curve.
 * The sequence depends only on frameSeed and the accumulated sample, never
 * pixel coordinates. The sample index follows a low-discrepancy sequence;
 * frameSeed only scrambles that sequence, so hosts that increment frameSeed and
 * the accumulated sample in lockstep do not collapse onto an even subsequence.
 */
export function sharedBdptWavelengthForSeed(
  seed: number,
  accumulatedSample = 0,
): SharedBdptWavelengthSample {
  const index = ((Math.trunc(accumulatedSample) >>> 0) + 1) >>> 0;
  const seedUint = Math.trunc(seed) >>> 0;
  const strategyScramble = mixUint32(seedUint ^ 0x9e3779b9);
  const wavelengthRotation = mixUint32(seedUint ^ 0x243f6a88) / 0x1_0000_0000;
  const uStrategy =
    ((reverseBits32(index) ^ strategyScramble) >>> 0) / 0x1_0000_0000;
  const uLambda = (radicalInverseBase3(index) + wavelengthRotation) % 1;
  const sampled = sampleHeroWavelengthMIS(uStrategy, uLambda);
  const wavelengthNm = Math.fround(sampled.lambdaNm);
  // The shared wavelength is quantized to f32 before entering the light-path
  // uniforms. Re-evaluate the same represented X/Y/Z mixture at that published
  // wavelength; ideal thirds would disagree with the 24-bit selector used by
  // both PCG and Sobol.
  const pdf = Math.fround(heroWavelengthMisMixturePdf(wavelengthNm));
  return { wavelengthNm, pdf };
}
