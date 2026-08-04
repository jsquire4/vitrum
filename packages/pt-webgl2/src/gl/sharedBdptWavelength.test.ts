import { describe, expect, it } from 'vitest';
import {
  X_CMF_INTEGRAL,
  Y_CMF_INTEGRAL,
  Z_CMF_INTEGRAL,
  heroWavelengthMisMixturePdf,
  wavelengthToRGB,
  xyzToLinearSRGB,
} from '@vitrum/shared-samplers';
import { sharedBdptWavelengthForSeed } from './sharedBdptWavelength.js';

describe('shared spectral + BDPT wavelength', () => {
  it('is reproducible, Float32-exact, positive, and varies with sample and seed', () => {
    const first = sharedBdptWavelengthForSeed(37, 4);
    expect(sharedBdptWavelengthForSeed(37, 4)).toEqual(first);
    expect(sharedBdptWavelengthForSeed(37, 5)).not.toEqual(first);
    expect(Math.fround(first.wavelengthNm)).toBe(first.wavelengthNm);
    expect(Math.fround(first.pdf)).toBe(first.pdf);
    expect(first.pdf).toBe(
      Math.fround(heroWavelengthMisMixturePdf(first.wavelengthNm)),
    );
    expect(first.wavelengthNm).toBeGreaterThanOrEqual(380);
    expect(first.wavelengthNm).toBeLessThanOrEqual(780);
    expect(first.pdf).toBeGreaterThan(0);

    const uniqueSamples = new Set(
      Array.from({ length: 128 }, (_, sample) =>
        sharedBdptWavelengthForSeed(37, sample).wavelengthNm,
      ),
    );
    expect(uniqueSamples.size).toBeGreaterThan(120);

    const uniqueSeeds = new Set(
      Array.from({ length: 128 }, (_, seed) =>
        sharedBdptWavelengthForSeed(seed).wavelengthNm,
      ),
    );
    expect(uniqueSeeds.size).toBeGreaterThan(120);

    for (let sample = 0; sample < 128; sample++) {
      const represented = sharedBdptWavelengthForSeed(37, sample);
      expect(represented.pdf).toBe(
        Math.fround(heroWavelengthMisMixturePdf(represented.wavelengthNm)),
      );
    }
  });

  it('preserves constant-spectrum MIS normalization over successive frames', () => {
    const expected = xyzToLinearSRGB(
      X_CMF_INTEGRAL / Y_CMF_INTEGRAL,
      1,
      Z_CMF_INTEGRAL / Y_CMF_INTEGRAL,
    );
    const count = 32768;
    for (const seed of [0, 1, 37, 0xffffffff]) {
      const sum = [0, 0, 0];
      for (let sample = 0; sample < count; sample++) {
        const { wavelengthNm, pdf } = sharedBdptWavelengthForSeed(seed, sample);
        const rgb = wavelengthToRGB(wavelengthNm, 1, pdf);
        sum[0] = (sum[0] ?? 0) + rgb[0];
        sum[1] = (sum[1] ?? 0) + rgb[1];
        sum[2] = (sum[2] ?? 0) + rgb[2];
      }
      expect(sum[0]! / count).toBeCloseTo(expected[0], 2);
      expect(sum[1]! / count).toBeCloseTo(expected[1], 2);
      expect(sum[2]! / count).toBeCloseTo(expected[2], 2);
    }
  });

  it('wires one Float32 uniform pair into both BDPT passes', async () => {
    const [{ readFile }, { fileURLToPath }] = await Promise.all([
      import('node:fs/promises'),
      import('node:url'),
    ]);
    const root = fileURLToPath(new URL('.', import.meta.url));
    const [builder, uploader] = await Promise.all([
      readFile(`${root}/BdptSubpathBuilder.ts`, 'utf8'),
      readFile(`${root}/uploadFrameUniforms.ts`, 'utf8'),
    ]);
    for (const name of ['uBdptSharedWavelength', 'uBdptSharedWavelengthPdf']) {
      expect(builder).toContain(`setFloat('${name}'`);
      expect(uploader).toContain(`setFloat('${name}'`);
    }
  });
});
