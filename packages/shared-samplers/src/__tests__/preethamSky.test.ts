import { describe, expect, it } from 'vitest';
import { luminance } from '../luminance.js';
import { bakePreethamSkyEquirect } from '../preethamSky.js';

describe('bakePreethamSkyEquirect', () => {
  const normalized = (
    value: readonly [number, number, number],
  ): readonly [number, number, number] => {
    const length = Math.hypot(value[0], value[1], value[2]);
    return [value[0] / length, value[1] / length, value[2] / length];
  };

  const brightestTexelDirection = (
    bake: ReturnType<typeof bakePreethamSkyEquirect>,
  ): readonly [number, number, number] => {
    let maxLuminance = -Infinity;
    let maxIndex = 0;
    for (let i = 0; i < bake.width * bake.height; i += 1) {
      const offset = i * 4;
      const value = luminance(
        bake.texels[offset]!,
        bake.texels[offset + 1]!,
        bake.texels[offset + 2]!,
      );
      if (value > maxLuminance) {
        maxLuminance = value;
        maxIndex = i;
      }
    }
    const y = Math.floor(maxIndex / bake.width);
    const x = maxIndex % bake.width;
    const theta = ((y + 0.5) / bake.height) * Math.PI;
    const phi = ((x + 0.5) / bake.width - 0.5) * (2 * Math.PI);
    return [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
  };

  it('bakes a finite 256x128 RGBA equirect map with a normalized CDF', () => {
    const bake = bakePreethamSkyEquirect();

    expect(bake.width).toBe(256);
    expect(bake.height).toBe(128);
    expect(bake.texels).toHaveLength(256 * 128 * 4);
    expect(bake.cdf).toHaveLength(256 * 128 + 1);
    expect(bake.cdf[0]).toBe(0);
    expect(bake.cdf[bake.cdf.length - 1]).toBe(1);
    expect(Array.from(bake.texels).every(Number.isFinite)).toBe(true);
    expect(Number.isFinite(bake.luminanceIntegral)).toBe(true);
    expect(bake.luminanceIntegral).toBeGreaterThan(0);
  });

  it('normalizes the stored per-steradian PDF over exact texel solid angles', () => {
    const bake = bakePreethamSkyEquirect({ width: 32, height: 16 });
    const deltaPhi = (2 * Math.PI) / bake.width;
    let integratedPdf = 0;
    let flattenedBins = 0;
    for (let y = 0; y < bake.height; y += 1) {
      const theta0 = (y / bake.height) * Math.PI;
      const theta1 = ((y + 1) / bake.height) * Math.PI;
      const texelSolidAngle = deltaPhi * (Math.cos(theta0) - Math.cos(theta1));
      for (let x = 0; x < bake.width; x += 1) {
        const index = y * bake.width + x;
        const effectivePmf = bake.cdf[index + 1]! - bake.cdf[index]!;
        const pdf = bake.texels[index * 4 + 3]!;
        expect(pdf).toBe(Math.fround(effectivePmf / texelSolidAngle));
        if (effectivePmf === 0) {
          flattenedBins += 1;
          expect(pdf).toBe(0);
        }
        integratedPdf += pdf * texelSolidAngle;
      }
    }
    expect(flattenedBins).toBeGreaterThan(0);
    expect(integratedPdf).toBeCloseTo(1, 6);
  });

  it('honors zero intensity instead of falling back to the default sky strength', () => {
    const bake = bakePreethamSkyEquirect({ intensity: 0 });
    let total = 0;
    for (let i = 0; i < bake.width * bake.height; i += 1) {
      total += luminance(bake.texels[i * 4]!, bake.texels[i * 4 + 1]!, bake.texels[i * 4 + 2]!);
    }

    expect(total).toBe(0);
    expect(bake.cdf[bake.cdf.length - 1]).toBe(0);
    expect(bake.luminanceIntegral).toBe(0);
  });

  it('carries the baked radiance scale into its luminance integral exactly once', () => {
    const unit = bakePreethamSkyEquirect({
      width: 32,
      height: 16,
      intensity: 1,
    });
    const scaled = bakePreethamSkyEquirect({
      width: 32,
      height: 16,
      intensity: 3.5,
    });

    expect(scaled.luminanceIntegral / unit.luminanceIntegral).toBeCloseTo(3.5, 5);
  });

  it('builds a normalized CDF for a dim but positive baked sky', () => {
    const bake = bakePreethamSkyEquirect({
      width: 32,
      height: 16,
      intensity: 1e-20,
    });

    expect(bake.luminanceIntegral).toBeGreaterThan(0);
    expect(Array.from(bake.texels).some((value, index) => index % 4 !== 3 && value > 0)).toBe(true);
    expect(bake.cdf[0]).toBe(0);
    expect(bake.cdf[bake.cdf.length - 1]).toBe(1);
  });

  it('rejects a mathematically positive sky that collapses entirely in Float32', () => {
    expect(() => bakePreethamSkyEquirect({
      width: 32,
      height: 16,
      rayleigh: 2 ** -160,
      intensity: 1,
    })).toThrow(/positive procedural-sky radiance underflows entirely/i);
  });

  it('keeps an explicitly zero-Rayleigh model black at positive intensity', () => {
    const bake = bakePreethamSkyEquirect({
      width: 32,
      height: 16,
      rayleigh: 0,
      intensity: 1,
    });

    expect(bake.luminanceIntegral).toBe(0);
    expect(bake.cdf[bake.cdf.length - 1]).toBe(0);
  });

  it('places the brightest texel near a low-angle sun direction', () => {
    const bake = bakePreethamSkyEquirect({ sunDirection: [1, 0.05, 0] });
    let maxLum = -Infinity;
    let maxIdx = 0;
    for (let i = 0; i < bake.width * bake.height; i += 1) {
      const lum = luminance(bake.texels[i * 4]!, bake.texels[i * 4 + 1]!, bake.texels[i * 4 + 2]!);
      if (lum > maxLum) {
        maxLum = lum;
        maxIdx = i;
      }
    }

    const pyMax = (maxIdx / bake.width) | 0;
    const pxMax = maxIdx % bake.width;
    const thetaMax = ((pyMax + 0.5) / bake.height) * Math.PI;
    const phiMax = ((pxMax + 0.5) / bake.width - 0.5) * (2 * Math.PI);

    expect(thetaMax).toBeGreaterThan(Math.PI / 2 - 0.3);
    expect(thetaMax).toBeLessThan(Math.PI / 2 + 0.3);
    expect(Math.abs(phiMax)).toBeLessThan(0.3);
  });

  it('integrates the sub-texel solar disc at the default resolution', () => {
    const bake = bakePreethamSkyEquirect({
      sunDirection: [0, 1, 0],
      mieCoefficient: 0,
    });
    const rowAverage = (row: number): number => {
      let sum = 0;
      for (let x = 0; x < bake.width; x += 1) {
        const i = (row * bake.width + x) * 4;
        sum += luminance(bake.texels[i]!, bake.texels[i + 1]!, bake.texels[i + 2]!);
      }
      return sum / bake.width;
    };

    // At 256×128 every texel centre lies outside the 0.00436-radian solar
    // radius. The integrated cap must nevertheless produce a clear first-row
    // contribution rather than silently omitting the sun.
    expect(rowAverage(0)).toBeGreaterThan(rowAverage(1) * 2);
  });

  it('preserves solar visibility and integrated energy across directions and resolutions', () => {
    const directions = [
      normalized([0, 1, 0]),
      normalized([1, 0.05, 0]),
      normalized([0.4, 0.7, -0.2]),
    ] as const;

    for (const sunDirection of directions) {
      const low = bakePreethamSkyEquirect({
        sunDirection,
        width: 64,
        height: 32,
      });
      const high = bakePreethamSkyEquirect({
        sunDirection,
        width: 256,
        height: 128,
      });

      // The cap is integrated in solid angle, so increasing the map
      // resolution must not create or destroy a material amount of energy.
      expect(high.luminanceIntegral / low.luminanceIntegral).toBeGreaterThan(0.94);
      expect(high.luminanceIntegral / low.luminanceIntegral).toBeLessThan(1.06);

      for (const bake of [low, high]) {
        const brightest = brightestTexelDirection(bake);
        const cosine =
          brightest[0] * sunDirection[0] +
          brightest[1] * sunDirection[1] +
          brightest[2] * sunDirection[2];
        const angularError = Math.acos(Math.max(-1, Math.min(1, cosine)));
        const texelDiagonal = Math.hypot(Math.PI / bake.height, (2 * Math.PI) / bake.width);

        // Even when the physical disc is much smaller than one texel, its
        // deposited energy remains visible at the authored sun direction.
        expect(angularError).toBeLessThan(texelDiagonal);
      }
    }
  });

  it('preserves authored Mie asymmetry beyond the former 0.9999 cutoff', () => {
    const options = {
      width: 32,
      height: 16,
      sunDirection: normalized([1, 0.05, 0]),
    } as const;
    const oldBoundary = bakePreethamSkyEquirect({
      ...options,
      mieDirectionalG: 0.9999,
    });
    const authoredTail = bakePreethamSkyEquirect({
      ...options,
      mieDirectionalG: 0.99995,
    });

    // The old local clamp flattened these two authored values to the same sky.
    expect(authoredTail.luminanceIntegral).not.toBe(oldBoundary.luminanceIntegral);
    expect(Array.from(authoredTail.texels)).not.toEqual(Array.from(oldBoundary.texels));
  });
});
