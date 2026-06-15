import { describe, expect, it } from 'vitest';
import { luminance } from '../luminance.js';
import { bakePreethamSkyEquirect } from '../preethamSky.js';

describe('bakePreethamSkyEquirect', () => {
  it('bakes a finite 256x128 RGBA equirect map with a normalized CDF', () => {
    const bake = bakePreethamSkyEquirect();

    expect(bake.width).toBe(256);
    expect(bake.height).toBe(128);
    expect(bake.texels).toHaveLength(256 * 128 * 4);
    expect(bake.cdf).toHaveLength(256 * 128 + 1);
    expect(bake.cdf[0]).toBe(0);
    expect(bake.cdf[bake.cdf.length - 1]).toBe(1);
    expect(Array.from(bake.texels).every(Number.isFinite)).toBe(true);
  });

  it('honors zero intensity instead of falling back to the default sky strength', () => {
    const bake = bakePreethamSkyEquirect({ intensity: 0 });
    let total = 0;
    for (let i = 0; i < bake.width * bake.height; i += 1) {
      total += luminance(bake.texels[i * 4]!, bake.texels[i * 4 + 1]!, bake.texels[i * 4 + 2]!);
    }

    expect(total).toBe(0);
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
    const phiMax = ((pxMax + 0.5) / bake.width) * (2 * Math.PI);

    expect(thetaMax).toBeGreaterThan(Math.PI / 2 - 0.3);
    expect(thetaMax).toBeLessThan(Math.PI / 2 + 0.3);
    expect(Math.min(phiMax, 2 * Math.PI - phiMax)).toBeLessThan(0.3);
  });
});
