import { describe, it, expect } from 'vitest';
import { linearTileIndexFromVarianceReadPixelsPy } from '../adaptiveTileWeights.js';

describe('linearTileIndexFromVarianceReadPixelsPy', () => {
  it('maps bottom readPixels row py=0 to largest fork ty', () => {
    const tilesX = 4;
    const tilesY = 3;
    expect(linearTileIndexFromVarianceReadPixelsPy(0, 0, tilesX, tilesY)).toBe((tilesY - 1) * tilesX);
    expect(linearTileIndexFromVarianceReadPixelsPy(0, 3, tilesX, tilesY)).toBe((tilesY - 1) * tilesX + 3);
  });

  it('maps top readPixels row to ty=0', () => {
    const tilesX = 4;
    const tilesY = 3;
    const pyTop = tilesY - 1;
    expect(linearTileIndexFromVarianceReadPixelsPy(pyTop, 2, tilesX, tilesY)).toBe(2);
  });

  it('is bijective over rows for fixed px', () => {
    const tilesX = 3;
    const tilesY = 5;
    const seen = new Set<number>();
    for (let py = 0; py < tilesY; py += 1) {
      seen.add(linearTileIndexFromVarianceReadPixelsPy(py, 1, tilesX, tilesY));
    }
    expect(seen.size).toBe(tilesY);
  });
});
