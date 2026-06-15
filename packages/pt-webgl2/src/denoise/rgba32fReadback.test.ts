import { describe, expect, it } from 'vitest';
import { rgba32fBottomLeftToRgbF32 } from './rgba32fReadback.js';

describe('pt-webgl2 OIDN RGBA32F readback conversion', () => {
  it('converts GL bottom-left RGBA rows to top-left RGB rows', () => {
    const rgba = new Float32Array([
      // GL row 0: bottom
      1, 2, 3, 99,
      4, 5, 6, 99,
      // GL row 1: top
      7, 8, 9, 99,
      10, 11, 12, 99,
    ]);

    expect(Array.from(rgba32fBottomLeftToRgbF32(rgba, 2, 2))).toEqual([
      7, 8, 9,
      10, 11, 12,
      1, 2, 3,
      4, 5, 6,
    ]);
  });

  it('applies optional RGB decode while dropping alpha', () => {
    const rgba = new Float32Array([
      0, 0.5, 1, 123,
    ]);

    expect(Array.from(rgba32fBottomLeftToRgbF32(rgba, 1, 1, (r, g, b) => [
      r * 2 - 1,
      g * 2 - 1,
      b * 2 - 1,
    ]))).toEqual([-1, 0, 1]);
  });
});
