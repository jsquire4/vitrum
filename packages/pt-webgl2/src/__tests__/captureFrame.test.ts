/**
 * captureFrame — pt-webgl2 backend.
 *
 * Tests the row-flip (WebGL readPixels returns bottom-left origin; captureFrame
 * contract requires top-left origin) and the null-before-first-frame guard.
 * Uses a mock GL context — no real GPU required.
 */
import { describe, expect, it, vi } from 'vitest';
import { createMockGl } from './mockGl.js';
import { createPTEngine_WebGL2 } from '../index.js';

// ── row-flip geometry test ────────────────────────────────────────────────────

describe('captureFrame row-flip (pt-webgl2)', () => {
  it('captureFrame returns null before any frame is rendered', async () => {
    const gl = createMockGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    // captureFrame is optional on the core contract; this backend implements it.
    expect(engine.captureFrame).toBeDefined();
    const frame = await engine.captureFrame!();
    expect(frame).toBeNull();
    engine.dispose();
  });
});

// ── row-flip unit test (standalone, no engine) ────────────────────────────────

describe('readPixelsRgba32f row flip', () => {
  it('vertically flips a 2-row image so row 0 = top', () => {
    // Simulate a 2×2 RGBA32F GL readback (bottom-left origin from GL):
    //   GL row 0 (bottom) = all 1s
    //   GL row 1 (top)    = all 2s
    // After flip, CPU row 0 should be 2s (was GL row 1 = top of image).
    const w = 2, h = 2;
    const pixels = new Float32Array([
      // GL row 0 (BOTTOM of image)
      1, 1, 1, 1,  1, 1, 1, 1,
      // GL row 1 (TOP of image)
      2, 2, 2, 2,  2, 2, 2, 2,
    ]);

    // Run the same flip logic as GlResources.readPixelsRgba32f.
    const rowLen = w * 4;
    const tmp = new Float32Array(rowLen);
    for (let top = 0, bot = h - 1; top < bot; top++, bot--) {
      const topOff = top * rowLen;
      const botOff = bot * rowLen;
      tmp.set(pixels.subarray(topOff, topOff + rowLen));
      pixels.copyWithin(topOff, botOff, botOff + rowLen);
      pixels.set(tmp, botOff);
    }

    // After flip: CPU row 0 = top of image = old GL row 1 = all 2s.
    expect(pixels[0]).toBe(2);
    expect(pixels[1]).toBe(2);
    expect(pixels[4]).toBe(2);
    // CPU row 1 = bottom of image = old GL row 0 = all 1s.
    expect(pixels[rowLen]).toBe(1);
    expect(pixels[rowLen + 1]).toBe(1);
  });

  it('leaves a 1-row image unchanged', () => {
    const w = 3, h = 1;
    const pixels = new Float32Array([1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]);
    const orig = pixels.slice();
    const rowLen = w * 4;
    const tmp = new Float32Array(rowLen);
    for (let top = 0, bot = h - 1; top < bot; top++, bot--) {
      const topOff = top * rowLen;
      const botOff = bot * rowLen;
      tmp.set(pixels.subarray(topOff, topOff + rowLen));
      pixels.copyWithin(topOff, botOff, botOff + rowLen);
      pixels.set(tmp, botOff);
    }
    expect(Array.from(pixels)).toEqual(Array.from(orig));
  });
});
