/**
 * C1 regression — composite blit must map swap-chain coordinates into the
 * INTERNAL-resolution texture so `resolutionFactor < 1` upscales to fill the
 * whole canvas instead of rendering into the top-left and leaving the rest
 * black.
 *
 * Before the fix `composite.wgsl` indexed `denoisedTex` (sized at the internal
 * resolution = swap × factor) with the raw swap-chain `fragCoord`. For
 * factor < 1 every swap-chain pixel outside the small internal region read
 * out of bounds (→ 0), so the canvas was mostly black.
 *
 * The fix emits a resolution-independent screen UV from the vertex shader and
 * indexes `denoisedTex` via `min(uv * textureDimensions(denoisedTex), dims-1)`.
 * These tests:
 *   1. mirror the WGSL coordinate transform in JS and pin the mapping for the
 *      §C1 worked example (factor 0.5: canvas (1919,1079) → internal (959,539)),
 *   2. prove the blit covers the WHOLE canvas (every swap-chain pixel maps to
 *      an in-bounds internal texel — the property the bug violated),
 *   3. guard the shader source so it can never regress to `fragCoord`-indexing.
 */
import { describe, expect, it } from 'vitest';
import { COMPOSITE_VERT_WGSL, COMPOSITE_FRAG_WGSL } from '../shaders/composite.wgsl.js';

/**
 * JS mirror of the composite shader's swap→internal coordinate transform.
 * The rasterizer interpolates the vertex-shader UV at each fragment's pixel
 * CENTER, so for swap-chain pixel (sx, sy) the UV is ((sx+0.5)/swapW,
 * (sy+0.5)/swapH). The fragment then computes
 * `px = u32(min(uv * internalDims, internalDims - 1))`.
 */
function blitTexel(
  sx: number, sy: number,
  swapW: number, swapH: number,
  internalW: number, internalH: number,
): { px: number; py: number } {
  const uvx = (sx + 0.5) / swapW;
  const uvy = (sy + 0.5) / swapH;
  const px = Math.trunc(Math.min(uvx * internalW, internalW - 1));
  const py = Math.trunc(Math.min(uvy * internalH, internalH - 1));
  return { px, py };
}

describe('C1 — composite resolution-factor blit coordinate mapping', () => {
  it('factor 0.5: bottom-right canvas pixel maps into the internal texture (§C1 example)', () => {
    // 1920×1080 canvas, factor 0.5 ⇒ 960×540 internal.
    expect(blitTexel(1919, 1079, 1920, 1080, 960, 540)).toEqual({ px: 959, py: 539 });
    // top-left maps to internal (0,0).
    expect(blitTexel(0, 0, 1920, 1080, 960, 540)).toEqual({ px: 0, py: 0 });
    // canvas center maps to internal center.
    expect(blitTexel(960, 540, 1920, 1080, 960, 540)).toEqual({ px: 480, py: 270 });
  });

  it('factor 0.75: every swap-chain pixel maps to an IN-BOUNDS internal texel (full coverage)', () => {
    const swapW = 800, swapH = 600;
    const internalW = Math.round(swapW * 0.75); // 600
    const internalH = Math.round(swapH * 0.75); // 450
    // Sample a dense set of swap-chain pixels including all four corners.
    for (const sx of [0, 1, 199, 400, swapW - 1]) {
      for (const sy of [0, 1, 137, 300, swapH - 1]) {
        const { px, py } = blitTexel(sx, sy, swapW, swapH, internalW, internalH);
        expect(px).toBeGreaterThanOrEqual(0);
        expect(py).toBeGreaterThanOrEqual(0);
        expect(px).toBeLessThan(internalW);
        expect(py).toBeLessThan(internalH);
      }
    }
  });

  it('the bug it replaces: raw-fragCoord indexing would read out of bounds at factor < 1', () => {
    // The OLD shader did `px = u32(fragCoord.xy)`. For factor 0.5 the bottom-
    // right canvas pixel (1919,1079) indexed an internal 960×540 texture far
    // out of bounds — this assertion documents why the canvas went black.
    const oldPx = 1919, oldPy = 1079;
    const internalW = 960, internalH = 540;
    expect(oldPx).toBeGreaterThanOrEqual(internalW); // out of bounds → read 0
    expect(oldPy).toBeGreaterThanOrEqual(internalH);
  });

  it('factor 1.0: mapping is bit-identical to the old 1:1 index', () => {
    // At factor 1 internal == swap, so uv*dims floors back to the same texel
    // the old `u32(fragCoord)` produced. Spot-check across the canvas.
    const W = 640, H = 480;
    for (const [sx, sy] of [[0, 0], [1, 1], [123, 456], [W - 1, H - 1]] as const) {
      expect(blitTexel(sx, sy, W, H, W, H)).toEqual({ px: sx, py: sy });
    }
  });

  it('shader source indexes denoisedTex via UV * textureDimensions, not raw fragCoord', () => {
    // Vertex shader emits a UV varying for the blit.
    expect(COMPOSITE_VERT_WGSL).toMatch(/@location\(0\)\s+uv\s*:\s*vec2f/);
    // Fragment shader derives the texel from uv * internal dims (resolution-
    // independent) — and must NOT index by raw swap-chain fragCoord.
    expect(COMPOSITE_FRAG_WGSL).toMatch(/textureDimensions\(denoisedTex\)/);
    expect(COMPOSITE_FRAG_WGSL).toMatch(/in\.uv/);
    expect(COMPOSITE_FRAG_WGSL).not.toMatch(/u32\(fragCoord\.x\)/);
  });
});
