import { describe, expect, it } from 'vitest';
import type * as THREE from 'three';
import { materialTextureToPayload } from '../texturePixels.js';

// materialTextureToPayload extracts RGBA-float linear pixels from a THREE material
// texture for the THREE-free atlas (G3.1). The DataTexture path + flip + sRGB decode
// are unit-testable; the canvas-readback (Image/ImageBitmap) path needs a DOM and is
// validated in the browser A/B harness — here we only assert it degrades to null
// without a DOM rather than throwing.

function fakeDataTex(
  data: ArrayLike<number>, w: number, h: number,
  opts: { flipY?: boolean; colorSpace?: string } = {},
): THREE.Texture {
  return {
    image: { data, width: w, height: h },
    flipY: opts.flipY ?? false,
    colorSpace: opts.colorSpace ?? '',
  } as unknown as THREE.Texture;
}

describe('materialTextureToPayload', () => {
  it('extracts RGBA float from a linear DataTexture, no flip (flipY false)', () => {
    const out = materialTextureToPayload(fakeDataTex(new Float32Array([0.5, 0.25, 0.75, 1]), 1, 1));
    expect(out).not.toBeNull();
    expect(Array.from(out!.data)).toEqual([0.5, 0.25, 0.75, 1]);
  });

  it('reverses rows when flipY is true (image textures default flipY=true)', () => {
    // 1×2: source row 0 = dark, row 1 = bright (f32-exact values)
    const data = new Float32Array([0.25, 0.25, 0.25, 1, 0.75, 0.75, 0.75, 1]);
    const out = materialTextureToPayload(fakeDataTex(data, 1, 2, { flipY: true }));
    expect(Array.from(out!.data.slice(0, 4))).toEqual([0.75, 0.75, 0.75, 1]); // out row 0 ← source row 1
  });

  it('sRGB-decodes colour channels but NOT alpha when colorSpace is srgb', () => {
    const out = materialTextureToPayload(fakeDataTex(new Float32Array([0.5, 0.5, 0.5, 0.5]), 1, 1, { colorSpace: 'srgb' }));
    const lin = ((0.5 + 0.055) / 1.055) ** 2.4; // ≈ 0.214
    expect(out!.data[0]).toBeCloseTo(lin, 5);
    expect(out!.data[3]).toBe(0.5); // alpha is linear
  });

  it('promotes an R-only (stride 1) DataTexture to RGBA (rr r 1)', () => {
    const out = materialTextureToPayload(fakeDataTex(new Float32Array([0.5]), 1, 1));
    expect(Array.from(out!.data)).toEqual([0.5, 0.5, 0.5, 1]);
  });

  it('returns null for a drawable image source without a DOM (Node)', () => {
    const tex = { image: { width: 4, height: 4 }, flipY: true, colorSpace: '' } as unknown as THREE.Texture;
    expect(materialTextureToPayload(tex)).toBeNull();
  });
});
