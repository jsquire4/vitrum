import { describe, expect, it } from 'vitest';
import type * as THREE from 'three';
import { equirectTextureToPayload } from '../environment.js';

// equirectTextureToPayload duck-types a THREE equirect DataTexture into the raw
// { width, height, data } RGB-float payload the THREE-free path tracers read. These
// pin the stride/flip/type handling that must mirror the fork's preprocessEnvMap so
// the IBL is oriented + scaled identically (the G2 data-bridge).

/** Minimal THREE.Texture-shaped stub (only the fields the extractor reads). */
function fakeTex(
  data: ArrayLike<number>,
  width: number,
  height: number,
  flipY = false,
): THREE.Texture {
  return { image: { data, width, height }, flipY } as unknown as THREE.Texture;
}

describe('equirectTextureToPayload', () => {
  it('extracts RGB from an RGBA Float32 texture (stride 4 → 3), no flip', () => {
    // 2×1 image, RGBA: px0 = (1,2,3,9), px1 = (4,5,6,9)
    const rgba = new Float32Array([1, 2, 3, 9, 4, 5, 6, 9]);
    const out = equirectTextureToPayload(fakeTex(rgba, 2, 1));
    expect(out).not.toBeNull();
    expect(out!.width).toBe(2);
    expect(out!.height).toBe(1);
    expect(Array.from(out!.data)).toEqual([1, 2, 3, 4, 5, 6]); // alpha dropped
  });

  it('reverses rows when flipY is true (mirrors the fork)', () => {
    // 1×2 image, RGBA: row0 = (1,1,1), row1 = (2,2,2)
    const rgba = new Float32Array([1, 1, 1, 0, 2, 2, 2, 0]);
    const out = equirectTextureToPayload(fakeTex(rgba, 1, 2, true));
    // flipY → output row 0 reads source row (height-1-0)=1, output row 1 reads source row 0
    expect(Array.from(out!.data)).toEqual([2, 2, 2, 1, 1, 1]);
  });

  it('decodes HalfFloat (Uint16) data', () => {
    // half 0x3c00 = 1.0, 0x4000 = 2.0; one RGBA pixel (1,2,1,_)
    const half = new Uint16Array([0x3c00, 0x4000, 0x3c00, 0x0000]);
    const out = equirectTextureToPayload(fakeTex(half, 1, 1));
    expect(Array.from(out!.data)).toEqual([1, 2, 1]);
  });

  it('handles an RGB (stride 3) source', () => {
    const rgb = new Float32Array([0.5, 0.25, 0.75]); // exactly representable in f32
    const out = equirectTextureToPayload(fakeTex(rgb, 1, 1));
    expect(Array.from(out!.data!)).toEqual([0.5, 0.25, 0.75]);
  });

  it('returns null when the texture has no readable CPU pixels', () => {
    expect(equirectTextureToPayload({ image: undefined, flipY: false } as unknown as THREE.Texture)).toBeNull();
    expect(equirectTextureToPayload({ image: { width: 0, height: 0 }, flipY: false } as unknown as THREE.Texture)).toBeNull();
  });
});
