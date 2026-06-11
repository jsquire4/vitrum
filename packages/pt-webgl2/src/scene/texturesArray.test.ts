import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packTextureAtlas, type TextureHandleHint } from './texturesArray.js';

// packTextureAtlas gathers material-map handles into a sampler2DArray + a
// handle→layer map. These pin the duck-typed pixel read (raw + DataTexture forms),
// layer assignment, and the no-texture short-circuit the G3 path depends on.

/** A DataTexture-shaped handle: { image: { data, width, height } }. */
function dataTexHandle(data: Float32Array, w: number, h: number): unknown {
  return { image: { data, width: w, height: h } };
}

function matWithBaseColorMap(handle: unknown): MaterialSpec {
  return { baseColor: [1, 1, 1], roughness: 1, metallic: 0, baseColorMap: { handle } };
}

describe('packTextureAtlas', () => {
  it('returns null when no material carries a texture', () => {
    const mats: MaterialSpec[] = [{ baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 }];
    expect(packTextureAtlas(mats)).toBeNull();
  });

  it('packs a DataTexture-shaped baseColorMap and assigns it layer 0', () => {
    // 1×1 red RGBA float
    const handle = dataTexHandle(new Float32Array([1, 0, 0, 1]), 1, 1);
    const atlas = packTextureAtlas([matWithBaseColorMap(handle)]);
    expect(atlas).not.toBeNull();
    expect(atlas!.layerCount).toBe(1);
    expect(atlas!.dim).toBe(1);
    expect(atlas!.layerOf.get(handle)).toBe(0);
    expect(Array.from(atlas!.data)).toEqual([1, 0, 0, 1]);
  });

  it('dedups a shared handle across materials to one layer', () => {
    const handle = dataTexHandle(new Float32Array([0.5, 0.5, 0.5, 1]), 1, 1);
    const atlas = packTextureAtlas([matWithBaseColorMap(handle), matWithBaseColorMap(handle)]);
    expect(atlas!.layerCount).toBe(1);
    expect(atlas!.layerOf.get(handle)).toBe(0);
  });

  it('assigns distinct layers to distinct handles and resamples to a common dim', () => {
    const h1 = dataTexHandle(new Float32Array([1, 1, 1, 1]), 1, 1); // 1×1
    const h2 = dataTexHandle(new Float32Array(2 * 2 * 4).fill(0.5), 2, 2); // 2×2
    const atlas = packTextureAtlas([matWithBaseColorMap(h1), matWithBaseColorMap(h2)]);
    expect(atlas!.layerCount).toBe(2);
    expect(atlas!.dim).toBe(2); // max source dim
    expect(atlas!.layerOf.get(h1)).toBe(0);
    expect(atlas!.layerOf.get(h2)).toBe(1);
    // layer 0 (the 1×1 white) nearest-upsampled to 2×2 → all white
    expect(Array.from(atlas!.data.slice(0, 16))).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  });
});

// D10.12: TextureHandleHint — explicit channels/dataType override
describe('TextureHandleHint: readHandlePixels uses explicit hints', () => {
  /** Wrap a raw payload with a __vitrum_hint__ hint. */
  function hintedHandle(
    data: ArrayLike<number>,
    w: number,
    h: number,
    hint: TextureHandleHint,
  ): unknown {
    return { width: w, height: h, data, __vitrum_hint__: hint };
  }

  function mat(handle: unknown): MaterialSpec {
    return { baseColor: [1, 1, 1], roughness: 1, metallic: 0, baseColorMap: { handle } };
  }

  it('explicit channels:4 on Uint8 data gives correct normalized RGBA output', () => {
    // 1×1 red pixel in Uint8 RGBA (255,0,0,255)
    const handle = hintedHandle(new Uint8Array([255, 0, 0, 255]), 1, 1, { channels: 4 });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    // Should decode to [1,0,0,1] after /255 normalization
    expect(atlas!.data[0]).toBeCloseTo(1, 5);
    expect(atlas!.data[1]).toBeCloseTo(0, 5);
    expect(atlas!.data[2]).toBeCloseTo(0, 5);
    expect(atlas!.data[3]).toBeCloseTo(1, 5);
  });

  it('explicit channels:1 expands a single-channel R value to (R,R,R,1)', () => {
    // 1×1 grayscale 128/255 ≈ 0.502
    const handle = hintedHandle(new Uint8Array([128]), 1, 1, { channels: 1 });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    const r = atlas!.data[0]!;
    expect(r).toBeCloseTo(128 / 255, 4);
    // Channels 1 and 2 mirror channel 0 (stride=1 path)
    expect(atlas!.data[1]!).toBe(r);
    expect(atlas!.data[2]!).toBe(r);
    // Alpha = 1 (default for stride < 4)
    expect(atlas!.data[3]).toBe(1);
  });

  it('explicit dataType:float32 skips normalization on Float32 data', () => {
    // 1×1 pixel, RGBA, float32 already in [0,1]
    const handle = hintedHandle(new Float32Array([0.5, 0.25, 0.75, 1.0]), 1, 1, { channels: 4, dataType: 'float32' });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    expect(atlas!.data[0]).toBeCloseTo(0.5, 5);
    expect(atlas!.data[1]).toBeCloseTo(0.25, 5);
    expect(atlas!.data[2]).toBeCloseTo(0.75, 5);
    expect(atlas!.data[3]).toBeCloseTo(1.0, 5);
  });

  it('ambiguous stride (3 channels) without hint emits a console.warn', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      // 1×1, 3-channel RGB (stride 3 is ambiguous without a hint)
      const handle = { width: 1, height: 1, data: new Uint8Array([255, 128, 0]) };
      packTextureAtlas([mat(handle)]);
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('ambiguous pixel stride'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('explicit channels:3 hint suppresses the ambiguous-stride warn', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const handle = hintedHandle(new Uint8Array([255, 128, 0]), 1, 1, { channels: 3 });
      packTextureAtlas([mat(handle)]);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});
