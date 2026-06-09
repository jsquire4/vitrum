import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packTextureAtlas } from './texturesArray.js';

// packTextureAtlas gathers material-map handles into a sampler2DArray + a
// handle→layer map. These pin the duck-typed pixel read (raw + DataTexture forms),
// layer assignment, and the no-texture short-circuit the G3 path depends on.

/** A DataTexture-shaped handle: { image: { data, width, height } }. */
function dataTexHandle(data: Float32Array, w: number, h: number): unknown {
  return { image: { data, width: w, height: h } };
}

function matWithBaseColorMap(handle: unknown): MaterialSpec {
  return { baseColor: [1, 1, 1], roughness: 1, metallic: 0, baseColorMap: { handle } } as MaterialSpec;
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
