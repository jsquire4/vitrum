import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packMaterialTextureAtlas } from '../pipeline/materialTextureAtlas.js';

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

describe('walkaround materialTextureAtlas', () => {
  it('decodes readable baseColorMap handles into a linear atlas layer', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 255, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const material: MaterialSpec = {
      baseColor: [0.5, 0.5, 0.5],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBaseColorLayerCount).toBe(1);
    expect(atlas.atlasDim).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(srgbToLinear(128 / 255), 5);
    expect(atlas.atlasData[1]).toBeCloseTo(1, 5);
    expect(atlas.atlasData[2]).toBeCloseTo(0, 5);
    expect(atlas.atlasData[3]).toBeCloseTo(1, 5);
    expect(atlas.baseColorMetaData[0]).toBe(0);
  });

  it('packs per-triangle wrap and KHR_texture_transform metadata', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const rotation = Math.PI / 2;
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: {
        handle,
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
        transform: {
          offset: [0.25, 0.5],
          scale: [2, 3],
          rotation,
        },
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.baseColorMetaData[0]).toBe(0);
    expect(atlas.baseColorMetaData[1]).toBe(1 + 2 * 4);
    expect(atlas.baseColorMetaData[2]).toBeCloseTo(0.25, 5);
    expect(atlas.baseColorMetaData[3]).toBeCloseTo(0.5, 5);
    expect(atlas.baseColorMetaData[4]).toBeCloseTo(2, 5);
    expect(atlas.baseColorMetaData[5]).toBeCloseTo(3, 5);
    expect(atlas.baseColorMetaData[6]).toBeCloseTo(Math.cos(rotation), 5);
    expect(atlas.baseColorMetaData[7]).toBeCloseTo(Math.sin(rotation), 5);
  });

  it('keeps texCoord 1 baseColorMap layers readable and records the uv-set selector', () => {
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: {
        handle: {
          width: 1,
          height: 1,
          data: new Uint8Array([255, 255, 255, 255]),
          __vitrum_hint__: { channels: 4, dataType: 'uint8' },
        },
        texCoord: 1,
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBaseColorLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[0]).toBe(0);
    expect(atlas.baseColorMetaData[1]).toBe(16);
  });
});
