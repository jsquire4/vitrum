import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packMaterialTextureAtlas } from '../pipeline/materialTextureAtlas.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';

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

  it('packs roughnessMap and metallicMap as linear scalar atlas slots', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([10, 128, 200, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      roughnessMap: {
        handle,
        wrapS: 'mirrored-repeat',
        wrapT: 'clamp-to-edge',
      },
      metallicMap: {
        handle,
        texCoord: 1,
        transform: {
          offset: [0.125, 0.25],
          scale: [2, 4],
          rotation: Math.PI,
        },
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBaseColorLayerCount).toBe(0);
    expect(atlas.readableRoughnessLayerCount).toBe(1);
    expect(atlas.readableMetallicLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(10 / 255, 5);
    expect(atlas.atlasData[1]).toBeCloseTo(128 / 255, 5);
    expect(atlas.atlasData[2]).toBeCloseTo(200 / 255, 5);

    // Slot layout: baseColor=[0,1], roughness=[2,3], metallic=[4,5].
    expect(atlas.baseColorMetaData[0]).toBe(-1);
    expect(atlas.baseColorMetaData[8]).toBe(0);
    expect(atlas.baseColorMetaData[9]).toBe(2 + 1 * 4);
    expect(atlas.baseColorMetaData[16]).toBe(0);
    expect(atlas.baseColorMetaData[17]).toBe(16);
    expect(atlas.baseColorMetaData[18]).toBeCloseTo(0.125, 5);
    expect(atlas.baseColorMetaData[19]).toBeCloseTo(0.25, 5);
    expect(atlas.baseColorMetaData[20]).toBeCloseTo(2, 5);
    expect(atlas.baseColorMetaData[21]).toBeCloseTo(4, 5);
    expect(atlas.baseColorMetaData[22]).toBeCloseTo(Math.cos(Math.PI), 5);
    expect(atlas.baseColorMetaData[23]).toBeCloseTo(Math.sin(Math.PI), 5);
  });

  it('shade samples roughness and metallic scalar maps from the atlas', () => {
    expect(SHADE_WGSL).toContain('const MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;');
    expect(SHADE_WGSL).toContain('const MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;');
    expect(SHADE_WGSL).toContain(
      'let rough    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, primaryHit.uv, uv1, rm.x);',
    );
    expect(SHADE_WGSL).toContain(
      'let metal    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_METALLIC, 2u, primaryHit.uv, uv1, rm.y);',
    );
  });
});
