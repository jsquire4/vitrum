import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packMaterialTextureAtlas } from '../pipeline/materialTextureAtlas.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';

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

    // Slot layout: baseColor=[0,1], roughness=[2,3], metallic=[4,5], ao=[6,7],
    // alpha=[8,9], alphaCoverage=[10], emissive=[11,12],
    // transmission=[13,14].
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
    expect(atlas.baseColorMetaData[24]).toBe(-1);
    expect(atlas.baseColorMetaData[32]).toBe(-1);
    expect(atlas.baseColorMetaData[44]).toBe(-1);
    expect(atlas.baseColorMetaData[52]).toBe(-1);
  });

  it('packs aoMap as a linear R-channel atlas slot', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([96, 128, 200, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      aoMap: {
        handle,
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableAoLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(96 / 255, 5);
    expect(atlas.atlasData[1]).toBeCloseTo(128 / 255, 5);
    expect(atlas.atlasData[2]).toBeCloseTo(200 / 255, 5);
    // Slot layout: baseColor=[0,1], roughness=[2,3], metallic=[4,5], ao=[6,7].
    expect(atlas.baseColorMetaData[0]).toBe(-1);
    expect(atlas.baseColorMetaData[8]).toBe(-1);
    expect(atlas.baseColorMetaData[16]).toBe(-1);
    expect(atlas.baseColorMetaData[24]).toBe(0);
    expect(atlas.baseColorMetaData[25]).toBe(1 + 2 * 4);
  });

  it('packs alphaMap as a linear atlas slot with coverage metadata', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([64, 128, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      alphaMode: 'mask',
      opacity: 0.75,
      alphaCutoff: 0.4,
      alphaMap: {
        handle,
        texCoord: 1,
        wrapS: 'mirrored-repeat',
        wrapT: 'clamp-to-edge',
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableAlphaLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(64 / 255, 5);
    expect(atlas.atlasData[1]).toBeCloseTo(128 / 255, 5);
    // alpha slot starts at texel 8, coverage scalars at texel 10.
    expect(atlas.baseColorMetaData[32]).toBe(0);
    expect(atlas.baseColorMetaData[33]).toBe(2 + 1 * 4 + 16);
    expect(atlas.baseColorMetaData[40]).toBe(1);
    expect(atlas.baseColorMetaData[41]).toBeCloseTo(0.75, 5);
    expect(atlas.baseColorMetaData[42]).toBeCloseTo(0.4, 5);
  });

  it('packs emissiveMap as an sRGB-decoded atlas slot for visible emitter glow', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emissive: [2, 2, 2],
      emissiveIntensity: 3,
      emissiveMap: {
        handle,
        texCoord: 1,
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableEmissiveLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(srgbToLinear(128 / 255), 5);
    expect(atlas.atlasData[1]).toBeCloseTo(srgbToLinear(64 / 255), 5);
    expect(atlas.atlasData[2]).toBeCloseTo(1, 5);
    // emissive slot starts at texel 11.
    expect(atlas.baseColorMetaData[44]).toBe(0);
    expect(atlas.baseColorMetaData[45]).toBe(1 + 2 * 4 + 16);
  });

  it('packs transmissionMap as a linear R-channel atlas slot for glass gating', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([192, 64, 32, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      transmission: 1,
      transmissionMap: {
        handle,
        texCoord: 1,
        wrapS: 'mirrored-repeat',
        wrapT: 'clamp-to-edge',
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableTransmissionLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(192 / 255, 5);
    expect(atlas.atlasData[1]).toBeCloseTo(64 / 255, 5);
    // transmission slot starts at texel 13.
    expect(atlas.baseColorMetaData[52]).toBe(0);
    expect(atlas.baseColorMetaData[53]).toBe(2 + 1 * 4 + 16);
  });

  it('shade and traversal sample material maps from the shared atlas module', () => {
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_AO: u32 = 3u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET: u32 = 13u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleEmissiveMap(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleTransmissionMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn traceSceneFirstHitAlphaMaskTextured(');
    expect(MATERIAL_ATLAS_WGSL).toContain('return coverage < cutoff;');
    expect(SHADE_WGSL).toContain(
      'let rough    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, primaryHit.uv, uv1, rm.x);',
    );
    expect(SHADE_WGSL).toContain(
      'let metal    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_METALLIC, 2u, primaryHit.uv, uv1, rm.y);',
    );
    expect(SHADE_WGSL).toContain(
      'let authoredAo = sampleAoMapFactor(primaryHit.indices.w, materialWord, primaryHit.uv, uv1);',
    );
    expect(SHADE_WGSL).toContain('traceSceneFirstHitAlphaMaskTextured(');
    expect(SHADE_WGSL).toContain('let Lo_emitterGlow = sampleEmissiveMap(');
    expect(SHADE_WGSL).toContain('sampleTransmissionMapForHit(primaryHit, scalarMatColor.a)');
    expect(SHADE_WGSL).toContain(') * authoredAo;');
  });
});
