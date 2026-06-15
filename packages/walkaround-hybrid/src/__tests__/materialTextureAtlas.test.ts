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
    // transmission=[13,14], normal=[15,16], normalScale=[17],
    // lightMap=[18,19], lightMapIntensity=[20].
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
    expect(atlas.baseColorMetaData[60]).toBe(-1);
    expect(atlas.baseColorMetaData[72]).toBe(-1);
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

  it('packs normalMap as a linear tangent-space atlas slot with normalScale metadata', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 128, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      normalMap: {
        handle,
        texCoord: 1,
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
      },
      normalScale: 0.5,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableNormalLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(128 / 255, 5);
    expect(atlas.atlasData[1]).toBeCloseTo(128 / 255, 5);
    expect(atlas.atlasData[2]).toBeCloseTo(1, 5);
    // normal slot starts at texel 15; normalScale metadata at texel 17.
    expect(atlas.baseColorMetaData[60]).toBe(0);
    expect(atlas.baseColorMetaData[61]).toBe(1 + 2 * 4 + 16);
    expect(atlas.baseColorMetaData[68]).toBeCloseTo(0.5, 5);
  });

  it('packs lightMap as a linear atlas slot with intensity metadata', () => {
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
      lightMap: {
        handle,
        texCoord: 1,
        wrapS: 'mirrored-repeat',
        wrapT: 'clamp-to-edge',
      },
      lightMapIntensity: 2.5,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableLightLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(64 / 255, 5);
    expect(atlas.atlasData[1]).toBeCloseTo(128 / 255, 5);
    expect(atlas.atlasData[2]).toBeCloseTo(1, 5);
    // lightMap slot starts at texel 18; lightMapIntensity metadata at texel 20.
    expect(atlas.baseColorMetaData[72]).toBe(0);
    expect(atlas.baseColorMetaData[73]).toBe(2 + 1 * 4 + 16);
    expect(atlas.baseColorMetaData[80]).toBeCloseTo(2.5, 5);
  });

  it('packs scalar specular controls into per-triangle material metadata', () => {
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      specularColor: [0.25, 0.5, 0.75],
      specularIntensity: 0.4,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    // specular metadata lives at texel 21 (vec4 lanes 84..87).
    expect(atlas.baseColorMetaData[84]).toBeCloseTo(0.25, 5);
    expect(atlas.baseColorMetaData[85]).toBeCloseTo(0.5, 5);
    expect(atlas.baseColorMetaData[86]).toBeCloseTo(0.75, 5);
    expect(atlas.baseColorMetaData[87]).toBeCloseTo(0.4, 5);
  });

  it('packs specular texture maps into atlas metadata with the expected color spaces', () => {
    const specularColorHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const specularIntensityHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([0, 0, 0, 128]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      specularColorMap: { handle: specularColorHandle },
      specularIntensityMap: { handle: specularIntensityHandle, texCoord: 1 },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableSpecularColorLayerCount).toBe(1);
    expect(atlas.readableSpecularIntensityLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[96]).toBe(0);  // specularColorMap layer
    expect(atlas.baseColorMetaData[104]).toBe(1); // specularIntensityMap layer
    expect(atlas.baseColorMetaData[105]).toBe(16); // uv1 selector
    expect(atlas.atlasData[0]).toBeCloseTo(srgbToLinear(128 / 255), 5);
    expect(atlas.atlasData[1]).toBeCloseTo(srgbToLinear(64 / 255), 5);
    expect(atlas.atlasData[2]).toBeCloseTo(1, 5);
    const intensityLayerBase = 4;
    expect(atlas.atlasData[intensityLayerBase + 3]).toBeCloseTo(128 / 255, 5);
  });

  it('packs clearcoat and sheen texture maps into atlas metadata with glTF channels', () => {
    const clearcoatHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 0, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const clearcoatRoughnessHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([0, 192, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const clearcoatNormalHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 192, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const sheenColorHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([64, 128, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const sheenRoughnessHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([0, 0, 0, 96]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      clearcoatMap: { handle: clearcoatHandle },
      clearcoatRoughnessMap: { handle: clearcoatRoughnessHandle, texCoord: 1 },
      clearcoatNormalMap: { handle: clearcoatNormalHandle },
      clearcoatNormalScale: 0.25,
      sheenColorMap: { handle: sheenColorHandle },
      sheenRoughnessMap: { handle: sheenRoughnessHandle, texCoord: 1 },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableClearcoatLayerCount).toBe(1);
    expect(atlas.readableClearcoatRoughnessLayerCount).toBe(1);
    expect(atlas.readableClearcoatNormalLayerCount).toBe(1);
    expect(atlas.readableSheenColorLayerCount).toBe(1);
    expect(atlas.readableSheenRoughnessLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[112]).toBe(0); // clearcoatMap layer
    expect(atlas.baseColorMetaData[120]).toBe(1); // clearcoatRoughnessMap layer
    expect(atlas.baseColorMetaData[121]).toBe(16); // uv1 selector
    expect(atlas.baseColorMetaData[144]).toBe(2); // clearcoatNormalMap layer
    expect(atlas.baseColorMetaData[152]).toBeCloseTo(0.25, 5); // clearcoatNormalScale
    expect(atlas.baseColorMetaData[128]).toBe(3); // sheenColorMap layer
    expect(atlas.baseColorMetaData[136]).toBe(4); // sheenRoughnessMap layer
    expect(atlas.baseColorMetaData[137]).toBe(16); // uv1 selector
    expect(atlas.atlasData[0]).toBeCloseTo(128 / 255, 5);
    expect(atlas.atlasData[4 + 1]).toBeCloseTo(192 / 255, 5);
    const clearcoatNormalLayerBase = 8;
    expect(atlas.atlasData[clearcoatNormalLayerBase + 1]).toBeCloseTo(192 / 255, 5);
    const sheenColorLayerBase = 12;
    expect(atlas.atlasData[sheenColorLayerBase]).toBeCloseTo(srgbToLinear(64 / 255), 5);
    expect(atlas.atlasData[sheenColorLayerBase + 1]).toBeCloseTo(srgbToLinear(128 / 255), 5);
    expect(atlas.atlasData[sheenColorLayerBase + 2]).toBeCloseTo(1, 5);
    const sheenRoughnessLayerBase = 16;
    expect(atlas.atlasData[sheenRoughnessLayerBase + 3]).toBeCloseTo(96 / 255, 5);
  });

  it('packs anisotropy controls and KHR anisotropy maps into atlas metadata', () => {
    const anisotropyHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([255, 128, 64, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      anisotropy: 0.5,
      anisotropyRotation: 0.25,
      anisotropyMap: { handle: anisotropyHandle, texCoord: 1 },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableAnisotropyLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[156]).toBe(0); // anisotropyMap layer (texel 39)
    expect(atlas.baseColorMetaData[157]).toBe(16); // uv1 selector
    expect(atlas.baseColorMetaData[164]).toBeCloseTo(0.5, 5); // anisotropy scalar (texel 41)
    expect(atlas.baseColorMetaData[165]).toBeCloseTo(0.25, 5); // anisotropyRotation
    expect(atlas.atlasData[0]).toBeCloseTo(1, 5); // direction.r
    expect(atlas.atlasData[1]).toBeCloseTo(128 / 255, 5); // direction.g
    expect(atlas.atlasData[2]).toBeCloseTo(64 / 255, 5); // strength.b
  });

  it('packs iridescence controls and KHR iridescence maps into atlas metadata', () => {
    const iridescenceHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 0, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const thicknessHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([0, 192, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      iridescence: 0.5,
      iridescenceIor: 2,
      iridescenceThicknessRange: [200, 800],
      iridescenceMap: { handle: iridescenceHandle },
      iridescenceThicknessMap: { handle: thicknessHandle, texCoord: 1 },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableIridescenceLayerCount).toBe(1);
    expect(atlas.readableIridescenceThicknessLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[168]).toBe(0); // iridescenceMap layer (texel 42)
    expect(atlas.baseColorMetaData[176]).toBe(1); // iridescenceThicknessMap layer (texel 44)
    expect(atlas.baseColorMetaData[177]).toBe(16); // uv1 selector
    expect(atlas.baseColorMetaData[184]).toBeCloseTo(0.5, 5); // scalar factor
    expect(atlas.baseColorMetaData[185]).toBeCloseTo(2, 5); // iridescence IOR
    expect(atlas.baseColorMetaData[186]).toBeCloseTo(200, 5); // min thickness nm
    expect(atlas.baseColorMetaData[187]).toBeCloseTo(800, 5); // max thickness nm
    expect(atlas.atlasData[0]).toBeCloseTo(128 / 255, 5); // factor.r
    expect(atlas.atlasData[4 + 1]).toBeCloseTo(192 / 255, 5); // thickness.g
  });

  it('packs scalar clearcoat controls into per-triangle material metadata', () => {
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      clearcoat: 0.65,
      clearcoatRoughness: 0.25,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    // clearcoat metadata lives at texel 22 (vec4 lanes 88..91).
    expect(atlas.baseColorMetaData[88]).toBeCloseTo(0.65, 5);
    expect(atlas.baseColorMetaData[89]).toBeCloseTo(0.25, 5);
    expect(atlas.baseColorMetaData[90]).toBe(0); // sheen
    expect(atlas.baseColorMetaData[91]).toBe(0); // sheenRoughness
    expect(atlas.baseColorMetaData[92]).toBe(0); // sheenColor.r
    expect(atlas.baseColorMetaData[93]).toBe(0); // sheenColor.g
    expect(atlas.baseColorMetaData[94]).toBe(0); // sheenColor.b
    expect(atlas.baseColorMetaData[95]).toBe(0);
  });

  it('packs scalar sheen controls into per-triangle material metadata', () => {
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      sheen: 0.7,
      sheenRoughness: 0.35,
      sheenColor: [0.25, 0.5, 0.75],
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    // sheen scalar metadata shares texel 22 (vec4 lanes 90..91);
    // sheen color metadata lives at texel 23 (vec4 lanes 92..95).
    expect(atlas.baseColorMetaData[90]).toBeCloseTo(0.7, 5);
    expect(atlas.baseColorMetaData[91]).toBeCloseTo(0.35, 5);
    expect(atlas.baseColorMetaData[92]).toBeCloseTo(0.25, 5);
    expect(atlas.baseColorMetaData[93]).toBeCloseTo(0.5, 5);
    expect(atlas.baseColorMetaData[94]).toBeCloseTo(0.75, 5);
    expect(atlas.baseColorMetaData[95]).toBe(0);
  });

  it('shade and traversal sample material maps from the shared atlas module', () => {
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 47u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_AO: u32 = 3u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET: u32 = 13u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_NORMAL_TEXEL_OFFSET: u32 = 15u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_LIGHT_TEXEL_OFFSET: u32 = 18u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SPECULAR_TEXEL_OFFSET: u32 = 21u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET: u32 = 22u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET: u32 = 23u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET: u32 = 24u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET: u32 = 26u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET: u32 = 28u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET: u32 = 30u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET: u32 = 32u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET: u32 = 34u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET: u32 = 36u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET: u32 = 38u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET: u32 = 39u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET: u32 = 41u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET: u32 = 42u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET: u32 = 44u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('const MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET: u32 = 46u;');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleEmissiveMap(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleTransmissionMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleLightMap(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleSpecularControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleClearcoatControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleSheenControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleSheenRoughness(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleAnisotropyControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleIridescenceControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyNormalMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyClearcoatNormalMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn traceSceneFirstHitAlphaMaskTextured(');
    expect(MATERIAL_ATLAS_WGSL).toContain('return coverage < cutoff;');
    expect(SHADE_WGSL).toContain(
      'let rough    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, primaryHit.uv, uv1, rm.x);',
    );
    expect(SHADE_WGSL).toContain(
      'let metal    = sampleMaterialScalarMap(primaryHit.indices.w, MATERIAL_MAP_SLOT_METALLIC, 2u, primaryHit.uv, uv1, rm.y);',
    );
    expect(SHADE_WGSL).toContain('let specular = sampleSpecularControls(primaryHit.indices.w, primaryHit.uv, uv1);');
    expect(SHADE_WGSL).toContain('let clearcoatNormal = applyClearcoatNormalMapForHit(primaryHit, smoothNormal, normal);');
    expect(SHADE_WGSL).toContain('let clearcoat = sampleClearcoatControls(primaryHit.indices.w, primaryHit.uv, uv1);');
    expect(SHADE_WGSL).toContain('let sheen = sampleSheenControls(primaryHit.indices.w, primaryHit.uv, uv1);');
    expect(SHADE_WGSL).toContain('let sheenRoughness = sampleSheenRoughness(primaryHit.indices.w, primaryHit.uv, uv1);');
    expect(SHADE_WGSL).toContain('let anisotropy = sampleAnisotropyControls(primaryHit.indices.w, primaryHit.uv, uv1);');
    expect(SHADE_WGSL).toContain('let iridescence = sampleIridescenceControls(primaryHit.indices.w, primaryHit.uv, uv1);');
    expect(SHADE_WGSL).toContain(
      'let authoredAo = sampleAoMapFactor(primaryHit.indices.w, materialWord, primaryHit.uv, uv1);',
    );
    expect(SHADE_WGSL).toContain('traceSceneFirstHitAlphaMaskTextured(');
    expect(SHADE_WGSL).toContain('let normal = applyNormalMapForHit(primaryHit, smoothNormal);');
    expect(SHADE_WGSL).toContain('let Lo_emitterGlow = sampleEmissiveMap(');
    expect(SHADE_WGSL).toContain('let Lo_lightMap = sampleLightMap(primaryHit.indices.w, primaryHit.uv, uv1);');
    expect(SHADE_WGSL).toContain('sampleTransmissionMapForHit(primaryHit, scalarMatColor.a)');
    expect(SHADE_WGSL).toContain(') * authoredAo;');
  });
});
