import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec, TextureRef } from '@vitrum/core';
import {
  BASE_COLOR_MAP_META_TEX_WIDTH,
  MATERIAL_MAP_META_TEXEL_OFFSETS,
  MATERIAL_MAP_META_TEXELS_PER_TRI,
  packMaterialTextureAtlas,
  uploadMaterialTextureAtlas,
} from '../pipeline/materialTextureAtlas.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';

const GLTF_TEXTURE_REF_SOURCE = Symbol('vitrum.gltf.textureRefSource');

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function parseMaterialAtlasU32Constants(wgsl: string): Map<string, number> {
  const constants = new Map<string, number>();
  const re = /const\s+([A-Z0-9_]+):\s*u32\s*=\s*(\d+)u;/g;
  let match: RegExpExecArray | null = re.exec(wgsl);
  while (match != null) {
    constants.set(match[1]!, Number(match[2]));
    match = re.exec(wgsl);
  }
  return constants;
}

describe('walkaround materialTextureAtlas', () => {
  it('rejects adversarial metadata dimensions before traversing material data', () => {
    const materialRead = vi.fn();
    const material = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      get baseColorMap() {
        materialRead();
        return undefined;
      },
    } as unknown as MaterialSpec;

    expect(() => packMaterialTextureAtlas(
      [material],
      new Uint32Array(0),
      Number.MAX_SAFE_INTEGER,
    )).toThrow(/material metadata atlas byte length exceeds the safe integer range/);
    expect(materialRead).not.toHaveBeenCalled();
  });

  it('preflights a 4096² decode plus its final atlas before either Float32Array allocation', () => {
    const pixelRead = vi.fn();
    const float32Allocation = vi.fn();
    const data = {
      length: 4,
      get 0() {
        pixelRead();
        return 255;
      },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: {
        handle: {
          width: 4096,
          height: 4096,
          data,
          __vitrum_hint__: { channels: 4, dataType: 'uint8' },
        },
      },
    };
    const triMaterialIds = new Uint32Array([0]);
    const NativeFloat32Array = globalThis.Float32Array;
    const TrackedFloat32Array = new Proxy(NativeFloat32Array, {
      construct(target, args, newTarget) {
        float32Allocation(args[0]);
        return Reflect.construct(target, args, newTarget);
      },
    });
    vi.stubGlobal('Float32Array', TrackedFloat32Array);
    try {
      expect(() => packMaterialTextureAtlas(
        [material],
        triMaterialIds,
        1,
      )).toThrow(
        /baseColorMap RGBA decode requires 268435456 CPU bytes .*above the .*aggregate staging budget/,
      );
      expect(float32Allocation).not.toHaveBeenCalled();
      expect(pixelRead).not.toHaveBeenCalled();
    } finally {
      vi.stubGlobal('Float32Array', NativeFloat32Array);
    }
  });

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

  it('treats Uint16Array atlas handles as normalized uint16 unless explicitly hinted as float16', () => {
    const normalized = {
      width: 1,
      height: 1,
      data: new Uint16Array([32768, 65535, 0, 65535]),
      __vitrum_hint__: { channels: 4, colorSpace: 'linear' },
    };
    const normalizedMaterial: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle: normalized },
    };

    const atlas = packMaterialTextureAtlas([normalizedMaterial], new Uint32Array([0]), 1);

    expect(atlas.atlasData[0]).toBeCloseTo(32768 / 65535, 5);
    expect(atlas.atlasData[1]).toBeCloseTo(1, 5);
    expect(atlas.atlasData[2]).toBeCloseTo(0, 5);
    expect(atlas.atlasData[3]).toBeCloseTo(1, 5);

    const halfFloat = {
      width: 1,
      height: 1,
      data: new Uint16Array([0x3800, 0x3c00, 0, 0x3c00]),
      __vitrum_hint__: { channels: 4, dataType: 'float16', colorSpace: 'linear' },
    };
    const halfMaterial: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle: halfFloat },
    };
    const halfAtlas = packMaterialTextureAtlas([halfMaterial], new Uint32Array([0]), 1);

    expect(halfAtlas.atlasData[0]).toBeCloseTo(0.5, 5);
    expect(halfAtlas.atlasData[1]).toBeCloseTo(1, 5);
    expect(halfAtlas.atlasData[2]).toBeCloseTo(0, 5);
    expect(halfAtlas.atlasData[3]).toBeCloseTo(1, 5);
  });

  it('reports unreadable atlas-backed map handles as diagnostics and disables metadata', () => {
    const baseColorMap = {
      handle: { id: 'gpu-only-texture' },
      [GLTF_TEXTURE_REF_SOURCE]: {
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        textureIndex: 2,
        imageIndex: 3,
        samplerIndex: 4,
        imageUri: 'albedo.webp',
        imageMimeType: 'image/webp',
        textureSourceExtension: 'EXT_texture_webp',
      },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBaseColorLayerCount).toBe(0);
    expect(atlas.baseColorMetaData[0]).toBe(-1);
    expect(atlas.diagnostics).toEqual([
      expect.objectContaining({
        code: 'unreadable-material-texture-map',
        materialIndex: 0,
        field: 'baseColorMap',
        colorSpace: 'srgb',
        sourcePath: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        textureIndex: 2,
        imageIndex: 3,
        samplerIndex: 4,
        imageUri: 'albedo.webp',
        imageMimeType: 'image/webp',
        textureSourceExtension: 'EXT_texture_webp',
      }),
    ]);
    expect(atlas.diagnostics[0]?.message).toContain('materials[0].pbrMetallicRoughness.baseColorTexture');
  });

  it('reports ambiguous raw texture strides as diagnostics without console output', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const baseColorMap = {
        handle: {
          width: 1,
          height: 1,
          data: new Uint8Array([64, 128, 255]),
        },
        [GLTF_TEXTURE_REF_SOURCE]: {
          path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
          textureIndex: 5,
          imageIndex: 6,
        },
      } as TextureRef;
      const material: MaterialSpec = {
        baseColor: [1, 1, 1],
        roughness: 1,
        metallic: 0,
        baseColorMap,
      };

      const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

      expect(warnSpy).not.toHaveBeenCalled();
      expect(atlas.readableBaseColorLayerCount).toBe(1);
      expect(atlas.baseColorMetaData[0]).toBe(0);
      expect(atlas.atlasData[0]).toBeCloseTo(srgbToLinear(64 / 255), 5);
      expect(atlas.atlasData[1]).toBeCloseTo(srgbToLinear(128 / 255), 5);
      expect(atlas.atlasData[2]).toBeCloseTo(1, 5);
      expect(atlas.atlasData[3]).toBeCloseTo(1, 5);
      expect(atlas.diagnostics).toEqual([
        expect.objectContaining({
          code: 'ambiguous-material-texture-stride',
          materialIndex: 0,
          field: 'baseColorMap',
          colorSpace: 'srgb',
          pixelStride: 3,
          valueCount: 3,
          width: 1,
          height: 1,
          sourcePath: 'materials[0].pbrMetallicRoughness.baseColorTexture',
          textureIndex: 5,
          imageIndex: 6,
        }),
      ]);
      expect(atlas.diagnostics[0]?.message).toContain('ambiguous pixel stride 3');
    } finally {
      warnSpy.mockRestore();
    }
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

  it('sanitizes non-finite texture transforms and reports a structured diagnostic', () => {
    const baseColorMap = {
      handle: {
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        __vitrum_hint__: { channels: 4, dataType: 'uint8' },
      },
      transform: {
        offset: [Number.NaN, 0.25],
        scale: [2, Number.POSITIVE_INFINITY],
        rotation: Number.NEGATIVE_INFINITY,
      },
      [GLTF_TEXTURE_REF_SOURCE]: {
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture.extensions.KHR_texture_transform',
        textureIndex: 7,
        imageIndex: 8,
      },
    } as TextureRef;
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.baseColorMetaData[0]).toBe(0);
    expect(atlas.baseColorMetaData[2]).toBe(0);
    expect(atlas.baseColorMetaData[3]).toBeCloseTo(0.25, 5);
    expect(atlas.baseColorMetaData[4]).toBeCloseTo(2, 5);
    expect(atlas.baseColorMetaData[5]).toBe(1);
    expect(atlas.baseColorMetaData[6]).toBeCloseTo(1, 5);
    expect(atlas.baseColorMetaData[7]).toBeCloseTo(0, 5);
    expect(Array.from(atlas.baseColorMetaData.slice(0, 8)).every(Number.isFinite)).toBe(true);
    expect(atlas.diagnostics).toEqual([
      expect.objectContaining({
        code: 'invalid-material-texture-transform',
        materialIndex: 0,
        field: 'baseColorMap',
        colorSpace: 'srgb',
        texCoord: 0,
        transformComponents: ['offset.x', 'scale.y', 'rotation'],
        sourcePath: 'materials[0].pbrMetallicRoughness.baseColorTexture.extensions.KHR_texture_transform',
        textureIndex: 7,
        imageIndex: 8,
      }),
    ]);
    expect(atlas.diagnostics[0]?.message).toContain('offset.x, scale.y, rotation');
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

  it('packs authored mag/min policy with mip filtering explicitly disabled', () => {
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
        magFilter: 'linear',
        minFilter: 'linear',
        mipFilter: 'none',
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBaseColorLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[0]).toBe(0);
    expect(atlas.baseColorMetaData[1]).toBe(1024 + 2048);
    expect(atlas.diagnostics).toEqual([]);
  });

  it('compacts arbitrary texCoord values and packs an exact per-triangle affine chart', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const baseColorMap = {
      handle,
      texCoord: 2,
      [GLTF_TEXTURE_REF_SOURCE]: {
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        textureIndex: 1,
      },
    } as TextureRef;
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap,
    };

    const uv0 = new Float32Array([0, 0, 1, 0, 0, 1]);
    const uv2 = new Float32Array([0.25, -0.5, 2.25, -0.5, 0.25, 2.5]);
    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1, {
      indices: new Uint32Array([0, 1, 2]),
      uv0,
      uvSets: new Map([[2, uv2]]),
    });

    expect(atlas.readableBaseColorLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[0]).toBe(0);
    expect(atlas.baseColorMetaData[1]).toBe(2 * 16);
    const affineBase = MATERIAL_MAP_META_TEXEL_OFFSETS.UV_AFFINE_BASE * 4;
    expect(Array.from(atlas.baseColorMetaData.slice(affineBase, affineBase + 8)))
      .toEqual([2, 0, 0.25, 0, 0, 3, -0.5, 1]);
    expect(atlas.diagnostics).toEqual([]);
  });

  it('fails explicitly when atlas-backed maps exceed the fourteen high-UV affine lanes', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' as const },
    };
    const materials: MaterialSpec[] = Array.from({ length: 15 }, (_, index) => ({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle, texCoord: index + 2 },
    }));

    expect(() => packMaterialTextureAtlas(
      materials,
      Uint32Array.from({ length: 15 }, (_, index) => index),
      15,
    )).toThrow(/15 atlas-backed high UV sets.*14-lane material UV budget/);
  });

  it('fails explicitly when a triangle material references a missing high UV stream', () => {
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
        texCoord: 7,
      },
    };

    expect(() => packMaterialTextureAtlas([material], new Uint32Array([0]), 1, {
      indices: new Uint32Array([0, 1, 2]),
      uv0: new Float32Array([0, 0, 1, 0, 0, 1]),
      uvSets: new Map(),
    })).toThrow(/texCoord 7.*UV stream is missing/);
  });

  it('packs footprint-dependent sampler policies with a complete atlas mip chain', () => {
    const handle = {
      width: 4,
      height: 4,
      data: new Uint8Array(4 * 4 * 4).fill(128),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const baseColorMap = {
      handle,
      magFilter: 'nearest',
      minFilter: 'nearest',
      mipFilter: 'linear',
      [GLTF_TEXTURE_REF_SOURCE]: {
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture.sampler',
        textureIndex: 2,
        imageIndex: 3,
        samplerIndex: 4,
      },
    } as TextureRef;
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBaseColorLayerCount).toBe(1);
    expect(atlas.atlasDim).toBe(4);
    expect(atlas.atlasMipLevelCount).toBe(3);
    expect(atlas.baseColorMetaData[0]).toBe(0);
    expect(atlas.baseColorMetaData[1]).toBe(2 * 256);
    expect(atlas.diagnostics).toEqual([]);
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

  it('preserves baseColorMap alpha for walkaround alpha coverage without requiring alphaMap', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 64]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      alphaMode: 'mask',
      alphaCutoff: 0.5,
      baseColorMap: { handle },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBaseColorLayerCount).toBe(1);
    expect(atlas.readableAlphaLayerCount).toBe(0);
    expect(atlas.atlasData[3]).toBeCloseTo(64 / 255, 5);
    expect(atlas.baseColorMetaData[0]).toBe(0);
    expect(atlas.baseColorMetaData[32]).toBe(-1);
    expect(atlas.baseColorMetaData[40]).toBe(1);
    expect(atlas.baseColorMetaData[41]).toBeCloseTo(1, 5);
    expect(atlas.baseColorMetaData[42]).toBeCloseTo(0.5, 5);
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

  it('packs thicknessMap as a linear G-channel atlas slot for Beer-Lambert tinting', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([0, 192, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      transmission: 1,
      attenuationColor: [0.5, 0.25, 1],
      attenuationDistance: 1,
      thickness: 0.5,
      thicknessMap: {
        handle,
        texCoord: 1,
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableThicknessLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[1]).toBeCloseTo(192 / 255, 5);
    // thicknessMap metadata starts at texel 47.
    expect(atlas.baseColorMetaData[188]).toBe(0);
    expect(atlas.baseColorMetaData[189]).toBe(1 + 2 * 4 + 16);
  });

  it('packs bumpMap as a linear height-field atlas slot with signed bumpScale metadata', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([96, 0, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      bumpMap: {
        handle,
        texCoord: 1,
        wrapS: 'mirrored-repeat',
        wrapT: 'clamp-to-edge',
      },
      bumpScale: -0.35,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBumpLayerCount).toBe(1);
    expect(atlas.atlasLayerCount).toBe(1);
    expect(atlas.atlasData[0]).toBeCloseTo(96 / 255, 5);
    // bumpMap metadata starts at texel 49; bumpScale metadata at texel 51.
    expect(atlas.baseColorMetaData[196]).toBe(0);
    expect(atlas.baseColorMetaData[197]).toBe(2 + 1 * 4 + 16);
    expect(atlas.baseColorMetaData[204]).toBeCloseTo(-0.35, 5);
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
    expect(atlas.baseColorMetaData[84]).toBeCloseTo(0.04 * 0.25, 5);
    expect(atlas.baseColorMetaData[85]).toBeCloseTo(0.04 * 0.5, 5);
    expect(atlas.baseColorMetaData[86]).toBeCloseTo(0.04 * 0.75, 5);
    expect(atlas.baseColorMetaData[87]).toBeCloseTo(0.4, 5);
  });

  it('packs IOR=0 F0 before preserving unbounded nonnegative specularColor', () => {
    const atlas = packMaterialTextureAtlas([{
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      ior: 0,
      specularColor: [2.5, 0.5, 0],
      specularIntensity: 0.25,
    }], new Uint32Array([0]), 1);

    // IOR=0 has base F0=1. Strength remains in alpha and is applied in WGSL
    // after the base F0 clamp; RGB factors above one remain intact.
    expect(Array.from(atlas.baseColorMetaData.slice(84, 88))).toEqual([
      2.5, 0.5, 0, 0.25,
    ]);
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

  it('packs envMapIntensity as non-negative per-triangle material metadata', () => {
    const authored: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      envMapIntensity: 2.5,
    };
    const defaulted: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
    };
    const clamped: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      envMapIntensity: -4,
    };

    expect(packMaterialTextureAtlas([authored], new Uint32Array([0]), 1).baseColorMetaData[208])
      .toBeCloseTo(2.5, 5);
    expect(packMaterialTextureAtlas([defaulted], new Uint32Array([0]), 1).baseColorMetaData[208])
      .toBe(1);
    expect(packMaterialTextureAtlas([clamped], new Uint32Array([0]), 1).baseColorMetaData[208])
      .toBe(0);
  });

  it('packs frontLayer/backLayer transmission, roughness, and layer-local normal maps', () => {
    const frontNormal = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 128, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const backNormal = {
      width: 1,
      height: 1,
      data: new Uint8Array([64, 192, 255, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.8,
      metallic: 0,
      frontLayer: {
        transmission: [1, 0.5, -2],
        roughness: 0.25,
        normalMap: { handle: frontNormal, texCoord: 1, wrapS: 'clamp-to-edge' },
        normalScale: 0.4,
      },
      backLayer: {
        transmission: [0.25, 2, 0.75],
        normalMap: { handle: backNormal, wrapT: 'mirrored-repeat' },
        normalScale: 0.75,
      },
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);
    const front = MATERIAL_MAP_META_TEXEL_OFFSETS.FRONT_LAYER * 4;
    const back = MATERIAL_MAP_META_TEXEL_OFFSETS.BACK_LAYER * 4;
    const frontNormalMeta = MATERIAL_MAP_META_TEXEL_OFFSETS.FRONT_LAYER_NORMAL * 4;
    const frontNormalScale = MATERIAL_MAP_META_TEXEL_OFFSETS.FRONT_LAYER_NORMAL_SCALE * 4;
    const backNormalMeta = MATERIAL_MAP_META_TEXEL_OFFSETS.BACK_LAYER_NORMAL * 4;
    const backNormalScale = MATERIAL_MAP_META_TEXEL_OFFSETS.BACK_LAYER_NORMAL_SCALE * 4;

    expect(atlas.readableNormalLayerCount).toBe(2);
    expect(atlas.atlasLayerCount).toBe(2);
    expect(atlas.baseColorMetaData[front]).toBe(1);
    expect(atlas.baseColorMetaData[front + 1]).toBe(0.5);
    expect(atlas.baseColorMetaData[front + 2]).toBe(0);
    expect(atlas.baseColorMetaData[front + 3]).toBe(0.25);
    expect(atlas.baseColorMetaData[back]).toBe(0.25);
    expect(atlas.baseColorMetaData[back + 1]).toBe(1);
    expect(atlas.baseColorMetaData[back + 2]).toBe(0.75);
    expect(atlas.baseColorMetaData[back + 3]).toBe(-1);
    expect(atlas.baseColorMetaData[frontNormalMeta]).toBe(0);
    expect(atlas.baseColorMetaData[frontNormalMeta + 1]).toBe(1 + 16);
    expect(atlas.baseColorMetaData[frontNormalScale]).toBeCloseTo(0.4, 5);
    expect(atlas.baseColorMetaData[backNormalMeta]).toBe(1);
    expect(atlas.baseColorMetaData[backNormalMeta + 1]).toBe(2 * 4);
    expect(atlas.baseColorMetaData[backNormalScale]).toBeCloseTo(0.75, 5);
  });

  it('packs volume scattering sigmaS and anisotropy metadata', () => {
    const scalar: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.8,
      metallic: 0,
      scatteringCoefficient: 0.4,
      scatteringAnisotropy: 1.5,
    };
    const rgb: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.8,
      metallic: 0,
      scatteringCoefficient: 0.4,
      scatteringCoefficientRGB: [0.1, -1, 2],
      scatteringAnisotropy: -2,
    };

    const scalarAtlas = packMaterialTextureAtlas([scalar], new Uint32Array([0]), 1);
    const rgbAtlas = packMaterialTextureAtlas([rgb], new Uint32Array([0]), 1);
    const offset = MATERIAL_MAP_META_TEXEL_OFFSETS.VOLUME_SCATTERING * 4;

    expect(scalarAtlas.baseColorMetaData[offset]).toBeCloseTo(0.4, 6);
    expect(scalarAtlas.baseColorMetaData[offset + 1]).toBeCloseTo(0.4, 6);
    expect(scalarAtlas.baseColorMetaData[offset + 2]).toBeCloseTo(0.4, 6);
    expect(scalarAtlas.baseColorMetaData[offset + 3]).toBeCloseTo(0.99, 6);
    expect(rgbAtlas.baseColorMetaData[offset]).toBeCloseTo(0.1, 6);
    expect(rgbAtlas.baseColorMetaData[offset + 1]).toBe(0);
    expect(rgbAtlas.baseColorMetaData[offset + 2]).toBe(2);
    expect(rgbAtlas.baseColorMetaData[offset + 3]).toBeCloseTo(-0.99, 6);
  });

  it('packs bump-map source dimensions next to bumpScale metadata', () => {
    const handle = {
      width: 3,
      height: 5,
      data: new Uint8Array(15).fill(128),
      __vitrum_hint__: { channels: 1, dataType: 'uint8', colorSpace: 'linear' },
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      bumpMap: { handle },
      bumpScale: 0.25,
    };

    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);

    expect(atlas.readableBumpLayerCount).toBe(1);
    expect(atlas.baseColorMetaData[204]).toBeCloseTo(0.25, 5);
    expect(atlas.baseColorMetaData[205]).toBe(3);
    expect(atlas.baseColorMetaData[206]).toBe(5);
    expect(atlas.baseColorMetaData[207]).toBe(0);
  });

  it('WGSL atlas metadata constants mirror the host packer offsets', () => {
    const constants = parseMaterialAtlasU32Constants(MATERIAL_ATLAS_WGSL);
    const offsets = MATERIAL_MAP_META_TEXEL_OFFSETS;
    const expected: Readonly<Record<string, number>> = {
      BASE_COLOR_MAP_META_TEX_WIDTH,
      MATERIAL_MAP_META_TEXELS_PER_TRI,
      MATERIAL_MAP_SLOT_BASE_COLOR: offsets.BASE_COLOR / 2,
      MATERIAL_MAP_SLOT_ROUGHNESS: offsets.ROUGHNESS / 2,
      MATERIAL_MAP_SLOT_METALLIC: offsets.METALLIC / 2,
      MATERIAL_MAP_SLOT_AO: offsets.AO / 2,
      MATERIAL_MAP_SLOT_ALPHA: offsets.ALPHA / 2,
      MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET: offsets.ALPHA_COVERAGE,
      MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: offsets.EMISSIVE,
      MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET: offsets.TRANSMISSION,
      MATERIAL_MAP_NORMAL_TEXEL_OFFSET: offsets.NORMAL,
      MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET: offsets.NORMAL_SCALE,
      MATERIAL_MAP_LIGHT_TEXEL_OFFSET: offsets.LIGHT,
      MATERIAL_MAP_LIGHT_INTENSITY_TEXEL_OFFSET: offsets.LIGHT_INTENSITY,
      MATERIAL_MAP_SPECULAR_TEXEL_OFFSET: offsets.SPECULAR,
      MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET: offsets.CLEARCOAT,
      MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET: offsets.SHEEN_COLOR,
      MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET: offsets.SPECULAR_COLOR,
      MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET: offsets.SPECULAR_INTENSITY,
      MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET: offsets.CLEARCOAT_FACTOR,
      MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET: offsets.CLEARCOAT_ROUGHNESS,
      MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET: offsets.SHEEN_COLOR_MAP,
      MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET: offsets.SHEEN_ROUGHNESS,
      MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET: offsets.CLEARCOAT_NORMAL,
      MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET: offsets.CLEARCOAT_NORMAL_SCALE,
      MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET: offsets.ANISOTROPY,
      MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET: offsets.ANISOTROPY_SCALAR,
      MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET: offsets.IRIDESCENCE,
      MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET: offsets.IRIDESCENCE_THICKNESS,
      MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET: offsets.IRIDESCENCE_SCALAR,
      MATERIAL_MAP_THICKNESS_TEXEL_OFFSET: offsets.THICKNESS,
      MATERIAL_MAP_BUMP_TEXEL_OFFSET: offsets.BUMP,
      MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET: offsets.BUMP_SCALE,
      MATERIAL_MAP_ENV_INTENSITY_TEXEL_OFFSET: offsets.ENV_INTENSITY,
      MATERIAL_MAP_FRONT_LAYER_TEXEL_OFFSET: offsets.FRONT_LAYER,
      MATERIAL_MAP_BACK_LAYER_TEXEL_OFFSET: offsets.BACK_LAYER,
      MATERIAL_MAP_VOLUME_SCATTERING_TEXEL_OFFSET: offsets.VOLUME_SCATTERING,
    };

    for (const [name, expectedValue] of Object.entries(expected)) {
      expect(constants.get(name), name).toBe(expectedValue);
    }
  });

  it('shade and traversal sample material maps from the shared atlas module', () => {
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleEmissiveMap(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleTransmissionMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleLightMap(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fallback * materialMapChannel(texelColor, channel)');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleSpecularControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleClearcoatControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleSheenControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleSheenRoughness(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleAnisotropyControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleIridescenceControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleEnvMapIntensity(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleFaceLayerControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn faceLayerTransmission(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn faceLayerRoughness(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleVolumeScatteringControls(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyHomogeneousVolumeSingleScatter(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyThicknessMapToBeerTint(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyFaceLayerNormalMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET');
    expect(MATERIAL_ATLAS_WGSL).toContain('MATERIAL_MAP_BACK_LAYER_NORMAL_TEXEL_OFFSET');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyNormalMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyBumpMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('let bumpTexelStep = vec2f(');
    expect(MATERIAL_ATLAS_WGSL).toContain('1.0 / max(scaleMeta.y, 1.0)');
    expect(MATERIAL_ATLAS_WGSL).toContain('1.0 / max(scaleMeta.z, 1.0)');
    expect(MATERIAL_ATLAS_WGSL).toContain('scaleMeta.y > 0.0 && scaleMeta.z > 0.0');
    expect(MATERIAL_ATLAS_WGSL).not.toContain('1.0 / 512.0');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn applyClearcoatNormalMapForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialAtlasLodForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('Bounded realtime footprint model');
    expect(MATERIAL_ATLAS_WGSL).toContain('propagated ray-differential model');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleMaterialAtlasRawAtOffsetForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialAtlasFilterMode(samplerPacked: u32, lod: f32)');
    expect(MATERIAL_ATLAS_WGSL).toContain('textureNumLevels(materialTextureAtlas)');
    expect(MATERIAL_ATLAS_WGSL).toContain('textureDimensions(materialTextureAtlas, level)');
    expect(MATERIAL_ATLAS_WGSL).not.toContain('sampleMaterialAtlasBaseLevel');
    expect(MATERIAL_ATLAS_WGSL).toContain('@group(1) @binding(23) var bvh_vertex_color: texture_2d<f32>;');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleVertexColorForHit(hit: IntersectionResult) -> vec4f');
    expect(MATERIAL_ATLAS_WGSL).toContain('struct MaterialAlphaCoverage');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialAlphaCoverageForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn traceSceneFirstHitAlphaMaskTextured(');
    expect(MATERIAL_ATLAS_WGSL).toContain('traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(');
    expect(MATERIAL_ATLAS_WGSL).not.toContain('fn traceSceneAnyAlphaMaskTextured(');
    expect(MATERIAL_ATLAS_WGSL).not.toContain('fn materialShadowOccluderForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialShadowTransmittanceForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn traceSceneAlphaTransmittanceTextured(');
    expect(MATERIAL_ATLAS_WGSL).toContain('if ((materialWord & 1u) != 0u)');
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'packedMaterialHasTransmission(hit.matColorPacked)',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('let baseColorAlpha = select(clamp(baseColorTexel.a, 0.0, 1.0), 1.0, baseColorTexel.x < 0.0);');
    expect(MATERIAL_ATLAS_WGSL).toContain('let alphaMapCoverage = select(clamp(alphaTexel.r, 0.0, 1.0), 1.0, alphaTexel.x < 0.0);');
    expect(MATERIAL_ATLAS_WGSL).toContain('let vertexColorAlpha = sampleVertexColorForHit(hit).a;');
    expect(MATERIAL_ATLAS_WGSL).toContain('out.coverage = clamp(opacity * vertexColorAlpha * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);');
    expect(MATERIAL_ATLAS_WGSL).toContain('return alpha.coverage < alpha.cutoff;');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialAlphaBlendCoverageHash(');
    expect(MATERIAL_ATLAS_WGSL).toContain('sampleSeed: u32,');
    expect(MATERIAL_ATLAS_WGSL).toContain('materialAlphaBlendCoverageHash(hit, ray, layer, sampleSeed) >= alpha.coverage;');
    expect(MATERIAL_ATLAS_WGSL).toContain('return clamp(1.0 - alpha.coverage, 0.0, 1.0);');
    expect(MATERIAL_ATLAS_WGSL).not.toContain('tau <= 0.001');
    expect(SHADE_WGSL).toContain('let vertexColor = sampleVertexColorForHit(primaryHit);');
    expect(SHADE_WGSL).toContain('let layerControls = sampleFaceLayerControls(primaryHit.indices.w, primaryHit.side >= 0.0);');
    expect(SHADE_WGSL).toContain('let layerTransmission = faceLayerTransmission(layerControls);');
    expect(SHADE_WGSL).toContain('let volumeScattering = sampleVolumeScatteringControls(primaryHit.indices.w);');
    expect(SHADE_WGSL).toContain(
      'let albedo   = sampleBaseColorMap(primaryHit, matColor.rgb * vertexColor.rgb);',
    );
    expect(SHADE_WGSL).toContain('let rough    = faceLayerRoughness(');
    expect(SHADE_WGSL).toContain('layerControls,');
    expect(SHADE_WGSL).toContain(
      'let metal    = sampleMaterialScalarMap(primaryHit, MATERIAL_MAP_SLOT_METALLIC, 2u, rm.y);',
    );
    expect(SHADE_WGSL).toContain('let specular = sampleSpecularControls(primaryHit);');
    expect(SHADE_WGSL).toContain('let normalMapped = applyNormalMapForHit(primaryHit, smoothNormal);');
    expect(SHADE_WGSL).toContain('let normal = applyBumpMapForHit(primaryHit, normalMapped);');
    expect(SHADE_WGSL).toContain('let clearcoatNormal = applyClearcoatNormalMapForHit(primaryHit, smoothNormal, normal);');
    expect(SHADE_WGSL).toContain('let clearcoat = sampleClearcoatControls(primaryHit);');
    expect(SHADE_WGSL).toContain('let sheen = sampleSheenControls(primaryHit);');
    expect(SHADE_WGSL).toContain('let sheenRoughness = sampleSheenRoughness(primaryHit);');
    expect(SHADE_WGSL).toContain('let anisotropy = sampleAnisotropyControls(primaryHit);');
    expect(SHADE_WGSL).toContain('let anisotropyFrame = materialTangentFrameForHit(primaryHit, normal, MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);');
    expect(SHADE_WGSL).toContain('let iridescence = sampleIridescenceControls(primaryHit);');
    expect(SHADE_WGSL).toContain('let envMapIntensity = sampleEnvMapIntensity(primaryHit.indices.w);');
    expect(SHADE_WGSL).toContain(
      'let authoredAo = sampleAoMapFactor(primaryHit, materialWord);',
    );
    expect(SHADE_WGSL).toContain('traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(');
    expect(SHADE_WGSL).toContain('let Lo_emitterGlow = sampleEmissiveMap(');
    expect(SHADE_WGSL).toContain('let lightMapIrradiance = sampleLightMap(primaryHit);');
    expect(SHADE_WGSL).toContain('let Lo_lightMap = albedo * INV_PI * lightMapIrradiance;');
    expect(SHADE_WGSL).toContain('envMapIntensity, isGlass, isMetal, &rng)');
    expect(SHADE_WGSL).toContain('sampleTransmissionMapForHit(primaryHit, scalarMatColor.a)');
    expect(SHADE_WGSL).toContain('lo_emit(matColor, normal, isGlass, primaryHit.uv, uv1');
    expect(SHADE_WGSL).toContain('lo_transmittedGI(pix, dims, isGlass)');
    expect(SHADE_WGSL).toContain(') * authoredAo;');
  });

  it('cleans up the first atlas texture when a later candidate allocation fails', () => {
    const firstDestroy = vi.fn();
    let allocation = 0;
    const device = {
      createTexture: vi.fn(() => {
        allocation += 1;
        if (allocation === 2) throw new Error('meta texture allocation failed');
        return {
          createView: vi.fn(() => ({})),
          destroy: firstDestroy,
        };
      }),
      queue: { writeTexture: vi.fn() },
    } as unknown as GPUDevice;
    const payload = packMaterialTextureAtlas([], new Uint32Array([0]), 1);

    expect(() => uploadMaterialTextureAtlas(device, payload))
      .toThrow('meta texture allocation failed');
    expect(firstDestroy).toHaveBeenCalledTimes(1);
  });
});
