import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES,
  materialTextureAtlasLayerCapacities,
  packMaterialTextureAtlases,
  packTextureAtlas,
  snapshotMaterialTextureInputs,
  textureAtlasLayerCapacity,
  textureAtlasLayerCapacityForStorage,
  textureAtlasMipElementCounts,
  textureAtlasStorageByteLength,
  uploadTextureAtlas,
  type TextureAtlas,
  type TextureHandleHint,
} from './texturesArray.js';
import { packMaterialsTexture } from './materialsTexture.js';
import { collectMaterialTexCoords } from './uvAttributeLayout.js';
import {
  FLOAT16_HALF_MIN_SUBNORMAL,
  FLOAT16_MAX_FINITE,
  FLOAT16_MIN_SUBNORMAL,
  float16BitsToFloat32,
} from './halfFloat.js';

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

function matWithEmissiveMap(
  handle: unknown,
  sampler: Omit<NonNullable<MaterialSpec['emissiveMap']>, 'handle'> = {},
): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 1,
    metallic: 0,
    emissive: [1, 1, 1],
    emissiveMap: { handle, ...sampler },
  };
}

function srgbToLinear(v: number): number {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

function ldrLinear(data: Uint8Array, index: number): number {
  return (data[index] ?? 0) / 255;
}

function ldrSrgbLinear(data: Uint8Array, index: number): number {
  return srgbToLinear(ldrLinear(data, index));
}

function oneShotUint8Handle(
  label: string,
  values: readonly [number, number, number, number],
): { readonly handle: unknown; readonly reads: ReadonlyMap<string, number> } {
  const reads = new Map<string, number>();
  const once = (target: object, key: PropertyKey, value: unknown, field: string): void => {
    Object.defineProperty(target, key, {
      enumerable: true,
      configurable: true,
      get() {
        const count = (reads.get(field) ?? 0) + 1;
        reads.set(field, count);
        if (count !== 1) throw new Error(`${label}.${field} was read more than once`);
        return value;
      },
    });
  };

  const data = {};
  once(data, 'length', values.length, 'data.length');
  values.forEach((value, index) => once(data, index, value, `data[${index}]`));

  const handle = {};
  once(handle, 'cpuMirror', undefined, 'cpuMirror');
  once(handle, 'data', data, 'data');
  once(handle, 'width', 1, 'width');
  once(handle, 'height', 1, 'height');
  once(handle, 'channels', 4, 'channels');
  once(handle, 'dataType', 'uint8', 'dataType');
  once(handle, 'colorSpace', undefined, 'colorSpace');
  once(handle, '__vitrum_hint__', undefined, '__vitrum_hint__');
  return { handle, reads };
}

function atlasSourceTexel(
  atlas: TextureAtlas,
  colorSpace: 'srgb' | 'linear',
  handle: unknown,
): readonly [number, number, number, number] {
  const placement = atlas.layerOfByColorSpace.placements?.[colorSpace].get(handle);
  if (placement == null) throw new Error(`missing ${colorSpace} test placement`);
  const offset =
    placement.layer * atlas.dim * atlas.dim * 4 +
    (placement.y * atlas.dim + placement.x) * 4;
  return [
    atlas.data[offset] ?? 0,
    atlas.data[offset + 1] ?? 0,
    atlas.data[offset + 2] ?? 0,
    atlas.data[offset + 3] ?? 0,
  ];
}

function makeTextureAtlasUploadGl(
  error = 0,
  limits: {
    readonly maxTextureSize?: number;
    readonly maxArrayTextureLayers?: number;
  } = {},
) {
  const texture = {} as WebGLTexture;
  const deleteTexture = vi.fn();
  const bindTexture = vi.fn();
  const texStorage3D = vi.fn();
  const texSubImage3D = vi.fn();
  const texImage3D = vi.fn();
  const gl = {
    TEXTURE_2D_ARRAY: 0x8c1a,
    RGBA8: 0x8058,
    RGBA16F: 0x881a,
    RGBA: 0x1908,
    UNSIGNED_BYTE: 0x1401,
    HALF_FLOAT: 0x140b,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    TEXTURE_BASE_LEVEL: 0x813c,
    TEXTURE_MAX_LEVEL: 0x813d,
    NEAREST: 0x2600,
    CLAMP_TO_EDGE: 0x812f,
    MAX_TEXTURE_SIZE: 0x0d33,
    MAX_ARRAY_TEXTURE_LAYERS: 0x88ff,
    NO_ERROR: 0,
    isContextLost: vi.fn(() => false),
    getParameter: vi.fn((parameter: number) =>
      parameter === 0x0d33
        ? (limits.maxTextureSize ?? 256)
        : (limits.maxArrayTextureLayers ?? 256),
    ),
    getError: vi.fn(() => error),
    createTexture: vi.fn(() => texture),
    deleteTexture,
    bindTexture,
    texParameteri: vi.fn(),
    texStorage3D,
    texSubImage3D,
    texImage3D,
  } as unknown as WebGL2RenderingContext;
  return { gl, texture, deleteTexture, bindTexture, texStorage3D, texSubImage3D, texImage3D };
}

describe('packTextureAtlas', () => {
  it('returns null when no material carries a texture', () => {
    const mats: MaterialSpec[] = [{ baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 }];
    expect(packTextureAtlas(mats)).toBeNull();
  });

  it('shares one immutable staging token across UV, atlas, descriptor, and HDR packing', () => {
    const reads = new Map<string, number>();
    const once = (
      target: object,
      key: PropertyKey,
      value: unknown,
      label: string,
    ): void => {
      Object.defineProperty(target, key, {
        enumerable: true,
        configurable: true,
        get() {
          reads.set(label, (reads.get(label) ?? 0) + 1);
          return value;
        },
      });
    };

    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 32, 255]),
    };
    const transform = {};
    once(transform, 'rotation', 0.25, 'transform.rotation');
    const ref = {};
    once(ref, 'handle', handle, 'ref.handle');
    once(ref, 'texCoord', 1, 'ref.texCoord');
    once(ref, 'transform', transform, 'ref.transform');
    once(ref, 'mipFilter', 'linear', 'ref.mipFilter');
    const material = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emissive: [1, 1, 1],
    } as unknown as MaterialSpec;
    once(material, 'baseColorMap', ref, 'material.baseColorMap');
    once(material, 'emissiveMap', ref, 'material.emissiveMap');

    const staged = snapshotMaterialTextureInputs([material]);
    expect(snapshotMaterialTextureInputs(staged)).toBe(staged);
    expect(Object.isFrozen(staged)).toBe(true);
    expect(Object.isFrozen(staged[0])).toBe(true);
    expect(Object.isFrozen(staged[0]!.baseColorMap)).toBe(true);

    expect(collectMaterialTexCoords(staged)).toContain(1);
    const atlases = packMaterialTextureAtlases(staged);
    packMaterialsTexture(staged, {
      ldr: atlases.ldr?.layerOfByColorSpace ?? null,
      hdr: atlases.hdr?.layerOfByColorSpace ?? null,
    });

    expect([...reads.entries()]).toEqual([
      ['material.baseColorMap', 1],
      ['material.emissiveMap', 1],
      ['ref.handle', 1],
      ['ref.texCoord', 1],
      ['ref.transform', 1],
      ['ref.mipFilter', 1],
      ['transform.rotation', 1],
    ]);
  });

  it('observes one raw handle once across LDR sRGB, LDR linear, and HDR sRGB entries', () => {
    const source = oneShotUint8Handle('shared', [128, 64, 32, 255]);
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emissive: [1, 1, 1],
      baseColorMap: { handle: source.handle },
      normalMap: { handle: source.handle },
      emissiveMap: { handle: source.handle },
    };

    const atlases = packMaterialTextureAtlases([material]);
    expect(atlases.ldr).not.toBeNull();
    expect(atlases.hdr).not.toBeNull();
    expect(atlasSourceTexel(atlases.ldr!, 'srgb', source.handle)).toEqual([
      128, 64, 32, 255,
    ]);
    expect(atlasSourceTexel(atlases.ldr!, 'linear', source.handle)).toEqual([
      128, 64, 32, 255,
    ]);

    const hdrTexel = atlasSourceTexel(atlases.hdr!, 'srgb', source.handle);
    expect(float16BitsToFloat32(hdrTexel[0])).toBeCloseTo(srgbToLinear(128 / 255), 3);
    expect(float16BitsToFloat32(hdrTexel[1])).toBeCloseTo(srgbToLinear(64 / 255), 3);
    expect(float16BitsToFloat32(hdrTexel[2])).toBeCloseTo(srgbToLinear(32 / 255), 3);
    expect(float16BitsToFloat32(hdrTexel[3])).toBe(1);
    expect([...source.reads.values()]).toEqual(new Array(source.reads.size).fill(1));
  });

  it('keeps distinct raw handles independently inspected and snapshotted', () => {
    const baseColor = oneShotUint8Handle('baseColor', [11, 22, 33, 255]);
    const emissive = oneShotUint8Handle('emissive', [201, 155, 99, 255]);
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emissive: [1, 1, 1],
      baseColorMap: { handle: baseColor.handle },
      emissiveMap: { handle: emissive.handle },
    };

    const atlases = packMaterialTextureAtlases([material]);
    expect(atlasSourceTexel(atlases.ldr!, 'srgb', baseColor.handle)).toEqual([
      11, 22, 33, 255,
    ]);
    const hdrTexel = atlasSourceTexel(atlases.hdr!, 'srgb', emissive.handle);
    expect(float16BitsToFloat32(hdrTexel[0])).toBeCloseTo(srgbToLinear(201 / 255), 3);
    expect(float16BitsToFloat32(hdrTexel[1])).toBeCloseTo(srgbToLinear(155 / 255), 3);
    expect(float16BitsToFloat32(hdrTexel[2])).toBeCloseTo(srgbToLinear(99 / 255), 3);
    expect([...baseColor.reads.values()]).toEqual(new Array(baseColor.reads.size).fill(1));
    expect([...emissive.reads.values()]).toEqual(new Array(emissive.reads.size).fill(1));
  });

  it('fails the whole atlas build when any authored handle is unreadable', () => {
    const readable = dataTexHandle(new Float32Array([1, 0, 0, 1]), 1, 1);
    const unreadable = { id: 'opaque-texture-without-cpu-mirror' };

    expect(() =>
      packTextureAtlas([matWithBaseColorMap(readable), matWithBaseColorMap(unreadable)], {
        warningMethod: 'setScene',
      }),
    ).toThrow(/authored material texture during setScene is not CPU-readable/);
  });

  it('admits three 2048² layers without decoded-source or spare-capacity CPU copies', () => {
    let numericElementReads = 0;
    const virtualFloat32Data = (): ArrayLike<number> =>
      new Proxy(
        {
          length: 2_048 * 2_048 * 4,
          [Symbol.toStringTag]: 'Float32Array',
        } as unknown as ArrayLike<number>,
        {
          get(target, property, receiver) {
            if (typeof property === 'string' && /^(0|[1-9]\d*)$/.test(property)) {
              numericElementReads += 1;
              throw new Error('source-read-reached');
            }
            return Reflect.get(target, property, receiver);
          },
        },
      );
    const sources = Array.from({ length: 3 }, () => ({
      width: 2_048,
      height: 2_048,
      data: virtualFloat32Data(),
    }));

    const NativeFloat32Array = globalThis.Float32Array;
    let decodedAllocations = 0;
    class CountingFloat32Array extends NativeFloat32Array {
      constructor(length: number) {
        decodedAllocations += 1;
        super(length);
      }
    }
    vi.stubGlobal('Float32Array', CountingFloat32Array);
    let thrown: unknown;
    try {
      packTextureAtlas(sources.map(matWithBaseColorMap), { maxArrayTextureLayers: 256 });
    } catch (error) {
      thrown = error;
    } finally {
      vi.unstubAllGlobals();
    }

    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).message).toBe('source-read-reached');
    expect(numericElementReads).toBe(1);
    expect(decodedAllocations).toBe(1);
  });

  it('decodes source pixels directly into retained atlas level zero', () => {
    const handles = [
      dataTexHandle(new Float32Array([1, 0, 0, 1]), 1, 1),
      dataTexHandle(new Float32Array([0, 1, 0, 1]), 1, 1),
      dataTexHandle(new Float32Array([0, 0, 1, 1]), 1, 1),
    ];
    const NativeUint8Array = globalThis.Uint8Array;
    const allocationLengths: number[] = [];
    class CountingUint8Array extends NativeUint8Array {
      constructor(length: number) {
        allocationLengths.push(length);
        super(length);
      }
    }
    vi.stubGlobal('Uint8Array', CountingUint8Array);
    try {
      const atlas = packTextureAtlas(handles.map(matWithBaseColorMap));
      expect(atlas?.layerCount).toBe(3);
      expect(Array.from(atlas?.data ?? [])).toHaveLength(12);
    } finally {
      vi.unstubAllGlobals();
    }

    expect(allocationLengths).toEqual([12]);
  });

  it('reserves spare GPU layers without expanding retained mip data on the CPU', () => {
    const handles = Array.from({ length: 3 }, (_, index) =>
      dataTexHandle(new Float32Array(2 * 2 * 4).fill((index + 1) / 4), 2, 2),
    );
    const atlas = packTextureAtlas(handles.map(matWithBaseColorMap))!;
    const { gl, texture, texStorage3D, texSubImage3D, texImage3D } = makeTextureAtlasUploadGl();

    expect(uploadTextureAtlas(gl, atlas, { layerCapacity: 4 })).toBe(texture);
    expect(texStorage3D).toHaveBeenCalledOnce();
    expect(texStorage3D).toHaveBeenCalledWith(gl.TEXTURE_2D_ARRAY, 2, gl.RGBA8, 2, 2, 4);
    expect(texImage3D).not.toHaveBeenCalled();
    expect(texSubImage3D).toHaveBeenCalledTimes(2);
    atlas.mipLevels.forEach((level, lod) => {
      const call = texSubImage3D.mock.calls[lod];
      expect(call?.slice(0, 10)).toEqual([
        gl.TEXTURE_2D_ARRAY,
        lod,
        0,
        0,
        0,
        level.dim,
        level.dim,
        3,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
      ]);
      expect(call?.[10]).toBe(level.data);
    });
  });

  it('uploads outgoing-radiance maps as RGBA16F/HALF_FLOAT', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([8, 4, 2, 1]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      emissiveMap: { handle },
    };
    const atlas = packTextureAtlas([material], { storageClass: 'hdr' })!;
    const { gl, texStorage3D, texSubImage3D } = makeTextureAtlasUploadGl();

    uploadTextureAtlas(gl, atlas, { layerCapacity: 2 });

    expect(texStorage3D).toHaveBeenCalledWith(
      gl.TEXTURE_2D_ARRAY,
      1,
      gl.RGBA16F,
      1,
      1,
      2,
    );
    expect(texSubImage3D.mock.calls[0]?.[9]).toBe(gl.HALF_FLOAT);
    expect(float16BitsToFloat32(atlas.data[0]!)).toBe(8);
  });

  it.each([
    ['maximum finite', FLOAT16_MAX_FINITE, 0x7bff],
    ['minimum positive subnormal', FLOAT16_MIN_SUBNORMAL, 0x0001],
  ])('stores the HDR %s boundary exactly', (_label, value, expectedBits) => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([value]),
      __vitrum_hint__: {
        channels: 1,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const atlas = packTextureAtlas(
      [matWithEmissiveMap(handle)],
      { storageClass: 'hdr' },
    )!;
    expect(atlas.data[0]).toBe(expectedBits);
  });

  it.each([
    ['positive half-minimum tie', FLOAT16_HALF_MIN_SUBNORMAL, /\+0/],
  ])('fails closed when an HDR level-0 %s encodes to signed zero', (
    _label,
    value,
    zeroPattern,
  ) => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([value]),
      __vitrum_hint__: {
        channels: 1,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    expect(() =>
      packTextureAtlas(
        [matWithEmissiveMap(handle)],
        { storageClass: 'hdr' },
      ),
    ).toThrow(zeroPattern);
  });

  it('rejects a reachable generated HDR mip that underflows at the half-minimum tie', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        FLOAT16_MIN_SUBNORMAL, 0, 0, 1,
        0, 0, 0, 1,
      ]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };

    // Omitted mipFilter follows the public linear-mip default, so the generated
    // half-minimum level is reachable and must fail closed. Only an explicit
    // level-0-only sampler makes it inert.
    expect(packTextureAtlas(
      [matWithEmissiveMap(handle, { mipFilter: 'none' })],
      { storageClass: 'hdr' },
    )).not.toBeNull();
    expect(() =>
      packTextureAtlas(
        [matWithEmissiveMap(handle)],
        { storageClass: 'hdr' },
      ),
    ).toThrow(/generated HDR mip 1 .* underflows to \+0/);
  });

  it.each([
    ['NaN', Number.NaN, /decoded pixel data must be finite/],
    ['positive infinity', Number.POSITIVE_INFINITY, /decoded pixel data must be finite/],
    ['negative infinity', Number.NEGATIVE_INFINITY, /decoded pixel data must be finite/],
    ['positive overflow', 65_520, /exceeds the finite RGBA16F range/],
  ])('rejects HDR %s before materialization', (_label, value, message) => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([value]),
      __vitrum_hint__: {
        channels: 1,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    expect(() =>
      packTextureAtlas(
        [matWithEmissiveMap(handle)],
        { storageClass: 'hdr' },
      ),
    ).toThrow(message);
  });

  it('rejects negative linear RGB for both outgoing-radiance atlas roles', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([-FLOAT16_MIN_SUBNORMAL, 0, 0, 1]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const lightMapMaterial: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      lightMap: { handle },
    };

    for (const material of [matWithEmissiveMap(handle), lightMapMaterial]) {
      expect(() =>
        packTextureAtlas(
          [material],
          { storageClass: 'hdr' },
        ),
      ).toThrow(/outgoing-radiance RGB value .* must be non-negative/);
    }
  });

  it.each([
    ['negative maximum finite', -FLOAT16_MAX_FINITE, 0xfbff],
    ['minimum negative subnormal', -FLOAT16_MIN_SUBNORMAL, 0x8001],
  ])('keeps the HDR alpha lane format-generic at the %s boundary', (
    _label,
    alpha,
    expectedBits,
  ) => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([0, 0, 0, alpha]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    const atlas = packTextureAtlas(
      [matWithEmissiveMap(handle)],
      { storageClass: 'hdr' },
    )!;
    expect(atlas.data[3]).toBe(expectedBits);
  });

  it.each([
    ['negative half-minimum tie', -FLOAT16_HALF_MIN_SUBNORMAL, /underflows to -0/],
    ['negative overflow', -65_520, /exceeds the finite RGBA16F range/],
  ])('checks the format-generic HDR alpha %s', (_label, alpha, message) => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([0, 0, 0, alpha]),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };
    expect(() =>
      packTextureAtlas(
        [matWithEmissiveMap(handle)],
        { storageClass: 'hdr' },
      ),
    ).toThrow(message);
  });

  it('deletes and unbinds an atlas candidate when immutable upload reports a GL error', () => {
    const atlas = packTextureAtlas([
      matWithBaseColorMap(dataTexHandle(new Float32Array([1, 1, 1, 1]), 1, 1)),
    ])!;
    const { gl, texture, deleteTexture, bindTexture } = makeTextureAtlasUploadGl(0x0505);

    expect(() => uploadTextureAtlas(gl, atlas)).toThrow(/WebGL error 0x505/);
    expect(deleteTexture).toHaveBeenCalledOnce();
    expect(deleteTexture).toHaveBeenCalledWith(texture);
    expect(bindTexture).toHaveBeenLastCalledWith(gl.TEXTURE_2D_ARRAY, null);
  });

  it('rejects over-budget immutable storage before creating a GL texture', () => {
    const placeholder = new Uint8Array(4);
    const atlas = {
      data: placeholder,
      storageClass: 'ldr',
      format: 'rgba8unorm',
      dim: 2_048,
      mipLevels: [{ data: placeholder, dim: 2_048 }],
      layerCount: 1,
      layerOfByColorSpace: { srgb: new Map(), linear: new Map() },
      sourceDimensions: [[1, 1]],
      sourcePlacements: [{ layer: 0, x: 0, y: 0, width: 1, height: 1 }],
    } satisfies TextureAtlas;
    const { gl, texStorage3D } = makeTextureAtlasUploadGl(0, {
      maxTextureSize: 2_048,
      maxArrayTextureLayers: 256,
    });

    expect(() => uploadTextureAtlas(gl, atlas, { layerCapacity: 32 })).toThrow(
      /allocation requests .* bytes .* exceeding the 536870912-byte storage budget/,
    );
    expect(gl.createTexture).not.toHaveBeenCalled();
    expect(texStorage3D).not.toHaveBeenCalled();
  });

  it('packs a DataTexture-shaped baseColorMap and assigns it layer 0', () => {
    // 1×1 red RGBA float
    const handle = dataTexHandle(new Float32Array([1, 0, 0, 1]), 1, 1);
    const atlas = packTextureAtlas([matWithBaseColorMap(handle)]);
    expect(atlas).not.toBeNull();
    expect(atlas!.layerCount).toBe(1);
    expect(atlas!.dim).toBe(1);
    expect(atlas!.mipLevels).toHaveLength(1);
    expect(atlas!.mipLevels[0]!.data).toBe(atlas!.data);
    expect(atlas!.layerOfByColorSpace.srgb.get(handle)).toBe(0);
    expect(Array.from(atlas!.data)).toEqual([255, 0, 0, 255]);
  });

  it('dedups a shared handle across materials to one layer', () => {
    const handle = dataTexHandle(new Float32Array([0.5, 0.5, 0.5, 1]), 1, 1);
    const atlas = packTextureAtlas([matWithBaseColorMap(handle), matWithBaseColorMap(handle)]);
    expect(atlas!.layerCount).toBe(1);
    expect(atlas!.layerOfByColorSpace.srgb.get(handle)).toBe(0);
  });

  it('stores heterogeneous handles at native extent without resampling', () => {
    const h1 = dataTexHandle(new Float32Array([1, 1, 1, 1]), 1, 1); // 1×1
    const h2 = dataTexHandle(new Float32Array(2 * 2 * 4).fill(0.5), 2, 2); // 2×2
    const atlas = packTextureAtlas([matWithBaseColorMap(h1), matWithBaseColorMap(h2)]);
    expect(atlas!.layerCount).toBe(2);
    expect(atlas!.dim).toBe(2); // max source dim
    expect(atlas!.layerOfByColorSpace.srgb.get(h2)).toBe(0);
    expect(atlas!.layerOfByColorSpace.srgb.get(h1)).toBe(1);
    // The largest source owns layer 0. The 1×1 source keeps its native texel in
    // layer 1; padding is never addressable through its packed placement.
    expect(Array.from(atlas!.data.slice(16, 32))).toEqual([
      255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    expect(atlas!.sourceDimensions[0]).toEqual([1, 1]);
    expect(atlas!.sourceDimensions[1]).toEqual([2, 2]);
    expect(atlas!.layerOfByColorSpace.dimensions?.get(h1)).toEqual([1, 1]);
    expect(atlas!.layerOfByColorSpace.dimensions?.get(h2)).toEqual([2, 2]);
    expect(atlas!.mipLevels).toHaveLength(2);
    expect(atlas!.mipLevels[1]!.dim).toBe(1);
    expect(ldrSrgbLinear(atlas!.mipLevels[1]!.data as Uint8Array, 0))
      .toBeCloseTo(0.5, 2);
    expect(ldrSrgbLinear(atlas!.mipLevels[1]!.data as Uint8Array, 1))
      .toBeCloseTo(0.5, 2);
    expect(ldrSrgbLinear(atlas!.mipLevels[1]!.data as Uint8Array, 2))
      .toBeCloseTo(0.5, 2);
    expect(ldrLinear(atlas!.mipLevels[1]!.data as Uint8Array, 3)).toBeCloseTo(0.5, 2);
    expect(Array.from(atlas!.mipLevels[1]!.data.slice(4, 8))).toEqual([0, 0, 0, 0]);
  });

  it('packs small sources together instead of charging each the largest source extent', () => {
    const large = dataTexHandle(new Float32Array(8 * 8 * 4).fill(0.25), 8, 8);
    const small = Array.from({ length: 16 }, (_, index) =>
      dataTexHandle(
        new Float32Array([
          (index + 1) / 32,
          (index + 2) / 32,
          (index + 3) / 32,
          1,
        ]),
        1,
        1,
      ),
    );
    const atlas = packTextureAtlas([
      matWithBaseColorMap(large),
      ...small.map(matWithBaseColorMap),
    ])!;

    expect(atlas.dim).toBe(8);
    expect(atlas.layerCount).toBe(2);
    expect(
      new Set(small.map((handle) => atlas.layerOfByColorSpace.srgb.get(handle))),
    ).toEqual(new Set([1]));
    const placementKeys = new Set(
      small.map((handle) => {
        const placement = atlas.layerOfByColorSpace.placements?.srgb.get(handle);
        return `${placement?.layer}:${placement?.x}:${placement?.y}`;
      }),
    );
    expect(placementKeys.size).toBe(16);

    const oneSourcePerLayerBytes = textureAtlasStorageByteLength(atlas.dim, 17);
    const packedBytes = textureAtlasStorageByteLength(atlas.dim, atlas.layerCount);
    expect(packedBytes * 8).toBeLessThan(oneSourcePerLayerBytes);
  });

  it('keeps HDR radiance above one and sRGB-coded LDR precision in separate mips', () => {
    const extentAnchor = {
      width: 8,
      height: 8,
      data: new Float32Array(8 * 8 * 4),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' } as const,
    };
    const hdr = {
      width: 2,
      height: 2,
      data: new Float32Array([
        4, 2, 1, 1,
        8, 4, 2, 1,
        12, 6, 3, 1,
        16, 8, 4, 1,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' } as const,
    };
    const encoded = {
      width: 2,
      height: 2,
      data: new Uint8Array([
        128, 64, 32, 255,
        128, 64, 32, 255,
        128, 64, 32, 255,
        128, 64, 32, 255,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8' } as const,
    };
    const hdrAtlas = packTextureAtlas([
      {
        baseColor: [1, 1, 1],
        roughness: 1,
        metallic: 0,
        lightMap: { handle: extentAnchor },
      },
      {
        baseColor: [1, 1, 1],
        roughness: 1,
        metallic: 0,
        emissiveMap: { handle: hdr },
      },
    ], { storageClass: 'hdr' })!;
    const hdrPlacement = hdrAtlas.layerOfByColorSpace.placements?.srgb.get(hdr);

    const readHdr = (
      level: number,
      placement: NonNullable<typeof hdrPlacement>,
      channel: number,
    ): number => {
      const mip = hdrAtlas.mipLevels[level]!;
      const x = Math.floor(placement.x / 2 ** level);
      const y = Math.floor(placement.y / 2 ** level);
      const bits =
        mip.data[(placement.layer * mip.dim * mip.dim + y * mip.dim + x) * 4 + channel]!;
      return float16BitsToFloat32(bits);
    };
    expect(readHdr(0, hdrPlacement!, 0)).toBe(4);
    expect(readHdr(1, hdrPlacement!, 0)).toBe(10);

    const ldrAtlas = packTextureAtlas([
      matWithBaseColorMap(extentAnchor),
      matWithBaseColorMap(encoded),
    ])!;
    const encodedPlacement = ldrAtlas.layerOfByColorSpace.placements?.srgb.get(encoded);
    const encodedOffset =
      ((encodedPlacement!.layer * ldrAtlas.dim + encodedPlacement!.y) * ldrAtlas.dim +
        encodedPlacement!.x) * 4;
    expect((ldrAtlas.data as Uint8Array)[encodedOffset]).toBe(128);
    expect(ldrSrgbLinear(ldrAtlas.data as Uint8Array, encodedOffset))
      .toBeCloseTo(srgbToLinear(128 / 255), 6);
  });

  it('builds each heterogeneous layer mip from only its native source rectangle', () => {
    const vertical = dataTexHandle(new Float32Array([1, 0, 0, 1, 0, 0, 1, 1]), 1, 2);
    const horizontal = dataTexHandle(new Float32Array([0, 1, 0, 1, 1, 1, 1, 1]), 2, 1);
    const atlas = packTextureAtlas([
      matWithBaseColorMap(vertical),
      matWithBaseColorMap(horizontal),
    ])!;
    expect(atlas.dim).toBe(2);
    expect(atlas.mipLevels[1]!.dim).toBe(1);
    const mip = atlas.mipLevels[1]!.data as Uint8Array;
    expect(ldrSrgbLinear(mip, 0)).toBeCloseTo(0.5, 2);
    expect(ldrSrgbLinear(mip, 1)).toBeCloseTo(0, 2);
    expect(ldrSrgbLinear(mip, 2)).toBeCloseTo(0.5, 2);
    expect(ldrLinear(mip, 3)).toBe(1);
    expect(ldrSrgbLinear(mip, 4)).toBeCloseTo(0.5, 2);
    expect(ldrSrgbLinear(mip, 5)).toBeCloseTo(1, 2);
    expect(ldrSrgbLinear(mip, 6)).toBeCloseTo(0.5, 2);
    expect(ldrLinear(mip, 7)).toBe(1);
  });

  it('averages every source texel when generating odd-sized mip levels', () => {
    const data = new Float32Array(3 * 3 * 4);
    for (let i = 0; i < 9; i += 1) {
      data[i * 4] = i / 8;
      data[i * 4 + 3] = 1;
    }
    const handle = dataTexHandle(data, 3, 3);
    const atlas = packTextureAtlas([
      {
        baseColor: [1, 1, 1],
        roughness: 1,
        metallic: 0,
        roughnessMap: { handle },
      },
    ]);

    expect(atlas).not.toBeNull();
    expect(atlas!.mipLevels).toHaveLength(2);
    expect(atlas!.mipLevels[1]!.dim).toBe(1);
    expect(ldrLinear(atlas!.mipLevels[1]!.data as Uint8Array, 0)).toBeCloseTo(0.5, 2);
    expect(ldrLinear(atlas!.mipLevels[1]!.data as Uint8Array, 3)).toBeCloseTo(1, 6);
  });

  it('collects front/back layer normal maps as linear atlas layers', () => {
    const front = dataTexHandle(new Float32Array([0.5, 0.5, 1, 1]), 1, 1);
    const back = dataTexHandle(new Float32Array([0.25, 0.5, 1, 1]), 1, 1);
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      frontLayer: { transmission: [1, 1, 1], normalMap: { handle: front } },
      backLayer: { transmission: [1, 1, 1], normalMap: { handle: back } },
    };
    const atlas = packTextureAtlas([material]);
    expect(atlas).not.toBeNull();
    expect(atlas!.layerCount).toBe(2);
    expect(atlas!.layerOfByColorSpace.linear.get(front)).toBe(0);
    expect(atlas!.layerOfByColorSpace.linear.get(back)).toBe(1);
    expect(atlas!.layerOfByColorSpace.srgb.size).toBe(0);
  });

  it('keeps authored sampler policy warning-free because filtering is shader-resolved', () => {
    const handle = dataTexHandle(new Float32Array([1, 1, 1, 1]), 1, 1);
    const atlas = packTextureAtlas(
      [
        {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          baseColorMap: {
            handle,
            magFilter: 'linear',
            minFilter: 'linear',
            mipFilter: 'linear',
          },
        },
      ],
      { warningMethod: 'setScene' },
    );

    expect(atlas).not.toBeNull();
  });
});

describe('textureAtlasLayerCapacity', () => {
  it('allocates exactly the live layer count because atlas growth rebuilds the scene', () => {
    expect(textureAtlasLayerCapacity(0, 256)).toBe(0);
    expect(textureAtlasLayerCapacity(1, 256)).toBe(1);
    expect(textureAtlasLayerCapacity(2, 256)).toBe(2);
    expect(textureAtlasLayerCapacity(3, 256)).toBe(3);
  });

  it('clamps the capacity seam to the device layer limit', () => {
    expect(textureAtlasLayerCapacity(3, 3)).toBe(3);
    expect(textureAtlasLayerCapacity(4, 6)).toBe(4);
    expect(textureAtlasLayerCapacity(7, 6)).toBe(6);
  });

  it('accounts for exact RGBA8/RGBA16F mip bytes at maximum ordinary extent', () => {
    expect(textureAtlasStorageByteLength(4, 2, 'ldr')).toBe(168);
    expect(textureAtlasStorageByteLength(4, 2, 'hdr')).toBe(336);
    expect(textureAtlasLayerCapacityForStorage(8_192, 1, 256, 'ldr')).toBe(1);
    expect(textureAtlasStorageByteLength(8_192, 1, 'ldr')).toBeLessThanOrEqual(
      MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES,
    );
    expect(textureAtlasStorageByteLength(8_192, 2, 'ldr')).toBeGreaterThan(
      MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES,
    );
    expect(textureAtlasStorageByteLength(4_096, 1, 'hdr')).toBeLessThanOrEqual(
      MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES,
    );
  });

  it('rejects live format-native storage that exceeds the byte budget', () => {
    expect(() => textureAtlasLayerCapacityForStorage(8_192, 2, 256, 'ldr')).toThrow(
      /bytes, exceeding the 536870912-byte storage budget/,
    );
    expect(() => textureAtlasLayerCapacityForStorage(8_192, 1, 256, 'hdr')).toThrow(
      /bytes, exceeding the 536870912-byte storage budget/,
    );
  });

  it('keeps the exact live pair under one 512 MiB ceiling', () => {
    const capacities = materialTextureAtlasLayerCapacities(
      { dim: 8_192, layerCount: 1 },
      { dim: 4_096, layerCount: 1 },
      256,
    );
    expect(capacities).toEqual({
      ldr: 1,
      hdr: 1,
      storageBytes: MATERIAL_TEXTURE_ATLAS_STORAGE_BUDGET_BYTES - 4,
    });
    expect(() =>
      materialTextureAtlasLayerCapacities(
        { dim: 8_192, layerCount: 1 },
        { dim: 4_096, layerCount: 2 },
        256,
      ),
    ).toThrow(/combined material texture atlases require .* exceeding the 536870912-byte storage budget/);
  });

  it('keeps huge valid capacities exact and rejects hostile layer limits', () => {
    expect(
      materialTextureAtlasLayerCapacities(
        { dim: 1, layerCount: 33_554_433 },
        { dim: 1, layerCount: 33_554_433 },
        134_217_728,
      ),
    ).toEqual({
      ldr: 33_554_433,
      hdr: 33_554_433,
      storageBytes: 402_653_196,
    });
    expect(() =>
      materialTextureAtlasLayerCapacities(
        { dim: 1, layerCount: 1 },
        null,
        Number.POSITIVE_INFINITY,
      ),
    ).toThrow(/MAX_ARRAY_TEXTURE_LAYERS must be a positive safe integer/);
  });
});

describe('textureAtlasMipElementCounts', () => {
  it('returns the exact preflight allocation for every mip', () => {
    expect(textureAtlasMipElementCounts(4, 2)).toEqual([128, 32, 8]);
  });

  it('rejects an aggregate mip chain that exceeds the CPU staging budget', () => {
    // Level zero is below the cap; mip one is what pushes the retained chain over it.
    expect(() => textureAtlasMipElementCounts(11_000, 1, 'ldr')).toThrow(
      /512 MiB CPU staging budget at mip 1/,
    );
  });

  it('rejects invalid dimensions and layer counts before any allocation', () => {
    expect(() => textureAtlasMipElementCounts(Number.POSITIVE_INFINITY, 1)).toThrow(
      /positive safe integers/,
    );
    expect(() => textureAtlasMipElementCounts(1, 0)).toThrow(/positive safe integers/);
  });
});

describe('packMaterialTextureAtlases shared CPU preflight', () => {
  it('rejects an over-budget pair before reading payload values or materializing either atlas', () => {
    let pixelReads = 0;
    const virtualFloat32Payload = (length: number): ArrayLike<number> =>
      new Proxy(
        {
          length,
          [Symbol.toStringTag]: 'Float32Array',
        } as unknown as ArrayLike<number>,
        {
          get(target, property, receiver) {
            if (typeof property === 'string' && /^\d+$/.test(property)) {
              pixelReads += 1;
              throw new Error('pixel payload was scanned before aggregate preflight');
            }
            return Reflect.get(target, property, receiver);
          },
        },
      );
    const ldrDim = 8_192;
    const hdrDim = 4_097;
    const ldrHandle = {
      width: ldrDim,
      height: ldrDim,
      data: virtualFloat32Payload(ldrDim * ldrDim * 4),
      __vitrum_hint__: { channels: 4, dataType: 'float32' } as const,
    };
    const hdrHandle = {
      width: hdrDim,
      height: hdrDim,
      data: virtualFloat32Payload(hdrDim * hdrDim * 4),
      __vitrum_hint__: {
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      } as const,
    };

    expect(() =>
      packMaterialTextureAtlases([
        {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          baseColorMap: { handle: ldrHandle },
          emissiveMap: { handle: hdrHandle },
        },
      ]),
    ).toThrow(
      /combined material texture atlas CPU mip chains require .* exceeding the shared 536870912-byte staging budget before allocation/,
    );
    expect(pixelReads).toBe(0);
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

  it('explicit channels:4 retains baseColorMap sRGB coding in RGBA8', () => {
    // 1×1 red pixel in Uint8 RGBA (255,0,0,255)
    const handle = hintedHandle(new Uint8Array([255, 0, 0, 255]), 1, 1, {
      channels: 4,
      dataType: 'uint8',
    });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    expect(Array.from(atlas!.data)).toEqual([255, 0, 0, 255]);
  });

  it('explicit channels:1 expands one sRGB code to (R,R,R,255)', () => {
    // 1×1 grayscale 128/255 ≈ 0.502
    const handle = hintedHandle(new Uint8Array([128]), 1, 1, { channels: 1 });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    const r = atlas!.data[0]!;
    expect(r).toBe(128);
    // Channels 1 and 2 mirror channel 0 (stride=1 path)
    expect(atlas!.data[1]!).toBe(r);
    expect(atlas!.data[2]!).toBe(r);
    // Alpha = 1 (default for stride < 4)
    expect(atlas!.data[3]).toBe(255);
  });

  it('explicit channels:2 expands native RG to (R,G,0,255)', () => {
    const handle = hintedHandle(new Uint8Array([64, 192]), 1, 1, { channels: 2 });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    expect(Array.from(atlas!.data)).toEqual([64, 192, 0, 255]);
  });

  it('snapshots getter-backed length and every authored element exactly once', () => {
    let lengthReads = 0;
    const elementReads = [0, 0, 0, 0];
    const firstValues = [10, 20, 30, 255];
    const data = {
      get length() {
        lengthReads += 1;
        return lengthReads === 1 ? 4 : Number.POSITIVE_INFINITY;
      },
      get 0() {
        elementReads[0]! += 1;
        return elementReads[0] === 1 ? firstValues[0]! : Number.POSITIVE_INFINITY;
      },
      get 1() {
        elementReads[1]! += 1;
        return elementReads[1] === 1 ? firstValues[1]! : Number.POSITIVE_INFINITY;
      },
      get 2() {
        elementReads[2]! += 1;
        return elementReads[2] === 1 ? firstValues[2]! : Number.POSITIVE_INFINITY;
      },
      get 3() {
        elementReads[3]! += 1;
        return elementReads[3] === 1 ? firstValues[3]! : Number.POSITIVE_INFINITY;
      },
    };
    const handle = hintedHandle(data, 1, 1, {
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    });

    const atlas = packTextureAtlas([mat(handle)])!;

    expect(lengthReads).toBe(1);
    expect(elementReads).toEqual([1, 1, 1, 1]);
    expect(Array.from(atlas.data)).toEqual(firstValues);
  });

  it('rejects SharedArrayBuffer-backed raw handles before atlas staging', () => {
    if (typeof SharedArrayBuffer === 'undefined') return;
    const cases = [
      ['uint8', new Uint8Array(new SharedArrayBuffer(Uint8Array.BYTES_PER_ELEMENT))],
      ['uint16', new Uint16Array(new SharedArrayBuffer(Uint16Array.BYTES_PER_ELEMENT))],
      ['float32', new Float32Array(new SharedArrayBuffer(Float32Array.BYTES_PER_ELEMENT))],
    ] as const;

    for (const [dataType, data] of cases) {
      const handle = hintedHandle(data, 1, 1, {
        channels: 1,
        dataType,
        colorSpace: 'linear',
      });
      expect(() => packTextureAtlas([{
        baseColor: [1, 1, 1],
        roughness: 1,
        metallic: 0,
        roughnessMap: { handle },
      }])).toThrow(/SharedArrayBuffer-backed material texels/);
    }
  });

  it('rejects role/channel mismatches before the first indexed payload read', () => {
    let elementReads = 0;
    const data = new Proxy({ length: 2 } as unknown as ArrayLike<number>, {
      get(target, property, receiver) {
        if (property === '0' || property === '1') elementReads += 1;
        return property === '0' || property === '1'
          ? 128
          : Reflect.get(target, property, receiver);
      },
    });
    const handle = hintedHandle(data, 1, 1, {
      channels: 2,
      dataType: 'uint8',
      colorSpace: 'linear',
    });
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      normalMap: { handle },
    };

    expect(() => packTextureAtlas([material])).toThrow(
      /normalMap requires source channels 3, 4 \(received 2\)/,
    );
    expect(elementReads).toBe(0);
  });

  it('enforces the earliest consumer-channel profile for mapped roles', () => {
    const makeHandle = (channels: 1 | 2 | 3 | 4): unknown => hintedHandle(
      new Uint8Array(channels).fill(128),
      1,
      1,
      { channels, dataType: 'uint8', colorSpace: 'linear' },
    );
    const base: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
    };
    const normal = (channels: 1 | 2 | 3 | 4): MaterialSpec => ({
      ...base,
      normalMap: { handle: makeHandle(channels) },
    });
    const anisotropy = (channels: 1 | 2 | 3 | 4): MaterialSpec => ({
      ...base,
      anisotropy: 1,
      anisotropyMap: { handle: makeHandle(channels) },
    });
    const sheenRoughness = (channels: 1 | 2 | 3 | 4): MaterialSpec => ({
      ...base,
      sheen: 1,
      sheenRoughness: 0.5,
      sheenRoughnessMap: { handle: makeHandle(channels) },
    });
    const specularIntensity = (channels: 1 | 2 | 3 | 4): MaterialSpec => ({
      ...base,
      specularIntensity: 1,
      specularIntensityMap: { handle: makeHandle(channels) },
    });
    const metallic = (channels: 1 | 2 | 3 | 4): MaterialSpec => ({
      ...base,
      metallicMap: { handle: makeHandle(channels) },
    });

    expect(() => packTextureAtlas([normal(2)])).toThrow(/normalMap requires source channels 3, 4/);
    expect(packTextureAtlas([normal(3)])).not.toBeNull();
    expect(() => packTextureAtlas([anisotropy(2)])).toThrow(/anisotropyMap requires source channels 3, 4/);
    expect(packTextureAtlas([anisotropy(3)])).not.toBeNull();
    expect(() => packTextureAtlas([sheenRoughness(3)])).toThrow(/sheenRoughnessMap requires source channels 4/);
    expect(packTextureAtlas([sheenRoughness(4)])).not.toBeNull();
    expect(() => packTextureAtlas([specularIntensity(3)])).toThrow(/specularIntensityMap requires source channels 4/);
    expect(packTextureAtlas([specularIntensity(4)])).not.toBeNull();
    expect(packTextureAtlas([metallic(1)])).not.toBeNull();
    expect(() => packTextureAtlas([metallic(2)])).toThrow(/metallicMap requires source channels 1, 3, 4/);
    expect(packTextureAtlas([metallic(3)])).not.toBeNull();
    expect(packTextureAtlas([metallic(4)])).not.toBeNull();
  });

  it('unions reused-handle role requirements while retaining one placement', () => {
    const handle3 = hintedHandle(new Uint8Array([128, 128, 255]), 1, 1, {
      channels: 3,
      dataType: 'uint8',
      colorSpace: 'linear',
    });
    const compatible: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      roughnessMap: { handle: handle3 },
      normalMap: { handle: handle3 },
    };
    const atlas = packTextureAtlas([compatible])!;
    expect(atlas.layerCount).toBe(1);
    expect(atlas.layerOfByColorSpace.linear.get(handle3)).toBe(0);

    const alphaStrict: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      roughnessMap: { handle: handle3 },
      sheen: 1,
      sheenRoughness: 0.5,
      sheenRoughnessMap: { handle: handle3 },
    };
    expect(() => packTextureAtlas([alphaStrict])).toThrow(
      /roughnessMap \+ sheenRoughnessMap requires source channels 4/,
    );

    const handle4 = hintedHandle(new Uint8Array([128, 128, 255, 128]), 1, 1, {
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'linear',
    });
    const compatibleAlpha: MaterialSpec = {
      ...alphaStrict,
      roughnessMap: { handle: handle4 },
      sheenRoughnessMap: { handle: handle4 },
    };
    expect(packTextureAtlas([compatibleAlpha])?.layerCount).toBe(1);
  });

  it('rejects nonzero generic float32 inputs that underflow during the raw snapshot', () => {
    const handle = hintedHandle({ length: 1, 0: Number.MIN_VALUE }, 1, 1, {
      channels: 1,
      dataType: 'float32',
      colorSpace: 'linear',
    });
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      roughnessMap: { handle },
    };
    expect(() => packTextureAtlas([material])).toThrow(
      /raw pixel value 0 is nonzero but underflows to zero in float32/,
    );
  });

  it('explicit colorSpace:linear keeps Float32 baseColorMap values already in linear light', () => {
    // 1×1 pixel, RGBA, float32 already in [0,1]
    const handle = hintedHandle(new Float32Array([0.5, 0.25, 0.75, 1.0]), 1, 1, {
      channels: 4,
      dataType: 'float32',
      colorSpace: 'linear',
    });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 0)).toBeCloseTo(0.5, 2);
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 1)).toBeCloseTo(0.25, 2);
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 2)).toBeCloseTo(0.75, 2);
    expect(ldrLinear(atlas!.data as Uint8Array, 3)).toBeCloseTo(1.0, 5);
  });

  it('uploads an explicit cpuMirror for an otherwise opaque texture handle', () => {
    const handle = {
      id: 'opaque-texture',
      cpuMirror: {
        width: 1,
        height: 1,
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
        data: new Float32Array([0.5, 0.25, 0.75, 1]),
      },
    };
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    expect(atlas!.layerOfByColorSpace.srgb.get(handle)).toBe(0);
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 0)).toBeCloseTo(0.5, 2);
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 1)).toBeCloseTo(0.25, 2);
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 2)).toBeCloseTo(0.75, 2);
    expect(ldrLinear(atlas!.data as Uint8Array, 3)).toBe(1);
  });

  it('treats Uint16Array handles as normalized uint16 unless explicitly hinted as float16', () => {
    const normalized = hintedHandle(new Uint16Array([32768, 65535, 0, 65535]), 1, 1, {
      channels: 4,
      dataType: 'uint16',
      colorSpace: 'linear',
    });
    const atlas = packTextureAtlas([mat(normalized)]);
    expect(atlas).not.toBeNull();
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 0)).toBeCloseTo(32768 / 65535, 2);
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 1)).toBeCloseTo(1, 5);
    expect(ldrSrgbLinear(atlas!.data as Uint8Array, 2)).toBeCloseTo(0, 5);
    expect(ldrLinear(atlas!.data as Uint8Array, 3)).toBeCloseTo(1, 5);

    const halfFloat = hintedHandle(new Uint16Array([0x3800, 0x3c00, 0, 0x3c00]), 1, 1, {
      channels: 4,
      dataType: 'float16',
      colorSpace: 'linear',
    });
    const halfAtlas = packTextureAtlas([mat(halfFloat)]);
    expect(halfAtlas).not.toBeNull();
    expect(ldrSrgbLinear(halfAtlas!.data as Uint8Array, 0)).toBeCloseTo(0.5, 2);
    expect(ldrSrgbLinear(halfAtlas!.data as Uint8Array, 1)).toBeCloseTo(1, 5);
    expect(ldrSrgbLinear(halfAtlas!.data as Uint8Array, 2)).toBeCloseTo(0, 5);
    expect(ldrLinear(halfAtlas!.data as Uint8Array, 3)).toBeCloseTo(1, 5);
  });

  it('rejects a dataType hint that does not match the typed-array backing', () => {
    const handle = hintedHandle(new Uint16Array([255, 0, 0, 255]), 1, 1, {
      channels: 4,
      dataType: 'uint8',
    });
    expect(() => packTextureAtlas([mat(handle)])).toThrow(
      /dataType "uint8" requires Uint8Array or Uint8ClampedArray, received \[object Uint16Array\]/,
    );
  });

  it.each([
    new Float64Array([1, 0, 0, 1]),
    new Int16Array([1, 0, 0, 1]),
    new Uint32Array([1, 0, 0, 1]),
  ])('rejects unsupported inferred backing %s', (data) => {
    const handle = { width: 1, height: 1, data };
    expect(() => packTextureAtlas([mat(handle)])).toThrow(
      /pixel backing .* cannot be inferred; use Uint8Array, Uint16Array, or Float32Array/,
    );
  });

  it('does not complete a partial cpuMirror from a separate image payload', () => {
    const handle = {
      cpuMirror: { width: 1, height: 1, channels: 4, dataType: 'float32' },
      image: { width: 1, height: 1, data: new Float32Array([1, 0, 0, 1]) },
    };
    expect(() => packTextureAtlas([mat(handle)])).toThrow(
      /no cpuMirror, raw data, or DataTexture-shaped image data was supplied/,
    );
  });

  it('fails closed on an ambiguous RGB stride without a channels hint', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      // 1×1, 3-channel RGB (stride 3 is ambiguous without a hint)
      const handle = { width: 1, height: 1, data: new Uint8Array([255, 128, 0]) };
      expect(() => packTextureAtlas([mat(handle)])).toThrow(
        /3-channel layout is ambiguous without __vitrum_hint__\.channels/,
      );
      expect(warnSpy).not.toHaveBeenCalled();
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

  it.each([
    [
      'non-finite dimensions',
      { width: Number.POSITIVE_INFINITY, height: 1, data: new Float32Array([1, 1, 1, 1]) },
      /width and height must be positive safe integers/,
    ],
    [
      'fractional dimensions',
      { width: 1.5, height: 1, data: new Float32Array([1, 1, 1, 1]) },
      /width and height must be positive safe integers/,
    ],
    [
      'a non-finite data length',
      { width: 1, height: 1, data: { length: Number.POSITIVE_INFINITY } },
      /data\.length must be a finite non-negative safe integer/,
    ],
    [
      'a hinted payload length mismatch',
      {
        width: 1,
        height: 1,
        data: new Float32Array([1, 1, 1]),
        __vitrum_hint__: { channels: 4, dataType: 'float32' },
      },
      /data length 3 does not equal width×height×channels \(4\)/,
    ],
    [
      'non-finite decoded pixels',
      {
        width: 1,
        height: 1,
        data: new Float32Array([Number.NaN, 0, 0, 1]),
        __vitrum_hint__: { channels: 4, dataType: 'float32' },
      },
      /decoded pixel data must be finite/,
    ],
  ])('rejects %s before producing an atlas', (_label, handle, message) => {
    expect(() => packTextureAtlas([mat(handle)])).toThrow(message);
  });

  it('preserves the darkest authored sRGB code instead of quantizing linear light to zero', () => {
    const handle = hintedHandle(new Uint8Array([1, 1, 1, 255]), 1, 1, { channels: 4 });
    const atlas = packTextureAtlas([mat(handle)])!;
    expect(Array.from(atlas.data)).toEqual([1, 1, 1, 255]);
    expect(ldrSrgbLinear(atlas.data as Uint8Array, 0)).toBeCloseTo(1 / 255 / 12.92, 8);
  });

  it('uses independent role/storage placements when one handle feeds LDR color, LDR data, and HDR', () => {
    const handle = hintedHandle(new Uint8Array([128, 64, 32, 255]), 1, 1, { channels: 4 });
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle },
      roughnessMap: { handle },
      emissiveMap: { handle },
    };
    const atlas = packTextureAtlas([material]);
    expect(atlas).not.toBeNull();
    expect(atlas!.layerCount).toBe(2);
    expect(atlas!.layerOfByColorSpace.srgb.get(handle)).toBe(0);
    expect(atlas!.layerOfByColorSpace.linear.get(handle)).toBe(1);

    const linearLayerBase = atlas!.dim * atlas!.dim * 4;
    expect(Array.from(atlas!.data.slice(0, 4))).toEqual([128, 64, 32, 255]);
    expect(Array.from(atlas!.data.slice(linearLayerBase, linearLayerBase + 4)))
      .toEqual([128, 64, 32, 255]);

    const hdrAtlas = packTextureAtlas([material], { storageClass: 'hdr' })!;
    expect(hdrAtlas.layerOfByColorSpace.srgb.get(handle)).toBe(0);
    expect(float16BitsToFloat32(hdrAtlas.data[0]!))
      .toBeCloseTo(srgbToLinear(128 / 255), 3);
  });
});
