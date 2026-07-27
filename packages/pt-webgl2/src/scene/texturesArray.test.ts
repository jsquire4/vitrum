import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  packTextureAtlas,
  textureAtlasLayerCapacity,
  textureAtlasMipElementCounts,
  uploadTextureAtlas,
  type TextureHandleHint,
} from './texturesArray.js';

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

function srgbToLinear(v: number): number {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

function makeTextureAtlasUploadGl(error = 0) {
  const texture = {} as WebGLTexture;
  const deleteTexture = vi.fn();
  const bindTexture = vi.fn();
  const texStorage3D = vi.fn();
  const texSubImage3D = vi.fn();
  const texImage3D = vi.fn();
  const gl = {
    TEXTURE_2D_ARRAY: 0x8c1a,
    RGBA32F: 0x8814,
    RGBA: 0x1908,
    FLOAT: 0x1406,
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
    getParameter: vi.fn(() => 256),
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

  it('fails the whole atlas build when any authored handle is unreadable', () => {
    const readable = dataTexHandle(new Float32Array([1, 0, 0, 1]), 1, 1);
    const unreadable = { id: 'opaque-texture-without-cpu-mirror' };
    const onWarning = vi.fn();

    expect(() =>
      packTextureAtlas([matWithBaseColorMap(readable), matWithBaseColorMap(unreadable)], {
        onWarning,
        warningPhase: 'setScene',
        warningMethod: 'setScene',
      }),
    ).toThrow(/authored material texture during setScene is not CPU-readable/);
    expect(onWarning).not.toHaveBeenCalled();
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
    expect(decodedAllocations).toBe(0);
  });

  it('decodes source pixels directly into retained atlas level zero', () => {
    const handles = [
      dataTexHandle(new Float32Array([1, 0, 0, 1]), 1, 1),
      dataTexHandle(new Float32Array([0, 1, 0, 1]), 1, 1),
      dataTexHandle(new Float32Array([0, 0, 1, 1]), 1, 1),
    ];
    const NativeFloat32Array = globalThis.Float32Array;
    const allocationLengths: number[] = [];
    class CountingFloat32Array extends NativeFloat32Array {
      constructor(length: number) {
        allocationLengths.push(length);
        super(length);
      }
    }
    vi.stubGlobal('Float32Array', CountingFloat32Array);
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
    expect(texStorage3D).toHaveBeenCalledWith(gl.TEXTURE_2D_ARRAY, 2, gl.RGBA32F, 2, 2, 4);
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
        gl.FLOAT,
      ]);
      expect(call?.[10]).toBe(level.data);
    });
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

  it('packs a DataTexture-shaped baseColorMap and assigns it layer 0', () => {
    // 1×1 red RGBA float
    const handle = dataTexHandle(new Float32Array([1, 0, 0, 1]), 1, 1);
    const atlas = packTextureAtlas([matWithBaseColorMap(handle)]);
    expect(atlas).not.toBeNull();
    expect(atlas!.layerCount).toBe(1);
    expect(atlas!.dim).toBe(1);
    expect(atlas!.mipLevels).toHaveLength(1);
    expect(atlas!.mipLevels[0]!.data).toBe(atlas!.data);
    expect(atlas!.layerOf.get(handle)).toBe(0);
    expect(Array.from(atlas!.data)).toEqual([1, 0, 0, 1]);
  });

  it('dedups a shared handle across materials to one layer', () => {
    const handle = dataTexHandle(new Float32Array([0.5, 0.5, 0.5, 1]), 1, 1);
    const atlas = packTextureAtlas([matWithBaseColorMap(handle), matWithBaseColorMap(handle)]);
    expect(atlas!.layerCount).toBe(1);
    expect(atlas!.layerOf.get(handle)).toBe(0);
  });

  it('stores heterogeneous handles at native extent without resampling', () => {
    const h1 = dataTexHandle(new Float32Array([1, 1, 1, 1]), 1, 1); // 1×1
    const h2 = dataTexHandle(new Float32Array(2 * 2 * 4).fill(0.5), 2, 2); // 2×2
    const atlas = packTextureAtlas([matWithBaseColorMap(h1), matWithBaseColorMap(h2)]);
    expect(atlas!.layerCount).toBe(2);
    expect(atlas!.dim).toBe(2); // max source dim
    expect(atlas!.layerOf.get(h1)).toBe(0);
    expect(atlas!.layerOf.get(h2)).toBe(1);
    // Layer 0 retains its native 1×1 texel at the origin; padding stays zero
    // and is never addressable because its native extent is packed per handle.
    expect(Array.from(atlas!.data.slice(0, 16))).toEqual([
      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    expect(atlas!.sourceDimensions[0]).toEqual([1, 1]);
    expect(atlas!.sourceDimensions[1]).toEqual([2, 2]);
    expect(atlas!.layerOfByColorSpace.dimensions?.get(h1)).toEqual([1, 1]);
    expect(atlas!.layerOfByColorSpace.dimensions?.get(h2)).toEqual([2, 2]);
    expect(atlas!.mipLevels).toHaveLength(2);
    expect(atlas!.mipLevels[1]!.dim).toBe(1);
    expect(Array.from(atlas!.mipLevels[1]!.data.slice(0, 4))).toEqual([1, 1, 1, 1]);
    expect(atlas!.mipLevels[1]!.data[4]).toBeCloseTo(srgbToLinear(0.5), 6);
    expect(atlas!.mipLevels[1]!.data[5]).toBeCloseTo(srgbToLinear(0.5), 6);
    expect(atlas!.mipLevels[1]!.data[6]).toBeCloseTo(srgbToLinear(0.5), 6);
    expect(atlas!.mipLevels[1]!.data[7]).toBeCloseTo(0.5, 6);
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
    expect(Array.from(atlas.mipLevels[1]!.data.slice(0, 4))).toEqual([0.5, 0, 0.5, 1]);
    expect(Array.from(atlas.mipLevels[1]!.data.slice(4, 8))).toEqual([0.5, 1, 0.5, 1]);
  });

  it('averages every source texel when generating odd-sized mip levels', () => {
    const data = new Float32Array(3 * 3 * 4);
    for (let i = 0; i < 9; i += 1) {
      data[i * 4] = i;
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
    expect(atlas!.mipLevels[1]!.data[0]).toBeCloseTo(4, 6);
    expect(atlas!.mipLevels[1]!.data[3]).toBeCloseTo(1, 6);
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
    const onWarning = vi.fn();
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
      { onWarning, warningPhase: 'setScene', warningMethod: 'setScene' },
    );

    expect(atlas).not.toBeNull();
    expect(onWarning).not.toHaveBeenCalled();
  });
});

describe('textureAtlasLayerCapacity', () => {
  it('keeps spare power-of-two capacity when the device limit allows it', () => {
    expect(textureAtlasLayerCapacity(0, 256)).toBe(0);
    expect(textureAtlasLayerCapacity(1, 256)).toBe(2);
    expect(textureAtlasLayerCapacity(2, 256)).toBe(4);
    expect(textureAtlasLayerCapacity(3, 256)).toBe(4);
  });

  it('clamps spare capacity to the device layer limit without rejecting exact fits', () => {
    expect(textureAtlasLayerCapacity(3, 3)).toBe(3);
    expect(textureAtlasLayerCapacity(4, 6)).toBe(6);
  });
});

describe('textureAtlasMipElementCounts', () => {
  it('returns the exact preflight allocation for every mip', () => {
    expect(textureAtlasMipElementCounts(4, 2)).toEqual([128, 32, 8]);
  });

  it('rejects an aggregate mip chain that exceeds the CPU staging budget', () => {
    // Level zero is below the cap; mip one is what pushes the retained chain over it.
    expect(() => textureAtlasMipElementCounts(5600, 1)).toThrow(
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

  it('explicit channels:4 on Uint8 data decodes baseColorMap sRGB into linear RGBA output', () => {
    // 1×1 red pixel in Uint8 RGBA (255,0,0,255)
    const handle = hintedHandle(new Uint8Array([255, 0, 0, 255]), 1, 1, {
      channels: 4,
      dataType: 'uint8',
    });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    // Should decode to linear [1,0,0,1] after /255 normalization + sRGB decode.
    expect(atlas!.data[0]).toBeCloseTo(1, 5);
    expect(atlas!.data[1]).toBeCloseTo(0, 5);
    expect(atlas!.data[2]).toBeCloseTo(0, 5);
    expect(atlas!.data[3]).toBeCloseTo(1, 5);
  });

  it('explicit channels:1 expands a single-channel sRGB baseColorMap value to linear (R,R,R,1)', () => {
    // 1×1 grayscale 128/255 ≈ 0.502
    const handle = hintedHandle(new Uint8Array([128]), 1, 1, { channels: 1 });
    const atlas = packTextureAtlas([mat(handle)]);
    expect(atlas).not.toBeNull();
    const r = atlas!.data[0]!;
    expect(r).toBeCloseTo(srgbToLinear(128 / 255), 4);
    // Channels 1 and 2 mirror channel 0 (stride=1 path)
    expect(atlas!.data[1]!).toBe(r);
    expect(atlas!.data[2]!).toBe(r);
    // Alpha = 1 (default for stride < 4)
    expect(atlas!.data[3]).toBe(1);
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
    expect(atlas!.data[0]).toBeCloseTo(0.5, 5);
    expect(atlas!.data[1]).toBeCloseTo(0.25, 5);
    expect(atlas!.data[2]).toBeCloseTo(0.75, 5);
    expect(atlas!.data[3]).toBeCloseTo(1.0, 5);
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
    expect(atlas!.layerOf.get(handle)).toBe(0);
    expect(Array.from(atlas!.data)).toEqual([0.5, 0.25, 0.75, 1]);
  });

  it('treats Uint16Array handles as normalized uint16 unless explicitly hinted as float16', () => {
    const normalized = hintedHandle(new Uint16Array([32768, 65535, 0, 65535]), 1, 1, {
      channels: 4,
      dataType: 'uint16',
      colorSpace: 'linear',
    });
    const atlas = packTextureAtlas([mat(normalized)]);
    expect(atlas).not.toBeNull();
    expect(atlas!.data[0]).toBeCloseTo(32768 / 65535, 5);
    expect(atlas!.data[1]).toBeCloseTo(1, 5);
    expect(atlas!.data[2]).toBeCloseTo(0, 5);
    expect(atlas!.data[3]).toBeCloseTo(1, 5);

    const halfFloat = hintedHandle(new Uint16Array([0x3800, 0x3c00, 0, 0x3c00]), 1, 1, {
      channels: 4,
      dataType: 'float16',
      colorSpace: 'linear',
    });
    const halfAtlas = packTextureAtlas([mat(halfFloat)]);
    expect(halfAtlas).not.toBeNull();
    expect(halfAtlas!.data[0]).toBeCloseTo(0.5, 5);
    expect(halfAtlas!.data[1]).toBeCloseTo(1, 5);
    expect(halfAtlas!.data[2]).toBeCloseTo(0, 5);
    expect(halfAtlas!.data[3]).toBeCloseTo(1, 5);
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

  it('uses separate atlas layers when the same handle is sampled as sRGB and linear data', () => {
    const handle = hintedHandle(new Uint8Array([128, 64, 32, 255]), 1, 1, { channels: 4 });
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle },
      roughnessMap: { handle },
    };
    const atlas = packTextureAtlas([material]);
    expect(atlas).not.toBeNull();
    expect(atlas!.layerCount).toBe(2);
    expect(atlas!.layerOfByColorSpace.srgb.get(handle)).toBe(0);
    expect(atlas!.layerOfByColorSpace.linear.get(handle)).toBe(1);

    const linearLayerBase = atlas!.dim * atlas!.dim * 4;
    expect(atlas!.data[0]).toBeCloseTo(srgbToLinear(128 / 255), 4);
    expect(atlas!.data[1]).toBeCloseTo(srgbToLinear(64 / 255), 4);
    expect(atlas!.data[2]).toBeCloseTo(srgbToLinear(32 / 255), 4);
    expect(atlas!.data[3]).toBe(1);
    expect(atlas!.data[linearLayerBase]).toBeCloseTo(128 / 255, 4);
    expect(atlas!.data[linearLayerBase + 1]).toBeCloseTo(64 / 255, 4);
    expect(atlas!.data[linearLayerBase + 2]).toBeCloseTo(32 / 255, 4);
    expect(atlas!.data[linearLayerBase + 3]).toBe(1);
  });
});
