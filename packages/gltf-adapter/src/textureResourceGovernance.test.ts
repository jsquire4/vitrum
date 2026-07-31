import { afterEach, describe, expect, it, vi } from 'vitest';
import type {
  MaterialSpec,
  MeshPrimitive,
  Scene,
  TextureRef,
} from '@vitrum/core';
import {
  createDecodeSceneTexturesContext,
  decodeSceneTextures,
  decodeSceneTexturesWithContext,
  inferDecodedDataType,
  type DecodeGltfTexturePixelsFn,
} from './texturePipeline.js';
import {
  decodeSceneTextures as decodeSceneTexturesFromPublicApi,
} from './index.js';
import {
  createBitmapFromRawImage,
  decodeRawImagePixelsWithPlatform,
  decodeRawJpegPixelsDeterministically,
  decodeRawPngPixelsDeterministically,
  decodeRawWebpPixelsWithNode,
  jpegDecodeOptionsForPixelLimit,
  normalizeCapturedRawImageForDecode,
} from './textureCodecs.js';
import {
  RawImageDimensionsError,
  readEncodedImageDimensions,
} from './rawImageDimensions.js';
import { buildTextureDecodeReport } from './textureDecodeReport.js';
import type { RawImageHandle } from './textures.js';

const jpegDecodeMock = vi.hoisted(() => vi.fn());

vi.mock('jpeg-js', () => {
  const module = { decode: jpegDecodeMock };
  return { ...module, default: module };
});

function pngHeader(width: number, height: number): Uint8Array {
  const data = new Uint8Array(24);
  data.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  data.set([0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52], 8);
  writeU32Be(data, 16, width);
  writeU32Be(data, 20, height);
  return data;
}

function jpegHeader(width: number, height: number): Uint8Array {
  const data = new Uint8Array(21);
  data.set([0xff, 0xd8, 0xff, 0xc0, 0, 17, 8]);
  data[7] = (height >>> 8) & 0xff;
  data[8] = height & 0xff;
  data[9] = (width >>> 8) & 0xff;
  data[10] = width & 0xff;
  return data;
}

function webpVp8xHeader(width: number, height: number): Uint8Array {
  const data = new Uint8Array(30);
  data.set([0x52, 0x49, 0x46, 0x46]);
  writeU32Le(data, 4, 22);
  data.set([0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x58], 8);
  writeU32Le(data, 16, 10);
  writeU24Le(data, 24, width - 1);
  writeU24Le(data, 27, height - 1);
  return data;
}

function raw(data = pngHeader(1, 1), mimeType = 'image/png'): RawImageHandle {
  return { kind: 'raw-image', data, mimeType };
}

function sceneForMaterials(materials: readonly MaterialSpec[]): Scene {
  return {
    primitives: materials.map((material, index): MeshPrimitive => ({
      kind: 'mesh',
      id: `texture-resource-${index}`,
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material,
    })),
    emitters: [],
    environment: { kind: 'none' },
  };
}

function materialWith(
  field: keyof MaterialSpec,
  handle: unknown,
): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 1,
    metallic: 0,
    [field]: { handle, texCoord: 0 },
  };
}

const codecContext = {
  materialField: 'baseColorMap' as const,
  path: 'materials[0].baseColorMap',
  colorSpace: 'srgb' as const,
  primitiveId: 'p',
  primitiveIndex: 0,
  maxDecodedTexturePixels: 16,
};

afterEach(() => {
  jpegDecodeMock.mockReset();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('raw image dimension preflight', () => {
  it('reads PNG/JPEG/WebP dimensions and rejects a truncated declared RIFF', () => {
    expect(readEncodedImageDimensions(pngHeader(123, 45))).toEqual({
      format: 'png',
      width: 123,
      height: 45,
    });
    expect(readEncodedImageDimensions(jpegHeader(321, 54))).toEqual({
      format: 'jpeg',
      width: 321,
      height: 54,
    });
    expect(readEncodedImageDimensions(webpVp8xHeader(777, 333))).toEqual({
      format: 'webp',
      width: 777,
      height: 333,
    });

    const truncated = webpVp8xHeader(4, 4);
    writeU32Le(truncated, 4, 100);
    expect(() => readEncodedImageDimensions(truncated)).toThrow(RawImageDimensionsError);
  });

  it('rejects decompression-bomb headers before platform or Node decoders run', async () => {
    const createImageBitmap = vi.fn();
    vi.stubGlobal('createImageBitmap', createImageBitmap);
    vi.stubGlobal('Blob', class {});

    await expect(
      decodeRawImagePixelsWithPlatform(raw(pngHeader(10_000, 10_000)), codecContext),
    ).rejects.toMatchObject({ code: 'decoded-texture-exceeds-pixel-budget' });
    await expect(
      decodeRawPngPixelsDeterministically(raw(pngHeader(10_000, 10_000)), codecContext),
    ).rejects.toMatchObject({ code: 'decoded-texture-exceeds-pixel-budget' });
    await expect(
      decodeRawJpegPixelsDeterministically(
        raw(jpegHeader(10_000, 10_000), 'image/jpeg'),
        codecContext,
      ),
    ).rejects.toMatchObject({ code: 'decoded-texture-exceeds-pixel-budget' });
    await expect(
      decodeRawWebpPixelsWithNode(
        raw(webpVp8xHeader(10_000, 10_000), 'image/webp'),
        codecContext,
      ),
    ).rejects.toMatchObject({ code: 'decoded-texture-exceeds-pixel-budget' });
    expect(jpegDecodeMock).not.toHaveBeenCalled();
    expect(createImageBitmap).not.toHaveBeenCalled();
  });

  it('translates the public pixel policy into explicit non-stricter jpeg-js limits', () => {
    const safeMemoryMiB = Math.floor(
      Number.MAX_SAFE_INTEGER / (1024 * 1024),
    );
    expect(jpegDecodeOptionsForPixelLimit(2_500_000)).toEqual({
      useTArray: true,
      maxResolutionInMP: 3,
      maxMemoryUsageInMB: safeMemoryMiB,
    });
    expect(jpegDecodeOptionsForPixelLimit(100_000_001)).toEqual({
      useTArray: true,
      maxResolutionInMP: 101,
      maxMemoryUsageInMB: safeMemoryMiB,
    });

    const zero = jpegDecodeOptionsForPixelLimit(0);
    expect(zero).toEqual({
      useTArray: true,
      maxResolutionInMP: Number.MAX_SAFE_INTEGER / 1_000_000,
      maxMemoryUsageInMB: safeMemoryMiB,
    });
    expect(zero.maxResolutionInMP * 1_000_000)
      .toBeGreaterThanOrEqual(Number.MAX_SAFE_INTEGER);
    expect(zero.maxMemoryUsageInMB * 1024 * 1024)
      .toBeLessThanOrEqual(Number.MAX_SAFE_INTEGER);
    expect(jpegDecodeOptionsForPixelLimit(Number.MAX_SAFE_INTEGER))
      .toEqual(zero);
    const roundingEdge = jpegDecodeOptionsForPixelLimit(129_650);
    expect(roundingEdge.maxResolutionInMP).toBe(1);
    expect(roundingEdge.maxResolutionInMP * 1_000_000)
      .toBeGreaterThanOrEqual(129_650);
    for (const invalid of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => jpegDecodeOptionsForPixelLimit(invalid)).toThrow(
        'must be a non-negative safe integer',
      );
    }
  });

  it.each([
    { label: 'explicit zero', maxDecodedTexturePixels: 0 },
    {
      label: 'a finite cap above jpeg-js defaults',
      maxDecodedTexturePixels: 100_000_001,
    },
  ])(
    'passes explicit decoder limits through the public decode API for $label',
    async ({ maxDecodedTexturePixels }) => {
      jpegDecodeMock.mockReturnValue({
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
      });
      const result = await decodeSceneTexturesFromPublicApi(
        sceneForMaterials([
          materialWith(
            'baseColorMap',
            raw(jpegHeader(1, 1), 'image/jpeg'),
          ),
        ]),
        {
          target: 'cpu-linear',
          maxDecodedTexturePixels,
        },
      );

      expect(result.decodedCount).toBe(1);
      expect(result.diagnostics).toEqual([]);
      expect(jpegDecodeMock).toHaveBeenCalledOnce();
      expect(jpegDecodeMock.mock.calls[0]?.[1]).toEqual({
        useTArray: true,
        maxResolutionInMP: maxDecodedTexturePixels === 0
          ? Number.MAX_SAFE_INTEGER / 1_000_000
          : 101,
        maxMemoryUsageInMB: Math.floor(
          Number.MAX_SAFE_INTEGER / (1024 * 1024),
        ),
      });
    },
  );

  it('routes corrupt PNG MIME to dimension preflight', async () => {

    const corrupt = await decodeSceneTextures(
      sceneForMaterials([
        materialWith(
          'baseColorMap',
          raw(new Uint8Array([0x89, 0x50, 0x4e, 0x47]), 'image/png'),
        ),
      ]),
      { target: 'cpu-linear' },
    );
    expect(corrupt.diagnostics).toContainEqual(expect.objectContaining({
      code: 'platform-image-decode-failed',
      message: expect.stringContaining('safely preflighted'),
    }));
    expect(corrupt.diagnostics).not.toContainEqual(expect.objectContaining({
      code: 'raw-image-decoder-missing',
    }));
  });
});

describe('decode context governance', () => {
  it('validates runtime options before touching a scene', () => {
    const invalid = [
      { target: 'other' },
      { target: 'cpu-linear', npotRepeatWrapPolicy: 'other' },
      { target: 'cpu-linear', maxTextureSize: Number.NaN },
      { target: 'cpu-linear', maxTextureSize: 1.5 },
      { target: 'cpu-linear', decodePixels: 1 },
      { target: 'cpu-linear', warnOnNpotRepeatWrap: 'yes' },
      { target: 'cpu-linear', onDiagnostic: {} },
      { target: 'cpu-linear', resourceLimits: null },
      { target: 'cpu-linear', resourceLimits: [] },
      { target: 'cpu-linear', resourceLimits: 'not-an-object' },
    ];
    for (const options of invalid) {
      expect(() => createDecodeSceneTexturesContext(options as never)).toThrow();
    }
  });

  it('rejects an invalid nested pixel cap even when a valid flat alias would override it', async () => {
    await expect(
      decodeSceneTextures(sceneForMaterials([]), {
        target: 'cpu-linear',
        resourceLimits: {
          maxDecodedTexturePixels: -1,
        },
        maxDecodedTexturePixels: 1,
      }),
    ).rejects.toThrow(
      'resourceLimits.maxDecodedTexturePixels must be a non-negative safe integer',
    );
  });

  it('deduplicates concurrent failures but evicts them for a later retry', async () => {
    const handle = raw();
    let calls = 0;
    const decodePixels: DecodeGltfTexturePixelsFn = async () => {
      calls += 1;
      await Promise.resolve();
      if (calls === 1) throw new Error('transient');
      return {
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        channels: 4,
        dataType: 'uint8',
      };
    };
    const context = createDecodeSceneTexturesContext({
      target: 'cpu-linear',
      decodePixels,
    });
    const duplicateScene = sceneForMaterials([
      materialWith('baseColorMap', handle),
      materialWith('baseColorMap', handle),
    ]);

    const failed = await decodeSceneTexturesWithContext(duplicateScene, context);
    expect(calls).toBe(1);
    expect(failed.unchangedCount).toBe(2);
    const retried = await decodeSceneTexturesWithContext(
      sceneForMaterials([materialWith('baseColorMap', handle)]),
      context,
    );
    expect(calls).toBe(2);
    expect(retried.decodedCount).toBe(1);
  });

  it('limits unique raw decodes and shares one owned snapshot across color spaces', async () => {
    const handles = Array.from({ length: 5 }, () => raw());
    let active = 0;
    let maxActive = 0;
    const seen = new Map<RawImageHandle, RawImageHandle>();
    const decodePixels: DecodeGltfTexturePixelsFn = async (_handle) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise<void>((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return {
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        channels: 4,
        dataType: 'uint8',
      };
    };
    const result = await decodeSceneTextures(
      sceneForMaterials(handles.map((handle) => materialWith('baseColorMap', handle))),
      {
        target: 'cpu-linear',
        decodePixels: (handle, context) => {
          seen.set(handles[context.primitiveIndex]!, handle);
          return decodePixels(handle, context);
        },
        maxImageDecodeConcurrency: 2,
      },
    );
    expect(result.decodedCount).toBe(5);
    expect(maxActive).toBe(2);
    for (const handle of handles) expect(seen.get(handle)).not.toBe(handle);

    const shared = raw();
    const snapshots: RawImageHandle[] = [];
    await decodeSceneTextures(
      sceneForMaterials([{
        ...materialWith('baseColorMap', shared),
        normalMap: { handle: shared, texCoord: 0 },
      }]),
      {
        target: 'cpu-linear',
        decodePixels: (handle, context) => {
          snapshots.push(handle);
          return decodePixels(handle, context);
        },
      },
    );
    expect(snapshots).toHaveLength(2);
    expect(snapshots[0]).toBe(snapshots[1]);
    expect(snapshots[0]).not.toBe(shared);
  });

  it('admits decoded textures in scene order regardless of decoder completion order', async () => {
    const run = async (delays: readonly number[]) => {
      const handles = delays.map((_, index) =>
        raw(new Uint8Array([index]), 'application/octet-stream')
      );
      const result = await decodeSceneTextures(
        sceneForMaterials(handles.map((handle) => materialWith('baseColorMap', handle))),
        {
          target: 'cpu-linear',
          maxImageDecodeConcurrency: 3,
          maxTotalDecodedTexturePixels: 4,
          decodePixels: async (handle) => {
            const index = handle.data[0]!;
            await new Promise<void>((resolve) => setTimeout(resolve, delays[index]));
            return {
              width: 2,
              height: 1,
              data: new Uint8Array(8).fill(255),
              channels: 4,
              dataType: 'uint8',
            };
          },
        },
      );
      return {
        decodedCount: result.decodedCount,
        accepted: result.scene.primitives.map((primitive, index) =>
          primitive.material.baseColorMap?.handle === handles[index] ? false : true
        ),
      };
    };

    expect(await run([30, 15, 0])).toEqual({
      decodedCount: 2,
      accepted: [true, true, false],
    });
    expect(await run([0, 15, 30])).toEqual({
      decodedCount: 2,
      accepted: [true, true, false],
    });
  });

  it('checks encoded bytes before copying or calling a custom decoder', async () => {
    const decodePixels = vi.fn<
      Parameters<DecodeGltfTexturePixelsFn>,
      ReturnType<DecodeGltfTexturePixelsFn>
    >();
    const handle = raw(pngHeader(1, 1));
    const blocked = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', handle)]),
      {
        target: 'cpu-linear',
        decodePixels,
        resourceLimits: { maxEncodedResourceBytes: 8 },
      },
    );
    expect(decodePixels).not.toHaveBeenCalled();
    expect(blocked.diagnostics).toContainEqual(expect.objectContaining({
      code: 'encoded-texture-exceeds-byte-budget',
    }));

    decodePixels.mockReturnValue({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 255]),
      channels: 4,
      dataType: 'uint8',
    });
    const unbounded = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', handle)]),
      {
        target: 'cpu-linear',
        decodePixels,
        resourceLimits: {
          maxEncodedResourceBytes: 0,
          maxTotalEncodedBytes: 0,
        },
      },
    );
    expect(unbounded.decodedCount).toBe(1);
  });

  it('charges and copies one captured raw-byte getter value without a TOCTOU re-read', async () => {
    const small = pngHeader(1, 1);
    const large = new Uint8Array(1024);
    let reads = 0;
    const switchingHandle = {
      kind: 'raw-image' as const,
      mimeType: 'image/png',
      get data(): Uint8Array {
        reads += 1;
        return reads === 1 ? small : large;
      },
    } as RawImageHandle;
    const decodePixels = vi.fn<
      Parameters<DecodeGltfTexturePixelsFn>,
      ReturnType<DecodeGltfTexturePixelsFn>
    >((snapshot) => {
      expect(snapshot.data.byteLength).toBe(small.byteLength);
      return {
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        channels: 4,
        dataType: 'uint8',
      };
    });
    const result = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', switchingHandle)]),
      {
        target: 'cpu-linear',
        decodePixels,
        resourceLimits: { maxEncodedResourceBytes: small.byteLength },
      },
    );
    expect(result.decodedCount).toBe(1);
    expect(reads).toBe(1);
    expect(decodePixels).toHaveBeenCalledTimes(1);
  });

  it('selects a built-in codec from the same once-captured bytes and MIME', async () => {
    let dataReads = 0;
    let mimeReads = 0;
    const switchingHandle = {
      kind: 'raw-image' as const,
      get data(): Uint8Array {
        dataReads += 1;
        return dataReads === 1 ? pngHeader(1, 1) : jpegHeader(1, 1);
      },
      get mimeType(): string {
        mimeReads += 1;
        return mimeReads === 1 ? 'image/png' : 'image/jpeg';
      },
    } as RawImageHandle;
    const result = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', switchingHandle)]),
      { target: 'cpu-linear' },
    );
    expect(dataReads).toBe(1);
    expect(mimeReads).toBe(1);
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      code: 'platform-image-decode-failed',
      message: expect.stringContaining('decoded as PNG'),
    }));
  });

  it('charges normalization and POT resize allocations against one aggregate budget', async () => {
    const handle = {
      width: 3,
      height: 1,
      data: new Uint8Array(12).fill(255),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    };
    const result = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', handle)]),
      {
        target: 'cpu-linear',
        npotRepeatWrapPolicy: 'resize-to-pot',
        maxTotalDecodedTexturePixels: 6,
      },
    );
    const ref = (result.scene.primitives[0] as MeshPrimitive).material
      .baseColorMap as TextureRef;
    expect(ref.handle).toMatchObject({ width: 3, height: 1 });
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      code: 'decoded-texture-exceeds-total-pixel-budget',
      maxTotalDecodedTexturePixels: 6,
    }));
  });

  it('reuses one POT derivation for shared source identity and charges it once', async () => {
    const handle = {
      width: 3,
      height: 1,
      data: new Uint8Array(12).fill(255),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    };
    const context = createDecodeSceneTexturesContext({
      target: 'cpu-linear',
      npotRepeatWrapPolicy: 'resize-to-pot',
      maxTotalDecodedTexturePixels: 7,
    });
    const result = await decodeSceneTexturesWithContext(
      sceneForMaterials([
        materialWith('baseColorMap', handle),
        materialWith('baseColorMap', handle),
      ]),
      context,
    );
    const first = (result.scene.primitives[0] as MeshPrimitive).material
      .baseColorMap as TextureRef;
    const second = (result.scene.primitives[1] as MeshPrimitive).material
      .baseColorMap as TextureRef;

    expect(first.handle).toBe(second.handle);
    expect(first.handle).toMatchObject({ width: 4, height: 1 });
    // 3 source-normalization pixels + one shared 4-pixel POT derivation.
    expect(context.resourceLedger.totalDecodedTexturePixels).toBe(7);
  });

  it('evicts a failed POT derivation so a raised import budget can retry it', async () => {
    const handle = {
      width: 3,
      height: 1,
      data: new Uint8Array(12).fill(255),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    };
    const scene = sceneForMaterials([materialWith('baseColorMap', handle)]);
    const context = createDecodeSceneTexturesContext({
      target: 'cpu-linear',
      npotRepeatWrapPolicy: 'resize-to-pot',
      maxTotalDecodedTexturePixels: 6,
    });

    const failed = await decodeSceneTexturesWithContext(scene, context);
    const failedRef = (failed.scene.primitives[0] as MeshPrimitive).material
      .baseColorMap as TextureRef;
    expect(failedRef.handle).toMatchObject({ width: 3, height: 1 });
    expect(failed.diagnostics).toContainEqual(expect.objectContaining({
      code: 'decoded-texture-exceeds-total-pixel-budget',
    }));
    expect(context.resourceLedger.totalDecodedTexturePixels).toBe(3);

    context.resourceLedger.reconfigureLimits({
      ...context.resourceLedger.limits,
      maxTotalDecodedTexturePixels: 7,
    });
    const retried = await decodeSceneTexturesWithContext(scene, context);
    const retriedRef = (retried.scene.primitives[0] as MeshPrimitive).material
      .baseColorMap as TextureRef;
    expect(retriedRef.handle).toMatchObject({ width: 4, height: 1 });
    expect(context.resourceLedger.totalDecodedTexturePixels).toBe(7);
  });

  it('keeps distinct source color configurations in distinct POT derivations', async () => {
    const handle = {
      width: 3,
      height: 1,
      data: new Uint8Array(12).fill(128),
      channels: 4 as const,
      dataType: 'uint8' as const,
    };
    const material: MaterialSpec = {
      ...materialWith('baseColorMap', handle),
      normalMap: { handle, texCoord: 0 },
    };
    const result = await decodeSceneTextures(
      sceneForMaterials([material]),
      {
        target: 'cpu-linear',
        npotRepeatWrapPolicy: 'resize-to-pot',
        maxTotalDecodedTexturePixels: 14,
      },
    );
    const decodedMaterial = (result.scene.primitives[0] as MeshPrimitive).material;
    const color = decodedMaterial.baseColorMap as TextureRef;
    const normal = decodedMaterial.normalMap as TextureRef;
    const colorData = (color.handle as { data: Float32Array }).data;
    const normalData = (normal.handle as { data: Float32Array }).data;

    expect(color.handle).not.toBe(normal.handle);
    expect(colorData[0]).not.toBeCloseTo(normalData[0]!, 4);
  });

  it('does not bake a malformed short CPU-linear spec-gloss payload', async () => {
    const malformed = {
      width: 2,
      height: 2,
      data: new Float32Array(4),
      __vitrum_hint__: {
        channels: 4 as const,
        dataType: 'float32' as const,
        colorSpace: 'linear' as const,
      },
    };
    const material: MaterialSpec = {
      ...materialWith('specularColorMap', malformed),
      extensions: {
        KHR_materials_pbrSpecularGlossiness: {
          glossinessFactor: 1,
          specularGlossinessTexture: {},
        },
      },
    };
    const result = await decodeSceneTextures(sceneForMaterials([material]), {
      target: 'cpu-linear',
      maxDecodedTexturePixels: 4,
    });
    const decodedMaterial = (result.scene.primitives[0] as MeshPrimitive).material;
    expect(decodedMaterial.roughnessMap).toBeUndefined();
    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'decode-pixels-invalid' }),
      expect.objectContaining({ code: 'spec-gloss-alpha-bake-unavailable' }),
    ]));
  });

  it('does not bake a non-finite CPU-linear spec-gloss payload', async () => {
    const data = new Float32Array(16).fill(1);
    data[3] = Number.NaN;
    const malformed = {
      width: 2,
      height: 2,
      data,
      __vitrum_hint__: {
        channels: 4 as const,
        dataType: 'float32' as const,
        colorSpace: 'linear' as const,
      },
    };
    const material: MaterialSpec = {
      ...materialWith('specularColorMap', malformed),
      extensions: {
        KHR_materials_pbrSpecularGlossiness: {
          glossinessFactor: 1,
          specularGlossinessTexture: {},
        },
      },
    };
    const result = await decodeSceneTextures(sceneForMaterials([material]), {
      target: 'cpu-linear',
      maxDecodedTexturePixels: 4,
    });
    const decodedMaterial = (result.scene.primitives[0] as MeshPrimitive).material;
    expect(decodedMaterial.roughnessMap).toBeUndefined();
    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'decode-pixels-invalid' }),
      expect.objectContaining({ code: 'spec-gloss-alpha-bake-unavailable' }),
    ]));
  });

  it('honors flat pixel aliases over resourceLimits and explicit zero opt-out', async () => {
    const handle = {
      width: 2,
      height: 2,
      data: new Uint8Array(16).fill(255),
      channels: 4 as const,
      dataType: 'uint8' as const,
    };
    const flatWins = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', handle)]),
      {
        target: 'cpu-linear',
        resourceLimits: { maxDecodedTexturePixels: 1 },
        maxDecodedTexturePixels: 4,
      },
    );
    expect(flatWins.decodedCount).toBe(1);
    const zeroWins = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', handle)]),
      {
        target: 'cpu-linear',
        resourceLimits: { maxDecodedTexturePixels: 1 },
        maxDecodedTexturePixels: 0,
      },
    );
    expect(zeroWins.decodedCount).toBe(1);

    const defaultGuard = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', raw())]),
      {
        target: 'cpu-linear',
        decodePixels: () => ({
          width: 5_000,
          height: 5_000,
          data: { length: 100_000_000 },
          channels: 4,
          dataType: 'uint8',
        }),
      },
    );
    expect(defaultGuard.diagnostics).toContainEqual(expect.objectContaining({
      code: 'decoded-texture-exceeds-pixel-budget',
      maxDecodedTexturePixels: 16_777_216,
    }));
  });
});

describe('hostile texture boundaries', () => {
  it.each([
    ['NaN', Number.NaN],
    ['positive infinity', Number.POSITIVE_INFINITY],
    ['negative infinity', Number.NEGATIVE_INFINITY],
  ])('rejects custom decoded pixels containing %s', async (_label, value) => {
    const source = raw();
    const result = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', source)]),
      {
        target: 'cpu-linear',
        decodePixels: () => ({
          width: 1,
          height: 1,
          data: new Float32Array([value, 0.25, 0.5, 1]),
          channels: 4,
          dataType: 'float32',
        }),
      },
    );

    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    expect((material.baseColorMap as TextureRef).handle).toBe(source);
    expect(result.decodedCount).toBe(0);
    expect(result.unchangedCount).toBe(1);
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      code: 'decode-pixels-invalid',
      message: expect.stringContaining('non-finite component at index 0'),
    }));
  });

  it('uses intrinsic byte-view offsets and ArrayBuffer.slice despite own shadows', async () => {
    const normalized = normalizeCapturedRawImageForDecode(
      pngHeader(1, 1),
      'image/png',
      'image',
    );
    const backing = normalized.data.buffer;
    Object.defineProperties(normalized.data, {
      buffer: { value: new ArrayBuffer(0) },
      byteOffset: { value: 999 },
      byteLength: { value: 0 },
    });
    Object.defineProperty(backing, 'slice', {
      value() {
        throw new Error('hostile own slice');
      },
    });
    const createImageBitmap = vi.fn(async () => ({ width: 1, height: 1 }));
    vi.stubGlobal('createImageBitmap', createImageBitmap);

    await expect(createBitmapFromRawImage(normalized, 'image')).resolves.toBeDefined();
    expect(createImageBitmap).toHaveBeenCalledTimes(1);
  });

  it('suppresses hostile bitmap close access without replacing a successful decode', async () => {
    const bitmap = {
      width: 1,
      height: 1,
      get close(): never {
        throw new Error('hostile close');
      },
    };
    vi.stubGlobal('createImageBitmap', vi.fn(async () => bitmap));
    vi.stubGlobal('OffscreenCanvas', class {
      width: number;
      height: number;

      constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
      }

      getContext(): unknown {
        return {
          drawImage() {},
          getImageData() {
            return { data: new Uint8ClampedArray([1, 2, 3, 4]) };
          },
        };
      }
    });

    await expect(
      decodeRawImagePixelsWithPlatform(raw(), {
        ...codecContext,
        maxDecodedTexturePixels: 1,
      }),
    ).resolves.toMatchObject({ width: 1, height: 1 });
  });

  it('turns hostile result getters and pixel index access into per-texture diagnostics', async () => {
    const throwingData = new Proxy({ length: 4 }, {
      get(target, key, receiver) {
        if (key === '0') throw new Error('hostile pixel');
        return Reflect.get(target, key, receiver);
      },
    });
    const result = await decodeSceneTextures(
      sceneForMaterials([materialWith('baseColorMap', raw())]),
      {
        target: 'cpu-linear',
        decodePixels: () => ({
          width: 1,
          height: 1,
          data: throwingData,
          channels: 4,
          dataType: 'float32',
        }),
      },
    );
    expect(result.unchangedCount).toBe(1);
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      code: 'decode-pixels-invalid',
      causeMessage: 'hostile pixel',
    }));
  });

  it('uses intrinsic typed-array branding and rejects shared pixel storage', async () => {
    const bytes = new Uint8Array([1, 2, 3, 4]);
    Object.defineProperty(bytes, Symbol.toStringTag, { value: 'Float32Array' });
    expect(inferDecodedDataType(bytes)).toBe('uint8');

    if (typeof SharedArrayBuffer !== 'undefined') {
      const shared = new Uint8Array(new SharedArrayBuffer(4));
      const result = await decodeSceneTextures(
        sceneForMaterials([materialWith('baseColorMap', raw())]),
        {
          target: 'cpu-linear',
          decodePixels: () => ({
            width: 1,
            height: 1,
            data: shared,
            channels: 4,
            dataType: 'uint8',
          }),
        },
      );
      expect(result.diagnostics).toContainEqual(expect.objectContaining({
        code: 'decode-pixels-invalid',
        message: expect.stringContaining('SharedArrayBuffer'),
      }));
    }
  });

  it('keeps texture reports total for opaque handles with throwing getters', () => {
    const hostile = new Proxy({}, {
      get(_target, key) {
        throw new Error(`hostile ${String(key)}`);
      },
    });
    const report = buildTextureDecodeReport(
      sceneForMaterials([materialWith('baseColorMap', hostile)]),
    );
    expect(report).toMatchObject({
      mapCount: 1,
      opaqueHandleCount: 1,
      cpuReadableCount: 0,
    });
    expect(report.entries[0]).toMatchObject({ handleKind: 'opaque' });
  });

  it('does not report malformed CPU or bitmap-like handles as backend-ready', () => {
    const malformedCpu = {
      width: Number.NaN,
      height: 1,
      data: new Uint8Array(4),
    };
    const malformedBitmap = {
      width: -1,
      height: 1.5,
      close() {},
    };
    const nonFiniteCpu = {
      width: 1,
      height: 1,
      data: new Float32Array([Number.NaN, 0, 0, 1]),
    };
    const report = buildTextureDecodeReport(sceneForMaterials([
      materialWith('baseColorMap', malformedCpu),
      materialWith('baseColorMap', malformedBitmap),
      materialWith('baseColorMap', nonFiniteCpu),
    ]));
    expect(report.entries).toEqual([
      expect.objectContaining({
        handleKind: 'opaque',
        backendReadiness: expect.objectContaining({ ptWebgpu: 'opaque' }),
      }),
      expect.objectContaining({
        handleKind: 'opaque',
        backendReadiness: expect.objectContaining({ ptWebgpu: 'opaque' }),
      }),
      expect.objectContaining({
        handleKind: 'opaque',
        backendReadiness: expect.objectContaining({ ptWebgpu: 'opaque' }),
      }),
    ]);
  });
});

function writeU32Be(data: Uint8Array, offset: number, value: number): void {
  data[offset] = (value >>> 24) & 0xff;
  data[offset + 1] = (value >>> 16) & 0xff;
  data[offset + 2] = (value >>> 8) & 0xff;
  data[offset + 3] = value & 0xff;
}

function writeU32Le(data: Uint8Array, offset: number, value: number): void {
  data[offset] = value & 0xff;
  data[offset + 1] = (value >>> 8) & 0xff;
  data[offset + 2] = (value >>> 16) & 0xff;
  data[offset + 3] = (value >>> 24) & 0xff;
}

function writeU24Le(data: Uint8Array, offset: number, value: number): void {
  data[offset] = value & 0xff;
  data[offset + 1] = (value >>> 8) & 0xff;
  data[offset + 2] = (value >>> 16) & 0xff;
}
