// texturePixelBudget.test.ts — pre-allocation decode ceilings (V3-2).
//
// The decoded-texture size ceiling (maxTextureSize) and the pixel-budget
// rejection (maxDecodedTexturePixels) must both be applied BEFORE the
// full-resolution Float32Array is allocated, so a hostile/huge texture can
// never force an unbounded allocation. These tests spy on Float32Array
// allocations to prove the full-resolution buffer is never created.

import { afterEach, describe, expect, it, vi } from 'vitest';
import type { MeshPrimitive, Scene, TextureRef } from '@vitrum/core';
import { decodeSceneTextures } from './texturePipeline.js';

interface PixelHandle {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array;
  readonly channels: 4;
  readonly dataType: 'uint8';
  readonly colorSpace: 'srgb';
}

function makePixelHandle(width: number, height: number): PixelHandle {
  // Deterministic ramp so filtered clamped output is checkable.
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    data[i * 4] = i % 256;
    data[i * 4 + 1] = (i * 2) % 256;
    data[i * 4 + 2] = (i * 3) % 256;
    data[i * 4 + 3] = 255;
  }
  return { width, height, data, channels: 4, dataType: 'uint8', colorSpace: 'srgb' };
}

function sceneWith(handle: PixelHandle): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'budget-mesh',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 1,
          metallic: 0,
          baseColorMap: { handle, texCoord: 0 },
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/**
 * Spy on Float32Array allocation sizes for the duration of `fn`. Returns the
 * largest allocation length observed (element count).
 */
async function withFloat32AllocSpy(fn: () => Promise<void>): Promise<{ maxLen: number; lens: number[] }> {
  const OriginalFloat32Array = Float32Array;
  const lens: number[] = [];
  const Spy = new Proxy(OriginalFloat32Array, {
    construct(target, args: unknown[], newTarget): object {
      if (typeof args[0] === 'number') lens.push(args[0]);
      return Reflect.construct(target, args, newTarget) as object;
    },
  });
  (globalThis as unknown as { Float32Array: typeof Float32Array }).Float32Array =
    Spy;
  try {
    await fn();
  } finally {
    (globalThis as unknown as { Float32Array: typeof Float32Array }).Float32Array = OriginalFloat32Array;
  }
  return { maxLen: lens.length > 0 ? Math.max(...lens) : 0, lens };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('decoded-texture pixel budget (maxDecodedTexturePixels)', () => {
  it('rejects an over-budget texture and never allocates the full-resolution buffer', async () => {
    // 64x64 = 4096 pixels; budget 1024 pixels → reject before allocation.
    const handle = makePixelHandle(64, 64);
    const diagnostics: unknown[] = [];
    let result: Awaited<ReturnType<typeof decodeSceneTextures>> | null = null;
    const spy = await withFloat32AllocSpy(async () => {
      result = await decodeSceneTextures(sceneWith(handle), {
        target: 'cpu-linear',
        maxDecodedTexturePixels: 1024,
        onDiagnostic: (d) => diagnostics.push(d),
      });
    });

    // The full-resolution RGBA float buffer would be 64*64*4 = 16384 elements.
    // With rejection-before-allocation, that allocation must never happen.
    expect(spy.maxLen).toBeLessThan(64 * 64 * 4);

    expect(result).not.toBeNull();
    const res = result!;
    // Texture left unchanged (still the original pixel handle).
    const material = (res.scene.primitives[0] as MeshPrimitive).material;
    const baseColor = material.baseColorMap as TextureRef;
    expect(baseColor.handle).toBe(handle);
    expect(res.decodedCount).toBe(0);
    expect(res.diagnostics).toContainEqual(
      expect.objectContaining({
        code: 'decoded-texture-exceeds-pixel-budget',
        width: 64,
        height: 64,
        maxDecodedTexturePixels: 1024,
      }),
    );
  });

  it('accepts a texture at exactly the pixel budget', async () => {
    const handle = makePixelHandle(32, 32); // 1024 pixels
    const result = await decodeSceneTextures(sceneWith(handle), {
      target: 'cpu-linear',
      maxDecodedTexturePixels: 1024,
    });
    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    const baseColor = material.baseColorMap as TextureRef;
    expect(baseColor.handle).not.toBe(handle); // decoded
    expect(result.decodedCount).toBe(1);
    expect(
      result.diagnostics.some((d) => d.code === 'decoded-texture-exceeds-pixel-budget'),
    ).toBe(false);
  });

  it('accepts a small texture under the safe default and explicit zero opt-out', async () => {
    const handle = makePixelHandle(8, 8);
    const result = await decodeSceneTextures(sceneWith(handle), { target: 'cpu-linear' });
    expect(result.decodedCount).toBe(1);
    const result0 = await decodeSceneTextures(sceneWith(handle), {
      target: 'cpu-linear',
      maxDecodedTexturePixels: 0,
    });
    expect(result0.decodedCount).toBe(1);
  });
});

describe('maxTextureSize clamp is applied BEFORE allocation (fused resize)', () => {
  it('only ever allocates the clamped buffer, never the full-resolution one', async () => {
    // Source 64x64 (16384 float elements at full-res), clamped to 8x8 (256).
    const handle = makePixelHandle(64, 64);
    let result: Awaited<ReturnType<typeof decodeSceneTextures>> | null = null;
    const spy = await withFloat32AllocSpy(async () => {
      result = await decodeSceneTextures(sceneWith(handle), {
        target: 'cpu-linear',
        maxTextureSize: 8,
      });
    });

    // The clamped destination is 8*8*4 = 256 elements. The pre-fuse code would
    // first allocate 64*64*4 = 16384 then resize; the fused path must not.
    expect(spy.maxLen).toBeLessThanOrEqual(8 * 8 * 4);
    expect(spy.maxLen).toBeGreaterThan(0);

    const res = result!;
    const material = (res.scene.primitives[0] as MeshPrimitive).material;
    const baseColor = material.baseColorMap as TextureRef;
    const decoded = baseColor.handle as {
      width: number;
      height: number;
      data: Float32Array;
      __vitrum_hint__: { originalWidth?: number; originalHeight?: number; maxTextureSize?: number };
    };
    expect(decoded.width).toBe(8);
    expect(decoded.height).toBe(8);
    expect(decoded.data.length).toBe(8 * 8 * 4);
    // Original dimensions are still recorded on the metadata for diagnostics.
    expect(decoded.__vitrum_hint__.originalWidth).toBe(64);
    expect(decoded.__vitrum_hint__.originalHeight).toBe(64);
    expect(decoded.__vitrum_hint__.maxTextureSize).toBe(8);
  });

  it('area-filters clamped output in linear light', async () => {
    // 2x1 source clamped to 1x1: red and green contribute equal energy.
    const handle: PixelHandle = {
      width: 2,
      height: 1,
      data: new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]),
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    };
    const result = await decodeSceneTextures(sceneWith(handle), {
      target: 'cpu-linear',
      maxTextureSize: 1,
    });
    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    const decoded = (material.baseColorMap as TextureRef).handle as {
      width: number;
      height: number;
      data: Float32Array;
    };
    expect(decoded.width).toBe(1);
    expect(decoded.height).toBe(1);
    expect(decoded.data[0]).toBeCloseTo(0.5, 5);
    expect(decoded.data[1]).toBeCloseTo(0.5, 5);
    expect(decoded.data[2]).toBeCloseTo(0, 5);
    expect(decoded.data[3]).toBe(1);
  });

  it('preserves checkerboard energy during area downsampling', async () => {
    const data = new Uint8Array(4 * 4 * 4);
    for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) {
        const value = (x + y) % 2 === 0 ? 0 : 255;
        const offset = (y * 4 + x) * 4;
        data.set([value, value, value, 255], offset);
      }
    }
    const result = await decodeSceneTextures(sceneWith({
      width: 4,
      height: 4,
      data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    }), {
      target: 'cpu-linear',
      maxTextureSize: 1,
    });
    const decoded = ((result.scene.primitives[0] as MeshPrimitive).material
      .baseColorMap as TextureRef).handle as { data: Float32Array };
    expect([...decoded.data]).toEqual([
      expect.closeTo(0.5, 6),
      expect.closeTo(0.5, 6),
      expect.closeTo(0.5, 6),
      1,
    ]);
  });

  it('re-encodes an area-filtered sRGB output after linear-light averaging', async () => {
    const result = await decodeSceneTextures(sceneWith({
      width: 2,
      height: 1,
      data: new Uint8Array([
        0, 0, 0, 255,
        255, 255, 255, 255,
      ]),
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    }), {
      target: 'webgpu',
      maxTextureSize: 1,
    });
    const decoded = ((result.scene.primitives[0] as MeshPrimitive).material
      .baseColorMap as TextureRef).handle as {
        data: Float32Array;
        __vitrum_hint__: { colorSpace: string };
      };
    const encodedLinearHalf = 1.055 * (0.5 ** (1 / 2.4)) - 0.055;
    expect(decoded.__vitrum_hint__.colorSpace).toBe('srgb');
    expect(decoded.data[0]).toBeCloseTo(encodedLinearHalf, 6);
    expect(decoded.data[1]).toBeCloseTo(encodedLinearHalf, 6);
    expect(decoded.data[2]).toBeCloseTo(encodedLinearHalf, 6);
    expect(decoded.data[0]).not.toBeCloseTo(0.5, 2);
  });

  it('uses pixel-centred bilinear reconstruction when NPOT policy upsamples', async () => {
    const result = await decodeSceneTextures(sceneWith({
      width: 3,
      height: 1,
      data: new Uint8Array([
        0, 0, 0, 255,
        255, 255, 255, 255,
        0, 0, 0, 255,
      ]),
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    }), {
      target: 'cpu-linear',
      npotRepeatWrapPolicy: 'resize-to-pot',
    });
    const decoded = ((result.scene.primitives[0] as MeshPrimitive).material
      .baseColorMap as TextureRef).handle as {
        width: number;
        height: number;
        data: Float32Array;
      };
    expect([decoded.width, decoded.height]).toEqual([4, 1]);
    expect([
      decoded.data[0],
      decoded.data[4],
      decoded.data[8],
      decoded.data[12],
    ]).toEqual([
      expect.closeTo(0, 6),
      expect.closeTo(0.625, 6),
      expect.closeTo(0.625, 6),
      expect.closeTo(0, 6),
    ]);
  });

  it('leaves a within-size texture unchanged in dimensions', async () => {
    const handle = makePixelHandle(4, 4);
    const result = await decodeSceneTextures(sceneWith(handle), {
      target: 'cpu-linear',
      maxTextureSize: 16,
    });
    const decoded = ((result.scene.primitives[0] as MeshPrimitive).material.baseColorMap as TextureRef)
      .handle as { width: number; height: number };
    expect(decoded.width).toBe(4);
    expect(decoded.height).toBe(4);
  });
});
