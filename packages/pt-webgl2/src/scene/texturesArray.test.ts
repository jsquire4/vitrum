import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packTextureAtlas, textureAtlasLayerCapacity, type TextureHandleHint } from './texturesArray.js';

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
    expect(atlas!.mipLevels).toHaveLength(2);
    expect(atlas!.mipLevels[1]!.dim).toBe(1);
    expect(Array.from(atlas!.mipLevels[1]!.data.slice(0, 4))).toEqual([1, 1, 1, 1]);
    expect(atlas!.mipLevels[1]!.data[4]).toBeCloseTo(srgbToLinear(0.5), 6);
    expect(atlas!.mipLevels[1]!.data[5]).toBeCloseTo(srgbToLinear(0.5), 6);
    expect(atlas!.mipLevels[1]!.data[6]).toBeCloseTo(srgbToLinear(0.5), 6);
    expect(atlas!.mipLevels[1]!.data[7]).toBeCloseTo(0.5, 6);
  });

  it('averages every source texel when generating odd-sized mip levels', () => {
    const data = new Float32Array(3 * 3 * 4);
    for (let i = 0; i < 9; i += 1) {
      data[i * 4] = i;
      data[i * 4 + 3] = 1;
    }
    const handle = dataTexHandle(data, 3, 3);
    const atlas = packTextureAtlas([{
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      roughnessMap: { handle },
    }]);

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
    const atlas = packTextureAtlas([
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
    ], { onWarning, warningPhase: 'setScene', warningMethod: 'setScene' });

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
    const handle = hintedHandle(new Uint8Array([255, 0, 0, 255]), 1, 1, { channels: 4 });
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

  it('ambiguous stride (3 channels) without hint emits a console.warn', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      // 1×1, 3-channel RGB (stride 3 is ambiguous without a hint)
      const handle = { width: 1, height: 1, data: new Uint8Array([255, 128, 0]) };
      packTextureAtlas([mat(handle)]);
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('ambiguous pixel stride'));
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
