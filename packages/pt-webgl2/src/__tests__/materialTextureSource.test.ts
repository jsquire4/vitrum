import { afterEach, describe, expect, it, vi } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  createPtWebgl2TextureSource,
  isPtWebgl2TextureSource,
} from '../materialTextureSource.js';
import { packTextureAtlas } from '../scene/texturesArray.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('pt-webgl2 immutable material texture sources', () => {
  it('copies raw payloads and feeds the existing heterogeneous atlas without aliasing host memory', () => {
    const pixels = new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 255,
    ]);
    const source = createPtWebgl2TextureSource({
      width: 2,
      height: 1,
      channels: 4,
      dataType: 'uint8',
      data: pixels,
    }, { colorSpace: 'linear' });

    pixels[0] = 0;
    expect(isPtWebgl2TextureSource(source)).toBe(true);
    expect(source.cpuMirror.data[0]).toBe(255);
    expect(() => Reflect.set(source.cpuMirror.data, '0', 0)).toThrow(/immutable/);

    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle: source },
    };
    const atlas = packTextureAtlas([material]);
    expect(atlas).not.toBeNull();
    expect(atlas?.sourceDimensions).toEqual([[2, 1]]);
    expect(Array.from(atlas?.data ?? [])).toEqual(Array.from(new Float32Array([
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ])));
  });

  it('snapshots loaded browser sources through an origin-clean 2D readback surface', () => {
    const drawImage = vi.fn();
    const getImageData = vi.fn(() => ({
      width: 2,
      height: 1,
      data: new Uint8ClampedArray([
        10, 20, 30, 255,
        40, 50, 60, 128,
      ]),
    }));
    const getContext = vi.fn(() => ({ drawImage, getImageData }));
    class FakeOffscreenCanvas {
      constructor(public width: number, public height: number) {}
      getContext = getContext;
    }
    vi.stubGlobal('OffscreenCanvas', FakeOffscreenCanvas);
    const browserSource = {
      naturalWidth: 2,
      naturalHeight: 1,
    } as unknown as TexImageSource;

    const source = createPtWebgl2TextureSource(browserSource, { colorSpace: 'srgb' });

    expect(drawImage).toHaveBeenCalledWith(browserSource, 0, 0, 2, 1);
    expect(getImageData).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(Array.from({ length: source.cpuMirror.data.length }, (_, index) =>
      source.cpuMirror.data[index])).toEqual([
      10, 20, 30, 255,
      40, 50, 60, 128,
    ]);
    expect(source).toMatchObject({
      ownership: 'immutable-snapshot',
      width: 2,
      height: 1,
      colorSpace: 'srgb',
    });

    const imageData = {
      width: 1,
      height: 1,
      data: new Uint8ClampedArray([7, 8, 9, 255]),
      [Symbol.toStringTag]: 'ImageData',
    } as unknown as ImageData;
    const imageDataSource = createPtWebgl2TextureSource(imageData, { colorSpace: 'linear' });
    expect(Array.from({ length: imageDataSource.cpuMirror.data.length }, (_, index) =>
      imageDataSource.cpuMirror.data[index])).toEqual([7, 8, 9, 255]);
    expect(drawImage).toHaveBeenCalledTimes(1);
  });

  it('fails closed for tainted/unloaded browser sources and malformed raw layouts', () => {
    class FailingOffscreenCanvas {
      constructor(public width: number, public height: number) {}
      getContext() {
        return {
          drawImage() {},
          getImageData() {
            throw new DOMException('tainted', 'SecurityError');
          },
        };
      }
    }
    vi.stubGlobal('OffscreenCanvas', FailingOffscreenCanvas);
    const browserSource = { width: 1, height: 1 } as unknown as TexImageSource;
    expect(() => createPtWebgl2TextureSource(browserSource, { colorSpace: 'srgb' }))
      .toThrow(/loaded and origin-clean/);

    expect(() => createPtWebgl2TextureSource({
      width: 1,
      height: 1,
      channels: 4,
      dataType: 'uint8',
      data: new Uint8Array(3),
    }, { colorSpace: 'linear' })).toThrow(/length must be exactly 4/);
    expect(() => createPtWebgl2TextureSource({
      width: 1,
      height: 1,
      channels: 1,
      dataType: 'float32',
      data: [Number.NaN],
    }, { colorSpace: 'linear' })).toThrow(/finite and representable as f32/);
  });
});
