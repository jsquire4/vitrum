import { describe, expect, it } from 'vitest';
import { readEnvironmentMapPixels } from '../environmentMapPixels.js';

describe('readEnvironmentMapPixels', () => {
  it('reads raw RGB float environment handles as linear RGBA pixels', () => {
    const pixels = readEnvironmentMapPixels({
      width: 2,
      height: 1,
      data: new Float32Array([
        1, 0.25, 0,
        0, 0.5, 1,
      ]),
    });

    expect(pixels).not.toBeNull();
    expect(pixels!.width).toBe(2);
    expect(pixels!.height).toBe(1);
    expect(pixels!.sourceChannels).toBe(3);
    expect(pixels!.explicitChannels).toBe(false);
    expect(Array.from(pixels!.data.slice(0, 8))).toEqual([
      1, 0.25, 0, 1,
      0, 0.5, 1, 1,
    ]);
  });

  it('reads DataTexture-shaped image payloads', () => {
    const pixels = readEnvironmentMapPixels({
      image: {
        width: 1,
        height: 2,
        data: new Float32Array([
          0.1, 0.2, 0.3, 1,
          0.4, 0.5, 0.6, 1,
        ]),
      },
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    });

    expect(pixels).not.toBeNull();
    expect(pixels!.width).toBe(1);
    expect(pixels!.height).toBe(2);
    expect(pixels!.sourceChannels).toBe(4);
    expect(pixels!.explicitChannels).toBe(true);
    expect(pixels!.data[0]).toBeCloseTo(0.1);
    expect(pixels!.data[4]).toBeCloseTo(0.4);
  });

  it('normalizes hinted uint8 sRGB payloads into linear-light pixels', () => {
    const pixels = readEnvironmentMapPixels({
      width: 1,
      height: 1,
      data: new Uint8Array([128, 255, 0, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'srgb' },
    });

    expect(pixels).not.toBeNull();
    expect(pixels!.sourceColorSpace).toBe('srgb');
    expect(pixels!.data[0]).toBeCloseTo(0.21586, 4);
    expect(pixels!.data[1]).toBeCloseTo(1);
    expect(pixels!.data[2]).toBeCloseTo(0);
    expect(pixels!.data[3]).toBeCloseTo(1);
  });
});
