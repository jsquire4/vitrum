import { describe, expect, it } from 'vitest';
import { resolveReadableTexture, srgbToLinear } from '../textureDecode.js';

describe('resolveReadableTexture color-space inference', () => {
  it('treats unhinted Float32 color-role texels as linear HDR', () => {
    const data = new Float32Array([0.25, 0.5, 4, 1]);
    const resolved = resolveReadableTexture(
      { width: 1, height: 1, data },
      'srgb',
      undefined,
      undefined,
      undefined,
    );

    expect(resolved).not.toBeNull();
    expect(resolved?.needsSrgbDecode).toBe(false);
    expect(resolved?.decode(data[0]!)).toBe(0.25);
    expect(resolved?.decode(data[1]!)).toBe(0.5);
    expect(resolved?.decode(data[2]!)).toBe(4);
  });

  it('applies an explicit sRGB hint even to Float32 payloads', () => {
    const data = new Float32Array([0.25, 0.5, 1, 1]);
    const resolved = resolveReadableTexture(
      { width: 1, height: 1, data },
      'srgb',
      4,
      'float32',
      'srgb',
    );

    expect(resolved).not.toBeNull();
    expect(resolved?.needsSrgbDecode).toBe(true);
    expect(srgbToLinear(resolved!.decode(data[0]!))).toBeCloseTo(
      srgbToLinear(0.25),
      12,
    );
    expect(srgbToLinear(resolved!.decode(data[1]!))).toBeCloseTo(
      srgbToLinear(0.5),
      12,
    );
  });

  it('uses explicit float32 metadata for immutable array-like mirrors', () => {
    const data: ArrayLike<number> = { 0: 0.25, 1: 0.5, 2: 8, 3: 1, length: 4 };
    const resolved = resolveReadableTexture(
      { width: 1, height: 1, data },
      'srgb',
      4,
      'float32',
      undefined,
    );

    expect(resolved).not.toBeNull();
    expect(resolved?.needsSrgbDecode).toBe(false);
    expect(resolved?.decode(Number(data[2]))).toBe(8);
  });
});
