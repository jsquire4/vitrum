import { describe, expect, it, vi } from 'vitest';

import {
  refreshEmissiveTexture,
  uploadEmissiveTexture,
} from '../bvhEmissiveTexture.js';

function mockDevice() {
  const writeTexture = vi.fn();
  return {
    device: {
      limits: { maxTextureDimension2D: 8192 },
      createTexture: vi.fn(() => ({ destroy: vi.fn() })),
      queue: { writeTexture },
    } as unknown as GPUDevice,
    writeTexture,
  };
}

function uploadedFloats(writeTexture: ReturnType<typeof vi.fn>): number[] {
  const buffer = writeTexture.mock.calls[0]?.[1];
  expect(buffer).toBeInstanceOf(ArrayBuffer);
  return Array.from(new Float32Array(buffer as ArrayBuffer));
}

describe('BVH emissive ownership texture uploads', () => {
  it('preserves every rgba lane on initial upload', () => {
    const { device, writeTexture } = mockDevice();
    const rgba = new Float32Array([
      1, 2, 3, 1,
      4, 5, 6, 0,
      7, 8, 9, 1,
    ]);

    uploadEmissiveTexture(device, rgba, 3);

    expect(uploadedFloats(writeTexture)).toEqual(Array.from(rgba));
  });

  it('preserves ownership alpha during full refresh and zero-pads only unused texels', () => {
    const { device, writeTexture } = mockDevice();
    const rgba = new Float32Array([
      0.25, 0.5, 0.75, 0,
      3, 2, 1, 1,
      10, 20, 30, 0,
    ]);

    refreshEmissiveTexture(
      device,
      { texture: {} as GPUTexture, width: 4, height: 1 },
      rgba,
      3,
    );

    expect(uploadedFloats(writeTexture)).toEqual([
      ...Array.from(rgba),
      0, 0, 0, 0,
    ]);
  });
});
