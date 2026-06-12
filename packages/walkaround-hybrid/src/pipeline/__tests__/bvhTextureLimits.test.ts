import { describe, expect, it, vi } from 'vitest';
import { uploadBeerTexture, refreshBeerTexture } from '../bvhBeerTexture.js';
import { uploadEmissiveTexture, refreshEmissiveTexture } from '../bvhEmissiveTexture.js';

function mockDevice(maxTextureDimension2D: number) {
  const createTexture = vi.fn(() => ({
    createView: vi.fn(() => ({})),
    destroy: vi.fn(),
  }));
  const writeTexture = vi.fn();
  return {
    device: {
      limits: { maxTextureDimension2D },
      createTexture,
      queue: { writeTexture },
    } as unknown as GPUDevice,
    createTexture,
    writeTexture,
  };
}

describe('BVH per-triangle texture dimension guards', () => {
  it('rejects beer textures whose computed height exceeds maxTextureDimension2D', () => {
    const { device, createTexture } = mockDevice(4096);
    const tooManyTriangles = 4096 * 4096 + 1;

    expect(() => uploadBeerTexture(device, new ArrayBuffer(0), tooManyTriangles))
      .toThrow(/bvhBeer texture requires 4096x4097 texels/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects emissive textures whose computed width exceeds maxTextureDimension2D', () => {
    const { device, createTexture } = mockDevice(4);

    expect(() => uploadEmissiveTexture(device, new Float32Array(0), 5))
      .toThrow(/bvhEmissive texture requires 5x1 texels/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('does not truncate very large triangle counts through 32-bit bitwise coercion', () => {
    const { device, createTexture } = mockDevice(8192);

    expect(() => uploadBeerTexture(device, new ArrayBuffer(0), 3_000_000_000))
      .toThrow(/3000000000 triangles/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects beer refreshes that no longer fit the existing texture capacity', () => {
    const { device, writeTexture } = mockDevice(4096);
    const texture = { texture: {} as GPUTexture, width: 4, height: 1 };

    expect(() => refreshBeerTexture(device, texture, new ArrayBuffer(0), 5))
      .toThrow(/bvhBeer texture refresh needs 5 triangles/);
    expect(writeTexture).not.toHaveBeenCalled();
  });

  it('rejects emissive refreshes that no longer fit the existing texture capacity', () => {
    const { device, writeTexture } = mockDevice(4096);
    const texture = { texture: {} as GPUTexture, width: 4, height: 1 };

    expect(() => refreshEmissiveTexture(device, texture, new Float32Array(0), 5))
      .toThrow(/bvhEmissive texture refresh needs 5 triangles/);
    expect(writeTexture).not.toHaveBeenCalled();
  });
});
