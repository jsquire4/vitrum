import { describe, expect, it, vi } from 'vitest';
import { createMaterialTextureArray } from '../scene/materialTextureArray.js';
import { installGpuConstStubs } from './gpuStub.js';

function makeDevice(): GPUDevice {
  return {
    limits: { maxTextureDimension2D: 8192 },
    queue: {
      writeTexture: vi.fn(),
      copyExternalImageToTexture: vi.fn(),
    },
    createTexture: vi.fn((desc: GPUTextureDescriptor) => {
      const size = desc.size as { width: number; height: number; depthOrArrayLayers?: number };
      return {
        label: desc.label,
        width: size.width,
        height: size.height,
        depthOrArrayLayers: size.depthOrArrayLayers ?? 1,
        format: desc.format,
        createView: vi.fn(() => ({})),
        destroy: vi.fn(),
      };
    }),
    createSampler: vi.fn(() => ({})),
  } as unknown as GPUDevice;
}

function rawImage(width: number, height: number): { width: number; height: number; data: Uint8Array } {
  return { width, height, data: new Uint8Array(width * height * 4) };
}

describe('createMaterialTextureArray', () => {
  it('reports per-layer UV-fit scales for heterogeneous source dimensions', () => {
    installGpuConstStubs();
    const array = createMaterialTextureArray(makeDevice(), [
      rawImage(2, 4),
      rawImage(4, 4),
      rawImage(4, 2),
    ]);

    expect(array.layerUvScales).toEqual([
      [0.5, 1],
      [1, 1],
      [1, 0.5],
    ]);
    expect(array.warnings).toContain(
      '[materialTextureArray] source 0 is 2×4 but the array is 4×4; copied at native size and sampled through a per-layer UV-fit scale. Use same-size textures when exact mip/border filtering parity is required.',
    );
    expect(array.warnings).toContain(
      '[materialTextureArray] source 2 is 4×2 but the array is 4×4; copied at native size and sampled through a per-layer UV-fit scale. Use same-size textures when exact mip/border filtering parity is required.',
    );
  });
});
