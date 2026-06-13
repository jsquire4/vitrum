import { describe, expect, it, vi } from 'vitest';
import { createMaterialTextureArray } from '../scene/materialTextureArray.js';
import { installGpuConstStubs } from './gpuStub.js';

function makeDevice(): {
  readonly device: GPUDevice;
  readonly writeTexture: ReturnType<typeof vi.fn>;
} {
  const writeTexture = vi.fn();
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    queue: {
      writeTexture,
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
  return { device, writeTexture };
}

function rawImage(width: number, height: number): { width: number; height: number; data: Uint8Array } {
  return { width, height, data: new Uint8Array(width * height * 4) };
}

describe('createMaterialTextureArray', () => {
  it('reports per-layer UV-fit scales for heterogeneous source dimensions', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const array = createMaterialTextureArray(device, [
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

  it('expands 8-bit RGB raw data to RGBA8 rows before writeTexture', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const array = createMaterialTextureArray(device, [
      { width: 2, height: 1, data: new Uint8Array([10, 20, 30, 40, 50, 60]) },
    ]);

    expect(array.warnings).toEqual([]);
    expect(writeTexture).toHaveBeenCalledTimes(1);
    const call = writeTexture.mock.calls[0] as [
      GPUImageCopyTexture,
      Uint8Array,
      GPUImageDataLayout,
      GPUExtent3D,
    ];
    expect(Array.from(call[1])).toEqual([10, 20, 30, 255, 40, 50, 60, 255]);
    expect(call[2]).toEqual({ bytesPerRow: 8, rowsPerImage: 1 });
    expect(call[3]).toEqual({ width: 2, height: 1 });
  });

  it('warns and leaves unsupported raw typed-array data black', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const array = createMaterialTextureArray(device, [
      { width: 1, height: 1, data: new Float32Array([1, 0, 0, 1]) },
    ]);

    expect(writeTexture).not.toHaveBeenCalled();
    expect(array.warnings).toContain(
      '[materialTextureArray] source 0 has raw data with unsupported byte layout (16 bytes for 1×1); expected 1, 2, 3, or 4 8-bit channel(s) per pixel. Layer left black.',
    );
  });
});
