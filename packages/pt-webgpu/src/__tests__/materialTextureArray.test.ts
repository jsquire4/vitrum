import { describe, expect, it, vi } from 'vitest';
import {
  createMaterialTextureArray,
  materialTextureMipLevelCount,
} from '../scene/materialTextureArray.js';
import { installGpuConstStubs } from './gpuStub.js';

function makeDevice() {
  const writeTexture = vi.fn();
  const beginRenderPass = vi.fn(() => ({
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    draw: vi.fn(),
    end: vi.fn(),
  }));
  const submit = vi.fn();
  const createTexture = vi.fn((desc: GPUTextureDescriptor) => {
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
  });
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    queue: {
      writeTexture,
      copyExternalImageToTexture: vi.fn(),
      submit,
    },
    createTexture,
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({})),
    createRenderPipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginRenderPass,
      finish: vi.fn(() => ({})),
    })),
  } as unknown as GPUDevice;
  return { device, writeTexture, createTexture, beginRenderPass, submit };
}

function rawImage(width: number, height: number): { width: number; height: number; data: Uint8Array } {
  return { width, height, data: new Uint8Array(width * height * 4) };
}

describe('createMaterialTextureArray', () => {
  it('computes full mip chain lengths from the max texture dimension', () => {
    expect(materialTextureMipLevelCount(1, 1)).toBe(1);
    expect(materialTextureMipLevelCount(2, 1)).toBe(2);
    expect(materialTextureMipLevelCount(4, 2)).toBe(3);
    expect(materialTextureMipLevelCount(8, 8)).toBe(4);
  });

  it('allocates and renders a mip chain for each material texture array layer', () => {
    installGpuConstStubs();
    const { device, createTexture, beginRenderPass, submit } = makeDevice();
    const array = createMaterialTextureArray(device, [
      rawImage(4, 4),
      rawImage(4, 4),
    ]);

    expect(array.mipLevelCount).toBe(3);
    expect(createTexture).toHaveBeenCalledWith(expect.objectContaining({
      mipLevelCount: 3,
      size: { width: 4, height: 4, depthOrArrayLayers: 2 },
    }));
    expect(beginRenderPass).toHaveBeenCalledTimes(4);
    expect(submit).toHaveBeenCalledTimes(1);
  });

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
