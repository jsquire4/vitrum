import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { createWalkaroundWebGpuTextureSource } from '../materialTextureSource.js';
import {
  MATERIAL_ATLAS_GENERATE_MIP_WGSL,
  MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
  packMaterialTextureAtlas,
  uploadMaterialTextureAtlas,
} from '../pipeline/materialTextureAtlas.js';

const TEXTURE_BINDING = 0x04;

function makeSourceTexture(overrides: Partial<{
  width: number;
  height: number;
  depthOrArrayLayers: number;
  mipLevelCount: number;
  sampleCount: number;
  dimension: GPUTextureDimension;
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
}> = {}) {
  const createView = vi.fn(() => ({}));
  const destroy = vi.fn();
  const texture = {
    width: 4,
    height: 4,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba8unorm',
    usage: TEXTURE_BINDING,
    createView,
    destroy,
    ...overrides,
  } as unknown as GPUTexture;
  return { texture, createView, destroy };
}

function materialWithMaps(baseColorHandle: unknown, normalHandle?: unknown): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 0.5,
    metallic: 0,
    baseColorMap: { handle: baseColorHandle },
    ...(normalHandle == null ? {} : { normalMap: { handle: normalHandle } }),
  };
}

function makeDestinationTexture(descriptor: GPUTextureDescriptor) {
  return {
    descriptor,
    createView: vi.fn(() => ({})),
    destroy: vi.fn(),
  };
}

function makeUploadHarness(failBeginComputePass = false) {
  const destinations: ReturnType<typeof makeDestinationTexture>[] = [];
  const createTexture = vi.fn((descriptor: GPUTextureDescriptor) => {
    const destination = makeDestinationTexture(descriptor);
    destinations.push(destination);
    return destination;
  });
  const createShaderModule = vi.fn((descriptor: GPUShaderModuleDescriptor) => descriptor);
  const createBindGroupLayout = vi.fn((descriptor: GPUBindGroupLayoutDescriptor) => descriptor);
  const createPipelineLayout = vi.fn((descriptor: GPUPipelineLayoutDescriptor) => descriptor);
  const createComputePipeline = vi.fn(() => ({}));
  const createBindGroup = vi.fn(() => ({}));
  const bufferDestroy = vi.fn();
  const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => ({
    getMappedRange: vi.fn(() => new ArrayBuffer(Number(descriptor.size))),
    unmap: vi.fn(),
    destroy: bufferDestroy,
  }));
  const dispatchWorkgroups = vi.fn();
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups,
    end: vi.fn(),
  };
  const beginComputePass = vi.fn(() => {
    if (failBeginComputePass) throw new Error('compute pass construction failed');
    return pass;
  });
  const finish = vi.fn(() => ({}));
  const writeTexture = vi.fn();
  const submit = vi.fn();
  const device = {
    limits: { maxTextureDimension2D: 4096, maxTextureArrayLayers: 256 },
    createTexture,
    createShaderModule,
    createBindGroupLayout,
    createPipelineLayout,
    createComputePipeline,
    createBindGroup,
    createBuffer,
    createCommandEncoder: vi.fn(() => ({ beginComputePass, finish })),
    queue: { writeTexture, submit },
  } as unknown as GPUDevice;
  return {
    device,
    destinations,
    createTexture,
    createShaderModule,
    createBindGroupLayout,
    createBuffer,
    bufferDestroy,
    dispatchWorkgroups,
    beginComputePass,
    writeTexture,
    submit,
  };
}

describe('walkaround explicit WebGPU material texture sources', () => {
  it('pins source format, transfer function, selected subresource, and identity', () => {
    const device = {} as GPUDevice;
    const { texture } = makeSourceTexture({
      width: 16,
      height: 8,
      depthOrArrayLayers: 3,
      mipLevelCount: 4,
    });
    const first = createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      baseMipLevel: 2,
      arrayLayer: 1,
    });
    const second = createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      baseMipLevel: 2,
      arrayLayer: 1,
    });

    expect(first).toMatchObject({
      device,
      texture,
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      dimension: '2d',
      ownership: 'host',
      baseMipLevel: 2,
      arrayLayer: 1,
      width: 4,
      height: 2,
    });
    expect(Object.isFrozen(first)).toBe(true);
    expect(second.sourceId).toBeGreaterThan(first.sourceId);
  });

  it('snapshots an exact immutable CPU mirror and retains its texels in the CPU atlas', () => {
    const device = {} as GPUDevice;
    const { texture } = makeSourceTexture({ width: 2, height: 2 });
    const pixels = new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
      128, 64, 32, 255,
    ]);
    const source = createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      cpuMirror: {
        width: 2,
        height: 2,
        channels: 4,
        dataType: 'uint8',
        colorSpace: 'linear',
        data: pixels,
      },
    });

    pixels[0] = 0;
    expect(source.cpuMirror?.data[0]).toBe(255);
    expect(() => Reflect.set(source.cpuMirror!.data, '0', 0)).toThrow(/immutable/);

    const payload = packMaterialTextureAtlas(
      [materialWithMaps(source)],
      new Uint32Array([0]),
      1,
    );
    expect(payload.gpuSourceLayers).toEqual([{ layer: 0, source, decodeSrgb: false }]);
    expect(Array.from(payload.atlasData)).toEqual(Array.from(new Float32Array([
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 1, 1,
      128 / 255, 64 / 255, 32 / 255, 1,
    ])));
  });

  it('rejects CPU mirrors that cannot exactly identify the selected GPU subresource', () => {
    const device = {} as GPUDevice;
    const { texture } = makeSourceTexture({ width: 2, height: 2 });
    const base = {
      width: 2,
      height: 2,
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'linear' as const,
      data: new Uint8Array(16),
    };

    expect(() => createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      cpuMirror: { ...base, width: 1 },
    })).toThrow(/dimensions must exactly match/);
    expect(() => createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      cpuMirror: base,
    })).toThrow(/colorSpace must match/);
    expect(() => createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      cpuMirror: { ...base, data: new Uint8Array(15) },
    })).toThrow(/length must be exactly 16/);
  });

  it('rejects guessed formats, incompatible native-sRGB declarations, and invalid subresources', () => {
    const device = {} as GPUDevice;
    const linear = makeSourceTexture();
    expect(() => createWalkaroundWebGpuTextureSource(device, linear.texture, {
      format: 'rgba16float',
      colorSpace: 'linear',
    })).toThrow(/does not match texture\.format/);

    const nativeSrgb = makeSourceTexture({ format: 'rgba8unorm-srgb' });
    expect(() => createWalkaroundWebGpuTextureSource(device, nativeSrgb.texture, {
      format: 'rgba8unorm-srgb',
      colorSpace: 'linear',
    })).toThrow(/cannot be declared as linear/);

    expect(() => createWalkaroundWebGpuTextureSource(device, linear.texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      baseMipLevel: 1,
    })).toThrow(/outside 1 texture mip levels/);
  });

  it('packs mixed CPU and GPU sources without CPU readback and records conversion work', () => {
    const harness = makeUploadHarness();
    const sourceTexture = makeSourceTexture();
    const source = createWalkaroundWebGpuTextureSource(harness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
    });
    const cpuHandle = {
      width: 2,
      height: 2,
      data: new Uint8Array(2 * 2 * 4).fill(255),
      __vitrum_hint__: { channels: 4 as const, dataType: 'uint8' as const },
    };

    const payload = packMaterialTextureAtlas(
      [materialWithMaps(cpuHandle, source)],
      new Uint32Array([0]),
      1,
    );

    expect(payload.atlasDim).toBe(4);
    expect(payload.atlasLayerCount).toBe(2);
    expect(payload.atlasMipLevelCount).toBe(3);
    expect(payload.readableBaseColorLayerCount).toBe(1);
    expect(payload.readableNormalLayerCount).toBe(1);
    expect(payload.gpuSourceLayers).toEqual([{ layer: 1, source, decodeSrgb: false }]);
    expect(Array.from(payload.atlasData.slice(4 * 4 * 4))).toEqual(
      Array.from(new Float32Array(4 * 4 * 4)),
    );
  });

  it('rejects raw GPUTexture handles and sRGB sources used for linear-data maps', () => {
    const device = {} as GPUDevice;
    const raw = makeSourceTexture();
    expect(() => packMaterialTextureAtlas(
      [materialWithMaps(raw.texture)],
      new Uint32Array([0]),
      1,
    )).toThrow(/Wrap it with createWalkaroundWebGpuTextureSource/);

    const encoded = createWalkaroundWebGpuTextureSource(device, raw.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    expect(() => packMaterialTextureAtlas(
      [materialWithMaps({ width: 1, height: 1, data: new Uint8Array(4) }, encoded)],
      new Uint32Array([0]),
      1,
    )).toThrow(/normalMap is a linear-data map.*declares srgb/);
  });

  it('uploads GPU layers, generates every mip, and preserves host source ownership', () => {
    const harness = makeUploadHarness();
    const sourceTexture = makeSourceTexture();
    const source = createWalkaroundWebGpuTextureSource(harness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const payload = packMaterialTextureAtlas(
      [materialWithMaps(source)],
      new Uint32Array([0]),
      1,
    );

    const uploaded = uploadMaterialTextureAtlas(harness.device, payload);

    expect(harness.destinations).toHaveLength(2);
    expect(harness.destinations[0]?.descriptor).toMatchObject({
      size: { width: 4, height: 4, depthOrArrayLayers: 1 },
      mipLevelCount: 3,
      format: 'rgba32float',
    });
    expect(Number(harness.destinations[0]?.descriptor.usage) & 0x08).toBe(0x08);
    expect(harness.writeTexture).toHaveBeenCalledTimes(2);
    expect(harness.createShaderModule.mock.calls.map((call) => call[0]?.code)).toEqual([
      MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
      MATERIAL_ATLAS_GENERATE_MIP_WGSL,
    ]);
    expect(harness.createBindGroupLayout.mock.calls.map((call) => call[0]?.entries)).toEqual([
      [
        expect.objectContaining({
          binding: 0,
          texture: expect.objectContaining({
            sampleType: 'unfilterable-float',
            viewDimension: '2d',
          }),
        }),
        expect.objectContaining({
          binding: 1,
          storageTexture: expect.objectContaining({
            access: 'write-only',
            format: 'rgba32float',
            viewDimension: '2d-array',
          }),
        }),
        expect.objectContaining({ binding: 2, buffer: { type: 'uniform' } }),
      ],
      [
        expect.objectContaining({
          binding: 0,
          texture: expect.objectContaining({
            sampleType: 'unfilterable-float',
            viewDimension: '2d-array',
          }),
        }),
        expect.objectContaining({
          binding: 1,
          storageTexture: expect.objectContaining({
            access: 'write-only',
            format: 'rgba32float',
            viewDimension: '2d-array',
          }),
        }),
      ],
    ]);
    expect(sourceTexture.createView).toHaveBeenCalledWith(expect.objectContaining({
      baseMipLevel: 0,
      baseArrayLayer: 0,
      arrayLayerCount: 1,
    }));
    expect(harness.dispatchWorkgroups.mock.calls).toEqual([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    expect(harness.submit).toHaveBeenCalledTimes(1);
    expect(harness.bufferDestroy).toHaveBeenCalledTimes(1);
    expect(sourceTexture.destroy).not.toHaveBeenCalled();
    expect(uploaded.atlasDim).toBe(4);
  });

  it('fails preflight before destination allocation for device, format, and usage mismatches', () => {
    const sourceHarness = makeUploadHarness();
    const otherHarness = makeUploadHarness();
    const sourceTexture = makeSourceTexture();
    const foreign = createWalkaroundWebGpuTextureSource(sourceHarness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const foreignPayload = packMaterialTextureAtlas(
      [materialWithMaps(foreign)], new Uint32Array([0]), 1,
    );
    expect(() => uploadMaterialTextureAtlas(otherHarness.device, foreignPayload))
      .toThrow(/different GPUDevice/);
    expect(otherHarness.createTexture).not.toHaveBeenCalled();

    const unsupportedTexture = makeSourceTexture({ format: 'depth24plus' });
    const unsupported = createWalkaroundWebGpuTextureSource(otherHarness.device, unsupportedTexture.texture, {
      format: 'depth24plus',
      colorSpace: 'linear',
    });
    const unsupportedPayload = packMaterialTextureAtlas(
      [materialWithMaps({ width: 1, height: 1, data: new Uint8Array(4) }, unsupported)],
      new Uint32Array([0]),
      1,
    );
    expect(() => uploadMaterialTextureAtlas(otherHarness.device, unsupportedPayload))
      .toThrow(/unsupported format depth24plus/);
    expect(otherHarness.createTexture).not.toHaveBeenCalled();

    const noBindingTexture = makeSourceTexture({ usage: 0 });
    const noBinding = createWalkaroundWebGpuTextureSource(otherHarness.device, noBindingTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const noBindingPayload = packMaterialTextureAtlas(
      [materialWithMaps(noBinding)], new Uint32Array([0]), 1,
    );
    expect(() => uploadMaterialTextureAtlas(otherHarness.device, noBindingPayload))
      .toThrow(/lacks GPUTextureUsage\.TEXTURE_BINDING/);
    expect(otherHarness.createTexture).not.toHaveBeenCalled();
  });

  it('destroys destination resources and transient buffers after compute setup failure', () => {
    const harness = makeUploadHarness(true);
    const sourceTexture = makeSourceTexture();
    const source = createWalkaroundWebGpuTextureSource(harness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const payload = packMaterialTextureAtlas(
      [materialWithMaps(source)], new Uint32Array([0]), 1,
    );

    expect(() => uploadMaterialTextureAtlas(harness.device, payload))
      .toThrow('compute pass construction failed');
    expect(harness.destinations).toHaveLength(2);
    expect(harness.destinations[0]?.destroy).toHaveBeenCalledTimes(1);
    expect(harness.destinations[1]?.destroy).toHaveBeenCalledTimes(1);
    expect(harness.createBuffer).toHaveBeenCalledTimes(1);
    expect(harness.bufferDestroy).toHaveBeenCalledTimes(1);
    expect(sourceTexture.destroy).not.toHaveBeenCalled();
    expect(harness.submit).not.toHaveBeenCalled();
  });
});
