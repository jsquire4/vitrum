import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import {
  createPtWebgpuTextureSource,
  isPtWebgpuTextureSource,
} from '../materialTextureSource.js';
import { packEmitterArrays } from '../scene/emitterPacking.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import {
  createMaterialTextureArray,
  MATERIAL_TEXTURE_GPU_SOURCE_BLIT_WGSL,
} from '../scene/materialTextureArray.js';
import { installGpuConstStubs } from './gpuStub.js';

interface MockTextureOptions {
  readonly width?: number;
  readonly height?: number;
  readonly depthOrArrayLayers?: number;
  readonly mipLevelCount?: number;
  readonly sampleCount?: number;
  readonly dimension?: GPUTextureDimension;
  readonly format?: GPUTextureFormat;
  readonly usage?: GPUTextureUsageFlags;
}

function makeTexture(options: MockTextureOptions = {}): GPUTexture {
  return {
    width: options.width ?? 16,
    height: options.height ?? 8,
    depthOrArrayLayers: options.depthOrArrayLayers ?? 1,
    mipLevelCount: options.mipLevelCount ?? 1,
    sampleCount: options.sampleCount ?? 1,
    dimension: options.dimension ?? '2d',
    format: options.format ?? 'rgba8unorm',
    usage: options.usage ?? GPUTextureUsage.TEXTURE_BINDING,
    label: '',
    createView: vi.fn(() => ({} as GPUTextureView)),
    destroy: vi.fn(),
  } as unknown as GPUTexture;
}

function makeUploadDevice() {
  installGpuConstStubs();
  const renderPasses: Array<{
    setPipeline: ReturnType<typeof vi.fn>;
    setBindGroup: ReturnType<typeof vi.fn>;
    setViewport: ReturnType<typeof vi.fn>;
    draw: ReturnType<typeof vi.fn>;
    end: ReturnType<typeof vi.fn>;
  }> = [];
  const beginRenderPass = vi.fn(() => {
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      setViewport: vi.fn(),
      draw: vi.fn(),
      end: vi.fn(),
    };
    renderPasses.push(pass);
    return pass;
  });
  const createTexture = vi.fn((descriptor: GPUTextureDescriptor) => {
    const size = descriptor.size as GPUExtent3DDict;
    return makeTexture({
      width: size.width,
      height: size.height ?? 1,
      depthOrArrayLayers: size.depthOrArrayLayers ?? 1,
      mipLevelCount: descriptor.mipLevelCount ?? 1,
      dimension: descriptor.dimension ?? '2d',
      format: descriptor.format,
      usage: descriptor.usage,
    });
  });
  const createRenderPipeline = vi.fn(() => ({
    getBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
  } as unknown as GPURenderPipeline));
  const submit = vi.fn();
  const device = {
    limits: { maxTextureDimension2D: 8192, maxTextureArrayLayers: 256 },
    queue: {
      writeTexture: vi.fn(),
      copyExternalImageToTexture: vi.fn(),
      submit,
    },
    createTexture,
    createSampler: vi.fn(() => ({} as GPUSampler)),
    createShaderModule: vi.fn(() => ({} as GPUShaderModule)),
    createBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
    createPipelineLayout: vi.fn(() => ({} as GPUPipelineLayout)),
    createRenderPipeline,
    createBindGroup: vi.fn(() => ({} as GPUBindGroup)),
    createCommandEncoder: vi.fn(() => ({
      beginRenderPass,
      copyTextureToTexture: vi.fn(),
      finish: vi.fn(() => ({} as GPUCommandBuffer)),
    } as unknown as GPUCommandEncoder)),
  } as unknown as GPUDevice;
  return { device, createTexture, createRenderPipeline, beginRenderPass, renderPasses, submit };
}

function emissiveTriangleScene(handle: unknown): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'gpu-emissive-panel',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      material: {
        baseColor: [0.5, 0.5, 0.5],
        roughness: 0.5,
        metallic: 0,
        emissive: [2, 2, 2],
        emissiveIntensity: 1,
        emissiveMap: { handle },
      },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('PtWebgpuTextureSource', () => {
  it('pins an immutable host-owned 2D subresource with a distinct upload identity', () => {
    installGpuConstStubs();
    const device = {} as GPUDevice;
    const texture = makeTexture({
      width: 64,
      height: 32,
      depthOrArrayLayers: 3,
      mipLevelCount: 5,
    });
    const first = createPtWebgpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      baseMipLevel: 2,
      arrayLayer: 1,
    });
    const second = createPtWebgpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      baseMipLevel: 2,
      arrayLayer: 1,
    });

    expect(first).toMatchObject({
      ownership: 'host',
      dimension: '2d',
      width: 16,
      height: 8,
      baseMipLevel: 2,
      arrayLayer: 1,
      texture,
      device,
    });
    expect(Object.isFrozen(first)).toBe(true);
    expect(second.sourceId).not.toBe(first.sourceId);
    expect(isPtWebgpuTextureSource(first)).toBe(true);
    expect(isPtWebgpuTextureSource({ ...first })).toBe(true);
    expect(isPtWebgpuTextureSource({ kind: first.kind })).toBe(false);
  });

  it('rejects ambiguous format, transfer, dimension, sample, and subresource declarations', () => {
    installGpuConstStubs();
    const device = {} as GPUDevice;
    expect(() => createPtWebgpuTextureSource(device, makeTexture(), {
      format: 'rgba16float',
      colorSpace: 'linear',
    })).toThrow(/does not match texture\.format/);
    expect(() => createPtWebgpuTextureSource(device, makeTexture({ format: 'rgba8unorm-srgb' }), {
      format: 'rgba8unorm-srgb',
      colorSpace: 'linear',
    })).toThrow(/cannot be declared as linear/);
    expect(() => createPtWebgpuTextureSource(device, makeTexture({ dimension: '3d' }), {
      format: 'rgba8unorm',
      colorSpace: 'linear',
    })).toThrow(/only 2d textures/);
    expect(() => createPtWebgpuTextureSource(device, makeTexture({ sampleCount: 4 }), {
      format: 'rgba8unorm',
      colorSpace: 'linear',
    })).toThrow(/multisampled textures/);
    expect(() => createPtWebgpuTextureSource(device, makeTexture({ mipLevelCount: 2 }), {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      baseMipLevel: 2,
    })).toThrow(/outside 2 texture mip levels/);
    expect(() => createPtWebgpuTextureSource(device, makeTexture({ depthOrArrayLayers: 2 }), {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      arrayLayer: 2,
    })).toThrow(/outside 2 texture layers/);
  });

  it('accepts an exact immutable CPU mirror for emissive-map NEE', () => {
    installGpuConstStubs();
    const device = {} as GPUDevice;
    const input = new Float32Array([0.25, 0.5, 1, 1]);
    const source = createPtWebgpuTextureSource(
      device,
      makeTexture({ width: 1, height: 1, format: 'rgba16float' }),
      {
        format: 'rgba16float',
        colorSpace: 'linear',
        cpuMirror: {
          width: 1,
          height: 1,
          channels: 4,
          dataType: 'float32',
          colorSpace: 'linear',
          data: input,
        },
      },
    );
    input[0] = 0;

    expect(Object.isFrozen(source.cpuMirror)).toBe(true);
    expect(Array.from(source.cpuMirror!.data)).toEqual([0.25, 0.5, 1, 1]);
    expect(() => {
      (source.cpuMirror!.data as { [index: number]: number })[0] = 0;
    }).toThrow(/immutable/);

    const packed = packEmitterArrays(emissiveTriangleScene(source));
    expect(packed.meshAreaLightCount).toBe(1);
    // The packed record carries the unmodulated base Le. The shader evaluates
    // the exact mapped texel at the sampled barycentric point; no quadrature
    // average is baked into the proposal record.
    expect(Array.from(packed.meshAreaLightsData.slice(12, 15))).toEqual([2, 2, 2]);
  });

  it('keeps explicit mapped mesh emission identical between camera hits and NEE', () => {
    installGpuConstStubs();
    const device = {} as GPUDevice;
    const source = createPtWebgpuTextureSource(
      device,
      makeTexture({ width: 1, height: 1, format: 'rgba16float' }),
      {
        format: 'rgba16float',
        colorSpace: 'linear',
        cpuMirror: {
          width: 1,
          height: 1,
          channels: 4,
          dataType: 'float32',
          colorSpace: 'linear',
          data: new Float32Array([0.25, 0.5, 1, 1]),
        },
      },
    );
    const scene = {
      ...emissiveTriangleScene(source),
      emitters: [{
        kind: 'mesh-area' as const,
        id: 'explicit-panel-light',
        meshId: 'gpu-emissive-panel',
        color: [3, 2, 1] as const,
        intensity: 2,
      }],
    };

    const packed = buildPackedScene(scene, { cameraVisibleEmitters: true });
    const cameraHitBase = Array.from(packed.materials.slice(4, 7));
    const sourceFactor = Array.from(
      packed.meshAreaLightSourceFactorsData.slice(0, 3),
    );
    const neeRadiance = Array.from(
      packed.meshAreaLightsData.slice(12, 15),
    );

    expect(cameraHitBase).toEqual([6, 4, 2]);
    // Positive-support proposal proxy only. Exact map filtering happens in
    // sampleMeshAreaLightRadiance at the sampled point.
    expect(sourceFactor).toEqual([1, 1, 1]);
    expect(neeRadiance).toEqual(cameraHitBase);
  });

  it('rejects a GPU-only emissive source before creating any GPU resource', async () => {
    installGpuConstStubs();
    const createBuffer = vi.fn();
    const device = {
      createCommandEncoder: vi.fn(),
      createBuffer,
      limits: {
        maxStorageBuffersPerShaderStage: 64,
        maxStorageTexturesPerShaderStage: 8,
      },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      lost: new Promise<never>(() => {}),
    } as unknown as GPUDevice;
    const source = createPtWebgpuTextureSource(
      device,
      makeTexture({ width: 1, height: 1, format: 'rgba16float' }),
      { format: 'rgba16float', colorSpace: 'linear' },
    );
    const engine = await createPTEngine_WebGPU({ device });

    expect(() => engine.setScene(emissiveTriangleScene(source))).toThrow(
      /emissiveMap without complete CPU-readable texels/,
    );
    expect(createBuffer).not.toHaveBeenCalled();
    engine.dispose();
  });

  it('rejects malformed or mismatched CPU mirrors at descriptor construction', () => {
    installGpuConstStubs();
    const device = {} as GPUDevice;
    const texture = makeTexture({ width: 2, height: 1 });
    expect(() => createPtWebgpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      cpuMirror: {
        width: 1,
        height: 1,
        channels: 4,
        dataType: 'uint8',
        colorSpace: 'srgb',
        data: new Uint8Array(4),
      },
    })).toThrow(/dimensions must exactly match/);
    expect(() => createPtWebgpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      cpuMirror: {
        width: 2,
        height: 1,
        channels: 4,
        dataType: 'uint8',
        colorSpace: 'linear',
        data: new Uint8Array(8),
      },
    })).toThrow(/colorSpace must match/);
    expect(() => createPtWebgpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      cpuMirror: {
        width: 2,
        height: 1,
        channels: 4,
        dataType: 'uint8',
        colorSpace: 'srgb',
        data: new Uint8Array(7),
      },
    })).toThrow(/length must be exactly 8/);
  });

  it('rejects a CPU mirror snapshot above its explicit budget before reading data', () => {
    installGpuConstStubs();
    const device = {} as GPUDevice;
    const width = 8192;
    const height = 8192;
    const length = width * height * 4;
    const unreadLargeInput = { length } as ArrayLike<number>;
    expect(() => createPtWebgpuTextureSource(
      device,
      makeTexture({ width, height, format: 'rgba16float' }),
      {
        format: 'rgba16float',
        colorSpace: 'linear',
        cpuMirror: {
          width,
          height,
          channels: 4,
          dataType: 'float32',
          colorSpace: 'linear',
          data: unreadLargeInput,
        },
      },
    )).toThrow(/immutable snapshot requires .* per-source budget/);
  });

  it('rejects raw GPUTexture handles before allocating an engine-owned array', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeUploadDevice();
    expect(() => createMaterialTextureArray(device, [makeTexture()])).toThrow(
      /Wrap the texture with createPtWebgpuTextureSource/,
    );
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('preflights device, usage, format, and engine-resource ownership before allocation', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeUploadDevice();
    const foreignDevice = {} as GPUDevice;
    const foreign = createPtWebgpuTextureSource(foreignDevice, makeTexture(), {
      format: 'rgba8unorm',
      colorSpace: 'linear',
    });
    expect(() => createMaterialTextureArray(device, [foreign])).toThrow(/different GPUDevice/);

    const withoutBinding = makeTexture({ usage: GPUTextureUsage.COPY_SRC });
    const noBindingSource = createPtWebgpuTextureSource(device, withoutBinding, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
    });
    expect(() => createMaterialTextureArray(device, [noBindingSource])).toThrow(/lacks GPUTextureUsage\.TEXTURE_BINDING/);

    const depthTexture = makeTexture({ format: 'depth24plus' });
    const depthSource = createPtWebgpuTextureSource(device, depthTexture, {
      format: 'depth24plus',
      colorSpace: 'linear',
    });
    expect(() => createMaterialTextureArray(device, [depthSource])).toThrow(/not a float-sampleable color format/);

    const aliasedTexture = makeTexture();
    const aliasSource = createPtWebgpuTextureSource(device, aliasedTexture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
    });
    expect(() => createMaterialTextureArray(
      device,
      [aliasSource],
      'rgba8unorm-srgb',
      undefined,
      new Set([aliasedTexture]),
    )).toThrow(/aliases an engine-owned scene texture/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('converts a selected same-device mip/layer without readback and preserves host ownership', () => {
    installGpuConstStubs();
    const { device, createRenderPipeline, beginRenderPass, renderPasses, submit } = makeUploadDevice();
    const sourceTexture = makeTexture({
      width: 8,
      height: 4,
      depthOrArrayLayers: 3,
      mipLevelCount: 3,
      format: 'rgba8unorm',
    });
    const source = createPtWebgpuTextureSource(device, sourceTexture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      baseMipLevel: 1,
      arrayLayer: 2,
    });

    const result = createMaterialTextureArray(device, [source], 'rgba8unorm-srgb');

    expect(result.layerUvScales).toEqual([[1, 1]]);
    expect(result.mipLevelCount).toBe(3);
    expect(sourceTexture.createView).toHaveBeenCalledWith(expect.objectContaining({
      dimension: '2d',
      baseMipLevel: 1,
      mipLevelCount: 1,
      baseArrayLayer: 2,
      arrayLayerCount: 1,
    }));
    expect(device.queue.writeTexture).not.toHaveBeenCalled();
    expect(device.queue.copyExternalImageToTexture).not.toHaveBeenCalled();
    expect(sourceTexture.destroy).not.toHaveBeenCalled();
    expect(createRenderPipeline).toHaveBeenCalledWith(expect.objectContaining({
      fragment: expect.objectContaining({ constants: { decodeSrgb: 1 } }),
    }));
    expect(device.createShaderModule).toHaveBeenCalledWith(expect.objectContaining({
      code: MATERIAL_TEXTURE_GPU_SOURCE_BLIT_WGSL,
    }));
    expect(beginRenderPass).toHaveBeenCalledTimes(3);
    expect(renderPasses[0]?.setViewport).toHaveBeenCalledWith(0, 0, 4, 2, 0, 1);
    // Source conversion, native-size mip generation, then exact per-mip copies
    // into the shared array are three ordered command-buffer submissions.
    expect(submit).toHaveBeenCalledTimes(3);
  });
});
