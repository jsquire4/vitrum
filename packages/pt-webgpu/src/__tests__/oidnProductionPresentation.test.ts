import { describe, expect, it, vi, type Mock } from 'vitest';
import type { FrameInput, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { GpuResources } from '../gpuResources.js';
import { installGpuConstStubs } from './gpuStub.js';

interface StubTexture {
  readonly label: string;
  readonly descriptor: GPUTextureDescriptor;
  readonly createView: ReturnType<typeof vi.fn>;
  readonly destroy: ReturnType<typeof vi.fn>;
}

interface GpuRecorder {
  readonly textures: StubTexture[];
  readonly bindGroupLayouts: GPUBindGroupLayoutDescriptor[];
  readonly bindGroups: Array<{
    readonly label: string;
    readonly entries: readonly GPUBindGroupEntry[];
  }>;
  readonly oidnUploads: Array<{
    readonly texture: StubTexture;
    readonly rgba: Float32Array;
  }>;
  readonly createTexture: Mock<[GPUTextureDescriptor], StubTexture>;
  readonly createComputePipeline: Mock<
    [GPUComputePipelineDescriptor],
    GPUComputePipeline
  >;
  readonly writeTexture: Mock<
    [GPUImageCopyTexture, GPUAllowSharedBufferSource],
    void
  >;
  readonly submit: Mock<[Iterable<GPUCommandBuffer>], void>;
}

function makeProductionDevice(): {
  readonly device: GPUDevice;
  readonly record: GpuRecorder;
} {
  installGpuConstStubs();
  const textures: StubTexture[] = [];
  const bindGroupLayouts: GPUBindGroupLayoutDescriptor[] = [];
  const bindGroups: GpuRecorder['bindGroups'] = [];
  const oidnUploads: GpuRecorder['oidnUploads'] = [];
  const createTexture = vi.fn((descriptor: GPUTextureDescriptor) => {
    const texture = {
      label: String(descriptor.label ?? ''),
      descriptor,
      createView: vi.fn(),
      destroy: vi.fn(),
    } as StubTexture;
    texture.createView.mockImplementation(() => ({ texture }));
    textures.push(texture);
    return texture;
  });
  const writeTexture = vi.fn(
    (destination: GPUImageCopyTexture, data: GPUAllowSharedBufferSource) => {
      const texture = destination.texture as unknown as StubTexture;
      if (texture.label !== 'vitrum.pt-webgpu.oidn.linear') return;
      const source = data as Float32Array;
      const rgba = new Float32Array(source.length);
      rgba.set(source);
      oidnUploads.push({ texture, rgba });
    },
  );
  const submit = vi.fn<[Iterable<GPUCommandBuffer>], void>();
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
  const device = {
    queue: { writeBuffer: vi.fn(), writeTexture, submit },
    createBuffer: vi.fn((descriptor?: GPUBufferDescriptor) => ({
      label: String(descriptor?.label ?? ''),
      destroy: vi.fn(),
    })),
    createTexture,
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createBindGroupLayout: vi.fn((descriptor: GPUBindGroupLayoutDescriptor) => {
      bindGroupLayouts.push(descriptor);
      return {};
    }),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    createBindGroup: vi.fn((descriptor: GPUBindGroupDescriptor) => {
      bindGroups.push({
        label: String(descriptor.label ?? ''),
        entries: Array.from(descriptor.entries),
      });
      return {};
    }),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => pass),
      clearBuffer: vi.fn(),
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
      maxTextureDimension2D: 8192,
      maxTextureArrayLayers: 256,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
  return {
    device,
    record: {
      textures,
      bindGroupLayouts,
      bindGroups,
      oidnUploads,
      createTexture,
      createComputePipeline: device.createComputePipeline as unknown as GpuRecorder['createComputePipeline'],
      writeTexture,
      submit,
    },
  };
}

function scene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'triangle',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [0.8, 0.4, 0.2], roughness: 0.4, metallic: 0 },
    }],
    emitters: [{
      kind: 'directional',
      id: 'sun',
      direction: [0, -1, 0],
      color: [1, 1, 1],
      intensity: 1,
    }],
    environment: { kind: 'none' },
  };
}

function identity(): Float32Array {
  const matrix = new Float32Array(16);
  matrix[0] = 1;
  matrix[5] = 1;
  matrix[10] = 1;
  matrix[15] = 1;
  return matrix;
}

function frame(samplesTarget: number, exposure = 1): FrameInput {
  return {
    viewMatrix: asMat4(identity()),
    projMatrix: asMat4(identity()),
    viewport: { width: 2, height: 2, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 7,
    quality: {
      samplesTarget,
      bounces: 2,
      resolutionFactor: 1,
      tonemap: 'aces',
      exposure,
      outputColorSpace: 'srgb',
    },
  };
}

async function flushAsyncWork(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
  await new Promise<void>((resolve) => setImmediate(resolve));
}

describe('pt-webgpu OIDN production presentation', () => {
  it('uses an explicit unfilterable-float source layout for rgba16f and rgba32f presentation', async () => {
    const { device, record } = makeProductionDevice();
    const engine = await createPTEngine_WebGPU({
      device,
      traceTier: 'full',
      denoiser: 'none',
    });
    engine.setScene(scene());
    engine.renderFrame(frame(1));

    const layout = record.bindGroupLayouts.find(
      (descriptor) =>
        descriptor.label === 'vitrum.pt-webgpu.present.bindGroupLayout',
    );
    expect(layout).toBeDefined();
    expect(Array.from(layout!.entries)).toEqual([
      expect.objectContaining({
        binding: 0,
        buffer: { type: 'uniform' },
      }),
      expect.objectContaining({
        binding: 1,
        texture: {
          sampleType: 'unfilterable-float',
          viewDimension: '2d',
        },
      }),
      expect.objectContaining({
        binding: 2,
        storageTexture: {
          access: 'write-only',
          format: 'rgba16float',
          viewDimension: '2d',
        },
      }),
    ]);
    const presentPipeline = record.createComputePipeline.mock.calls
      .map(([descriptor]) => descriptor)
      .find((descriptor) =>
        descriptor.label === 'vitrum.pt-webgpu.present.pipeline',
      );
    expect(presentPipeline?.layout).not.toBe('auto');
    engine.dispose();
  });

  it('re-presents changed dials from the accumulator on converged and paused frames', async () => {
    const { device, record } = makeProductionDevice();
    const engine = await createPTEngine_WebGPU({
      device,
      traceTier: 'full',
      denoiser: 'none',
    });
    engine.setScene(scene());
    const first = engine.renderFrame(frame(1, 1));
    expect(first.kind).toBe('rendered');
    const accumulator = record.textures.find(
      (texture) => texture.label === 'vitrum.pt-webgpu.accum',
    );
    expect(accumulator).toBeDefined();
    const sourceForLastPresent = () => record.bindGroups
      .filter((group) => group.label === 'vitrum.pt-webgpu.present.bindgroup')
      .at(-1)?.entries.find((entry) => entry.binding === 1)
      ?.resource as { readonly texture?: StubTexture } | undefined;

    const submitsAfterFirst = record.submit.mock.calls.length;
    const converged = engine.renderFrame(frame(1, 2));
    expect(converged.kind).toBe('rendered');
    expect(converged.samplesAccumulated).toBe(1);
    expect(record.submit.mock.calls.length).toBe(submitsAfterFirst + 1);
    expect(sourceForLastPresent()?.texture).toBe(accumulator);

    engine.pause();
    const submitsBeforePausedChange = record.submit.mock.calls.length;
    const paused = engine.renderFrame(frame(1, 3));
    expect(paused.kind).toBe('rendered');
    expect(paused.samplesAccumulated).toBe(1);
    expect(record.submit.mock.calls.length).toBe(submitsBeforePausedChange + 1);
    expect(sourceForLastPresent()?.texture).toBe(accumulator);

    const submitsBeforeUnchangedPause = record.submit.mock.calls.length;
    engine.renderFrame(frame(1, 3));
    expect(record.submit.mock.calls.length).toBe(submitsBeforeUnchangedPause);
    engine.dispose();
  });

  it('queues CPU output, then uploads and presents it only on render cadence', async () => {
    const { device, record } = makeProductionDevice();
    let resolveFirst!: (rgb: Float32Array) => void;
    const firstInference = new Promise<Float32Array>((resolve) => {
      resolveFirst = resolve;
    });
    const denoised = new Float32Array([
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12,
    ]);
    const denoiseFinal = vi.fn()
      .mockImplementationOnce(async () => firstInference)
      .mockImplementationOnce(async () => denoised);
    const engine = await createPTEngine_WebGPU({
      device,
      traceTier: 'full',
      denoiser: 'oidn-final',
      oidn: { modelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx' },
      oidnBridgeLoader: async () => ({
        denoiseFinal,
        preloadOIDNModel: vi.fn(async () => undefined),
        releaseOIDNCacheEntry: vi.fn(),
      }),
      oidnReadbackFn: async (_device, _sources, width, height) => ({
        color: new Float32Array(width * height * 3).fill(0.25),
        albedo: new Float32Array(width * height * 3).fill(0.5),
        normal: new Float32Array(width * height * 3).fill(1),
        width,
        height,
      }),
    });
    engine.setScene(scene());
    expect(engine.renderFrame(frame(1)).kind).toBe('rendered');
    await flushAsyncWork();
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    const boundary = {
      textures: record.createTexture.mock.calls.length,
      uploads: record.oidnUploads.length,
      submits: record.submit.mock.calls.length,
    };
    resolveFirst(denoised);
    await firstInference;
    await flushAsyncWork();
    expect(record.createTexture.mock.calls.length).toBe(boundary.textures);
    expect(record.oidnUploads).toHaveLength(boundary.uploads);
    expect(record.submit.mock.calls.length).toBe(boundary.submits);
    expect(engine.getDenoisedFrame?.()?.rgb).toBe(denoised);

    const presented = engine.renderFrame(frame(1));
    expect(record.oidnUploads).toHaveLength(1);
    const upload = record.oidnUploads[0]!;
    expect(upload.texture.descriptor.format).toBe('rgba32float');
    expect(Array.from(upload.rgba)).toEqual([
      1, 2, 3, 1, 4, 5, 6, 1,
      7, 8, 9, 1, 10, 11, 12, 1,
    ]);
    const presentTexture = record.textures.find(
      (texture) => texture.label === 'vitrum.pt-webgpu.present',
    );
    if (presented.kind === 'rendered') {
      expect(presented.primaryRadiance).toBe(presentTexture);
    }
    const sourceForLastPresent = () => record.bindGroups
      .filter((group) => group.label === 'vitrum.pt-webgpu.present.bindgroup')
      .at(-1)?.entries.find((entry) => entry.binding === 1)
      ?.resource as { readonly texture?: StubTexture } | undefined;
    expect(sourceForLastPresent()?.texture).toBe(upload.texture);

    const uploadsBeforeExposure = record.oidnUploads.length;
    const submitsBeforeExposure = record.submit.mock.calls.length;
    engine.renderFrame(frame(1, 2));
    expect(record.oidnUploads).toHaveLength(uploadsBeforeExposure);
    expect(denoiseFinal).toHaveBeenCalledTimes(1);
    expect(record.submit.mock.calls.length).toBe(submitsBeforeExposure + 1);
    expect(sourceForLastPresent()?.texture).toBe(upload.texture);

    engine.renderFrame(frame(2, 2));
    expect(upload.texture.destroy).toHaveBeenCalledTimes(1);
    await flushAsyncWork();
    expect(denoiseFinal).toHaveBeenCalledTimes(2);
    engine.dispose();
  });

  it('rejects an OIDN candidate aliased to any owned live texture before upload', () => {
    const { device, record } = makeProductionDevice();
    const gpu = new GpuResources(device, 'full', false);
    gpu.ensureAccumResources(2, 2);
    const liveNormal = gpu.normalDepthTexture as unknown as StubTexture;
    const writesBefore = record.writeTexture.mock.calls.length;
    record.createTexture.mockImplementationOnce(() => liveNormal);

    expect(() => gpu.presentDenoisedResult(
      { rgb: new Float32Array(12), width: 2, height: 2 },
      0,
      1,
      0,
    )).toThrow(/aliased a live GPU texture/);
    expect(record.writeTexture.mock.calls.length).toBe(writesBefore);
    expect(liveNormal.destroy).not.toHaveBeenCalled();
    expect(gpu.hasDenoisedResult()).toBe(false);
    gpu.dispose();
  });

  it('destroys a fresh OIDN candidate when upload fails before publication', () => {
    const { device, record } = makeProductionDevice();
    const gpu = new GpuResources(device, 'full', false);
    gpu.ensureAccumResources(2, 2);
    const texturesBefore = record.textures.length;
    record.writeTexture.mockImplementationOnce(() => {
      throw new Error('synthetic OIDN upload failure');
    });

    expect(() => gpu.presentDenoisedResult(
      { rgb: new Float32Array(12), width: 2, height: 2 },
      0,
      1,
      0,
    )).toThrow(/synthetic OIDN upload failure/);
    const candidate = record.textures[texturesBefore]!;
    expect(candidate.label).toBe('vitrum.pt-webgpu.oidn.linear');
    expect(candidate.destroy).toHaveBeenCalledTimes(1);
    expect(gpu.hasDenoisedResult()).toBe(false);
    gpu.dispose();
  });
});
