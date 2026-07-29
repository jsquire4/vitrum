import { describe, expect, it, vi } from 'vitest';
import { GpuResources } from '../gpuResources.js';

interface PipelineDescriptor {
  readonly label?: string;
  readonly layout: GPUPipelineLayout | 'auto';
  readonly compute: {
    readonly module: GPUShaderModule;
    readonly entryPoint: string;
  };
}

interface TrackedBuffer {
  readonly label: string | undefined;
  readonly size: number;
  readonly destroy: ReturnType<typeof vi.fn>;
}

interface FailureHooks {
  readonly shaderModule?: (descriptor: GPUShaderModuleDescriptor) => boolean;
  readonly pipeline?: (descriptor: PipelineDescriptor) => boolean;
  readonly sampler?: (descriptor: GPUSamplerDescriptor) => boolean;
  readonly buffer?: (descriptor: GPUBufferDescriptor) => boolean;
  readonly bindGroup?: (descriptor: GPUBindGroupDescriptor) => boolean;
  readonly writeBuffer?: () => boolean;
  readonly commandEncoder?: (descriptor: GPUCommandEncoderDescriptor) => boolean;
  readonly beginComputePass?: (descriptor: GPUComputePassDescriptor) => boolean;
  readonly finish?: () => boolean;
  readonly submit?: () => boolean;
}

function installWebGpuConstants(): void {
  const globals = globalThis as unknown as {
    GPUBufferUsage?: Record<string, number>;
    GPUShaderStage?: Record<string, number>;
  };
  globals.GPUBufferUsage ??= {
    UNIFORM: 1 << 0,
    COPY_DST: 1 << 1,
    STORAGE: 1 << 2,
    COPY_SRC: 1 << 3,
  };
  globals.GPUShaderStage ??= { COMPUTE: 1 << 0 };
}

function makeDevice(
  failure: ((descriptor: PipelineDescriptor) => boolean) | FailureHooks = {},
) {
  const hooks: FailureHooks = typeof failure === 'function'
    ? { pipeline: failure }
    : failure;
  const buffers: TrackedBuffer[] = [];
  let failureAvailable = true;
  const maybeFail = (matches: boolean, message: string): void => {
    if (failureAvailable && matches) {
      failureAvailable = false;
      throw new Error(`transient GPU failure: ${message}`);
    }
  };
  const createComputePipeline = vi.fn((descriptor: PipelineDescriptor) => {
    maybeFail(
      hooks.pipeline?.(descriptor) === true,
      descriptor.label ?? descriptor.compute.entryPoint,
    );
    return {
      label: descriptor.label,
      getBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
    } as unknown as GPUComputePipeline;
  });
  const createBindGroup = vi.fn((descriptor: GPUBindGroupDescriptor) => {
    maybeFail(hooks.bindGroup?.(descriptor) === true, descriptor.label ?? 'bind group');
    return { label: descriptor.label } as unknown as GPUBindGroup;
  });
  const submit = vi.fn(() => {
    maybeFail(hooks.submit?.() === true, 'queue.submit');
  });
  const writeBuffer = vi.fn(() => {
    maybeFail(hooks.writeBuffer?.() === true, 'queue.writeBuffer');
  });
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      maybeFail(hooks.buffer?.(descriptor) === true, descriptor.label ?? 'buffer');
      const buffer = { label: descriptor.label, size: descriptor.size, destroy: vi.fn() };
      buffers.push(buffer);
      return buffer as unknown as GPUBuffer;
    }),
    createShaderModule: vi.fn((descriptor: GPUShaderModuleDescriptor) => {
      maybeFail(hooks.shaderModule?.(descriptor) === true, descriptor.label ?? 'shader module');
      return {} as GPUShaderModule;
    }),
    createSampler: vi.fn((descriptor: GPUSamplerDescriptor) => {
      maybeFail(hooks.sampler?.(descriptor) === true, descriptor.label ?? 'sampler');
      return { label: descriptor.label } as unknown as GPUSampler;
    }),
    createBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
    createPipelineLayout: vi.fn(() => ({} as GPUPipelineLayout)),
    createComputePipeline,
    createBindGroup,
    createCommandEncoder: vi.fn((descriptor: GPUCommandEncoderDescriptor = {}) => {
      maybeFail(hooks.commandEncoder?.(descriptor) === true, descriptor.label ?? 'command encoder');
      return {
        clearBuffer: vi.fn(),
        beginComputePass: vi.fn((passDescriptor: GPUComputePassDescriptor = {}) => {
          maybeFail(
            hooks.beginComputePass?.(passDescriptor) === true,
            passDescriptor.label ?? 'compute pass',
          );
          return {
            setPipeline: vi.fn(),
            setBindGroup: vi.fn(),
            dispatchWorkgroups: vi.fn(),
            end: vi.fn(),
          } as unknown as GPUComputePassEncoder;
        }),
        finish: vi.fn(() => {
          maybeFail(hooks.finish?.() === true, 'commandEncoder.finish');
          return {} as GPUCommandBuffer;
        }),
      } as unknown as GPUCommandEncoder;
    }),
    queue: { writeBuffer, submit },
    limits: {
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
    },
  } as unknown as GPUDevice;
  return { device, buffers, createComputePipeline, createBindGroup, writeBuffer, submit };
}

function sceneBuffers(resource: object): import('../scene/uploadSceneBuffers.js').UploadedSceneBuffers {
  return new Proxy({}, {
    get: () => resource,
  }) as import('../scene/uploadSceneBuffers.js').UploadedSceneBuffers;
}

function prepareFullBindGroupInputs(
  gpu: GpuResources,
  resource: GPUBuffer,
): void {
  const view = {} as GPUTextureView;
  gpu.accumView = view;
  gpu.normalDepthView = view;
  gpu.albedoView = view;
  gpu.varianceView = view;
  gpu.motionVectorsView = view;
  gpu.accumBuffer = resource;
  gpu.varianceMomentsBuffer = resource;
  gpu.sppm.sppmPhotonCellsBuffer = resource;
  gpu.sppm.sppmCellCountersBuffer = resource;
  gpu.sppm.sppmStatsBuffer = resource;
  gpu.sppm.sppmPixelStatsBuffer = resource;
}

describe('pt-webgpu pipeline initialization transactions', () => {
  installWebGpuConstants();

  it('publishes the main pipeline cohort atomically and retries cleanly', () => {
    const stub = makeDevice(
      (descriptor) => descriptor.compute.entryPoint === 'main',
    );
    const gpu = new GpuResources(stub.device, 'full', true);

    expect(() => gpu.ensurePipeline()).toThrow(/transient GPU failure/);
    expect(gpu.paramsBuffer).toBeNull();
    expect(gpu.computePipeline).toBeNull();
    expect(gpu.bindGroupLayout).toBeNull();
    expect(gpu.bindGroupLayout1).toBeNull();
    expect(gpu.bindGroupLayout2).toBeNull();
    expect(gpu.bindGroupLayout3).toBeNull();
    expect(stub.buffers).toHaveLength(1);
    expect(stub.buffers[0]!.destroy).toHaveBeenCalledTimes(1);

    gpu.ensurePipeline();
    expect(gpu.paramsBuffer).toBe(stub.buffers[1]);
    expect(gpu.computePipeline).not.toBeNull();
    expect(gpu.bindGroupLayout).not.toBeNull();
    expect(gpu.bindGroupLayout1).not.toBeNull();
    expect(gpu.bindGroupLayout2).not.toBeNull();
    expect(gpu.bindGroupLayout3).not.toBeNull();
    expect(stub.buffers[1]!.destroy).not.toHaveBeenCalled();

    const callsAfterRecovery = stub.createComputePipeline.mock.calls.length;
    gpu.ensurePipeline();
    expect(stub.createComputePipeline).toHaveBeenCalledTimes(callsAfterRecovery);
  });

  it('publishes the ReSTIR-PT pipeline cohort atomically and retries a later-pass failure', () => {
    const stub = makeDevice(
      (descriptor) => descriptor.label === 'vitrum.pt-webgpu.restirPt.temporal',
    );
    const gpu = new GpuResources(stub.device, 'full', false, true);
    gpu.ensurePipeline();

    expect(() => gpu.ensureReservoirPipelines()).toThrow(/transient GPU failure/);
    expect(gpu.reservoir.rptGroup0Layout).toBeNull();
    expect(gpu.reservoir.rptProducerPipeline).toBeNull();
    expect(gpu.reservoir.rptTemporalPipeline).toBeNull();
    expect(gpu.reservoir.rptSpatialPipeline).toBeNull();
    expect(gpu.reservoir.rptResolvePipeline).toBeNull();
    expect(gpu.reservoir.rptCompositePipeline).toBeNull();

    gpu.ensureReservoirPipelines();
    expect(gpu.reservoir.rptGroup0Layout).not.toBeNull();
    expect(gpu.reservoir.rptProducerPipeline).not.toBeNull();
    expect(gpu.reservoir.rptTemporalPipeline).not.toBeNull();
    expect(gpu.reservoir.rptSpatialPipeline).not.toBeNull();
    expect(gpu.reservoir.rptResolvePipeline).not.toBeNull();
    expect(gpu.reservoir.rptCompositePipeline).not.toBeNull();

    const callsAfterRecovery = stub.createComputePipeline.mock.calls.length;
    gpu.ensureReservoirPipelines();
    expect(stub.createComputePipeline).toHaveBeenCalledTimes(callsAfterRecovery);
  });

  it('does not publish ReSTIR buffers until their clear submission succeeds', () => {
    const stub = makeDevice({ submit: () => true });
    const gpu = new GpuResources(stub.device, 'full', false, true);

    expect(() => gpu.ensureReservoirBuffers(2, 2)).toThrow(/queue\.submit/);
    expect(gpu.reservoir.rptReservoirCur).toBeNull();
    expect(gpu.reservoir.rptReservoirPrev).toBeNull();
    expect(gpu.reservoir.rptResultBuffer).toBeNull();
    expect(gpu.reservoir.rptParamsBuffer).toBeNull();
    expect(gpu.reservoir.rptReservoirByteSize).toBe(0);
    expect(gpu.reservoir.rptResultByteSize).toBe(0);
    expect(stub.buffers).toHaveLength(4);
    for (const buffer of stub.buffers) {
      expect(buffer.destroy).toHaveBeenCalledTimes(1);
    }

    expect(gpu.ensureReservoirBuffers(2, 2)).toBe(true);
    expect(gpu.reservoir.rptReservoirCur).toBe(stub.buffers[4]);
    expect(gpu.reservoir.rptParamsBuffer).toBe(stub.buffers[7]);
    const buffersAfterRecovery = stub.buffers.length;
    expect(gpu.ensureReservoirBuffers(2, 2)).toBe(true);
    expect(stub.buffers).toHaveLength(buffersAfterRecovery);
  });

  it('rejects malformed ReSTIR dimensions before allocating any buffer', () => {
    const stub = makeDevice();
    const gpu = new GpuResources(stub.device, 'full', false, true);
    for (const [width, height] of [
      [0, 1],
      [1, 0],
      [1.5, 1],
      [1, Number.NaN],
      [0x1_0000_0000, 1],
      [1, 0x1_0000_0000],
    ] as const) {
      expect(() => gpu.ensureReservoirBuffers(width, height)).toThrow(/positive u32 integers/);
    }
    expect(stub.buffers).toHaveLength(0);
  });

  it('rejects absent or unrepresentable ReSTIR parameter state before queue writes', () => {
    const stub = makeDevice();
    const gpu = new GpuResources(stub.device, 'full', false, true);
    expect(() => gpu.writeReservoirParams(1, 1, 1)).toThrow(/params buffer is absent/);
    expect(stub.writeBuffer).not.toHaveBeenCalled();

    gpu.ensureReservoirBuffers(1, 1);
    for (const [width, height, mClamp] of [
      [0, 1, 1],
      [1.5, 1, 1],
      [1, 1, 0],
      [1, 1, 4096],
    ] as const) {
      expect(() => gpu.writeReservoirParams(width, height, mClamp)).toThrow();
    }
    expect(stub.writeBuffer).not.toHaveBeenCalled();
    gpu.writeReservoirParams(1, 1, 4095);
    expect(stub.writeBuffer).toHaveBeenCalledTimes(1);
  });

  it('preserves the prior ReSTIR cohort when a resized candidate fails midway', () => {
    let failResize = false;
    const stub = makeDevice({
      buffer: (descriptor) =>
        failResize &&
        descriptor.label === 'vitrum.pt-webgpu.restirPt.result',
    });
    const gpu = new GpuResources(stub.device, 'full', false, true);
    expect(gpu.ensureReservoirBuffers(1, 1)).toBe(true);
    const previous = [
      gpu.reservoir.rptReservoirCur,
      gpu.reservoir.rptReservoirPrev,
      gpu.reservoir.rptResultBuffer,
      gpu.reservoir.rptParamsBuffer,
    ] as const;

    failResize = true;
    expect(() => gpu.ensureReservoirBuffers(2, 1)).toThrow(/restirPt\.result/);
    expect([
      gpu.reservoir.rptReservoirCur,
      gpu.reservoir.rptReservoirPrev,
      gpu.reservoir.rptResultBuffer,
      gpu.reservoir.rptParamsBuffer,
    ]).toEqual(previous);
    expect(gpu.reservoir.rptReservoirByteSize).toBe(GpuResources.RESERVOIR_PT_HERO_BYTES);
    for (const buffer of stub.buffers.slice(0, 4)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
    for (const buffer of stub.buffers.slice(4)) {
      expect(buffer.destroy).toHaveBeenCalledTimes(1);
    }

    expect(gpu.ensureReservoirBuffers(2, 1)).toBe(true);
    for (const buffer of stub.buffers.slice(0, 5)) {
      expect(buffer.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('publishes all full-tier path-trace bind groups together and retries group-2 failure', () => {
    const stub = makeDevice({
      bindGroup: (descriptor) =>
        descriptor.label === 'vitrum.pt-webgpu.pathTrace.bindgroup2.full',
    });
    const gpu = new GpuResources(stub.device, 'full', false);
    gpu.ensurePipeline();
    const resource = stub.device.createBuffer({
      label: 'test.shared',
      size: 64,
      usage: GPUBufferUsage.STORAGE,
    });
    prepareFullBindGroupInputs(gpu, resource);
    const sb = sceneBuffers(resource);

    expect(() => gpu.buildBindGroups(sb)).toThrow(/bindgroup2/);
    expect(gpu.pathTraceBindGroup).toBeNull();
    expect(gpu.pathTraceBindGroup1).toBeNull();
    expect(gpu.pathTraceBindGroup2).toBeNull();
    expect(gpu.pathTraceBindGroup3).toBeNull();

    const group0 = gpu.buildBindGroups(sb);
    expect(gpu.pathTraceBindGroup).toBe(group0);
    expect(gpu.pathTraceBindGroup1).not.toBeNull();
    expect(gpu.pathTraceBindGroup2).not.toBeNull();
    expect(gpu.pathTraceBindGroup3).not.toBeNull();
    const callsAfterRecovery = stub.createBindGroup.mock.calls.length;
    expect(gpu.buildBindGroups(sb)).toBe(group0);
    expect(stub.createBindGroup).toHaveBeenCalledTimes(callsAfterRecovery);
  });

  it('retries the atomic shared ReSTIR bind group after creation failure', () => {
    const stub = makeDevice({
      bindGroup: (descriptor) =>
        descriptor.label === 'vitrum.pt-webgpu.restirPt.bindgroup0.temporal',
    });
    const gpu = new GpuResources(stub.device, 'full', false, true);
    gpu.ensurePipeline();
    gpu.ensureReservoirPipelines();
    gpu.ensureReservoirBuffers(1, 1);
    const resource = stub.device.createBuffer({
      label: 'test.shared',
      size: 64,
      usage: GPUBufferUsage.STORAGE,
    });
    prepareFullBindGroupInputs(gpu, resource);
    const sb = sceneBuffers(resource);

    expect(() => gpu.buildReservoirBindGroups(sb)).toThrow(/restirPt\.bindgroup0/);
    expect(gpu.reservoir.rptProducerGroup0).toBeNull();
    expect(gpu.reservoir.rptTemporalGroup0).toBeNull();
    expect(gpu.reservoir.rptSpatialGroup0).toBeNull();
    expect(gpu.reservoir.rptResolveGroup0).toBeNull();

    gpu.buildReservoirBindGroups(sb);
    const groups = [
      gpu.reservoir.rptProducerGroup0,
      gpu.reservoir.rptTemporalGroup0,
      gpu.reservoir.rptSpatialGroup0,
      gpu.reservoir.rptResolveGroup0,
    ];
    expect(groups.every((group) => group != null)).toBe(true);
    expect(new Set(groups).size).toBe(4);
    const callsAfterRecovery = stub.createBindGroup.mock.calls.length;
    gpu.buildReservoirBindGroups(sb);
    expect(stub.createBindGroup).toHaveBeenCalledTimes(callsAfterRecovery);
  });

  it('publishes all SPPM placeholders together and retries a middle allocation failure', () => {
    const stub = makeDevice({
      buffer: (descriptor) =>
        descriptor.label === 'vitrum.pt-webgpu.sppm.cellCounters.placeholder',
    });
    const gpu = new GpuResources(stub.device, 'full', false);

    expect(() => gpu.ensureSppmBuffers(false)).toThrow(/cellCounters\.placeholder/);
    expect(gpu.sppm.sppmPhotonCellsBuffer).toBeNull();
    expect(gpu.sppm.sppmCellCountersBuffer).toBeNull();
    expect(gpu.sppm.sppmStatsBuffer).toBeNull();
    expect(gpu.sppm.sppmPixelStatsBuffer).toBeNull();
    expect(stub.buffers).toHaveLength(1);
    expect(stub.buffers[0]!.destroy).toHaveBeenCalledTimes(1);

    expect(gpu.ensureSppmBuffers(false)).toBe(false);
    expect(gpu.sppm.sppmPhotonCellsBuffer).not.toBeNull();
    expect(gpu.sppm.sppmCellCountersBuffer).not.toBeNull();
    expect(gpu.sppm.sppmStatsBuffer).not.toBeNull();
    expect(gpu.sppm.sppmPixelStatsBuffer).not.toBeNull();
    const buffersAfterRecovery = stub.buffers.length;
    expect(gpu.ensureSppmBuffers(false)).toBe(false);
    expect(stub.buffers).toHaveLength(buffersAfterRecovery);
  });

  it('keeps SPPM placeholders live until the full buffers and photon pipeline succeed', () => {
    const stub = makeDevice({
      pipeline: (descriptor) =>
        descriptor.label === 'vitrum.pt-webgpu.sppm.photonPass.pipeline',
    });
    const gpu = new GpuResources(stub.device, 'full', false);
    gpu.ensurePipeline();
    gpu.ensureSppmBuffers(false);
    const placeholders = [
      gpu.sppm.sppmPhotonCellsBuffer,
      gpu.sppm.sppmCellCountersBuffer,
      gpu.sppm.sppmStatsBuffer,
    ] as const;

    expect(() => gpu.ensureSppmBuffers(true)).toThrow(/photonPass\.pipeline/);
    expect([
      gpu.sppm.sppmPhotonCellsBuffer,
      gpu.sppm.sppmCellCountersBuffer,
      gpu.sppm.sppmStatsBuffer,
    ]).toEqual(placeholders);
    expect(gpu.sppm.sppmPhotonPipeline).toBeNull();
    expect(gpu.sppm.sppmBuffersReady).toBe(false);
    for (const placeholder of stub.buffers.filter((buffer) => buffer.label?.endsWith('.placeholder'))) {
      expect(placeholder.destroy).not.toHaveBeenCalled();
    }
    for (const candidate of stub.buffers.filter((buffer) =>
      buffer.label?.startsWith('vitrum.pt-webgpu.sppm.') &&
      !buffer.label.endsWith('.placeholder'),
    )) {
      expect(candidate.destroy).toHaveBeenCalledTimes(1);
    }

    expect(gpu.ensureSppmBuffers(true)).toBe(true);
    expect(gpu.sppm.sppmBuffersReady).toBe(true);
    expect(gpu.sppm.sppmPhotonPipeline).not.toBeNull();
    for (const placeholder of stub.buffers.filter((buffer) =>
      buffer.label !== 'vitrum.pt-webgpu.sppm.pixelStats.placeholder' &&
      buffer.label?.endsWith('.placeholder'),
    )) {
      expect(placeholder.destroy).toHaveBeenCalledTimes(1);
    }
    const callsAfterRecovery = stub.createComputePipeline.mock.calls.length;
    expect(gpu.ensureSppmBuffers(true)).toBe(true);
    expect(stub.createComputePipeline).toHaveBeenCalledTimes(callsAfterRecovery);
  });

  it('publishes resized SPPM pixel statistics only after the GPU clear submits', () => {
    const stub = makeDevice({ submit: () => true });
    const gpu = new GpuResources(stub.device, 'full', false);
    gpu.ensureSppmBuffers(false);
    const placeholder = gpu.sppm.sppmPixelStatsBuffer;
    const placeholderRecord = stub.buffers.find(
      (buffer) => buffer.label === 'vitrum.pt-webgpu.sppm.pixelStats.placeholder',
    )!;

    expect(() => gpu.ensureSppmPixelStatsBuffer(2, 2)).toThrow(/queue\.submit/);
    expect(gpu.sppm.sppmPixelStatsBuffer).toBe(placeholder);
    expect(gpu.sppm.sppmPixelStatsWidth).toBe(0);
    expect(gpu.sppm.sppmPixelStatsHeight).toBe(0);
    expect(placeholderRecord.destroy).not.toHaveBeenCalled();
    const failedCandidate = stub.buffers.find(
      (buffer) => buffer.label === 'vitrum.pt-webgpu.sppm.pixelStats',
    )!;
    expect(failedCandidate.destroy).toHaveBeenCalledTimes(1);

    expect(gpu.ensureSppmPixelStatsBuffer(2, 2)).toBe(true);
    expect(gpu.sppm.sppmPixelStatsBuffer).not.toBe(placeholder);
    expect(gpu.sppm.sppmPixelStatsWidth).toBe(2);
    expect(gpu.sppm.sppmPixelStatsHeight).toBe(2);
    expect(placeholderRecord.destroy).toHaveBeenCalledTimes(1);
  });

  it('preserves a real SPPM pixel-statistics buffer when a later request exceeds device limits', () => {
    const stub = makeDevice();
    const gpu = new GpuResources(stub.device, 'full', false);
    gpu.ensureSppmBuffers(false);
    expect(gpu.ensureSppmPixelStatsBuffer(2, 2)).toBe(true);
    const previous = gpu.sppm.sppmPixelStatsBuffer;
    const previousRecord = stub.buffers.find(
      (buffer) => buffer.label === 'vitrum.pt-webgpu.sppm.pixelStats',
    )!;

    const limits = stub.device.limits as unknown as {
      maxBufferSize: number;
      maxStorageBufferBindingSize: number;
    };
    limits.maxBufferSize = 64;
    limits.maxStorageBufferBindingSize = 64;
    expect(gpu.ensureSppmPixelStatsBuffer(4, 4)).toBe(false);
    expect(gpu.sppm.sppmPixelStatsBuffer).toBe(previous);
    expect(gpu.sppm.sppmPixelStatsWidth).toBe(2);
    expect(gpu.sppm.sppmPixelStatsHeight).toBe(2);
    expect(previousRecord.destroy).not.toHaveBeenCalled();

    limits.maxBufferSize = 256 * 1024 * 1024;
    limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
    expect(gpu.ensureSppmPixelStatsBuffer(4, 4)).toBe(true);
    expect(gpu.sppm.sppmPixelStatsBuffer).not.toBe(previous);
    expect(previousRecord.destroy).toHaveBeenCalledTimes(1);
  });

  const seedCohortFailureCases: readonly (readonly [string, FailureHooks])[] = [
    ['shader-module', {
      shaderModule: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit',
    }],
    ['pipeline', {
      pipeline: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit.pipeline',
    }],
    ['sampler', {
      sampler: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit.sampler',
    }],
    ['params-buffer', {
      buffer: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit.params',
    }],
    ['variance-placeholder', {
      buffer: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit.varPlaceholder',
    }],
    ['uniform-write', { writeBuffer: () => true }],
    ['bind-group', {
      bindGroup: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit.bindgroup',
    }],
    ['command-encoder', {
      commandEncoder: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit.encoder',
    }],
    ['compute-pass', {
      beginComputePass: (descriptor) => descriptor.label === 'vitrum.pt-webgpu.seedBlit.pass',
    }],
    ['finish', { finish: () => true }],
    ['submit', { submit: () => true }],
  ];

  it.each(seedCohortFailureCases)(
    'keeps seed-blit construction transactional at failure index %s and retries',
    (_name, hooks) => {
      const stub = makeDevice(hooks);
      const gpu = new GpuResources(stub.device, 'lite', false);
      const accum = stub.device.createBuffer({
        label: 'test.seed.accum',
        size: 64,
        usage: GPUBufferUsage.STORAGE,
      });
      gpu.accumBuffer = accum;
      gpu.accumBufferByteSize = 64;
      const seedTex = { createView: vi.fn(() => ({} as GPUTextureView)) } as unknown as GPUTexture;

      expect(() => gpu.seedAccumBuffer(seedTex, 2, 2, 2)).toThrow(/transient GPU failure/);
      expect(gpu.present.seedBlitPipeline).toBeNull();
      expect(gpu.present.seedBlitSampler).toBeNull();
      expect(gpu.present.seedBlitParamsBuffer).toBeNull();
      expect(gpu.present.seedBlitVarPlaceholder).toBeNull();
      expect(gpu.present.seedBlitVarPlaceholderByteSize).toBe(0);
      for (const candidate of stub.buffers.filter((buffer) =>
        buffer.label?.startsWith('vitrum.pt-webgpu.seedBlit.'),
      )) {
        expect(candidate.destroy).toHaveBeenCalledTimes(1);
      }

      gpu.seedAccumBuffer(seedTex, 2, 2, 2);
      expect(gpu.present.seedBlitPipeline).not.toBeNull();
      expect(gpu.present.seedBlitSampler).not.toBeNull();
      expect(gpu.present.seedBlitParamsBuffer).not.toBeNull();
      expect(gpu.present.seedBlitVarPlaceholder).not.toBeNull();
      expect(gpu.present.seedBlitVarPlaceholderByteSize).toBe(64);
    },
  );

  it('preserves a complete seed-blit cohort when a later submission fails', () => {
    let failSubmit = false;
    const stub = makeDevice({ submit: () => failSubmit });
    const gpu = new GpuResources(stub.device, 'lite', false);
    gpu.accumBuffer = stub.device.createBuffer({
      label: 'test.seed.accum',
      size: 64,
      usage: GPUBufferUsage.STORAGE,
    });
    gpu.accumBufferByteSize = 64;
    const seedTex = { createView: vi.fn(() => ({} as GPUTextureView)) } as unknown as GPUTexture;
    gpu.seedAccumBuffer(seedTex, 1, 2, 2);
    const previous = {
      pipeline: gpu.present.seedBlitPipeline,
      sampler: gpu.present.seedBlitSampler,
      params: gpu.present.seedBlitParamsBuffer,
      placeholder: gpu.present.seedBlitVarPlaceholder,
    };
    const priorBuffers = stub.buffers.slice();

    failSubmit = true;
    expect(() => gpu.seedAccumBuffer(seedTex, 1, 2, 2)).toThrow(/queue\.submit/);
    expect(gpu.present.seedBlitPipeline).toBe(previous.pipeline);
    expect(gpu.present.seedBlitSampler).toBe(previous.sampler);
    expect(gpu.present.seedBlitParamsBuffer).toBe(previous.params);
    expect(gpu.present.seedBlitVarPlaceholder).toBe(previous.placeholder);
    for (const buffer of priorBuffers) expect(buffer.destroy).not.toHaveBeenCalled();

    gpu.seedAccumBuffer(seedTex, 1, 2, 2);
    expect(gpu.present.seedBlitParamsBuffer).toBe(previous.params);
    expect(gpu.present.seedBlitVarPlaceholder).toBe(previous.placeholder);
  });

  it('replaces a resized seed-blit cohort only after the new dispatch submits', () => {
    let failBindGroup = false;
    const stub = makeDevice({
      bindGroup: (descriptor) =>
        failBindGroup && descriptor.label === 'vitrum.pt-webgpu.seedBlit.bindgroup',
    });
    const gpu = new GpuResources(stub.device, 'lite', false);
    gpu.accumBuffer = stub.device.createBuffer({
      label: 'test.seed.accum',
      size: 128,
      usage: GPUBufferUsage.STORAGE,
    });
    gpu.accumBufferByteSize = 64;
    const seedTex = { createView: vi.fn(() => ({} as GPUTextureView)) } as unknown as GPUTexture;
    gpu.seedAccumBuffer(seedTex, 1, 2, 2);
    const previousParams = gpu.present.seedBlitParamsBuffer;
    const previousPlaceholder = gpu.present.seedBlitVarPlaceholder;
    const previousRecords = stub.buffers.filter((buffer) =>
      buffer.label?.startsWith('vitrum.pt-webgpu.seedBlit.'),
    );

    gpu.accumBufferByteSize = 128;
    failBindGroup = true;
    expect(() => gpu.seedAccumBuffer(seedTex, 1, 4, 2)).toThrow(/seedBlit\.bindgroup/);
    expect(gpu.present.seedBlitParamsBuffer).toBe(previousParams);
    expect(gpu.present.seedBlitVarPlaceholder).toBe(previousPlaceholder);
    expect(gpu.present.seedBlitVarPlaceholderByteSize).toBe(64);
    for (const buffer of previousRecords) expect(buffer.destroy).not.toHaveBeenCalled();
    for (const candidate of stub.buffers.filter((buffer) =>
      buffer.label?.startsWith('vitrum.pt-webgpu.seedBlit.') &&
      !previousRecords.includes(buffer),
    )) {
      expect(candidate.destroy).toHaveBeenCalledTimes(1);
    }

    gpu.seedAccumBuffer(seedTex, 1, 4, 2);
    expect(gpu.present.seedBlitParamsBuffer).not.toBe(previousParams);
    expect(gpu.present.seedBlitVarPlaceholder).not.toBe(previousPlaceholder);
    expect(gpu.present.seedBlitVarPlaceholderByteSize).toBe(128);
    for (const buffer of previousRecords) expect(buffer.destroy).toHaveBeenCalledTimes(1);
  });

  it('publishes the present pipeline and params UBO together after buffer creation succeeds', () => {
    const stub = makeDevice({
      buffer: (descriptor) =>
        descriptor.label === 'vitrum.pt-webgpu.present.params',
    });
    const gpu = new GpuResources(stub.device, 'full', false);

    expect(() => gpu.ensurePresentPipeline()).toThrow(/present\.params/);
    expect(gpu.present.presentPipeline).toBeNull();
    expect(gpu.present.presentParamsBuffer).toBeNull();

    gpu.ensurePresentPipeline();
    expect(gpu.present.presentPipeline).not.toBeNull();
    expect(gpu.present.presentParamsBuffer).not.toBeNull();
    const callsAfterRecovery = stub.createComputePipeline.mock.calls.length;
    gpu.ensurePresentPipeline();
    expect(stub.createComputePipeline).toHaveBeenCalledTimes(callsAfterRecovery);
  });
});
