import { describe, expect, it } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { compilePipelines } from '../pipelineCompiler.js';
import {
  HYBRID_LITE_LIMITS,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  NRC_REQUIRED_MAX_BIND_GROUPS,
  NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
  NRC_WEBGPU_REQUIRED_LIMITS,
  nrcWebGpuRequiredFeaturesForConfig,
  nrcWebGpuRequiredLimitsForConfig,
} from '../WalkaroundGPUPipeline.js';
import type { RisGiNrcConfig } from '../../shaders/risGiNrc.wgsl.js';

installWebGPUPolyfills();

interface RecordedBgl {
  readonly entries: readonly GPUBindGroupLayoutEntry[];
}
interface RecordedShaderModule {
  readonly code: string;
}
interface RecordedLayout {
  readonly groups: readonly RecordedBgl[];
}
interface PipelineCounts {
  storageBuffers: number;
  storageTextures: number;
  sampledTextures: number;
  samplers: number;
  uniformBuffers: number;
  bindGroups: number;
}
interface RecordedPipeline {
  readonly label: string;
  readonly counts: PipelineCounts;
}

function countLayout(layout: RecordedLayout): PipelineCounts {
  const counts: PipelineCounts = {
    storageBuffers: 0,
    storageTextures: 0,
    sampledTextures: 0,
    samplers: 0,
    uniformBuffers: 0,
    bindGroups: layout.groups.length,
  };
  for (const group of layout.groups) {
    for (const entry of group.entries) {
      if (entry.storageTexture != null) counts.storageTextures += 1;
      else if (entry.texture != null) counts.sampledTextures += 1;
      else if (entry.sampler != null) counts.samplers += 1;
      else if (entry.buffer?.type === 'uniform') counts.uniformBuffers += 1;
      else if (entry.buffer != null) counts.storageBuffers += 1;
    }
  }
  return counts;
}

function countAutoLayoutWgsl(code: string): PipelineCounts {
  const counts: PipelineCounts = {
    storageBuffers: 0,
    storageTextures: 0,
    sampledTextures: 0,
    samplers: 0,
    uniformBuffers: 0,
    bindGroups: 0,
  };
  const groups = new Set<number>();
  const declaration = /((?:@\w+\([^)]*\)\s*)+)var(?:<([^>]+)>)?\s+\w+\s*:\s*([^;]+);/g;
  for (const match of code.matchAll(declaration)) {
    const attrs = match[1] ?? '';
    const group = /@group\((\d+)\)/.exec(attrs)?.[1];
    if (group == null) continue;
    groups.add(Number(group));
    const addressSpace = (match[2] ?? '').trim();
    const type = (match[3] ?? '').trim();
    if (addressSpace.startsWith('storage')) counts.storageBuffers += 1;
    else if (addressSpace.startsWith('uniform')) counts.uniformBuffers += 1;
    else if (type.startsWith('texture_storage_')) counts.storageTextures += 1;
    else if (type.startsWith('texture_')) counts.sampledTextures += 1;
    else if (type.startsWith('sampler')) counts.samplers += 1;
  }
  counts.bindGroups = groups.size;
  return counts;
}

function recordPipeline(
  pipelines: RecordedPipeline[],
  label: string | undefined,
  layout: GPUPipelineLayout | 'auto',
  module: RecordedShaderModule,
): void {
  const counts = layout === 'auto'
    ? countAutoLayoutWgsl(module.code)
    : countLayout(layout as unknown as RecordedLayout);
  pipelines.push({ label: label ?? '<unlabelled>', counts });
}

function recordingDevice(
  pipelines: RecordedPipeline[],
  bgls: RecordedBgl[],
): GPUDevice {
  const shaderModule = (code: string) => ({
    code,
    getCompilationInfo: async () => ({ messages: [] as GPUCompilationMessage[] }),
  });
  return {
    createShaderModule: (desc: GPUShaderModuleDescriptor) => shaderModule(desc.code),
    createBindGroupLayout: (desc: GPUBindGroupLayoutDescriptor) => {
      const bgl = { entries: [...desc.entries] };
      bgls.push(bgl);
      return bgl;
    },
    createPipelineLayout: (desc: GPUPipelineLayoutDescriptor) => ({
      groups: [...desc.bindGroupLayouts],
    }),
    createComputePipelineAsync: async (desc: GPUComputePipelineDescriptor) => {
      recordPipeline(pipelines, desc.label, desc.layout,
        desc.compute.module as unknown as RecordedShaderModule);
      return {} as GPUComputePipeline;
    },
    createRenderPipelineAsync: async (desc: GPURenderPipelineDescriptor) => {
      const module = (desc.fragment?.module ?? desc.vertex.module) as unknown as RecordedShaderModule;
      recordPipeline(pipelines, desc.label, desc.layout, module);
      return {} as GPURenderPipeline;
    },
  } as unknown as GPUDevice;
}

const NRC_CONFIG: RisGiNrcConfig = {
  levels: 8,
  featuresPerEntry: 2,
  oneBlobBins: 8,
  width: 64,
  outWidth: 3,
  hidden: 6,
};

async function derive(
  options: NonNullable<Parameters<typeof compilePipelines>[3]> = {},
): Promise<{
  peaks: PipelineCounts;
  maxBindingsPerBindGroup: number;
  pipelines: readonly RecordedPipeline[];
}> {
  const pipelines: RecordedPipeline[] = [];
  const bgls: RecordedBgl[] = [];
  await compilePipelines(
    recordingDevice(pipelines, bgls),
    {},
    'bgra8unorm',
    options,
  );
  const peak = (key: keyof PipelineCounts) =>
    Math.max(...pipelines.map((pipeline) => pipeline.counts[key]));
  return {
    peaks: {
      storageBuffers: peak('storageBuffers'),
      storageTextures: peak('storageTextures'),
      sampledTextures: peak('sampledTextures'),
      samplers: peak('samplers'),
      uniformBuffers: peak('uniformBuffers'),
      bindGroups: peak('bindGroups'),
    },
    maxBindingsPerBindGroup: Math.max(...bgls.map((bgl) => bgl.entries.length)),
    pipelines,
  };
}

describe('walkaround explicit pipeline device-limit derivation', () => {
  it('derives the base limits from the layouts compilePipelines actually selects', async () => {
    const { peaks, maxBindingsPerBindGroup, pipelines } = await derive({
      ppgEnabled: true,
      regirEnabled: true,
    });
    expect(peaks).toEqual({
      storageBuffers: 8,
      storageTextures: 7,
      sampledTextures: 16,
      samplers: 3,
      uniformBuffers: 4,
      bindGroups: 4,
    });
    expect(pipelines.map((pipeline) => pipeline.label)).toEqual(expect.arrayContaining([
      'ppg-update',
      'regir-build',
    ]));
    expect(maxBindingsPerBindGroup).toBe(16);
    expect(HYBRID_WEBGPU_REQUIRED_LIMITS).toEqual({
      maxStorageBuffersPerShaderStage: peaks.storageBuffers,
      maxStorageTexturesPerShaderStage: peaks.storageTextures,
      maxSampledTexturesPerShaderStage: peaks.sampledTextures,
    });
  });

  it('does not advertise a fictitious lower lite layout floor', () => {
    expect(HYBRID_LITE_LIMITS).toEqual(HYBRID_WEBGPU_REQUIRED_LIMITS);
  });

  it('keeps NRC GI-RIS at the four-group portable arena layout', async () => {
    const { peaks, pipelines } = await derive({
      nrcConfig: NRC_CONFIG,
      ppgEnabled: true,
      regirEnabled: true,
    });
    expect(peaks.storageBuffers).toBe(8);
    expect(peaks.bindGroups).toBe(4);
    expect(peaks.storageTextures).toBe(7);
    expect(peaks.sampledTextures).toBe(16);
    expect(pipelines.map((pipeline) => pipeline.label)).toEqual(expect.arrayContaining([
      'ppg-update',
      'regir-build',
    ]));
    expect(NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE).toBe(peaks.storageBuffers);
    expect(NRC_REQUIRED_MAX_BIND_GROUPS).toBe(peaks.bindGroups);
    expect(NRC_WEBGPU_REQUIRED_LIMITS).toMatchObject({
      ...HYBRID_WEBGPU_REQUIRED_LIMITS,
      maxBindGroups: peaks.bindGroups,
      maxStorageBuffersPerShaderStage: peaks.storageBuffers,
    });
  });

  it('derives trainer workgroup storage and shader-f16 from the requested NRC shape', () => {
    expect(nrcWebGpuRequiredLimitsForConfig({
      width: 64,
      tileB: 64,
      useF16: false,
    })).toMatchObject({
      maxComputeWorkgroupStorageSize: 32_768,
    });
    expect(nrcWebGpuRequiredFeaturesForConfig({
      width: 64,
      tileB: 64,
      useF16: false,
    })).toEqual([]);

    expect(nrcWebGpuRequiredLimitsForConfig({
      width: 64,
      tileB: 64,
      useF16: true,
    })).toMatchObject({
      maxComputeWorkgroupStorageSize: 16_384,
    });
    expect(nrcWebGpuRequiredFeaturesForConfig({
      width: 64,
      tileB: 64,
      useF16: true,
    })).toEqual(['shader-f16']);
  });
});
