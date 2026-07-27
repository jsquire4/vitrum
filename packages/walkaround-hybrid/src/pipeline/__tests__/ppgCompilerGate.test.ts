import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { PPGUpdatePass } from '../passes/PPGUpdatePass.js';
import { compilePipelines } from '../pipelineCompiler.js';

const GPU_SHADER_STAGE = Object.freeze({
  VERTEX: 1,
  FRAGMENT: 2,
  COMPUTE: 4,
});

let previousGPUShaderStage: unknown;

beforeEach(() => {
  previousGPUShaderStage = (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage;
  Object.defineProperty(globalThis, 'GPUShaderStage', {
    value: GPU_SHADER_STAGE,
    configurable: true,
  });
});

afterEach(() => {
  if (previousGPUShaderStage === undefined) {
    delete (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage;
  } else {
    Object.defineProperty(globalThis, 'GPUShaderStage', {
      value: previousGPUShaderStage,
      configurable: true,
    });
  }
});

interface RecordedComputePipeline {
  readonly label: string | undefined;
  readonly layout: GPUPipelineLayout | GPUAutoLayoutMode;
  readonly entryPoint: string;
}

function makeCompileStub(): {
  readonly device: GPUDevice;
  readonly shaderSources: Map<string, string>;
  readonly computePipelines: RecordedComputePipeline[];
} {
  const shaderSources = new Map<string, string>();
  const computePipelines: RecordedComputePipeline[] = [];

  const device = {
    createShaderModule(desc: GPUShaderModuleDescriptor): GPUShaderModule {
      const label = String(desc.label ?? '');
      shaderSources.set(label, desc.code);
      return {
        label,
        getCompilationInfo: async () => ({ messages: [] }),
      } as unknown as GPUShaderModule;
    },
    createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
      return {
        label: desc.label,
        entries: desc.entries,
      } as unknown as GPUBindGroupLayout;
    },
    createPipelineLayout(desc: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
      return {
        label: desc.label,
        bindGroupLayouts: desc.bindGroupLayouts,
      } as unknown as GPUPipelineLayout;
    },
    async createComputePipelineAsync(desc: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> {
      computePipelines.push({
        label: String(desc.label ?? ''),
        layout: desc.layout,
        entryPoint: desc.compute.entryPoint ?? 'main',
      });
      return {
        label: desc.label,
        getBindGroupLayout: () => ({} as GPUBindGroupLayout),
      } as unknown as GPUComputePipeline;
    },
    async createRenderPipelineAsync(desc: GPURenderPipelineDescriptor): Promise<GPURenderPipeline> {
      return { label: desc.label } as unknown as GPURenderPipeline;
    },
  } as unknown as GPUDevice;

  return { device, shaderSources, computePipelines };
}

describe('PPG compiler/runtime gates', () => {
  it('omits the PPG update pipeline by default', async () => {
    const stub = makeCompileStub();

    const compiled = await compilePipelines(stub.device, {}, 'bgra8unorm');

    expect(compiled.ppgUpdatePipeline).toBeUndefined();
    expect(stub.shaderSources.has('ppg-update')).toBe(false);
    expect(stub.computePipelines.some((pipeline) => pipeline.label === 'ppg-update')).toBe(false);
  });

  it('compiles the PPG update pipeline only when ppgEnabled is true', async () => {
    const stub = makeCompileStub();

    const compiled = await compilePipelines(stub.device, {}, 'bgra8unorm', {
      ppgEnabled: true,
      ppgMaxDTreeNodesPerCell: 128,
    });

    expect(compiled.ppgUpdatePipeline).toBeDefined();
    expect(stub.computePipelines.some((pipeline) =>
      pipeline.label === 'ppg-update' &&
      pipeline.layout === 'auto' &&
      pipeline.entryPoint === 'ppgUpdateMain'
    )).toBe(true);
    const source = stub.shaderSources.get('ppg-update') ?? '';
    expect(source).toMatch(/MAX_DTREE_NODES_PER_CELL\s*:\s*u32\s*=\s*128u/);
    expect(source).toMatch(/RESERVOIR_GI_STRIDE_LOCAL\s*:\s*u32\s*=\s*20u/);
  });

  it('threads the GRIS reservoir stride into the PPG update shader when DDGI-proxy GRIS reuse is enabled', async () => {
    const stub = makeCompileStub();

    await compilePipelines(stub.device, {}, 'bgra8unorm', {
      ppgEnabled: true,
      grisReuse: true,
    });

    const source = stub.shaderSources.get('ppg-update') ?? '';
    expect(source).toMatch(/RESERVOIR_GI_STRIDE_LOCAL\s*:\s*u32\s*=\s*28u/);
  });

  it('PPGUpdatePass gates training on ppgEnabled and ppgTrainThisFrame', () => {
    const pass = new PPGUpdatePass({} as GPUComputePipeline);
    const base = {
      denoiserMode: 'atrous-variance',
      ppgEnabled: true,
    };

    expect(pass.gates({ ...base, ppgEnabled: false })).toBe(false);
    expect(pass.gates(base)).toBe(true);
    expect(pass.gates({ ...base, ppgTrainThisFrame: true })).toBe(true);
    expect(pass.gates({ ...base, ppgTrainThisFrame: false })).toBe(false);
  });
});
