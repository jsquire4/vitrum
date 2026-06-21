// nrcStructuralGate.test.ts — compile-time NRC layout/source gate.
//
// NRC is allowed to change the gi-ris pipeline structure only when explicitly
// opted in. The default ReSTIR-GI path must keep the 4-group DDGI-estimate
// module; the NRC variant adds @group(4) plus a 5th bind group layout.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { compilePipelines } from '../pipelineCompiler.js';
import type { BGLCache } from '../../bglTypes.js';
import type { RisGiNrcConfig } from '../../shaders/risGiNrc.wgsl.js';

interface RecordedLayout {
  readonly label: string | undefined;
  readonly bindGroupLayouts: readonly GPUBindGroupLayout[];
}

interface RecordedComputePipeline {
  readonly label: string | undefined;
  readonly layout: GPUPipelineLayout | GPUAutoLayoutMode;
  readonly entryPoint: string;
}

const GPU_SHADER_STAGE = Object.freeze({
  VERTEX: 1,
  FRAGMENT: 2,
  COMPUTE: 4,
});

const nrcConfig: RisGiNrcConfig = {
  levels: 2,
  featuresPerEntry: 2,
  oneBlobBins: 4,
  width: 8,
  outWidth: 3,
  hidden: 1,
};

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
      return { label: desc.label } as unknown as GPUComputePipeline;
    },
    async createRenderPipelineAsync(desc: GPURenderPipelineDescriptor): Promise<GPURenderPipeline> {
      return { label: desc.label } as unknown as GPURenderPipeline;
    },
  } as unknown as GPUDevice;

  return { device, shaderSources, computePipelines };
}

function risGiGroupCount(pipelines: readonly RecordedComputePipeline[]): number {
  const risGi = pipelines.find((pipeline) => pipeline.label === 'risGi');
  expect(risGi).toBeDefined();
  const layout = risGi!.layout as unknown as RecordedLayout;
  return layout.bindGroupLayouts.length;
}

describe('NRC structural gate', () => {
  it('keeps the default gi-ris pass on the 4-group non-NRC module', async () => {
    const stub = makeCompileStub();

    await compilePipelines(stub.device, {} as BGLCache, 'bgra8unorm');

    expect(risGiGroupCount(stub.computePipelines)).toBe(4);
    const source = stub.shaderSources.get('risGi') ?? '';
    expect(source).not.toContain('@group(4)');
    expect(source).not.toContain('nrcWeights');
    expect(source).not.toContain('nrcWriteRecord');
  });

  it('adds the 5th NRC bind group and NRC shader symbols only when nrcConfig is provided', async () => {
    const stub = makeCompileStub();

    await compilePipelines(stub.device, {} as BGLCache, 'bgra8unorm', { nrcConfig });

    expect(risGiGroupCount(stub.computePipelines)).toBe(5);
    const source = stub.shaderSources.get('risGi') ?? '';
    expect(source).toContain('@group(4)');
    expect(source).toContain('nrcWeights');
    expect(source).toContain('nrcWriteRecord');
  });
});
