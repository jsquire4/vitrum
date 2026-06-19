import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { GpuResources } from '../gpuResources.js';
import {
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import {
  PT_WEBGPU_TRACE_LITE_WGSL,
  composePtWebgpuTraceLiteWgsl,
} from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import {
  composeRestirPtProducerWgsl,
  composePtWebgpuReuseWgsl,
} from '../wgsl/pathTrace/restirPtCompose.wgsl.js';
import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from '../webgpuLimits.js';
import { installGpuConstStubs } from './gpuStub.js';

function makeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 8,
    },
    createCommandEncoder: vi.fn(() => ({ finish: vi.fn(() => ({})) })),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makePipelineDevice() {
  installGpuConstStubs();
  const shaderModules: Array<{ label?: string; code: string }> = [];
  const device = {
    createBuffer: vi.fn((desc?: { label?: string }) => ({ label: desc?.label ?? '', destroy: vi.fn() })),
    createShaderModule: vi.fn((desc: { label?: string; code: string }) => {
      shaderModules.push(desc);
      return {};
    }),
    createBindGroupLayout: vi.fn((desc: { label?: string }) => ({ label: desc.label })),
    createPipelineLayout: vi.fn((desc: { label?: string }) => ({ label: desc.label })),
    createComputePipeline: vi.fn((desc: { label?: string; compute: { entryPoint: string } }) => ({
      label: desc.label,
      entryPoint: desc.compute.entryPoint,
    })),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
  return { device, shaderModules };
}

describe('pt-webgpu sampling options', () => {
  it('keeps PCG as the default full and lite shader composition', () => {
    expect(composePtWebgpuTraceWgsl(false)).toContain('fn pcgNext(state: ptr<function, u32>) -> u32');
    expect(composePtWebgpuTraceWgsl(false)).not.toContain('ptSobolNextU32');
    expect(composePtWebgpuTraceLiteWgsl()).toBe(PT_WEBGPU_TRACE_LITE_WGSL);
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('ptSobolNextU32');
  });

  it('composes the binding-free Sobol RNG for full, lite, and ReSTIR-PT paths when requested', () => {
    const full = composePtWebgpuTraceWgsl(false, { sampling: 'sobol' });
    const lite = composePtWebgpuTraceLiteWgsl({ sampling: 'sobol' });
    const restirProducer = composeRestirPtProducerWgsl({ sampling: 'sobol' });
    const restirCombined = composePtWebgpuReuseWgsl({ sampling: 'sobol' });

    for (const wgsl of [full, lite, restirProducer, restirCombined]) {
      expect(wgsl).toContain('fn ptSobolNextU32(state: ptr<function, u32>) -> u32');
      expect(wgsl).toContain('fn pcgInit(px: u32, py: u32, frameSeed: u32) -> u32');
      expect(wgsl).toContain('fn rand_f32(state: ptr<function, u32>) -> f32');
      expect(wgsl).not.toContain('(*state) = (*state) * 747796405u + 2891336453u;');
    }
  });

  it('builds full and lite path-trace modules from the selected Sobol RNG', () => {
    const fullStub = makePipelineDevice();
    const full = new GpuResources(fullStub.device, 'full', false, false, undefined, 'sobol');
    full.ensurePipeline();
    expect(fullStub.shaderModules.find((m) => m.label === 'vitrum.pt-webgpu.pathTrace.full')?.code)
      .toContain('ptSobolNextU32');

    const liteStub = makePipelineDevice();
    const lite = new GpuResources(liteStub.device, 'lite', false, false, undefined, 'sobol');
    lite.ensurePipeline();
    expect(liteStub.shaderModules.find((m) => m.label === 'vitrum.pt-webgpu.pathTrace.lite')?.code)
      .toContain('ptSobolNextU32');
  });

  it('surfaces opt-in Sobol as an experimental capability with structured warning', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeDevice(),
      sampling: 'sobol',
      onWarning: (w) => structured.push(w),
    });

    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-sobol-sampling')).toBe(true);
    expect(structured.some((w) =>
      w.code === 'pt-webgpu.sobol-sampling-experimental' &&
      w.details?.sampling === 'sobol',
    )).toBe(true);
    expect(warn.mock.calls.some((c) => c.join(' ').includes("sampling:'sobol'"))).toBe(true);
    warn.mockRestore();
  });
});
