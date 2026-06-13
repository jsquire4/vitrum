import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

interface Recorder {
  readonly computePassLabels: string[];
}

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.2, 0.2, 0.2], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function identityMat(): Float32Array {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

function frameInput(size = 1) {
  return {
    viewMatrix: asMat4(identityMat()),
    projMatrix: asMat4(identityMat()),
    cameraPosition: [0, 0, 1] as [number, number, number],
    viewport: { width: size, height: size, devicePixelRatio: 1 },
    frameIndex: 0,
    frameSeed: 1,
    quality: { samplesTarget: 1, bounces: 1, resolutionFactor: 1 },
  };
}

function makeRenderAndReadbackDevice(rec: Recorder): GPUDevice {
  installGpuConstStubs();
  (globalThis as unknown as { GPUMapMode: { READ: number } }).GPUMapMode ??= { READ: 1 };
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
  const encoder = {
    beginComputePass: vi.fn((desc?: { label?: string }) => {
      rec.computePassLabels.push(desc?.label ?? '');
      return pass;
    }),
    clearBuffer: vi.fn(),
    copyTextureToBuffer: vi.fn(),
    finish: vi.fn(() => ({})),
  };
  return {
    queue: { writeBuffer: vi.fn(), writeTexture: vi.fn(), submit: vi.fn() },
    createBuffer: vi.fn((desc?: { label?: string; size?: number }) => {
      const size = Math.max(16, desc?.size ?? 256);
      return {
        label: desc?.label ?? '',
        destroy: vi.fn(),
        mapAsync: vi.fn(async () => undefined),
        getMappedRange: vi.fn(() => new ArrayBuffer(size)),
        unmap: vi.fn(),
      };
    }),
    ...textureStubMethods(),
    createShaderModule: vi.fn(() => ({ getCompilationInfo: vi.fn(async () => ({ messages: [] })) })),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
    createSampler: vi.fn(() => ({})),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => encoder),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
      maxTextureDimension2D: 8192,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

describe('pt-webgpu inverse render while host-paused', () => {
  it('bypasses the public paused-frame fast-out only for inverse-session renders', async () => {
    const rec: Recorder = { computePassLabels: [] };
    const engine = await createPTEngine_WebGPU({ device: makeRenderAndReadbackDevice(rec) });
    engine.setScene(makeScene());
    engine.renderFrame(frameInput());

    const session = engine.createInverseSession!({
      target: {
        data: new Float32Array([0.4, 0.4, 0.4]),
        width: 1,
        height: 1,
        channels: 3,
      },
      parameters: [{ path: 'materials.panel.baseColor', kind: 'rgb' }],
      samplesPerStep: 1,
      optimizer: { learningRate: 0.01, fdEpsilon: 1e-3 },
    });

    engine.pause();
    expect(engine.state).toBe('paused');
    const before = rec.computePassLabels.filter((label) => label === 'vitrum.pt-webgpu.pathTrace.pass').length;

    await session.step();

    const after = rec.computePassLabels.filter((label) => label === 'vitrum.pt-webgpu.pathTrace.pass').length;
    expect(after).toBeGreaterThan(before);
    expect(engine.state).toBe('paused');

    session.dispose();
    engine.dispose();
  });
});
