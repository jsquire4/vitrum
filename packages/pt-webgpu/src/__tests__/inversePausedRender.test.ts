import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import {
  FRAME_PARAMS_BYTE_SIZE,
  FrameParamsSlot,
} from '../scene/frameParamsLayout.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

interface Recorder {
  readonly computePassLabels: string[];
  readonly directLightingModes: number[];
  bufferCopies: number;
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

function makeMultiLightScene(): Scene {
  return {
    ...makeScene(),
    emitters: [
      {
        kind: 'point',
        id: 'lamp',
        position: [0, 1, 1],
        color: [1, 1, 1],
        intensity: 2,
      },
    ],
    environment: {
      kind: 'hdri',
      hdri: {
        width: 1,
        height: 1,
        data: new Float32Array([1, 1, 1]),
      },
    },
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
    copyBufferToBuffer: vi.fn(() => { rec.bufferCopies += 1; }),
    finish: vi.fn(() => ({})),
  };
  return {
    queue: {
      writeBuffer: vi.fn((_buffer: GPUBuffer, _offset: number, data: BufferSource) => {
        if (data instanceof ArrayBuffer && data.byteLength >= FRAME_PARAMS_BYTE_SIZE) {
          rec.directLightingModes.push(new Uint32Array(data)[FrameParamsSlot.directLightingMode] ?? -1);
        }
      }),
      writeTexture: vi.fn(),
      submit: vi.fn(),
    },
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
    const rec: Recorder = { computePassLabels: [], directLightingModes: [], bufferCopies: 0 };
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
    expect(rec.directLightingModes).toContain(0);
    expect(rec.directLightingModes).toContain(1);
    expect(rec.bufferCopies).toBeGreaterThan(0);
    expect(engine.state).toBe('paused');

    session.dispose();
    engine.dispose();
  });

  it('rejects uncertified real-engine replay fields and accepts certified replay', async () => {
    const rec: Recorder = { computePassLabels: [], directLightingModes: [], bufferCopies: 0 };
    const warnings: unknown[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeRenderAndReadbackDevice(rec),
      onWarning: (warning) => warnings.push(warning),
    });
    const scene = makeMultiLightScene();
    engine.setScene(scene);
    engine.renderFrame(frameInput());
    const retainedScene = engine.getScene!();
    const retainedMaterial = retainedScene!.primitives[0]!.material;

    const common = {
      target: {
        data: new Float32Array([0.4, 0.4, 0.4]),
        width: 1,
        height: 1,
        channels: 3 as const,
      },
      method: 'path-replay' as const,
      samplesPerStep: 1,
      optimizer: { learningRate: 0.01, fdEpsilon: 1e-3 },
    };

    for (const [field, kind] of [
      ['specularColor', 'rgb'],
      ['clearcoat', 'scalar'],
    ] as const) {
      const reported: unknown[] = [];
      expect(() => engine.createInverseSession!({
          ...common,
          parameters: [{
            path: `materials.panel.${field}`,
            kind,
            initial: kind === 'rgb' ? [0.9, 0.8, 0.7] : [0.1],
          }],
          onDiagnostic: (diagnostic) => reported.push(diagnostic),
        }))
        .toThrow(/requested path-replay is outside the certified pt-webgpu domain/);
      const expected = expect.objectContaining({
        code: 'path-replay-unsupported-field',
        path: `materials.panel.${field}`,
        details: expect.objectContaining({ proof: 'missing-end-to-end-gpu-fit' }),
      });
      expect(reported).toContainEqual(expected);
      expect(engine.getScene!()).toBe(retainedScene);
      expect(retainedMaterial.baseColor).toEqual([0.2, 0.2, 0.2]);
      expect(retainedMaterial.roughness).toBe(0.5);
    }

    const emissive = engine.createInverseSession!({
      ...common,
      parameters: [{ path: 'materials.panel.emissive', kind: 'rgb' }],
    });
    expect(emissive.method).toBe('path-replay');
    expect(emissive.diagnostics).toEqual([]);
    expect(warnings).toEqual([]);

    emissive.dispose();
    engine.dispose();
  });
});
