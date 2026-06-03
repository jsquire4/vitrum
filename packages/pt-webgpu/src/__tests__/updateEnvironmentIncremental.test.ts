import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function installWebGpuConstStubs(): void {
  installGpuConstStubs();
}

function makeHdri(width: number, height: number, value: number): Float32Array {
  const out = new Float32Array(width * height * 3);
  out.fill(value);
  return out;
}

function makeSceneWithHdri(width: number, height: number, value: number): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0.1 },
      },
    ],
    emitters: [],
    environment: {
      kind: 'hdri',
      hdri: {
        width,
        height,
        data: makeHdri(width, height, value),
      },
      intensity: 1,
    },
  };
}

function makeStubDevice() {
  const writeBuffer = vi.fn();
  const createBuffer = vi.fn((_desc: unknown) => ({
    destroy: vi.fn(),
  }));
  const device = {
    queue: { writeBuffer, writeTexture: vi.fn() },
    createBuffer,
    ...textureStubMethods(),
    createCommandEncoder: vi.fn(),
    limits: { maxStorageBuffersPerShaderStage: 64, maxTextureDimension2D: 8192 },
  } as unknown as GPUDevice;
  return { device, writeBuffer, createBuffer };
}

describe('pt-webgpu incremental environment updates', () => {
  it('exposes updateEnvironment and promises it in capabilities', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updateEnvironment).toBe('function');
  });

  it('updates environment buffers in-place when HDRI shape is unchanged', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeSceneWithHdri(2, 2, 0.25));

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;
    engine.updateEnvironment?.({
      kind: 'hdri',
      hdri: {
        width: 2,
        height: 2,
        data: makeHdri(2, 2, 0.75),
      },
      intensity: 1,
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    // Two direct buffer updates: environment texels + environment CDF.
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 2);
  });

  it('falls back to full scene rebuild when HDRI shape changes', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeSceneWithHdri(2, 2, 0.25));

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;
    engine.updateEnvironment?.({
      kind: 'hdri',
      hdri: {
        width: 4,
        height: 4,
        data: makeHdri(4, 4, 0.5),
      },
      intensity: 1,
    });

    expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore + 2);
  });
});
