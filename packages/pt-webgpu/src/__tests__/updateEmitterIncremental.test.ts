import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function installWebGpuConstStubs(): void {
  installGpuConstStubs();
}

function makeScene(): Scene {
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
    emitters: [
      {
        kind: 'point',
        id: 'point-a',
        position: [1, 2, 3],
        color: [1, 1, 1],
        intensity: 2,
      },
      {
        kind: 'directional',
        id: 'sun',
        direction: [0, -1, 0],
        color: [1, 0.9, 0.8],
        intensity: 1.5,
      },
    ],
    environment: { kind: 'none' },
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

describe('pt-webgpu incremental emitter updates', () => {
  it('advertises emitter incremental patch support', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    const patchSupport = engine.capabilities.incrementalPatchSupport;
    expect(engine.capabilities.supportsIncrementalScene).toBe(true);
    expect(patchSupport?.emitter).toBe(true);
  });

  it('updates light buffers in-place without rebuilding scene buffers', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updateEmitter).toBe('function');
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updateEmitter?.('point-a', {
      intensity: 4,
      color: [0.5, 0.25, 0.125],
    });

    // Light count is unchanged by this material-only patch, so the light-tree
    // buffer is rewritten in place — no new GPU buffer is allocated.
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    // Dynamic emitter buffers upload all populated light classes.  This scene
    // has a directional + a point light, so uploadEmitterArrays writes both
    // (directional data is re-written even though only the point changed — the
    // per-class optimization is future work tracked separately). The WS2
    // light-tree is also re-uploaded because the patched emitter's radiance
    // changes the leaf powers.  Total: directional(1) + point(1) + lightTree(1) = 3.
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 3);
  });
});
