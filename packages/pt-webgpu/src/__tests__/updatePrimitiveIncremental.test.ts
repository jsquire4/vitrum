import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { MATERIAL_FLOAT_STRIDE } from '../scene/materialPacking.js';

function installWebGpuConstStubs(): void {
  const g = globalThis as unknown as { GPUBufferUsage?: Record<string, number> };
  if (g.GPUBufferUsage == null) {
    g.GPUBufferUsage = { STORAGE: 1 << 0, COPY_DST: 1 << 1 };
  }
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
      {
        kind: 'mesh',
        id: 'mesh-b',
        positions: new Float32Array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.1, 0.4, 0.9], roughness: 0.6, metallic: 0.2 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makeInstancedScene(): Scene {
  return {
    primitives: [
      {
        kind: 'instanced-mesh',
        id: 'instanced-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.7, 0.7, 0.7], roughness: 0.4, metallic: 0.1 },
        instances: [
          asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
          ])),
          asMat4(new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            2, 0, 0, 1,
          ])),
        ],
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makeAnalyticScene(): Scene {
  return {
    primitives: [
      {
        kind: 'analytic',
        id: 'analytic-a',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 0.5]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.5, metallic: 0.1 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makeStubDevice() {
  const writeBuffer = vi.fn();
  const createBuffer = vi.fn((_desc: unknown) => ({
    destroy: vi.fn(),
  }));
  const device = {
    queue: { writeBuffer },
    createBuffer,
    createCommandEncoder: vi.fn(),
    limits: { maxStorageBuffersPerShaderStage: 64 },
  } as unknown as GPUDevice;
  return { device, writeBuffer, createBuffer };
}

describe('pt-webgpu incremental primitive updates', () => {
  it('advertises material-only incremental patch support', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    const patchSupport = engine.capabilities.incrementalPatchSupport;
    expect(engine.capabilities.supportsIncrementalScene).toBe(true);
    expect(patchSupport?.material).toBe(true);
    expect(patchSupport?.positions).toBe(false);
    expect(patchSupport?.transform).toBe(true);
  });

  it('updates material slot in-place without rebuilding scene buffers', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updatePrimitive).toBe('function');
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('mesh-b', {
      material: { baseColor: [0.2, 0.7, 0.9], roughness: 0.05, metallic: 0.4 },
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 1);

    const lastWrite = writeBuffer.mock.calls[writeBuffer.mock.calls.length - 1];
    const writeByteOffset = lastWrite?.[1];
    expect(writeByteOffset).toBe(1 * MATERIAL_FLOAT_STRIDE * Float32Array.BYTES_PER_ELEMENT);
  });

  it('falls back to full scene rebuild for non-material patches', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updatePrimitive).toBe('function');
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('mesh-b', {
      positions: new Float32Array([0, 0, 2, 1, 0, 2, 0, 1, 2]),
    });

    expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore + 1);
  });

  it('updates TLAS buffers in-place for transform-only mesh patches', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('mesh-b', {
      transform: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        2, 0, 0, 1,
      ])),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 5);
  });

  it('updates TLAS buffers in-place for instanced transform patches', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeInstancedScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('instanced-a', {
      instances: [
        asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          1, 0, 0, 1,
        ])),
        asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          3, 0, 0, 1,
        ])),
      ],
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 5);
  });

  it('falls back to full rebuild when instanced transform count changes', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeInstancedScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('instanced-a', {
      instances: [
        asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          1, 0, 0, 1,
        ])),
      ],
    });

    expect(createBuffer.mock.calls.length).toBeGreaterThan(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBeGreaterThan(writesBefore + 5);
  });

  it('updates analytic transform buffers in-place', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeAnalyticScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    engine.updatePrimitive?.('analytic-a', {
      transform: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 0, 1,
      ])),
    });

    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(writesBefore + 2);
  });
});
