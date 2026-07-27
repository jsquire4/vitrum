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
  const writeTexture = vi.fn();
  const createBuffer = vi.fn((desc: GPUBufferDescriptor) => ({
    label: desc.label ?? '',
    size: Number(desc.size),
    usage: Number(desc.usage),
    destroy: vi.fn(),
  }));
  const copyBufferToBuffer = vi.fn();
  const finish = vi.fn(() => ({}));
  const createCommandEncoder = vi.fn(() => ({ copyBufferToBuffer, finish }));
  const submit = vi.fn();
  const device = {
    queue: { writeBuffer, writeTexture, submit },
    createBuffer,
    ...textureStubMethods(),
    createCommandEncoder,
    limits: { maxStorageBuffersPerShaderStage: 64, maxStorageTexturesPerShaderStage: 8, maxTextureDimension2D: 8192 },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
  return {
    device,
    writeBuffer,
    writeTexture,
    createBuffer,
    createCommandEncoder,
    copyBufferToBuffer,
    finish,
    submit,
  };
}

describe('pt-webgpu incremental emitter updates', () => {
  it('advertises emitter incremental patch support', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    const patchSupport = engine.capabilities.incrementalPatchSupport;
    expect(engine.capabilities.supportsIncrementalScene).toBe(true);
    expect(patchSupport?.emitter).toBe(true);
  });

  it('copies only dirty emitter and light-tree words into live buffers', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updateEmitter).toBe('function');
    engine.setScene(makeScene());

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;

    const copiesBefore = copyBufferToBuffer.mock.calls.length;
    const submitsBefore = submit.mock.calls.length;

    engine.updateEmitter?.('point-a', {
      intensity: 4,
      color: [0.5, 0.25, 0.125],
    });

    const created = createBuffer.mock.results.slice(buffersBefore)
      .filter((result) => result.type === 'return')
      .map((result) => result.value as {
        readonly label: string;
        readonly size: number;
        destroy: ReturnType<typeof vi.fn>;
      });
    expect(created).toHaveLength(1);
    const staging = created[0]!;
    expect(staging.label).toBe('vitrum.pt-webgpu.scene.incremental-staging');
    expect(staging.size).toBe(28);
    expect(staging.destroy).toHaveBeenCalledTimes(1);

    const writes = writeBuffer.mock.calls.slice(writesBefore);
    expect(writes).toHaveLength(1);
    expect(writes[0]?.[0]).toBe(staging);
    expect(writes[0]?.[1]).toBe(0);
    expect(writes[0]?.[4]).toBe(28);

    const copies = copyBufferToBuffer.mock.calls.slice(copiesBefore);
    expect(copies.map((call) =>
      `${(call[2] as { label: string }).label}@${call[3]}+${call[4]}`,
    )).toEqual([
      'vitrum.pt-webgpu.scene.pointLights@20+8',
      'vitrum.pt-webgpu.scene.lightTree@4+4',
      'vitrum.pt-webgpu.scene.lightTree@64+8',
      'vitrum.pt-webgpu.scene.lightTree@128+8',
    ]);
    expect(copies.map((call) => call[1])).toEqual([0, 8, 12, 20]);
    expect(copies.reduce((sum, call) => sum + Number(call[4]), 0)).toBe(28);
    const liveBuffers = new Set(createBuffer.mock.results.slice(0, buffersBefore)
      .filter((result) => result.type === 'return')
      .map((result) => result.value));
    for (const [source, sourceOffset, destination, destinationOffset, size] of copies) {
      expect(source).toBe(staging);
      expect(liveBuffers.has(destination)).toBe(true);
      expect(Number(sourceOffset) % 4).toBe(0);
      expect(Number(destinationOffset) % 4).toBe(0);
      expect(Number(size) % 4).toBe(0);
    }
    expect(submit.mock.calls.length - submitsBefore).toBe(1);
    for (const result of createBuffer.mock.results.slice(0, buffersBefore)) {
      if (result.type === 'return') {
        expect((result.value as { destroy: ReturnType<typeof vi.fn> }).destroy)
          .not.toHaveBeenCalled();
      }
    }
  });

  it('treats an empty patch as a validated constant-work no-op', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, writeTexture, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeScene());

    const sceneBefore = engine.getScene?.();
    const buffersBefore = createBuffer.mock.calls.length;
    const bufferWritesBefore = writeBuffer.mock.calls.length;
    const textureWritesBefore = writeTexture.mock.calls.length;
    const destroyCount = (): number => createBuffer.mock.results.reduce(
      (total, result) => total + (
        result.type === 'return'
          ? (result.value as { destroy: ReturnType<typeof vi.fn> }).destroy.mock.calls.length
          : 0
      ),
      0,
    );
    const destroysBefore = destroyCount();
    const reset = vi.spyOn(engine, 'reset');

    engine.updateEmitter?.('point-a', {});
    expect(engine.getScene?.()).toBe(sceneBefore);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(bufferWritesBefore);
    expect(writeTexture.mock.calls.length).toBe(textureWritesBefore);
    expect(destroyCount()).toBe(destroysBefore);
    expect(reset).not.toHaveBeenCalled();

    expect(() => engine.updateEmitter?.('missing', {})).toThrow(/not found/);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(bufferWritesBefore);
    expect(writeTexture.mock.calls.length).toBe(textureWritesBefore);
    expect(destroyCount()).toBe(destroysBefore);
    expect(reset).not.toHaveBeenCalled();
    expect(() =>
      engine.updateEmitter?.('point-a', { id: null } as never),
    ).toThrow(/id cannot be changed/);
    expect(() =>
      engine.updateEmitter?.('point-a', { kind: null } as never),
    ).toThrow(/kind cannot change/);
    expect(() =>
      engine.updateEmitter?.('point-a', { radius: 4 }),
    ).toThrow(/is not a known contract field/);
    expect(engine.getScene?.()).toBe(sceneBefore);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(bufferWritesBefore);
    expect(writeTexture.mock.calls.length).toBe(textureWritesBefore);
  });


  it('refreshes lite sampled light/environment textures after updateEmitter', async () => {
    installWebGpuConstStubs();
    const { device, writeTexture } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device, traceTier: 'lite' });
    engine.setScene(makeScene());

    const writesBefore = writeTexture.mock.calls.length;
    engine.updateEmitter?.('point-a', { intensity: 5 });

    const labels = writeTexture.mock.calls
      .slice(writesBefore)
      .map((call) => ((call[0] as { texture?: { label?: string } }).texture?.label ?? ''));
    expect(labels).toEqual([
      'vitrum.pt-webgpu.lite.envTex',
      'vitrum.pt-webgpu.lite.envCdfTex',
      'vitrum.pt-webgpu.lite.lightTex',
    ]);
  });

  it('rejects unsupported lite lighting replacements before mutating live state', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, writeTexture, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device, traceTier: 'lite' });
    engine.setScene(makeScene());

    const sceneBefore = engine.getScene?.();
    const buffersBefore = createBuffer.mock.calls.length;
    const bufferWritesBefore = writeBuffer.mock.calls.length;
    const textureWritesBefore = writeTexture.mock.calls.length;

    expect(() => engine.updateLighting?.({
      emitters: [{
        kind: 'mesh-area',
        id: 'area',
        meshId: 'mesh-a',
        color: [1, 1, 1],
        intensity: 2,
      }],
    })).toThrow(/Lite tier cannot render.*mesh-area/);

    expect(engine.getScene?.()).toBe(sceneBefore);
    expect(createBuffer.mock.calls.length).toBe(buffersBefore);
    expect(writeBuffer.mock.calls.length).toBe(bufferWritesBefore);
    expect(writeTexture.mock.calls.length).toBe(textureWritesBefore);
  });
});
