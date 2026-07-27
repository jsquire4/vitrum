import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { installGpuConstStubs } from './gpuStub.js';

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
  const writeTexture = vi.fn();
  const createBuffer = vi.fn((desc: GPUBufferDescriptor) => ({
    label: desc.label ?? '',
    size: Number(desc.size),
    usage: Number(desc.usage),
    destroy: vi.fn(),
  }));
  const createTexture = vi.fn((desc: GPUTextureDescriptor) => ({
    label: desc.label ?? '',
    size: desc.size,
    createView: vi.fn(() => ({})),
    destroy: vi.fn(),
  }));
  const createSampler = vi.fn(() => ({}));
  const copyBufferToBuffer = vi.fn();
  const clearBuffer = vi.fn();
  const finish = vi.fn(() => ({}));
  const createCommandEncoder = vi.fn(() => ({ copyBufferToBuffer, clearBuffer, finish }));
  const submit = vi.fn();
  const destroyDevice = vi.fn();
  const device = {
    queue: { writeBuffer, writeTexture, submit },
    createBuffer,
    createTexture,
    createSampler,
    createCommandEncoder,
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
      maxTextureDimension2D: 8192,
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    destroy: destroyDevice,
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
  return {
    device,
    writeBuffer,
    writeTexture,
    createBuffer,
    createTexture,
    createCommandEncoder,
    copyBufferToBuffer,
    clearBuffer,
    finish,
    submit,
    destroyDevice,
  };
}

describe('pt-webgpu incremental environment updates', () => {
  it('exposes updateEnvironment and promises it in capabilities', async () => {
    const { device } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    expect(typeof engine.updateEnvironment).toBe('function');
  });

  it('copies only dirty same-shaped HDRI words into live buffers', async () => {
    installWebGpuConstStubs();
    const {
      device,
      writeBuffer,
      createBuffer,
      copyBufferToBuffer,
      submit,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeSceneWithHdri(2, 2, 0.25));

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;
    const copiesBefore = copyBufferToBuffer.mock.calls.length;
    const submitsBefore = submit.mock.calls.length;
    engine.updateEnvironment?.({
      kind: 'hdri',
      hdri: {
        width: 2,
        height: 2,
        data: makeHdri(2, 2, 0.75),
      },
      intensity: 1,
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
    // A pure radiance scale changes RGB only. Float64 CDF construction keeps
    // normalized pdf/CDF lanes bit-identical, so no probability words are dirty.
    expect(staging.size).toBe(48);
    expect(staging.destroy).toHaveBeenCalledTimes(1);

    const writes = writeBuffer.mock.calls.slice(writesBefore);
    expect(writes).toHaveLength(1);
    expect(writes[0]?.[0]).toBe(staging);
    expect(writes[0]?.[1]).toBe(0);
    expect(writes[0]?.[4]).toBe(48);

    const copies = copyBufferToBuffer.mock.calls.slice(copiesBefore);
    expect(copies.map((call) =>
      `${(call[2] as { label: string }).label}@${call[3]}+${call[4]}`,
    )).toEqual([
      'vitrum.pt-webgpu.scene.environmentMapTexels@0+12',
      'vitrum.pt-webgpu.scene.environmentMapTexels@16+12',
      'vitrum.pt-webgpu.scene.environmentMapTexels@32+12',
      'vitrum.pt-webgpu.scene.environmentMapTexels@48+12',
    ]);
    expect(copies.map((call) => call[1])).toEqual([0, 12, 24, 36]);
    expect(copies.reduce((sum, call) => sum + Number(call[4]), 0)).toBe(48);
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

  it('replaces only environment/light-tree buffers when HDRI shape changes', async () => {
    installWebGpuConstStubs();
    const { device, writeBuffer, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    engine.setScene(makeSceneWithHdri(2, 2, 0.25));

    const writesBefore = writeBuffer.mock.calls.length;
    const buffersBefore = createBuffer.mock.calls.length;
    const oldBuffers = createBuffer.mock.results.slice(0, buffersBefore)
      .filter((result) => result.type === 'return')
      .map((result) => result.value as {
        readonly label: string;
        destroy: ReturnType<typeof vi.fn>;
      });
    engine.updateEnvironment?.({
      kind: 'hdri',
      hdri: {
        width: 4,
        height: 4,
        data: makeHdri(4, 4, 0.5),
      },
      intensity: 1,
    });

    const createdLabels = createBuffer.mock.results.slice(buffersBefore)
      .filter((result) => result.type === 'return')
      .map((result) => (result.value as { readonly label: string }).label);
    expect(createdLabels).toEqual([
      'vitrum.pt-webgpu.scene.environmentMapTexels',
      'vitrum.pt-webgpu.scene.environmentMapCdf',
      'vitrum.pt-webgpu.scene.lightTree',
    ]);
    expect(writeBuffer.mock.calls.length - writesBefore).toBe(3);
    expect(createdLabels.some((label) =>
      /positions|normals|indices|bvh|tlas|materials/i.test(label),
    )).toBe(false);

    const destroyedOldLabels = oldBuffers
      .filter((buffer) => buffer.destroy.mock.calls.length > 0)
      .map((buffer) => buffer.label);
    expect(destroyedOldLabels).toEqual([
      'vitrum.pt-webgpu.scene.environmentMapTexels',
      'vitrum.pt-webgpu.scene.environmentMapCdf',
      'vitrum.pt-webgpu.scene.lightTree',
    ]);
  });

  it('refreshes lite sampled light/environment textures after same-shaped updateEnvironment', async () => {
    installWebGpuConstStubs();
    const { device, writeTexture } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device, traceTier: 'lite' });
    engine.setScene(makeSceneWithHdri(2, 2, 0.25));

    const writesBefore = writeTexture.mock.calls.length;
    engine.updateEnvironment?.({
      kind: 'hdri',
      hdri: {
        width: 2,
        height: 2,
        data: makeHdri(2, 2, 0.75),
      },
      intensity: 1,
    });

    const labels = writeTexture.mock.calls
      .slice(writesBefore)
      .map((call) => ((call[0] as { texture?: { label?: string } }).texture?.label ?? ''));
    expect(labels).toEqual([
      'vitrum.pt-webgpu.lite.envTex',
      'vitrum.pt-webgpu.lite.envCdfTex',
      'vitrum.pt-webgpu.lite.lightTex',
    ]);
  });

  it('supports eager, transactional setSize without changing host-device ownership', async () => {
    installWebGpuConstStubs();
    const {
      device,
      createBuffer,
      createTexture,
      submit,
      destroyDevice,
    } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });

    engine.pause();
    engine.setSize?.(32, 16);
    expect(engine.state).toBe('paused');
    expect(createTexture.mock.calls.slice(0, 6).map((call) => call[0].label)).toEqual([
      'vitrum.pt-webgpu.accum',
      'vitrum.pt-webgpu.normalDepth',
      'vitrum.pt-webgpu.albedo',
      'vitrum.pt-webgpu.variance',
      'vitrum.pt-webgpu.motionVectors',
      'vitrum.pt-webgpu.present',
    ]);
    expect(createBuffer.mock.calls.map((call) => call[0].label)).toEqual([
      'vitrum.pt-webgpu.accum.buffer',
      'vitrum.pt-webgpu.varianceMoments.buffer',
    ]);
    expect(submit).toHaveBeenCalledTimes(1);

    const textureCount = createTexture.mock.calls.length;
    const bufferCount = createBuffer.mock.calls.length;
    engine.setSize?.(32, 16);
    expect(createTexture).toHaveBeenCalledTimes(textureCount);
    expect(createBuffer).toHaveBeenCalledTimes(bufferCount);
    expect(submit).toHaveBeenCalledTimes(1);

    const firstTextures = createTexture.mock.results.slice(0, 6)
      .map((result) => result.value as { destroy: ReturnType<typeof vi.fn> });
    const firstBuffers = createBuffer.mock.results.slice(0, 2)
      .map((result) => result.value as { destroy: ReturnType<typeof vi.fn> });
    engine.setSize?.(64, 8);
    expect(createTexture).toHaveBeenCalledTimes(textureCount + 6);
    expect(createBuffer).toHaveBeenCalledTimes(bufferCount + 2);
    for (const resource of [...firstTextures, ...firstBuffers]) {
      expect(resource.destroy).toHaveBeenCalledTimes(1);
    }

    const allocationsBeforeInvalid = createTexture.mock.calls.length + createBuffer.mock.calls.length;
    for (const [width, height] of [
      [0, 1],
      [1.5, 1],
      [Number.NaN, 1],
      [8193, 1],
    ] as const) {
      expect(() => engine.setSize?.(width, height)).toThrow();
    }
    expect(createTexture.mock.calls.length + createBuffer.mock.calls.length)
      .toBe(allocationsBeforeInvalid);

    engine.dispose();
    expect(destroyDevice).not.toHaveBeenCalled();
    expect(() => engine.setSize?.(1, 1)).toThrow('setSize: engine is disposed');
  });

  it('updates all lighting atomically without rebuilding geometry', async () => {
    installWebGpuConstStubs();
    const warnings: EngineWarning[] = [];
    const { device, createBuffer, submit } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({
      device,
      onWarning: (warning) => warnings.push(warning),
    });
    const getScene = engine.getScene?.bind(engine);
    if (getScene == null) throw new Error('pt-webgpu must expose getScene');
    engine.setSize?.(16, 16);
    engine.setScene(makeSceneWithHdri(2, 2, 0.25));

    const buffersBefore = createBuffer.mock.calls.length;
    const submitsBefore = submit.mock.calls.length;
    const nextEmitters: Scene['emitters'] = [
      { kind: 'point', id: 'lamp', position: [0, 2, 0], color: [1, 0.5, 0.25], intensity: 4 },
      { kind: 'mesh-area', id: 'mesh-light', meshId: 'mesh-a', color: [0.25, 0.5, 1], intensity: 3 },
    ];
    const nextEnvironment: NonNullable<Scene['environment']> = {
      kind: 'hdri',
      hdri: { width: 4, height: 4, data: makeHdri(4, 4, 0.75) },
      intensity: 2,
    };
    engine.updateLighting?.({
      emitters: nextEmitters,
      environment: nextEnvironment,
    });

    expect(getScene()?.emitters).toEqual(nextEmitters);
    expect(getScene()?.emitters).not.toBe(nextEmitters);
    expect(getScene()?.environment).toEqual(nextEnvironment);
    const createdLabels = createBuffer.mock.calls.slice(buffersBefore)
      .map((call) => String(call[0].label));
    expect(createdLabels.length).toBeGreaterThan(0);
    expect(createdLabels).toContain('vitrum.pt-webgpu.scene.environmentMapTexels');
    expect(createdLabels).toContain('vitrum.pt-webgpu.scene.materials');
    expect(createdLabels.some((label) =>
      /positions|normals|indices|bvh|tlas/i.test(label),
    )).toBe(false);
    expect(submit.mock.calls.length).toBeGreaterThan(submitsBefore);

    const buffersBeforeNoop = createBuffer.mock.calls.length;
    const sceneAfterUpdate = getScene();
    engine.updateLighting?.({});
    expect(createBuffer).toHaveBeenCalledTimes(buffersBeforeNoop);
    expect(getScene()).toBe(sceneAfterUpdate);

    expect(() => engine.updateLighting?.({ legacyAmbient: 1 })).toThrow(
      /options contains unknown key.*legacyAmbient/,
    );
    expect(createBuffer).toHaveBeenCalledTimes(buffersBeforeNoop);
    expect(getScene()).toBe(sceneAfterUpdate);
    expect(warnings.some((warning) => warning.code === 'pt-webgpu.update-lighting-unknown-key')).toBe(false);
  });

  it('rejects lighting/environment misuse before mutating live state or allocating', async () => {
    installWebGpuConstStubs();
    const { device, createBuffer } = makeStubDevice();
    const engine = await createPTEngine_WebGPU({ device });
    const getScene = engine.getScene?.bind(engine);
    if (getScene == null) throw new Error('pt-webgpu must expose getScene');

    expect(() => engine.updateEnvironment?.(null)).toThrow(
      'updateEnvironment: call setScene() before updateEnvironment',
    );
    expect(() => engine.updateLighting?.({ emitters: [] })).toThrow(
      'updateLighting: call setScene() before updateLighting',
    );

    engine.setScene(makeSceneWithHdri(2, 2, 0.25));
    const liveScene = getScene();
    const buffersBefore = createBuffer.mock.calls.length;
    expect(() => engine.updateLighting?.({
      emitters: [
        { kind: 'point', id: 'bad', position: [0, 0, 0], color: [1, 1, 1], intensity: Number.NaN },
      ],
    })).toThrow();
    expect(() => engine.updateLighting?.({
      environment: {
        kind: 'hdri',
        hdri: { width: 1, height: 1, data: makeHdri(1, 1, 0.5) },
        intensity: Number.NaN,
      },
    })).toThrow();
    expect(() => engine.updateEnvironment?.({
      kind: 'none',
      hdri: { width: 1, height: 1, data: makeHdri(1, 1, 0.5) },
    } as never)).toThrow(/is not a known contract field/);
    expect(createBuffer).toHaveBeenCalledTimes(buffersBefore);
    expect(getScene()).toBe(liveScene);

    engine.dispose();
    expect(() => engine.updateEnvironment?.(null)).toThrow(
      'updateEnvironment: engine is disposed',
    );
    expect(() => engine.updateLighting?.({ emitters: [] })).toThrow(
      'updateLighting: engine is disposed',
    );
  });
});
