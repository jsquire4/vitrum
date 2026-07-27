import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Engine, FrameInput, Scene } from '@vitrum/core';

const constructWalkaroundMock = vi.hoisted(() => vi.fn());
const constructPathTracerWebGPUMock = vi.hoisted(() => vi.fn());

vi.mock('../createEngine.js', () => ({
  constructWalkaround: constructWalkaroundMock,
  constructPathTracerWebGPU: constructPathTracerWebGPUMock,
}));

import { createProgressiveEngine } from '../createProgressiveEngine.js';

const scene: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };
const sceneWithPrimitive: Scene = {
  primitives: [
    {
      kind: 'mesh',
      id: 'p',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
    },
  ],
  emitters: [],
  environment: { kind: 'none' },
};
const originalNavigator = globalThis.navigator;

function makeEngine(capability: 'seed-source' | 'seed-sink'): Engine {
  return {
    state: 'ready',
    capabilities: {
      ...(capability === 'seed-source' ? { supportsProgressiveSeedSource: true } : {}),
      ...(capability === 'seed-sink' ? { supportsAccumulatorSeed: true } : {}),
    },
    setScene: vi.fn(),
    renderFrame: vi.fn((_input: FrameInput) => ({
      kind: 'skipped',
      samplesAccumulated: 0,
      isConverged: false,
    })),
    ...(capability === 'seed-source'
      ? { getProgressiveSeedTexture: vi.fn(() => ({ texture: {}, width: 1, height: 1 })) }
      : {}),
    ...(capability === 'seed-sink' ? { seedAccumulator: vi.fn() } : {}),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
  } as unknown as Engine;
}

function makeDevice(): GPUDevice {
  return {
    destroy: vi.fn(),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 16,
      maxSampledTexturesPerShaderStage: 64,
    },
  } as unknown as GPUDevice;
}

function makeAdapter(device: GPUDevice): GPUAdapter {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 16,
      maxSampledTexturesPerShaderStage: 64,
    },
    requestDevice: vi.fn(async () => device),
  } as unknown as GPUAdapter;
}

function makeThrowingConfigureCanvas(error: Error): HTMLCanvasElement {
  return {
    width: 1,
    height: 1,
    getContext: vi.fn((kind: string) => (
      kind === 'webgpu'
        ? { configure: vi.fn(() => { throw error; }) }
        : null
    )),
  } as unknown as HTMLCanvasElement;
}

function makeCanvas(): HTMLCanvasElement {
  return {
    width: 1,
    height: 1,
    getContext: vi.fn(() => null),
  } as unknown as HTMLCanvasElement;
}

describe('createProgressiveEngine canvas plumbing diagnostics', () => {
  beforeEach(() => {
    constructWalkaroundMock.mockReset();
    constructPathTracerWebGPUMock.mockReset();
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  });

  it('reports final canvas-configure failures through the structured onError callback', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: () => 'rgba8unorm',
        },
      },
      configurable: true,
    });
    constructWalkaroundMock.mockResolvedValue(makeEngine('seed-source'));
    constructPathTracerWebGPUMock.mockResolvedValue(makeEngine('seed-sink'));
    const configureError = new Error('configure failed');
    const onError = vi.fn();

    const handle = await createProgressiveEngine({
      canvas: makeThrowingConfigureCanvas(configureError),
      scene,
      onError,
    });

    expect(onError).toHaveBeenCalledWith(configureError, {
      phase: 'canvas-configure',
      backend: 'walkaround-hybrid',
      recoverable: true,
    });

    handle.dispose();
  });

  it('seeds the coordinator scene fallback for progressive mutation patches', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: () => 'rgba8unorm',
        },
      },
      configurable: true,
    });
    const realtime = makeEngine('seed-source');
    const converged = makeEngine('seed-sink');
    const realtimeSetScene = realtime.setScene as unknown as ReturnType<typeof vi.fn>;
    constructWalkaroundMock.mockResolvedValue(realtime);
    constructPathTracerWebGPUMock.mockResolvedValue(converged);
    const nextPositions = new Float32Array([2, 0, 0, 1, 0, 0, 0, 1, 0]);

    const handle = await createProgressiveEngine({
      canvas: makeThrowingConfigureCanvas(new Error('ignored configure failure')),
      scene: sceneWithPrimitive,
    });

    handle.coordinator.updatePrimitive('p', { positions: nextPositions });

    expect(realtime.setScene).toHaveBeenCalledTimes(1);
    expect(converged.setScene).toHaveBeenCalledTimes(1);
    const patched = realtimeSetScene.mock.calls[0]![0] as Scene;
    expect((patched.primitives[0] as { positions: Float32Array }).positions).toBe(nextPositions);
    expect(converged.setScene).toHaveBeenCalledWith(patched);

    handle.dispose();
  });

  it('rejects invalid handoff configuration before requesting an adapter', async () => {
    const requestAdapter = vi.fn();
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter } },
      configurable: true,
    });

    await expect(createProgressiveEngine({
      canvas: makeCanvas(),
      scene,
      stillFramesBeforeHandoff: Number.NaN,
    })).rejects.toThrow(/stillFramesBeforeHandoff/);

    expect(requestAdapter).not.toHaveBeenCalled();
    expect(constructWalkaroundMock).not.toHaveBeenCalled();
    expect(constructPathTracerWebGPUMock).not.toHaveBeenCalled();
  });

  it('rejects malformed scenes before requesting an adapter', async () => {
    const requestAdapter = vi.fn();
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter } },
      configurable: true,
    });
    const invalidScene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'invalid',
        positions: new Float32Array([Number.NaN, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    await expect(createProgressiveEngine({
      canvas: makeCanvas(),
      scene: invalidScene,
    })).rejects.toThrow(/finite/i);

    expect(requestAdapter).not.toHaveBeenCalled();
    expect(constructWalkaroundMock).not.toHaveBeenCalled();
    expect(constructPathTracerWebGPUMock).not.toHaveBeenCalled();
  });

  it('rejects accessor-backed facade options without invoking the getter', async () => {
    const requestAdapter = vi.fn();
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter } },
      configurable: true,
    });
    const getter = vi.fn(() => true);
    const options: Record<string, unknown> = { canvas: makeCanvas(), scene };
    Object.defineProperty(options, 'debug', {
      enumerable: true,
      configurable: true,
      get: getter,
    });

    await expect(createProgressiveEngine(
      options as unknown as Parameters<typeof createProgressiveEngine>[0],
    )).rejects.toThrow(/own data property/i);

    expect(getter).not.toHaveBeenCalled();
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it('labels each non-empty backend option bag for the synthesized sub-build', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: () => 'rgba8unorm',
        },
      },
      configurable: true,
    });
    constructWalkaroundMock.mockResolvedValue(makeEngine('seed-source'));
    constructPathTracerWebGPUMock.mockResolvedValue(makeEngine('seed-sink'));

    const handle = await createProgressiveEngine({
      canvas: makeCanvas(),
      scene,
      realtimeOptions: { gpuSkinning: false },
      convergedOptions: { maxBounces: 3 },
    });

    expect(constructWalkaroundMock.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        advanced: { gpuSkinning: false },
        advancedBackend: 'walkaround-hybrid',
      }),
    );
    expect(constructPathTracerWebGPUMock.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        advanced: { maxBounces: 3 },
        advancedBackend: 'pt-webgpu',
      }),
    );
    handle.dispose();
  });

  it('reports adapter acquisition rejection as a progressive construction error', async () => {
    const adapterError = new Error('adapter request failed');
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => { throw adapterError; }) } },
      configurable: true,
    });
    const onError = vi.fn();

    await expect(createProgressiveEngine({
      canvas: makeCanvas(),
      scene,
      onError,
    })).rejects.toBe(adapterError);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(adapterError, {
      phase: 'create:progressive', backend: 'progressive', recoverable: false,
    });
  });

  it('reports device acquisition rejection before constructing either engine', async () => {
    const deviceError = new Error('device request failed');
    const adapter = {
      limits: {
        maxStorageBuffersPerShaderStage: 64,
        maxStorageTexturesPerShaderStage: 16,
        maxSampledTexturesPerShaderStage: 64,
      },
      requestDevice: vi.fn(async () => { throw deviceError; }),
    } as unknown as GPUAdapter;
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => adapter) } },
      configurable: true,
    });
    const onError = vi.fn();

    await expect(createProgressiveEngine({
      canvas: makeCanvas(),
      scene,
      onError,
    })).rejects.toBe(deviceError);
    expect(onError).toHaveBeenCalledWith(deviceError, {
      phase: 'create:progressive', backend: 'progressive', recoverable: false,
    });
    expect(constructWalkaroundMock).not.toHaveBeenCalled();
    expect(constructPathTracerWebGPUMock).not.toHaveBeenCalled();
  });

  it('forwards construction warning callbacks into both sub-engine option bags', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: () => 'rgba8unorm',
        },
      },
      configurable: true,
    });
    constructWalkaroundMock.mockResolvedValue(makeEngine('seed-source'));
    constructPathTracerWebGPUMock.mockResolvedValue(makeEngine('seed-sink'));
    const onWarning = vi.fn();

    const handle = await createProgressiveEngine({
      canvas: makeCanvas(),
      scene,
      onWarning,
    });

    expect(constructWalkaroundMock.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({ onWarning }),
    );
    expect(constructPathTracerWebGPUMock.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({ onWarning }),
    );
    handle.dispose();
  });

  it('rejects advertised seed support without a callable method and cleans all resources', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: () => 'rgba8unorm',
        },
      },
      configurable: true,
    });
    const realtime = makeEngine('seed-source');
    const converged = makeEngine('seed-sink');
    delete (realtime as Partial<Engine>).getProgressiveSeedTexture;
    constructWalkaroundMock.mockResolvedValue(realtime);
    constructPathTracerWebGPUMock.mockResolvedValue(converged);

    await expect(createProgressiveEngine({
      canvas: makeCanvas(),
      scene,
    })).rejects.toThrow(/callable, advertised progressive seed source/);

    expect(realtime.dispose).toHaveBeenCalledTimes(1);
    expect(converged.dispose).toHaveBeenCalledTimes(1);
    expect(device.destroy).toHaveBeenCalledTimes(1);
  });

  it('preflights realtime NRC device floors before requesting a device', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: () => 'rgba8unorm',
        },
      },
      configurable: true,
    });
    const onError = vi.fn();

    await expect(createProgressiveEngine({
      canvas: makeCanvas(),
      scene,
      realtimeOptions: { nrcEnabled: true },
      onError,
    })).rejects.toThrow(/maxBindGroups/);
    expect(adapter.requestDevice).not.toHaveBeenCalled();
    expect(constructWalkaroundMock).not.toHaveBeenCalled();
    expect(constructPathTracerWebGPUMock).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(expect.any(Error), {
      phase: 'create:progressive', backend: 'progressive', recoverable: false,
    });
  });
});
