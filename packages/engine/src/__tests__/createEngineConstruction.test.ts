import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Engine, Scene } from '@vitrum/core';
import type { SceneAABB } from '../sceneAABB.js';

const hybridFactory = vi.hoisted(() => vi.fn());
const ptFactory = vi.hoisted(() => vi.fn());
const ptRequiredLimits = vi.hoisted(() => vi.fn(() => ({
  maxStorageBuffersPerShaderStage: 28,
  maxStorageTexturesPerShaderStage: 5,
})));

vi.mock('@vitrum/walkaround-hybrid', () => ({
  HYBRID_WEBGPU_REQUIRED_LIMITS: {
    maxStorageBuffersPerShaderStage: 16,
    maxStorageTexturesPerShaderStage: 8,
  },
  HYBRID_LITE_LIMITS: {
    maxStorageBuffersPerShaderStage: 10,
    maxStorageTexturesPerShaderStage: 6,
  },
  createWalkaroundEngine_Hybrid: hybridFactory,
}));

vi.mock('@vitrum/pt-webgpu', () => ({
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE: 28,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE: 32,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE: 5,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE: 8,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE: 4,
  selectPtWebgpuTraceTier: (limits: Record<string, number>) =>
    (limits.maxStorageBuffersPerShaderStage ?? 0) >= 28 &&
    (limits.maxStorageTexturesPerShaderStage ?? 0) >= 5
      ? 'full'
      : 'lite',
  ptWebgpuRequiredLimitsForAdapter: ptRequiredLimits,
  createPTEngine_WebGPU: ptFactory,
}));

import {
  constructPathTracerWebGPU,
  constructWalkaround,
  type CreateEngineOptions,
  type SharedDeviceCtx,
} from '../createEngine.js';

const scene: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };
const ORIG_NAVIGATOR = globalThis.navigator;
const aabb: SceneAABB = {
  min: [-0.5, -0.5, -0.5],
  max: [0.5, 0.5, 0.5],
  center: [0, 0, 0],
  extent: [1, 1, 1],
  diagonal: 1,
  triangleCount: 0,
};

function makeCanvas(): HTMLCanvasElement {
  return {
    width: 64,
    height: 64,
    getContext: vi.fn(() => null),
  } as unknown as HTMLCanvasElement;
}

function makeEngine(setScene: (scene: Scene) => void = vi.fn()): Engine {
  return {
    state: 'ready',
    capabilities: {},
    setScene,
    renderFrame: vi.fn(),
    dispose: vi.fn(),
  } as unknown as Engine;
}

function makeDevice(): GPUDevice {
  return {
    destroy: vi.fn(),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
    },
  } as unknown as GPUDevice;
}

function makeAdapter(device: GPUDevice): GPUAdapter {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
    },
    requestDevice: vi.fn(async () => device),
  } as unknown as GPUAdapter;
}

function makeOptions(advanced?: CreateEngineOptions['advanced']): CreateEngineOptions {
  return {
    canvas: makeCanvas(),
    scene,
    ...(advanced != null ? { advanced } : {}),
  };
}

describe('createEngine backend construction safety', () => {
  beforeEach(() => {
    hybridFactory.mockReset();
    ptFactory.mockReset();
    ptRequiredLimits.mockClear();
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', { value: ORIG_NAVIGATOR, configurable: true });
  });

  it('destroys an owned WebGPU device when walkaround setScene throws', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => adapter) } },
      configurable: true,
    });
    const engine = makeEngine(vi.fn((_scene: Scene) => { throw new Error('scene failed'); }));
    hybridFactory.mockResolvedValue(engine);

    await expect(
      constructWalkaround(makeOptions(), scene, aabb, false),
    ).rejects.toThrow(/scene failed/);

    expect(engine.dispose).toHaveBeenCalledTimes(1);
    expect(device.destroy).toHaveBeenCalledTimes(1);
  });

  it('forces full walkaround tier on the shared-device progressive path', async () => {
    const device = makeDevice();
    const shared: SharedDeviceCtx = {
      adapter: makeAdapter(device),
      device,
      ownsDeviceLifecycle: false,
    };
    hybridFactory.mockResolvedValue(makeEngine());

    await constructWalkaround(
      makeOptions({ tier: 'lite' } as unknown as CreateEngineOptions['advanced']),
      scene,
      aabb,
      false,
      shared,
    );

    expect(hybridFactory.mock.calls[0]?.[0]?.tier).toBe('full');
    expect(device.destroy).not.toHaveBeenCalled();
  });

  it('passes ReSTIR-PT reuse into pt-webgpu device-limit negotiation', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => adapter) } },
      configurable: true,
    });
    ptFactory.mockResolvedValue(makeEngine());

    await constructPathTracerWebGPU(
      makeOptions({ restirPtReuse: true } as unknown as CreateEngineOptions['advanced']),
      scene,
    );

    expect(ptRequiredLimits).toHaveBeenCalledWith(adapter, { restirPtReuse: true });
  });

  it('forces full pt-webgpu trace tier on the shared-device progressive path', async () => {
    const device = makeDevice();
    const shared: SharedDeviceCtx = {
      adapter: makeAdapter(device),
      device,
      ownsDeviceLifecycle: false,
    };
    ptFactory.mockResolvedValue(makeEngine());

    await constructPathTracerWebGPU(
      makeOptions({ traceTier: 'lite' } as unknown as CreateEngineOptions['advanced']),
      scene,
      shared,
    );

    expect(ptFactory.mock.calls[0]?.[0]?.traceTier).toBe('full');
    expect(device.destroy).not.toHaveBeenCalled();
  });

  it('destroys an owned WebGPU device when pt-webgpu setScene throws', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => adapter) } },
      configurable: true,
    });
    const engine = makeEngine(vi.fn((_scene: Scene) => { throw new Error('pt scene failed'); }));
    ptFactory.mockResolvedValue(engine);

    await expect(
      constructPathTracerWebGPU(makeOptions(), scene),
    ).rejects.toThrow(/pt scene failed/);

    expect(engine.dispose).toHaveBeenCalledTimes(1);
    expect(device.destroy).toHaveBeenCalledTimes(1);
  });
});
