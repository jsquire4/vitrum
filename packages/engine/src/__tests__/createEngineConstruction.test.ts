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
  stripOwnershipCriticalKeys,
  warnCrossBackendAdvanced,
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
      makeOptions({ restirPtReuse: true }),
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

describe('Bug3 fix — advanced.device ownership guard (stripOwnershipCriticalKeys)', () => {
  it('strips device from advanced bag and emits a console.warn', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const fakeDevice = {} as GPUDevice;
    const advanced = { device: fakeDevice, maxBounces: 4 };
    const stripped = stripOwnershipCriticalKeys(advanced as unknown as Record<string, unknown>, 'walkaround-hybrid');
    expect((stripped as Record<string, unknown>).device).toBeUndefined();
    // Non-ownership keys are preserved.
    expect((stripped as Record<string, unknown>).maxBounces).toBe(4);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0]?.[0]).toMatch(/device/);
    expect(warnSpy.mock.calls[0]?.[0]).toMatch(/walkaround-hybrid/);
    warnSpy.mockRestore();
  });

  it('strips canvas and context keys too', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const advanced = { canvas: {}, context: {}, traceTier: 'full' };
    const stripped = stripOwnershipCriticalKeys(advanced as unknown as Record<string, unknown>, 'pt-webgpu');
    expect((stripped as Record<string, unknown>).canvas).toBeUndefined();
    expect((stripped as Record<string, unknown>).context).toBeUndefined();
    expect((stripped as Record<string, unknown>).traceTier).toBe('full');
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });

  it('returns the bag unchanged (no warn) when no ownership keys are present', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const advanced = { maxBounces: 8, spectral: true };
    const stripped = stripOwnershipCriticalKeys(advanced as unknown as Record<string, unknown>, 'pt-webgpu');
    expect(stripped).toEqual(advanced);
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('returns empty object for undefined advanced', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const stripped = stripOwnershipCriticalKeys(undefined, 'pt-webgpu');
    expect(stripped).toEqual({});
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('walkaround constructor strips device from advanced before passing to factory', async () => {
    hybridFactory.mockReset();
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => adapter) } },
      configurable: true,
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const impostor = makeDevice(); // a DIFFERENT device than the factory-minted one
    hybridFactory.mockResolvedValue(makeEngine());

    await constructWalkaround(
      makeOptions({ device: impostor }),
      scene,
      aabb,
      false,
    );

    // calls[0] is safe — hybridFactory was reset at the start of this test.
    const passedDevice = hybridFactory.mock.calls[0]?.[0]?.device;
    expect(passedDevice).toBe(device);
    expect(passedDevice).not.toBe(impostor);
    // A warn must have been emitted.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });

  it('pt-webgpu constructor strips device from advanced before passing to factory', async () => {
    ptFactory.mockReset();
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => adapter) } },
      configurable: true,
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const impostor = makeDevice();
    ptFactory.mockResolvedValue(makeEngine());

    await constructPathTracerWebGPU(
      makeOptions({ device: impostor }),
      scene,
    );

    // calls[0] is safe — ptFactory was reset at the start of this test.
    const passedDevice = ptFactory.mock.calls[0]?.[0]?.device;
    expect(passedDevice).toBe(device);
    expect(passedDevice).not.toBe(impostor);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });
});

describe('Bug4 fix — cross-backend advanced fallback warning (warnCrossBackendAdvanced)', () => {
  it('emits a console.warn when advanced is non-empty and backends differ', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    warnCrossBackendAdvanced(
      { maxBounces: 4 },
      'walkaround-hybrid',
      'pt-webgpu',
    );
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = warnSpy.mock.calls[0]?.[0] as string;
    expect(msg).toMatch(/walkaround-hybrid/);
    expect(msg).toMatch(/pt-webgpu/);
    expect(msg).toMatch(/maxBounces/);
    warnSpy.mockRestore();
  });

  it('does NOT warn when advanced is null/undefined', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    warnCrossBackendAdvanced(undefined, 'walkaround-hybrid', 'pt-webgl2');
    warnCrossBackendAdvanced(null as never, 'walkaround-hybrid', 'pt-webgl2');
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('does NOT warn when advanced is an empty object', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    warnCrossBackendAdvanced({}, 'pt-webgpu', 'pt-webgl2');
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('does NOT warn when advanced only has undefined-valued keys', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    warnCrossBackendAdvanced({ maxBounces: undefined } as unknown as CreateEngineOptions['advanced'], 'pt-webgpu', 'pt-webgl2');
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});
