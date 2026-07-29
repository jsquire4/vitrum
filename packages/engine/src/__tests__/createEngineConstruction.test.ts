import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { resetGpuDetectionCache, type Engine, type EngineWarning, type Scene } from '@vitrum/core';
import type { SceneAABB } from '../sceneAABB.js';

const hybridFactory = vi.hoisted(() => vi.fn());
const ptFactory = vi.hoisted(() => vi.fn());
const webglFactory = vi.hoisted(() => vi.fn());
const hybridAdvancedValidator = vi.hoisted(() => vi.fn());
const ptAdvancedValidator = vi.hoisted(() => vi.fn());
const webglAdvancedValidator = vi.hoisted(() => vi.fn());
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
  resolveHybridNrcConfig: vi.fn(() => ({ useF16: false })),
  nrcWebGpuRequiredLimitsForConfig: vi.fn(() => ({
    maxStorageBuffersPerShaderStage: 16,
    maxStorageTexturesPerShaderStage: 8,
  })),
  validateHybridEngineAdvancedOptions: hybridAdvancedValidator,
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
  validatePtWebgpuAdvancedOptions: ptAdvancedValidator,
}));

vi.mock('@vitrum/pt-webgl2', () => ({
  createPTEngine_WebGL2: webglFactory,
  validateWebgl2AdvancedOptions: webglAdvancedValidator,
}));

import {
  createEngine,
  constructPathTracerWebGPU,
  constructWalkaround,
  resolveAdvancedForBackend,
  stripOwnershipCriticalKeys,
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
  const webgpuContext = { configure: vi.fn() };
  return {
    width: 64,
    height: 64,
    getContext: vi.fn((kind: string) => kind === 'webgpu' ? webgpuContext : null),
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
    features: new Set(),
    requestDevice: vi.fn(async () => device),
  } as unknown as GPUAdapter;
}

function makeOptions(
  advanced?: CreateEngineOptions['advanced'],
  advancedBackend?: CreateEngineOptions['advancedBackend'],
): CreateEngineOptions {
  return {
    canvas: makeCanvas(),
    scene,
    ...(advanced != null ? { advanced } : {}),
    ...(advancedBackend != null ? { advancedBackend } : {}),
  };
}

describe('createEngine backend construction safety', () => {
  beforeEach(() => {
    resetGpuDetectionCache();
    hybridFactory.mockReset();
    ptFactory.mockReset();
    webglFactory.mockReset();
    hybridAdvancedValidator.mockReset();
    ptAdvancedValidator.mockReset();
    webglAdvancedValidator.mockReset();
    ptRequiredLimits.mockClear();
  });

  afterEach(() => {
    resetGpuDetectionCache();
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
      makeOptions({ tier: 'lite' } as unknown as CreateEngineOptions['advanced'], 'walkaround-hybrid'),
      scene,
      aabb,
      false,
      shared,
    );

    expect(hybridFactory.mock.calls[0]?.[0]?.tier).toBe('full');
    expect(device.destroy).not.toHaveBeenCalled();
  });

  it('guards throwing onAdapterProfile callbacks during walkaround construction', async () => {
    const device = makeDevice();
    const shared: SharedDeviceCtx = {
      adapter: makeAdapter(device),
      device,
      ownsDeviceLifecycle: false,
    };
    hybridFactory.mockResolvedValue(makeEngine());
    const onAdapterProfile = vi.fn(() => {
      throw new Error('host profile callback failed');
    });

    await expect(
      constructWalkaround(
        { ...makeOptions(undefined, 'walkaround-hybrid'), onAdapterProfile },
        scene,
        aabb,
        false,
        shared,
      ),
    ).resolves.toBeDefined();

    expect(onAdapterProfile).toHaveBeenCalledTimes(1);
    expect(hybridFactory).toHaveBeenCalledTimes(1);
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
      makeOptions({ oneEdgeReconnectionReuse: true }, 'pt-webgpu'),
      scene,
    );

    expect(ptRequiredLimits).toHaveBeenCalledWith(adapter, {
      bdpt: false,
      oneEdgeReconnectionReuse: true,
      cwbvhClosest: false,
    });
  });

  it.each(['pt-webgpu', 'pt-webgpu-lite'] as const)(
    'forwards %s profile and active-feature identity through createEngine',
    async (profileId) => {
      const device = makeDevice();
      const adapter = makeAdapter(device);
      Object.defineProperty(globalThis, 'navigator', {
        value: {
          gpu: {
            requestAdapter: vi.fn(async () => adapter),
            getPreferredCanvasFormat: vi.fn(() => 'bgra8unorm'),
          },
        },
        configurable: true,
      });
      const activeFeatures = new Set([
        profileId === 'pt-webgpu' ? 'pt-webgpu-spectral' : 'pt-webgpu-sobol-sampling',
      ] as const);
      const backendEngine = Object.assign(makeEngine(), {
        capabilities: { activeFeatures },
        backendProfileId: profileId,
        profileId,
      });
      ptFactory.mockResolvedValue(backendEngine);

      const result = await createEngine({
        canvas: makeCanvas(),
        scene,
        prefer: 'quality-webgpu',
      });

      expect(result.backendId).toBe('pt-webgpu');
      expect(result.backendProfileId).toBe(profileId);
      expect(result.profileId).toBe(profileId);
      expect(result.capabilities.activeFeatures).toBe(activeFeatures);
    },
  );

  it('keeps auto-selected approximate material fields on walkaround without a routing warning', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: vi.fn(() => 'bgra8unorm'),
        },
      },
      configurable: true,
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const warnings: EngineWarning[] = [];
    const engine = makeEngine();
    hybridFactory.mockResolvedValue(engine);
    const materialRouteScene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'volume-triangle',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.2,
          metallic: 0,
          scatteringCoefficient: 0.12,
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await createEngine({
      canvas: makeCanvas(),
      scene: materialRouteScene,
      prefer: 'auto',
      onWarning: (warning) => warnings.push(warning),
    });

    expect(result.backendId).toBe('walkaround-hybrid');
    expect(hybridFactory).toHaveBeenCalledTimes(1);
    expect(ptFactory).not.toHaveBeenCalled();
    expect(engine.setScene).toHaveBeenCalledWith(materialRouteScene);
    expect(warnings).toEqual([]);
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('keeps approximate material fields on walkaround when a glTF hint has no recommended backend', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: vi.fn(() => 'bgra8unorm'),
        },
      },
      configurable: true,
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const warnings: EngineWarning[] = [];
    const engine = makeEngine();
    hybridFactory.mockResolvedValue(engine);
    const materialRouteScene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'gltf-hint-volume-triangle',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.2,
          metallic: 0,
          scatteringCoefficient: 0.12,
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await createEngine({
      canvas: makeCanvas(),
      scene: materialRouteScene,
      prefer: 'auto',
      gltfAsset: {},
      onWarning: (warning) => warnings.push(warning),
    });

    expect(result.backendId).toBe('walkaround-hybrid');
    expect(hybridFactory).toHaveBeenCalledTimes(1);
    expect(ptFactory).not.toHaveBeenCalled();
    expect(engine.setScene).toHaveBeenCalledWith(materialRouteScene);
    expect(warnings).toEqual([]);
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
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
      makeOptions({ traceTier: 'lite' } as unknown as CreateEngineOptions['advanced'], 'pt-webgpu'),
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
  it('does not reinterpret a backend implementation error as hardware unavailability', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    const requestAdapter = vi.fn(async () => adapter);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter,
          getPreferredCanvasFormat: vi.fn(() => 'bgra8unorm'),
        },
      },
      configurable: true,
    });
    const implementationError = new Error('pipeline invariant failed');
    ptFactory.mockRejectedValue(implementationError);
    webglFactory.mockResolvedValue(makeEngine());
    const canvas = makeCanvas();

    await expect(createEngine({
      canvas,
      scene,
      prefer: 'quality-webgpu',
    })).rejects.toBe(implementationError);

    expect(requestAdapter).toHaveBeenCalledTimes(2);
    expect(webglFactory).not.toHaveBeenCalled();
    expect(canvas.getContext).not.toHaveBeenCalled();
  });

  it('falls back only when adapter acquisition raises typed backend unavailability', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    const requestAdapter = vi.fn()
      .mockResolvedValueOnce(adapter)
      .mockRejectedValueOnce(new Error('adapter disappeared'));
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter,
          getPreferredCanvasFormat: vi.fn(() => 'bgra8unorm'),
        },
      },
      configurable: true,
    });
    const gl = { createFramebuffer: vi.fn() } as unknown as WebGL2RenderingContext;
    const canvas = {
      width: 64,
      height: 64,
      getContext: vi.fn((kind: string) => kind === 'webgl2' ? gl : null),
    } as unknown as HTMLCanvasElement;
    webglFactory.mockResolvedValue(makeEngine());
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const result = await createEngine({
      canvas,
      scene,
      prefer: 'quality-webgpu',
    });

    expect(result.backendId).toBe('pt-webgl2');
    expect(webglFactory).toHaveBeenCalledTimes(1);
    expect(canvas.getContext).toHaveBeenCalledWith('webgl2', {
      antialias: false,
      preserveDrawingBuffer: false,
    });
    warnSpy.mockRestore();
  });

  it('warns when prefer:realtime resolves directly to the converged WebGL2 backend', async () => {
    Object.defineProperty(globalThis, 'navigator', {
      value: {},
      configurable: true,
    });
    const gl = { createFramebuffer: vi.fn() } as unknown as WebGL2RenderingContext;
    const canvas = {
      width: 64,
      height: 64,
      getContext: vi.fn((kind: string) => kind === 'webgl2' ? gl : null),
    } as unknown as HTMLCanvasElement;
    webglFactory.mockResolvedValue(makeEngine());
    const warnings: EngineWarning[] = [];
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const result = await createEngine({
      canvas,
      scene,
      prefer: 'realtime',
      onWarning: (warning) => warnings.push(warning),
    });

    expect(result.backendId).toBe('pt-webgl2');
    expect(hybridFactory).not.toHaveBeenCalled();
    expect(warnings).toContainEqual(expect.objectContaining({
      code: 'createEngine.realtime-unavailable-fallback',
      phase: 'fallback',
      details: expect.objectContaining({
        preferredBackend: 'walkaround-hybrid',
        resolvedBackend: 'pt-webgl2',
        reason: 'webgpu-unavailable',
      }),
    }));
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });

  it('rejects ambiguous advanced options before GPU detection', async () => {
    const requestAdapter = vi.fn();
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter } },
      configurable: true,
    });

    await expect(createEngine({
      canvas: makeCanvas(),
      scene,
      advanced: { maxBounces: 4 },
    })).rejects.toThrow(/requires advancedBackend/i);

    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it('runs backend option validation before GPU detection', async () => {
    const requestAdapter = vi.fn();
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter } },
      configurable: true,
    });
    ptAdvancedValidator.mockImplementation(() => {
      throw new RangeError('invalid pt-webgpu option');
    });

    await expect(createEngine({
      canvas: makeCanvas(),
      scene,
      advancedByBackend: { 'pt-webgpu': { maxBounces: 999 } },
    })).rejects.toThrow(/invalid pt-webgpu option/);

    expect(ptAdvancedValidator).toHaveBeenCalledWith({ maxBounces: 999 });
    expect(requestAdapter).not.toHaveBeenCalled();
  });
});

describe('advanced ownership guard', () => {
  it('rejects device in an advanced bag without warning or mutation', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const fakeDevice = {} as GPUDevice;
    const advanced = { device: fakeDevice, maxBounces: 4 };
    expect(() => stripOwnershipCriticalKeys(
      advanced as unknown as Record<string, unknown>,
      'walkaround-hybrid',
    )).toThrow(/ownership-critical key.*device/i);
    expect(advanced).toEqual({ device: fakeDevice, maxBounces: 4 });
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('rejects canvas and context keys too', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const advanced = { canvas: {}, context: {}, traceTier: 'full' };
    expect(() => stripOwnershipCriticalKeys(
      advanced as unknown as Record<string, unknown>,
      'pt-webgpu',
    )).toThrow(/canvas, context/);
    expect(warnSpy).not.toHaveBeenCalled();
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

  it('walkaround constructor rejects an injected device before adapter acquisition', async () => {
    hybridFactory.mockReset();
    const device = makeDevice();
    const adapter = makeAdapter(device);
    const requestAdapter = vi.fn(async () => adapter);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter } },
      configurable: true,
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const impostor = makeDevice(); // a DIFFERENT device than the factory-minted one
    hybridFactory.mockResolvedValue(makeEngine());

    await expect(constructWalkaround(
      makeOptions({ device: impostor }, 'walkaround-hybrid'),
      scene,
      aabb,
      false,
    )).rejects.toThrow(/ownership-critical key.*device/i);

    expect(requestAdapter).not.toHaveBeenCalled();
    expect(hybridFactory).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('pt-webgpu constructor rejects an injected device before adapter acquisition', async () => {
    ptFactory.mockReset();
    const device = makeDevice();
    const adapter = makeAdapter(device);
    const requestAdapter = vi.fn(async () => adapter);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter } },
      configurable: true,
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const impostor = makeDevice();
    ptFactory.mockResolvedValue(makeEngine());

    await expect(constructPathTracerWebGPU(
      makeOptions({ device: impostor }, 'pt-webgpu'),
      scene,
    )).rejects.toThrow(/ownership-critical key.*device/i);

    expect(requestAdapter).not.toHaveBeenCalled();
    expect(ptFactory).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});

describe('H31 fix — backend-scoped advanced resolution', () => {
  beforeEach(() => {
    ptAdvancedValidator.mockReset();
  });

  it('selects the matching advancedByBackend bag for the resolved backend', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const advanced = resolveAdvancedForBackend(
      {
        prefer: 'auto',
        advancedByBackend: {
          'walkaround-hybrid': { qualityTier: 'medium' },
          'pt-webgpu': { restirPtReuse: true },
          'pt-webgl2': { maxBounces: 12 },
        },
      },
      'pt-webgpu',
    );
    expect(advanced).toEqual({ restirPtReuse: true });
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('ignores legacy advanced when it is tagged for a different backend', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const advanced = resolveAdvancedForBackend(
      {
        advanced: { maxBounces: 8 },
        advancedBackend: 'pt-webgl2',
        onWarning: (w) => structured.push(w),
      },
      'walkaround-hybrid',
    );
    expect(advanced).toBeUndefined();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(structured.some((w) =>
      w.code === 'createEngine.advanced-target-backend-mismatch' &&
      w.details?.advancedBackend === 'pt-webgl2' &&
      w.details?.selectedBackend === 'walkaround-hybrid',
    )).toBe(true);
    warnSpy.mockRestore();
  });

  it('rejects legacy advanced under auto selection without a target', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const advanced = { maxBounces: 8 };
    expect(() => resolveAdvancedForBackend(
      {
        prefer: 'auto',
        advanced,
      },
      'pt-webgl2',
    )).toThrow(/require advancedBackend/i);
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('rejects advancedByBackend combined with legacy advanced', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(() => resolveAdvancedForBackend(
      {
        advanced: { maxBounces: 8 },
        advancedByBackend: {
          'walkaround-hybrid': { qualityTier: 'low' },
        },
      },
      'walkaround-hybrid',
    )).toThrow(/cannot be combined/i);
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('uses backend-scoped advanced options in pt-webgpu device-limit negotiation', async () => {
    ptFactory.mockReset();
    ptRequiredLimits.mockClear();
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: vi.fn(async () => adapter) } },
      configurable: true,
    });
    ptFactory.mockResolvedValue(makeEngine());

    await constructPathTracerWebGPU(
      {
        ...makeOptions(),
        advancedByBackend: {
          'walkaround-hybrid': { qualityTier: 'medium' },
          'pt-webgpu': { oneEdgeReconnectionReuse: true },
        },
      },
      scene,
    );

    expect(ptRequiredLimits).toHaveBeenCalledWith(adapter, {
      bdpt: false,
      oneEdgeReconnectionReuse: true,
      cwbvhClosest: false,
    });
    expect(ptFactory.mock.calls[0]?.[0]?.oneEdgeReconnectionReuse).toBe(true);
    Object.defineProperty(globalThis, 'navigator', { value: ORIG_NAVIGATOR, configurable: true });
  });
});
