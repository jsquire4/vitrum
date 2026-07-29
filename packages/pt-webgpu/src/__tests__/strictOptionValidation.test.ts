import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU, type PTEngineWebGPUOptions } from '../index.js';

function validationHarness() {
  const createBuffer = vi.fn();
  const device = {
    createCommandEncoder: vi.fn(),
    createBuffer,
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
  return { device, createBuffer };
}

function liteValidationHarness() {
  const harness = validationHarness();
  Object.assign((harness.device).limits, {
    maxStorageBuffersPerShaderStage: 8,
    maxStorageTexturesPerShaderStage: 4,
  });
  return harness;
}

describe('pt-webgpu strict construction options', () => {
  it.each([
    ['traceTier', { traceTier: 'automatic' }, /traceTier is unsupported/],
    ['denoiser', { denoiser: 'magic' }, /denoiser is unsupported/],
    ['causticStrategy', { causticStrategy: 'mlt' }, /causticStrategy is unsupported/],
    ['sampling', { sampling: 'halton' }, /sampling is unsupported/],
    ['bvhTraversal', { bvhTraversal: 'rope' }, /bvhTraversal is unsupported/],
  ] as const)('rejects invalid %s enum values before allocation', async (_label, invalid, error) => {
    const { device, createBuffer } = validationHarness();
    await expect(createPTEngine_WebGPU({
      device,
      ...invalid,
    } as unknown as PTEngineWebGPUOptions)).rejects.toThrow(error);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it.each([
    ['top level', { spectrall: true }, /options contains unknown key.*spectrall/],
    ['bdptOptions', { bdpt: true, bdptOptions: { maxLightBounce: 2 } }, /bdptOptions contains unknown key/],
    ['causticOptions', {
      causticStrategy: 'manifold-nee',
      causticOptions: { maxIterations: 8 },
    }, /causticOptions contains unknown key/],
    ['restirPtReuseOptions', {
      restirPtReuse: true,
      restirPtReuseOptions: { temporalClamp: 8 },
    }, /restirPtReuseOptions contains unknown key/],
    ['oidn', {
      denoiser: 'oidn-final',
      oidn: { modelUrl: '/model.onnx', provider: 'wasm' },
    }, /oidn contains unknown key/],
    ['extensions', {
      extensions: { 'vitrum.ptWebgpu.future': true },
    }, /extensions contains unsupported key/],
  ] as const)('rejects unknown %s keys before allocation', async (_label, invalid, error) => {
    const { device, createBuffer } = validationHarness();
    await expect(createPTEngine_WebGPU({
      device,
      ...invalid,
    } as unknown as PTEngineWebGPUOptions)).rejects.toThrow(error);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('rejects accessor-backed top-level and nested options without invoking getters', async () => {
    const { device, createBuffer } = validationHarness();
    const topGetter = vi.fn(() => true);
    const top = { device } as Record<string, unknown>;
    Object.defineProperty(top, 'spectral', {
      enumerable: true,
      get: topGetter,
    });
    await expect(
      createPTEngine_WebGPU(top as unknown as PTEngineWebGPUOptions),
    ).rejects.toThrow(/spectral must be an enumerable own data property/);
    expect(topGetter).not.toHaveBeenCalled();

    const nestedGetter = vi.fn(() => 8);
    const restirPtReuseOptions = {} as Record<string, unknown>;
    Object.defineProperty(restirPtReuseOptions, 'mClamp', {
      enumerable: true,
      get: nestedGetter,
    });
    await expect(createPTEngine_WebGPU({
      device,
      restirPtReuse: true,
      restirPtReuseOptions,
    })).rejects.toThrow(
      /restirPtReuseOptions.mClamp must be an enumerable own data property/,
    );
    expect(nestedGetter).not.toHaveBeenCalled();
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it.each([
    ['bdptOptions', { bdptOptions: { maxLightBounces: 2 } }, /requires bdpt:true/],
    ['causticOptions', {
      causticStrategy: 'photon-map',
      causticOptions: { mneeMaxIterations: 8 },
    }, /requires causticStrategy="manifold-nee"/],
    ['restirPtReuseOptions', {
      restirPtReuseOptions: { mClamp: 8 },
    }, /requires oneEdgeReconnectionReuse:true/],
    ['oidn', {
      denoiser: 'none',
      oidn: { modelUrl: '/model.onnx' },
    }, /require denoiser:'oidn-final' or 'auto'/],
  ] as const)('rejects non-empty ignored %s tuning', async (_label, invalid, error) => {
    const { device, createBuffer } = validationHarness();
    await expect(createPTEngine_WebGPU({
      device,
      ...invalid,
    })).rejects.toThrow(error);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('accepts empty tuning bags without treating them as enabled work', async () => {
    const { device } = validationHarness();
    const engine = await createPTEngine_WebGPU({
      device,
      bdptOptions: {},
      causticOptions: {},
      restirPtReuseOptions: {},
      extensions: {},
    });
    engine.dispose();
  });

  it.each([
    ['manifold caustics', { causticStrategy: 'manifold-nee' }, /requires traceTier "full"/],
    ['photon caustics', { causticStrategy: 'photon-map' }, /requires traceTier "full"/],
    ['light tree', { lightTreeImportanceSampling: true }, /requires traceTier "full"/],
    ['camera-visible mesh emitters', { cameraVisibleEmitters: true }, /requires traceTier "full"/],
  ] as const)('rejects explicit lite-tier %s requests instead of ignoring them', async (_label, options, error) => {
    const { device, createBuffer } = liteValidationHarness();
    await expect(createPTEngine_WebGPU({
      device,
      traceTier: 'lite',
      ...options,
    })).rejects.toThrow(error);
    expect(createBuffer).not.toHaveBeenCalled();
  });

  it('rejects forced lite on an adapter below the lite binding floor', async () => {
    const { device, createBuffer } = liteValidationHarness();
    Object.assign((device).limits, {
      maxStorageBuffersPerShaderStage: 7,
      maxStorageTexturesPerShaderStage: 3,
    });
    await expect(createPTEngine_WebGPU({ device, traceTier: 'lite' })).rejects.toThrow(
      /below the lite tier/,
    );
    expect(createBuffer).not.toHaveBeenCalled();
  });
});
