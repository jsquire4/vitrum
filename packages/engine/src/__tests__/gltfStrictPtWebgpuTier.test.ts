import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => {
  const scene = Object.freeze({ primitives: [], emitters: [] });
  const state = {
    selectedBackend: 'pt-webgpu' as 'pt-webgpu' | 'pt-webgl2' | 'walkaround-hybrid',
  };
  const asset = Object.freeze({
    scene,
    gltf: Object.freeze({ asset: Object.freeze({ version: '2.0' }) }),
    sceneIndex: 0,
    warnings: Object.freeze([]),
    animations: Object.freeze([]),
    animationTargets: Object.freeze([]),
    textureDecodeReport: Object.freeze({ textures: Object.freeze([]), warnings: Object.freeze([]) }),
  });
  return {
    state,
    asset,
    createEngine: vi.fn(async () => ({
      backendId: 'pt-webgpu',
      dispose: vi.fn(),
      renderFrame: vi.fn(),
      setScene: vi.fn(),
    })),
    loadGltfForEngine: vi.fn(async (_input: unknown, options: Record<string, unknown>) => {
      const factory = options['createEngine'] as
        | ((args: { scene: unknown; backend: string; asset: unknown; options: object }) => Promise<unknown>)
        | undefined;
      const factoryOptions = (options['engineOptions'] as object | undefined) ?? {};
      const engine = factory == null
        ? undefined
        : await factory({
          scene,
          backend: state.selectedBackend,
          asset,
          options: factoryOptions,
        });
      return {
        asset,
        backend: state.selectedBackend,
        ...(engine != null ? { engine } : {}),
        controller: { warnings: [], attachEngine: vi.fn() },
        attached: engine != null,
        textureDecodeReport: asset.textureDecodeReport,
        warnings: [],
      };
    }),
    probeAdapterProfile: vi.fn(async () => ({
      hasWebGPU: true,
      hybridCapable: true,
      hybridLiteCapable: true,
      ptWebgpuTier: 'lite',
      maxStorageBuffersPerStage: 16,
      maxStorageTexturesPerStage: 8,
      isSoftwareAdapter: false,
      adapterKind: 'hardware',
      hasWebGL2: true,
      recommendedRealtimeTier: 'ultra',
      recommendedHeroBackend: 'pt-webgpu-lite',
      limits: Object.freeze({}),
    })),
  };
});

vi.mock('@vitrum/gltf-adapter', () => ({
  loadGltfForEngine: mocks.loadGltfForEngine,
}));

vi.mock('../adapterProfile.js', () => ({
  probeAdapterProfile: mocks.probeAdapterProfile,
}));

vi.mock('../createEngine.js', () => ({
  createEngine: mocks.createEngine,
}));

import { loadGltfWithEngine } from '../gltf.js';

describe('loadGltfWithEngine strict pt-webgpu tier guard', () => {
  beforeEach(() => {
    mocks.state.selectedBackend = 'pt-webgpu';
    mocks.createEngine.mockClear();
    mocks.loadGltfForEngine.mockClear();
    mocks.probeAdapterProfile.mockClear();
    mocks.probeAdapterProfile.mockResolvedValue({
      hasWebGPU: true,
      hybridCapable: true,
      hybridLiteCapable: true,
      ptWebgpuTier: 'lite',
      maxStorageBuffersPerStage: 16,
      maxStorageTexturesPerStage: 8,
      isSoftwareAdapter: false,
      adapterKind: 'hardware',
      hasWebGL2: true,
      recommendedRealtimeTier: 'ultra',
      recommendedHeroBackend: 'pt-webgpu-lite',
      limits: Object.freeze({}),
    });
  });

  it('rejects reject-degraded glTF loads when selected pt-webgpu is lite tier', async () => {
    await expect(
      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-degraded' }),
    ).rejects.toThrow(/pt-webgpu.*lite.*glTF strict mode/);

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).not.toHaveBeenCalled();
  });

  it('allows non-strict pt-webgpu glTF loads to proceed on lite tier', async () => {
    await expect(
      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-unsupported' }),
    ).resolves.toMatchObject({ backend: 'pt-webgpu', attached: true });

    expect(mocks.probeAdapterProfile).not.toHaveBeenCalled();
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
  });

  it('allows reject-degraded pt-webgpu glTF loads on full tier adapters', async () => {
    mocks.probeAdapterProfile.mockResolvedValueOnce({
      hasWebGPU: true,
      hybridCapable: true,
      hybridLiteCapable: true,
      ptWebgpuTier: 'full',
      maxStorageBuffersPerStage: 28,
      maxStorageTexturesPerStage: 5,
      isSoftwareAdapter: false,
      adapterKind: 'hardware',
      hasWebGL2: true,
      recommendedRealtimeTier: 'ultra',
      recommendedHeroBackend: 'pt-webgpu-full',
      limits: Object.freeze({}),
    });

    await expect(
      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-degraded' }),
    ).resolves.toMatchObject({ backend: 'pt-webgpu', attached: true });

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
  });

  it('does not probe adapter tier for strict non-pt-webgpu selections', async () => {
    mocks.state.selectedBackend = 'pt-webgl2';

    await expect(
      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-degraded' }),
    ).resolves.toMatchObject({ backend: 'pt-webgl2', attached: true });

    expect(mocks.probeAdapterProfile).not.toHaveBeenCalled();
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
  });
});
