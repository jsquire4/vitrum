import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => {
  const scene = Object.freeze({ primitives: [], emitters: [] });
  const textureDecodeReport = Object.freeze({
    mapCount: 0,
    uniqueHandleCount: 0,
    rawImageCount: 0,
    imageBitmapCount: 0,
    opaqueHandleCount: 0,
    cpuReadableCount: 0,
    rawImageRefs: Object.freeze([]),
    imageBitmapRefs: Object.freeze([]),
    entries: Object.freeze([]),
  });
  const unsupportedLiteIssue = Object.freeze({
    category: 'material',
    name: 'baseColorMap',
    support: 'unsupported',
    path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
    message: 'pt-webgpu lite lacks full-tier material texture bindings.',
  });
  const approximateLiteIssue = Object.freeze({
    category: 'material',
    name: 'doubleSided',
    support: 'approximate',
    path: 'materials[0].doubleSided',
    message: 'double-sided shading is approximate on this profile.',
  });
	  const khrMeshoptHookIssue = Object.freeze({
	    category: 'extension',
	    name: 'KHR_meshopt_compression',
	    support: 'requires-hook',
	    path: 'bufferViews[0].extensions.KHR_meshopt_compression',
	    message: 'KHR_meshopt_compression requires a host meshopt decoder.',
	  });
	  const textureReadinessIssue = Object.freeze({
	    category: 'texture',
	    name: 'texture-readiness:pt-webgpu',
	    support: 'requires-hook',
	    path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
	    message: 'pt-webgpu requires host-confirmed opaque texture handles.',
	  });
  const state = {
    selectedBackend: 'pt-webgpu' as 'pt-webgpu' | 'pt-webgl2' | 'walkaround-hybrid',
    liteIssues: [unsupportedLiteIssue] as readonly object[],
  };
  const makeCompatibility = () => Object.freeze([
    Object.freeze({
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu',
      traceTier: 'full',
      unsupportedCount: 0,
      approximateCount: 0,
      nativeCount: 1,
      requiresHookCount: 0,
      issues: Object.freeze([]),
      isCompatible: true,
    }),
    Object.freeze({
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu-lite',
      traceTier: 'lite',
      unsupportedCount: state.liteIssues.filter((issue) =>
        (issue as { support?: string }).support === 'unsupported'
      ).length,
      approximateCount: state.liteIssues.filter((issue) =>
        (issue as { support?: string }).support === 'approximate'
      ).length,
      nativeCount: 0,
      requiresHookCount: state.liteIssues.filter((issue) =>
        (issue as { support?: string }).support === 'requires-hook'
      ).length,
      issues: Object.freeze([...state.liteIssues]),
      isCompatible: !state.liteIssues.some((issue) => (issue as { support?: string }).support === 'unsupported'),
    }),
    Object.freeze({
      backend: 'pt-webgl2',
      profileId: 'pt-webgl2',
      unsupportedCount: 0,
      approximateCount: 0,
      nativeCount: 1,
      requiresHookCount: 0,
      issues: Object.freeze([]),
      isCompatible: true,
    }),
  ]);
  const makeAsset = () => Object.freeze({
    scene,
    gltf: Object.freeze({ asset: Object.freeze({ version: '2.0' }) }),
    sceneIndex: 0,
    warnings: Object.freeze([]),
    diagnostics: Object.freeze([]),
    animations: Object.freeze([]),
    animationTargets: Object.freeze([]),
    backendCompatibility: makeCompatibility(),
    recommendedBackend: makeCompatibility()[0],
    textureDecodeReport,
  });
  const attachEngine = vi.fn();
  return {
    state,
	    unsupportedLiteIssue,
	    approximateLiteIssue,
	    khrMeshoptHookIssue,
	    textureReadinessIssue,
	    makeAsset,
    createEngine: vi.fn(async () => ({
      backendId: state.selectedBackend,
      dispose: vi.fn(),
      renderFrame: vi.fn(),
      setScene: vi.fn(),
    })),
    attachEngine,
    loadGltfForEngine: vi.fn(async (_input: unknown, options: Record<string, unknown>) => {
      const factory = options['createEngine'] as
        | ((args: { scene: unknown; backend: string; asset: unknown; options: object }) => Promise<unknown>)
        | undefined;
      const factoryOptions = (options['engineOptions'] as object | undefined) ?? {};
      const asset = makeAsset();
      const engine = factory == null
        ? undefined
        : await factory({
          scene,
          backend: state.selectedBackend,
          asset,
          options: factoryOptions,
        });
      const actualBackend = (
        engine as { readonly backendId?: 'pt-webgpu' | 'pt-webgl2' | 'walkaround-hybrid' } | undefined
      )?.backendId ?? state.selectedBackend;
      return {
        asset,
        backend: actualBackend,
        profileId: actualBackend === 'pt-webgpu' ? 'pt-webgpu' : actualBackend,
        ...(engine != null ? { engine } : {}),
        controller: { warnings: [], attachEngine },
        attached: engine != null,
        textureDecodeReport: asset.textureDecodeReport,
        decodedTextureCount: 0,
        unchangedTextureCount: 0,
        textureDecodeDiagnostics: [],
        textureDecodeWarnings: [],
        warnings: [],
        diagnostics: [],
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
  GltfCompatibilityError: class GltfCompatibilityError extends Error {
    readonly code: string;
    readonly backend?: string;
    readonly profileId?: string;
    readonly runtimeProfile?: string;
    readonly compatibilityMode?: string;
    readonly label?: string;
    readonly failures: readonly string[];

    constructor(init: {
      readonly code: string;
      readonly message: string;
      readonly backend?: string;
      readonly profileId?: string;
      readonly runtimeProfile?: string;
      readonly compatibilityMode?: string;
      readonly label?: string;
      readonly failures?: readonly string[];
    }) {
      super(init.message);
      this.name = 'GltfCompatibilityError';
      this.code = init.code;
      if (init.backend !== undefined) this.backend = init.backend;
      if (init.profileId !== undefined) this.profileId = init.profileId;
      if (init.runtimeProfile !== undefined) this.runtimeProfile = init.runtimeProfile;
      if (init.compatibilityMode !== undefined) this.compatibilityMode = init.compatibilityMode;
      if (init.label !== undefined) this.label = init.label;
      this.failures = [...(init.failures ?? [])];
    }
  },
  loadGltfForEngine: mocks.loadGltfForEngine,
}));

vi.mock('../adapterProfile.js', () => ({
  probeAdapterProfile: mocks.probeAdapterProfile,
}));

vi.mock('../createEngine.js', () => ({
  createEngine: mocks.createEngine,
}));

import { loadGltfWithEngine } from '../gltf.js';
import type { EngineWithBackendId } from '../createEngine.js';

function makeExistingPtWebgpuEngine(): EngineWithBackendId {
  return {
    backendId: 'pt-webgpu',
    state: 'ready',
    capabilities: {} as EngineWithBackendId['capabilities'],
    dispose: vi.fn(),
    renderFrame: vi.fn(),
    setScene: vi.fn(),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
  };
}

describe('loadGltfWithEngine strict pt-webgpu tier guard', () => {
  beforeEach(() => {
    mocks.state.selectedBackend = 'pt-webgpu';
    mocks.state.liteIssues = [mocks.unsupportedLiteIssue];
    mocks.createEngine.mockClear();
    mocks.attachEngine.mockClear();
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
    ).rejects.toThrow(/pt-webgpu.*lite.*reject-degraded.*baseColorMap=unsupported/);

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).not.toHaveBeenCalled();
  });

	  it('rejects reject-unsupported glTF loads when the runtime lite profile has unsupported rows', async () => {
	    await expect(
	      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-unsupported' }),
	    ).rejects.toThrow(/pt-webgpu.*lite.*reject-unsupported.*baseColorMap=unsupported/);

	    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
	    expect(mocks.createEngine).not.toHaveBeenCalled();
	  });

	  it('rejects strict pt-webgpu loads when the probed adapter is below lite tier', async () => {
	    mocks.probeAdapterProfile.mockResolvedValueOnce({
	      hasWebGPU: true,
	      hybridCapable: false,
	      hybridLiteCapable: false,
	      ptWebgpuTier: 'none',
	      maxStorageBuffersPerStage: 4,
	      maxStorageTexturesPerStage: 2,
	      isSoftwareAdapter: false,
	      adapterKind: 'hardware',
	      hasWebGL2: true,
	      recommendedRealtimeTier: 'unavailable',
	      recommendedHeroBackend: 'pt-webgl2',
	      limits: Object.freeze({}),
	    });

	    await expect(
	      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-unsupported' }),
	    ).rejects.toMatchObject({
	      code: 'GLTF_COMPATIBILITY_REJECTED',
	      backend: 'pt-webgpu',
	      failures: ['runtime:pt-webgpu=unsupported at adapterProfile.ptWebgpuTier'],
	    });

	    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
	    expect(mocks.createEngine).not.toHaveBeenCalled();
	  });

	  it('honors explicit pt-webgpu-lite runtimeProfile without probing', async () => {
	    await expect(
      loadGltfWithEngine('asset.glb', {
        runtimeProfile: 'pt-webgpu-lite',
        compatibilityMode: 'reject-unsupported',
      }),
    ).rejects.toMatchObject({
      code: 'GLTF_COMPATIBILITY_REJECTED',
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu-lite',
      runtimeProfile: 'pt-webgpu-lite',
      compatibilityMode: 'reject-unsupported',
      failures: [
        'material:baseColorMap=unsupported at materials[0].pbrMetallicRoughness.baseColorTexture',
      ],
    });

    expect(mocks.probeAdapterProfile).not.toHaveBeenCalled();
    expect(mocks.createEngine).not.toHaveBeenCalled();
  });

  it('reports explicit pt-webgpu-lite runtimeProfile on best-effort loads without probing', async () => {
    await expect(
      loadGltfWithEngine('asset.glb', {
        runtimeProfile: 'pt-webgpu-lite',
        compatibilityMode: 'best-effort',
      }),
    ).resolves.toMatchObject({
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu-lite',
      attached: true,
    });

    expect(mocks.probeAdapterProfile).not.toHaveBeenCalled();
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
  });

  it('rejects existing pt-webgpu engines before controller attachment when runtime tier is degraded', async () => {
    const existingEngine = makeExistingPtWebgpuEngine();

    await expect(
      loadGltfWithEngine('asset.glb', {
        engine: existingEngine,
        compatibilityMode: 'reject-degraded',
      }),
    ).rejects.toThrow(/pt-webgpu.*lite.*reject-degraded.*baseColorMap=unsupported/);

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).not.toHaveBeenCalled();
    expect(mocks.attachEngine).not.toHaveBeenCalled();
    expect(existingEngine.setScene).not.toHaveBeenCalled();
    expect(mocks.loadGltfForEngine).toHaveBeenCalledWith(
      'asset.glb',
      expect.objectContaining({
        backend: 'pt-webgpu',
        attachScene: false,
      }),
    );
    expect(mocks.loadGltfForEngine.mock.calls[0]?.[1]).not.toHaveProperty('engine');
  });

  it('attaches existing pt-webgpu engines after strict runtime tier accepts the asset', async () => {
    mocks.state.liteIssues = [mocks.approximateLiteIssue];
    const existingEngine = makeExistingPtWebgpuEngine();

    await expect(
      loadGltfWithEngine('asset.glb', {
        engine: existingEngine,
        compatibilityMode: 'reject-unsupported',
      }),
    ).resolves.toMatchObject({ backend: 'pt-webgpu', engine: existingEngine, attached: true });

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).not.toHaveBeenCalled();
    expect(mocks.attachEngine).toHaveBeenCalledWith(existingEngine, { setScene: true });
  });

  it('reports the runtime lite profile for best-effort existing pt-webgpu engines', async () => {
    const existingEngine = makeExistingPtWebgpuEngine();

    await expect(
      loadGltfWithEngine('asset.glb', {
        engine: existingEngine,
        compatibilityMode: 'best-effort',
      }),
    ).resolves.toMatchObject({
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu-lite',
      engine: existingEngine,
      attached: true,
    });

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).not.toHaveBeenCalled();
    expect(mocks.attachEngine).toHaveBeenCalledWith(existingEngine, { setScene: true });
  });

  it('allows reject-unsupported pt-webgpu glTF loads on lite tier when rows are degraded but supported', async () => {
    mocks.state.liteIssues = [mocks.approximateLiteIssue];

    await expect(
      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-unsupported' }),
    ).resolves.toMatchObject({ backend: 'pt-webgpu', attached: true });

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
  });

  it('accepts adapter-refined lite rows for bakeable COLOR_0 under reject-unsupported', async () => {
    mocks.state.liteIssues = [];

    await expect(
      loadGltfWithEngine('constant-color.glb', { compatibilityMode: 'reject-unsupported' }),
    ).resolves.toMatchObject({
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu-lite',
      attached: true,
    });

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
  });

	  it('treats KHR_meshopt_compression as satisfied by the engine meshoptDecode hook', async () => {
	    mocks.state.liteIssues = [mocks.khrMeshoptHookIssue];

    await expect(
      loadGltfWithEngine('asset.glb', {
        compatibilityMode: 'reject-degraded',
        meshoptDecode: vi.fn(),
      }),
    ).resolves.toMatchObject({ backend: 'pt-webgpu', attached: true });

	    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
	    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
	  });

	  it('treats pt-webgpu texture readiness rows as satisfied by host-ready opaque handles', async () => {
	    mocks.state.liteIssues = [mocks.textureReadinessIssue];

	    await expect(
	      loadGltfWithEngine('asset.glb', {
	        compatibilityMode: 'reject-degraded',
	        opaqueTextureHandlesReady: ['pt-webgpu'],
	      }),
	    ).resolves.toMatchObject({ backend: 'pt-webgpu', attached: true });

	    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
	    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
	  });

	  it('reports the runtime lite profile for best-effort pt-webgpu glTF loads without rejecting', async () => {
	    await expect(
      loadGltfWithEngine('asset.glb', { compatibilityMode: 'best-effort' }),
    ).resolves.toMatchObject({ backend: 'pt-webgpu', profileId: 'pt-webgpu-lite', attached: true });

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
  });

  it('forwards texture decode options into the adapter-owned bridge', async () => {
    const decodePixels = vi.fn(() => ({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 255]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    }));

    await expect(
      loadGltfWithEngine('asset.glb', {
        decodeTextures: true,
        decodePixels,
        maxTextureSize: 256,
        warnOnNpotRepeatWrap: true,
      }),
    ).resolves.toMatchObject({ backend: 'pt-webgpu', attached: true });

    expect(mocks.loadGltfForEngine).toHaveBeenCalledWith(
      'asset.glb',
      expect.objectContaining({
        decodeTextures: true,
        decodePixels,
        maxTextureSize: 256,
        warnOnNpotRepeatWrap: true,
      }),
    );
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

  it('revalidates actual pt-webgpu fallback engines against the runtime lite profile', async () => {
    mocks.state.selectedBackend = 'walkaround-hybrid';
    const dispose = vi.fn();
    mocks.createEngine.mockResolvedValueOnce({
      backendId: 'pt-webgpu',
      dispose,
      renderFrame: vi.fn(),
      setScene: vi.fn(),
    });

    await expect(
      loadGltfWithEngine('asset.glb', { compatibilityMode: 'reject-unsupported' }),
    ).rejects.toMatchObject({
      code: 'GLTF_COMPATIBILITY_REJECTED',
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu-lite',
      compatibilityMode: 'reject-unsupported',
      failures: [
        'material:baseColorMap=unsupported at materials[0].pbrMetallicRoughness.baseColorTexture',
      ],
    });

    expect(mocks.probeAdapterProfile).toHaveBeenCalledTimes(1);
    expect(mocks.createEngine).toHaveBeenCalledTimes(1);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(mocks.attachEngine).not.toHaveBeenCalled();
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
