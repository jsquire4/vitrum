import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';
import { PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE } from '../webgpuLimits.js';

function makeStubDevice(
  limits: Partial<GPUSupportedLimits> = {
    maxStorageBuffersPerShaderStage: 64,
    maxStorageTexturesPerShaderStage: 8,
  },
): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    limits,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

describe('createPTEngine_WebGPU', () => {
  it('reports requested caustic strategy in capabilities', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'manifold-nee',
    });
    expect(engine.capabilities.causticStrategy).toBe('manifold-nee');
    // A4: old approximate tag gone; SPPM tag not present for non-photon-map strategies.
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-photon-map-sppm')).toBe(false);
  });

  it('supports photon-map capability reporting path', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'photon-map',
    });
    expect(engine.capabilities.causticStrategy).toBe('photon-map');
    // A4: real SPPM (not approximate) — tag updated to 'pt-webgpu-photon-map-sppm'.
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-photon-map-sppm')).toBe(true);
    const details = engine.capabilities.supportDetails?.causticStrategies?.['photon-map'];
    expect(details?.estimatorScope).toMatch(/production delta surface event/);
    expect(details?.emitterKinds).toEqual({
      directional: 'native',
      point: 'native',
      spot: 'native',
      'rect-area': 'native',
      'disc-area': 'native',
      'mesh-area': 'native',
      environment: 'native',
    });
    expect(details?.volumeScattering).toBe('native');
    expect(details?.incompatibleFeatures).toEqual([]);
    engine.dispose();
  });

  it('constructs every formerly guarded advanced-estimator composition', async () => {
    const sppmComposite = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'photon-map',
      spectral: true,
      bdpt: true,
      restirPtReuse: true,
    });
    expect(sppmComposite.capabilities.activeFeatures?.has('pt-webgpu-spectral')).toBe(true);
    expect(sppmComposite.capabilities.activeFeatures?.has('pt-webgpu-bdpt')).toBe(true);
    expect(sppmComposite.capabilities.activeFeatures?.has('pt-webgpu-one-edge-gris-reconnection')).toBe(true);
    expect(sppmComposite.capabilities.activeFeatures?.has('pt-webgpu-photon-map-sppm')).toBe(true);
    sppmComposite.dispose();

    const mneeBdpt = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'manifold-nee',
      bdpt: true,
    });
    expect(mneeBdpt.capabilities.causticStrategy).toBe('manifold-nee');
    expect(mneeBdpt.capabilities.activeFeatures?.has('pt-webgpu-bdpt')).toBe(true);
    mneeBdpt.dispose();
  });

  it('rejects SPPM on the lite tier before construction', async () => {
    const liteDevice = makeStubDevice({
      maxStorageBuffersPerShaderStage: 10,
      maxStorageTexturesPerShaderStage: 4,
    });
    await expect(createPTEngine_WebGPU({
      device: liteDevice,
      causticStrategy: 'photon-map',
    })).rejects.toThrow(/requires traceTier "full"/);
  });

  it('reports current incremental patch support matrix', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    const patch = engine.capabilities.incrementalPatchSupport;
    expect(engine.capabilities.supportsIncrementalScene).toBe(true);
    expect(patch).toEqual({
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    });
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-spectral')).toBe(false);
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-cwbvh-closest-traversal')).toBe(
      false,
    );
  });

  it('reports spectral transport only when it is selected', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      spectral: true,
    });
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-spectral')).toBe(true);
    engine.dispose();
  });

  it('accepts spectral transport with hero-aware MNEE and ReSTIR-PT reuse', async () => {
    const mnee = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      spectral: true,
      causticStrategy: 'manifold-nee',
      causticOptions: { mneeMaxChainLength: 8 },
    });
    expect(mnee.capabilities.causticStrategy).toBe('manifold-nee');
    expect(mnee.capabilities.activeFeatures?.has('pt-webgpu-spectral')).toBe(true);
    mnee.dispose();

    const restir = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      spectral: true,
      restirPtReuse: true,
    });
    expect(restir.capabilities.activeFeatures?.has('pt-webgpu-spectral')).toBe(true);
    expect(restir.capabilities.activeFeatures?.has('pt-webgpu-one-edge-gris-reconnection')).toBe(true);
    restir.dispose();
  });

  it('exposes frame/progress telemetry subscriptions', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    expect(typeof engine.onFrame).toBe('function');
    expect(typeof engine.onProgress).toBe('function');
    const offFrame = engine.onFrame?.(() => {});
    const offProgress = engine.onProgress?.(() => {});
    expect(typeof offFrame).toBe('function');
    expect(typeof offProgress).toBe('function');
    offFrame?.();
    offProgress?.();
  });

  it('transitions state ready → disposed across the lifecycle', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    expect(engine.state).toBe('ready');
    engine.dispose();
    expect(engine.state).toBe('disposed');
    // After dispose, lifecycle methods throw rather than no-op.
    expect(() => engine.pause()).toThrow(/disposed/);
    expect(() => engine.resume()).toThrow(/disposed/);
    expect(() => engine.renderFrame({} as never)).toThrow();
  });

  it('pause / resume toggle state when live', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
    });
    engine.pause();
    expect(engine.state).toBe('paused');
    engine.resume();
    expect(engine.state).toBe('ready');
    engine.dispose();
  });

  it('accepts lite-tier adapters (SwiftShader-class limits)', async () => {
    const liteDevice = {
      createCommandEncoder: vi.fn(),
      limits: {
        maxStorageBuffersPerShaderStage: 10,
        maxStorageTexturesPerShaderStage: 4,
      },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      lost: new Promise<never>(() => {}),
    } as unknown as GPUDevice;
    const engine = await createPTEngine_WebGPU({ device: liteDevice });
    expect((engine).backendProfileId).toBe(
      'pt-webgpu-lite',
    );
    expect([...(engine.capabilities.activeFeatures ?? [])]).not.toContain('pt-webgpu-lite-tier');
    engine.dispose();
  });

  it('rejects devices below lite storage-buffer limit', async () => {
    const lowLimitDevice = {
      createCommandEncoder: vi.fn(),
      limits: {
        maxStorageBuffersPerShaderStage: 4,
        maxStorageTexturesPerShaderStage: 4,
      },
    } as unknown as GPUDevice;
    await expect(
      createPTEngine_WebGPU({
        device: lowLimitDevice,
      }),
    ).rejects.toThrow(/below the lite tier/i);
  });

  it('advertises stable opt-in CWBVH traversal without a prototype warning', async () => {
    const warnings: string[] = [];
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      bvhTraversal: 'cwbvh-closest',
      onWarning: (warning) => warnings.push(warning.code),
    });
    expect(engine.capabilities.activeFeatures?.has('pt-webgpu-cwbvh-closest-traversal')).toBe(true);
    expect(warnings).not.toContain('pt-webgpu.cwbvh-closest');
    engine.dispose();
  });

  it('rejects CWBVH traversal on lite-tier devices', async () => {
    await expect(
      createPTEngine_WebGPU({
        device: makeStubDevice({
          maxStorageBuffersPerShaderStage: 10,
          maxStorageTexturesPerShaderStage: 4,
        }),
        bvhTraversal: 'cwbvh-closest',
      }),
    ).rejects.toThrow(/requires traceTier "full"/);
  });

  it('rejects CWBVH traversal when the full-tier device limit floor is too low', async () => {
    await expect(
      createPTEngine_WebGPU({
        device: makeStubDevice({
          maxStorageBuffersPerShaderStage:
            PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE - 1,
          maxStorageTexturesPerShaderStage: 5,
        }),
        bvhTraversal: 'cwbvh-closest',
      }),
    ).rejects.toThrow(
      new RegExp(
        `requires maxStorageBuffersPerShaderStage >= ` +
        `${PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE}`,
      ),
    );
  });
});
