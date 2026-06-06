import { describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGPU } from '../index.js';

function makeStubDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
  } as unknown as GPUDevice;
}

describe('createPTEngine_WebGPU', () => {
  it('reports requested caustic strategy in capabilities', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'manifold-nee',
    });
    expect(engine.capabilities.causticStrategy).toBe('manifold-nee');
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-photon-map-approximate')).toBe(false);
  });

  it('supports photon-map capability reporting path', async () => {
    const engine = await createPTEngine_WebGPU({
      device: makeStubDevice(),
      causticStrategy: 'photon-map',
    });
    expect(engine.capabilities.causticStrategy).toBe('photon-map');
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-photon-map-approximate')).toBe(true);
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
    expect(engine.capabilities.experimentalFeatures?.has('experimental-backend')).toBe(true);
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
    } as unknown as GPUDevice;
    const engine = await createPTEngine_WebGPU({ device: liteDevice });
    expect(engine.capabilities.experimentalFeatures?.has('pt-webgpu-lite-tier')).toBe(true);
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
});
