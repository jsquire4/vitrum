import { describe, expect, it, vi } from 'vitest';
import type { PipelineInitHost } from '../HybridEngineLifecycle.js';
import { HybridEngine } from '../HybridEngine.js';
import type { HybridEngineOptions } from '../HybridEngine.js';
import type { WalkaroundGPUPipeline } from '../pipeline/WalkaroundGPUPipeline.js';

function makeStubDevice(): GPUDevice {
  const noop = () => undefined;
  return {
    createCommandEncoder: noop,
    createBuffer: () => ({ destroy: noop }),
    createTexture: () => ({ createView: noop, destroy: noop }),
    createBindGroupLayout: noop,
    createBindGroup: noop,
    createShaderModule: noop,
    createComputePipeline: noop,
    queue: { submit: noop, writeBuffer: noop },
    features: new Set<string>(),
    limits: {},
    addEventListener: noop,
    removeEventListener: noop,
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeEngineOpts(
  overrides: Partial<HybridEngineOptions> = {},
): HybridEngineOptions {
  return {
    device: makeStubDevice(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
    ...overrides,
  };
}

describe('HybridEngine runtime controls', () => {
  it('retains a pre-init PPG interval in the coordinator host reused by rebuilds', () => {
    const engine = new HybridEngine(makeEngineOpts({
      ppgEnabled: true,
      ppgDispatchInterval: 2,
    }));
    try {
      // PipelineInitCoordinator captures this host once in the constructor and
      // reuses it for every startInit(), including reset/rebuild starts.
      const coordinator = engine['_initCoordinator'] as unknown as {
        readonly host: PipelineInitHost;
      };
      expect(coordinator.host.ppgDispatchInterval).toBe(2);

      engine.setPpgDispatchInterval(7.9);
      expect(coordinator.host.ppgDispatchInterval).toBe(7);

      const setPipelineInterval = vi.fn();
      const pipeline = {
        setPpgDispatchInterval: setPipelineInterval,
        dispose: vi.fn(),
      } as unknown as WalkaroundGPUPipeline;
      coordinator.host.publishPipeline(pipeline);
      expect(setPipelineInterval).toHaveBeenCalledWith(7);

      engine.setPpgDispatchInterval(5.9);
      expect(setPipelineInterval).toHaveBeenCalledWith(5);
      expect(coordinator.host.ppgDispatchInterval).toBe(5);

      expect(() => engine.setPpgDispatchInterval(0)).toThrow(RangeError);
      expect(coordinator.host.ppgDispatchInterval).toBe(5);

      engine.enableFrameBudget({ targetMs: 16, maxPpgDispatchInterval: 8 });
      expect(engine['_frameBudget']?.snapshot().ppgDispatchInterval).toBe(5);
    } finally {
      engine.dispose();
    }
  });

  it('accepts only the implemented ddgi layer and validates JS callers', () => {
    const engine = new HybridEngine(makeEngineOpts());
    try {
      engine.setLayerEnabled('ddgi', false);
      expect(engine['_buildFrameDeps']().flags.isLayerEnabled('ddgi')).toBe(false);

      const callFromJs = engine.setLayerEnabled.bind(engine) as unknown as (
        layer: string,
        enabled: unknown,
      ) => void;
      expect(() => callFromJs('nrc', true)).toThrow(/unsupported layer "nrc"/i);
      expect(() => callFromJs('ddgi', 1)).toThrow(/enabled must be a boolean/i);
      expect(engine['_buildFrameDeps']().flags.isLayerEnabled('ddgi')).toBe(false);
    } finally {
      engine.dispose();
    }
  });

  it('releases frame and progress subscribers during disposal', () => {
    const engine = new HybridEngine(makeEngineOpts());
    const unsubscribeFrame = engine.onFrame(() => undefined);
    const unsubscribeProgress = engine.onProgress(() => undefined);

    expect(engine['_frameSubs']).toHaveLength(1);
    expect(engine['_progressSubs']).toHaveLength(1);

    engine.dispose();

    expect(engine['_frameSubs']).toHaveLength(0);
    expect(engine['_progressSubs']).toHaveLength(0);
    expect(() => unsubscribeFrame()).not.toThrow();
    expect(() => unsubscribeProgress()).not.toThrow();
  });
});
