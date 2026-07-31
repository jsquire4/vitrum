import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
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
  it('uses authored scene sun direction until updateLighting establishes a host override', () => {
    const engine = new HybridEngine(makeEngineOpts({
      primaryLightDir: [1, 0, 0],
    }));
    const scene: Scene = {
      primitives: [],
      emitters: [{
        kind: 'directional',
        id: 'scene-sun',
        direction: [0, 3, 4],
        color: [1, 1, 1],
        intensity: 2,
      }],
      environment: { kind: 'none' },
    };
    try {
      engine['_lastScene'] = scene;
      engine['_renderScene'] = scene;

      expect(engine['_lightingSnapshot']().primaryLightDir[0]).toBeCloseTo(0);
      expect(engine['_lightingSnapshot']().primaryLightDir[1]).toBeCloseTo(0.6);
      expect(engine['_lightingSnapshot']().primaryLightDir[2]).toBeCloseTo(0.8);
      expect(engine['_lightingSnapshot']().primaryLightDirOverride).toBeUndefined();
      const initHost = engine['_buildInitHost']();
      expect(initHost.primaryLightDir[0]).toBeCloseTo(0);
      expect(initHost.primaryLightDir[1]).toBeCloseTo(0.6);
      expect(initHost.primaryLightDir[2]).toBeCloseTo(0.8);
      expect(initHost.primaryLightDirOverride).toBeUndefined();

      engine.updateLighting({ primaryLightDir: [0, 0, -1] });

      expect(engine['_lightingSnapshot']().primaryLightDir).toEqual([0, 0, -1]);
      expect(engine['_lightingSnapshot']().primaryLightDirOverride).toEqual([0, 0, -1]);
      expect(initHost.primaryLightDir).toEqual([0, 0, -1]);
      expect(initHost.primaryLightDirOverride).toEqual([0, 0, -1]);
    } finally {
      engine.dispose();
    }
  });

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
      const updateDirectionalEnvironment = vi.fn();
      const prepareDirectionalEnvironment = vi.fn((
        data: unknown,
        rotationY: number,
        intensity: number,
      ) => ({
        envBindings: null,
        commit: vi.fn(() => {
          updateDirectionalEnvironment(data, rotationY, intensity);
        }),
        rollback: vi.fn(),
        finalize: vi.fn(),
      }));
      const pipeline = {
        setPpgDispatchInterval: setPipelineInterval,
        setDenoiserPassEnabled: vi.fn(),
        getEnvBindings: vi.fn(() => null),
        prepareDirectionalEnvironment,
        updateDirectionalEnvironment,
        requestAccumReset: vi.fn(),
        dispose: vi.fn(),
      } as unknown as WalkaroundGPUPipeline;
      coordinator.host.publishPipeline(pipeline);
      expect(setPipelineInterval).toHaveBeenCalledWith(7);
      expect(updateDirectionalEnvironment).toHaveBeenCalledWith(null, 0, 0);

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

  it('rejects a PPG cadence mutation when the PPG subsystem is disabled', () => {
    const engine = new HybridEngine(makeEngineOpts());
    try {
      expect(() => engine.setPpgDispatchInterval(2))
        .toThrow(/requires construction with ppgEnabled:true/i);
    } finally {
      engine.dispose();
    }
  });

  it('applies the latest resolved SceneEnvironment before an async pipeline generation is published', () => {
    const engine = new HybridEngine(makeEngineOpts());
    try {
      engine.updateEnvironment({
        kind: 'hdri',
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([2, 1, 0.5]),
        },
        intensity: 1.75,
        rotationY: 0.35,
      });

      const coordinator = engine['_initCoordinator'] as unknown as {
        readonly host: PipelineInitHost;
      };
      const resolved = engine['_resolvedEnvironment'];
      expect(resolved.directional).toBeDefined();

      const textureView = {} as GPUTextureView;
      const sampler = {} as GPUSampler;
      const updateDirectionalEnvironment = vi.fn();
      const getEnvBindings = vi.fn(() => null);
      const preparedCommit = vi.fn();
      const preparedFinalize = vi.fn();
      const prepareDirectionalEnvironment = vi.fn((
        data: unknown,
        rotationY: number,
        intensity: number,
      ) => ({
        envBindings: {
          textureView,
          sampler,
          rotationY,
          intensity,
          hasDirectionalEnvironment: true,
        },
        commit: vi.fn(() => {
          preparedCommit();
          updateDirectionalEnvironment(data, rotationY, intensity);
        }),
        rollback: vi.fn(),
        finalize: preparedFinalize,
      }));
      const pipeline = {
        setPpgDispatchInterval: vi.fn(),
        setDenoiserPassEnabled: vi.fn(),
        prepareDirectionalEnvironment,
        updateDirectionalEnvironment,
        getEnvBindings,
        requestAccumReset: vi.fn(),
        dispose: vi.fn(),
      } as unknown as WalkaroundGPUPipeline;
      const setDdgiEnvironment = vi
        .spyOn(engine['_ddgi'], 'setEnvironment')
        .mockImplementation(() => undefined);

      coordinator.host.publishPipeline(pipeline);

      expect(updateDirectionalEnvironment).toHaveBeenCalledWith(
        resolved.directional,
        resolved.rotationY,
        1.75,
      );
      expect(prepareDirectionalEnvironment).toHaveBeenCalledWith(
        resolved.directional,
        resolved.rotationY,
        1.75,
      );
      expect(getEnvBindings).toHaveBeenCalledTimes(1);
      expect(preparedCommit).toHaveBeenCalledTimes(1);
      expect(preparedFinalize).toHaveBeenCalledTimes(1);
      expect(setDdgiEnvironment).toHaveBeenCalledWith(
        textureView,
        sampler,
        resolved.rotationY,
        1.75,
        true,
      );
      expect(engine['_pipeline']).toBe(pipeline);
    } finally {
      engine.dispose();
    }
  });

  it('accepts only the implemented ddgi layer and validates JS callers', () => {
    const engine = new HybridEngine(makeEngineOpts());
    try {
      expect(engine.capabilities.supportDetails?.bounceSemantics).toEqual({
        kind: 'ddgi-feedback',
        directOnlyValue: 1,
        multiBounceEquilibriumValue: 2,
        inactiveWhenLayerDisabled: 'ddgi',
      });
      expect(engine.getBounceSemantics()).toEqual({
        kind: 'ddgi-feedback',
        configuredMaxBounces: 2,
        active: 'multi-bounce-equilibrium',
      });

      engine.setLayerEnabled('ddgi', false);
      const disabledFlags = engine['_buildFrameDeps']().flags;
      expect(disabledFlags.isLayerEnabled('ddgi')).toBe(false);
      expect(disabledFlags).not.toHaveProperty('ddgiOn');
      expect(engine.getBounceSemantics()).toEqual({
        kind: 'ddgi-feedback',
        configuredMaxBounces: 2,
        active: 'disabled',
      });

      engine.setLayerEnabled('ddgi', true);
      expect(engine.getBounceSemantics().active).toBe('multi-bounce-equilibrium');

      const callFromJs = engine.setLayerEnabled.bind(engine) as unknown as (
        layer: string,
        enabled: unknown,
      ) => void;
      expect(() => callFromJs('nrc', true)).toThrow(/unsupported layer "nrc"/i);
      expect(() => callFromJs('ddgi', 1)).toThrow(/enabled must be a boolean/i);
      expect(engine['_buildFrameDeps']().flags.isLayerEnabled('ddgi')).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('reports the direct-only DDGI bounce regime without calling it path depth', () => {
    const engine = new HybridEngine(makeEngineOpts({ maxBounces: 1 }));
    try {
      expect(engine.getBounceSemantics()).toEqual({
        kind: 'ddgi-feedback',
        configuredMaxBounces: 1,
        active: 'direct-only',
      });
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

  it('rejects every public runtime mutator consistently after dispose and retires frame-budget state', () => {
    const engine = new HybridEngine(makeEngineOpts());
    engine.enableFrameBudget();
    expect(engine['_frameBudget']).not.toBeNull();

    engine.dispose();

    expect(engine['_frameBudget']).toBeNull();
    const calls: ReadonlyArray<readonly [string, () => unknown]> = [
      ['setDdgiUpdateDivisor', () => engine.setDdgiUpdateDivisor(2)],
      ['setPpgDispatchInterval', () => engine.setPpgDispatchInterval(2)],
      ['enableFrameBudget', () => engine.enableFrameBudget()],
      ['disableFrameBudget', () => engine.disableFrameBudget()],
      ['tickFrameBudget', () => engine.tickFrameBudget(16)],
      ['setLayerEnabled', () => engine.setLayerEnabled('ddgi', false)],
    ];
    for (const [method, call] of calls) {
      expect(call).toThrow(`HybridEngine.${method}: engine is disposed.`);
    }
  });

  it('remembers pause during initialization and lets resume clear it before publication', () => {
    const engine = new HybridEngine(makeEngineOpts());
    try {
      const coordinator = engine['_initCoordinator'] as unknown as {
        readonly host: PipelineInitHost;
      };

      coordinator.host.setState('initializing');
      engine.pause();
      expect(engine.state).toBe('initializing');
      expect(engine['_pauseRequested']).toBe(true);
      coordinator.host.setState('ready');
      expect(engine.state).toBe('paused');

      coordinator.host.setState('initializing');
      engine.pause();
      engine.resume();
      expect(engine['_pauseRequested']).toBe(false);
      coordinator.host.setState('ready');
      expect(engine.state).toBe('ready');
    } finally {
      engine.dispose();
    }
  });

  it('re-emits pre-first-frame GPU errors after a wall-clock window while suppressing bursts', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-28T00:00:00Z'));
    const engine = new HybridEngine(makeEngineOpts());
    const errors: unknown[] = [];
    engine.onError((error) => errors.push(error));
    const handler = engine['_onUncapturedError']!;
    const validationError = {
      message: 'repeat before first frame',
      constructor: { name: 'GPUValidationError' },
    };

    try {
      handler({ error: validationError } as unknown as Event);
      handler({ error: validationError } as unknown as Event);
      expect(errors).toHaveLength(1);

      vi.advanceTimersByTime(1_001);
      handler({ error: validationError } as unknown as Event);
      expect(errors).toHaveLength(2);

      handler({
        error: {
          message: validationError.message,
          constructor: { name: 'GPUInternalError' },
        },
      } as unknown as Event);
      expect(errors).toHaveLength(3);
    } finally {
      engine.dispose();
      vi.useRealTimers();
    }
  });

  it('deduplicates environment warnings by warning identity and scene generation', () => {
    let resolverWarning = 'first approximation';
    const engine = new HybridEngine(makeEngineOpts({
      extensions: {
        'walkaround-hybrid': {
          resolveEnvironmentMap: () => ({
            kind: 'scalar-only',
            skyTint: [1, 1, 1],
            skyIrradiance: 1,
            warnings: [resolverWarning],
          }),
        },
      },
    }));
    const warnings: string[] = [];
    engine.onWarning((warning) => {
      if (warning.code === 'walkaround-hybrid.environment-approximation') {
        warnings.push(String(warning.details?.['warning']));
      }
    });
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const hdri = { opaque: true };
    const scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'hdri', hdri, intensity: 1 },
    } as Scene;

    try {
      engine.setScene(scene);
      engine.updateEnvironment(scene.environment);
      expect(warnings).toEqual(['first approximation']);

      resolverWarning = 'second approximation';
      engine.updateEnvironment(scene.environment);
      expect(warnings).toEqual([
        'first approximation',
        'second approximation',
      ]);

      resolverWarning = 'first approximation';
      engine.updateEnvironment(scene.environment);
      expect(warnings).toHaveLength(2);

      engine.setScene(scene);
      expect(warnings).toEqual([
        'first approximation',
        'second approximation',
        'first approximation',
      ]);
    } finally {
      consoleWarn.mockRestore();
      engine.dispose();
    }
  });
});
