import { describe, expect, it, vi } from 'vitest';
import {
  asMat4,
  BACKEND_PROMISE_LEDGER,
  type CapturedFrame,
  type Engine,
  type EngineCapabilities,
  type EngineState,
  type FrameInput,
  type FrameOutput,
  type Scene,
  type SceneEnvironment,
} from '@vitrum/core';
import {
  composeProgressiveCapabilities,
  progressiveHandleAsEngine,
  type ProgressiveEngineHandle,
} from '../createProgressiveEngine.js';
import { ProgressiveHandoffCoordinator } from '../progressiveHandoff.js';
import { stubCapabilities, stubEngine } from './fixtures/stubEngine.js';

const EMPTY_SCENE: Scene = {
  primitives: [],
  emitters: [{
    kind: 'directional',
    id: 'sun',
    color: [1, 1, 1],
    intensity: 1,
    direction: [0, 1, 0],
  }],
  environment: { kind: 'none' },
};

function input(): FrameInput {
  return {
    viewMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    projMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    cameraPosition: [0, 0, 0],
  } as unknown as FrameInput;
}

function rendered(samplesAccumulated: number): FrameOutput {
  return {
    kind: 'rendered',
    samplesAccumulated,
    isConverged: false,
    primaryRadiance: {},
  } as unknown as FrameOutput;
}

function capabilities(overrides: Partial<EngineCapabilities>): EngineCapabilities {
  return stubCapabilities(overrides);
}

function mutableEngine(caps: EngineCapabilities) {
  let state: EngineState = 'ready';
  const engine: Engine = {
    ...stubEngine(caps),
    get state() { return state; },
  };
  return { engine, setState(next: EngineState) { state = next; } };
}

function handleFor(realtime: Engine, converged: Engine): ProgressiveEngineHandle {
  return {
    coordinator: new ProgressiveHandoffCoordinator({
      realtime,
      converged,
      scene: EMPTY_SCENE,
      stillFramesBeforeHandoff: 1,
    }),
    realtime,
    converged,
    dispose: vi.fn(() => {
      realtime.dispose();
      converged.dispose();
    }),
  };
}

describe('progressive facade contract', () => {
  it('uses an honest composite identity and composes conservative capabilities', () => {
    const realtimeCaps = capabilities({
      supportsAuxBuffers: true,
      maxSamplesPerPixel: Number.POSITIVE_INFINITY,
      maxBounces: 4,
      supportedAnalyticShapes: new Set(['sphere', 'box']),
      supportedEmitterKinds: new Set(['directional', 'point']),
      supportedPrimitiveKinds: new Set(['mesh', 'analytic']),
      supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'procedural-sky']),
      activeFeatures: new Set(['walkaround-hybrid-denoiser-atrous-variance'] as const),
      causticStrategy: 'none',
    });
    const convergedCaps = capabilities({
      supportsAuxBuffers: false,
      maxSamplesPerPixel: 1024,
      maxBounces: 8,
      supportedAnalyticShapes: new Set(['box', 'cylinder']),
      supportedEmitterKinds: new Set(['directional']),
      supportedPrimitiveKinds: new Set(['mesh', 'skinned-mesh']),
      supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri']),
      activeFeatures: new Set(['pt-webgpu-spectral'] as const),
      causticStrategy: 'manifold-nee',
    });
    const realtime = mutableEngine(realtimeCaps).engine;
    const converged = Object.assign(mutableEngine(convergedCaps).engine, {
      backendProfileId: 'pt-webgpu' as const,
      profileId: 'pt-webgpu' as const,
    });
    const facade = progressiveHandleAsEngine(handleFor(realtime, converged));

    expect(facade.backendId).toBe('progressive');
    expect(facade.backendProfileId).toBe('pt-webgpu');
    expect(facade.profileId).toBe('pt-webgpu');
    expect(facade.capabilities).not.toBe(realtimeCaps);
    expect(facade.capabilities.supportsAuxBuffers).toBe(false);
    expect(facade.capabilities.maxSamplesPerPixel).toBe(1024);
    expect(facade.capabilities.maxBounces).toBe(4);
    expect([...facade.capabilities.supportedAnalyticShapes]).toEqual(['box']);
    expect([...facade.capabilities.supportedEmitterKinds]).toEqual(['directional']);
    expect([...facade.capabilities.supportedPrimitiveKinds ?? []]).toEqual(['mesh']);
    expect([...facade.capabilities.supportedEnvironmentKinds ?? []]).toEqual(['none']);
    expect([...facade.capabilities.activeFeatures ?? []]).toEqual([
      'walkaround-hybrid-denoiser-atrous-variance',
      'pt-webgpu-spectral',
    ]);
    expect(Object.isFrozen(facade.capabilities.supportedAnalyticShapes)).toBe(true);
    expect(Object.isFrozen(facade.capabilities.activeFeatures)).toBe(true);
    expect(() => (
      facade.capabilities.supportedAnalyticShapes as unknown as Set<string>
    ).add('sphere')).toThrow(TypeError);
    expect(() => (
      facade.capabilities.activeFeatures as unknown as Set<string>
    ).add('invented-feature')).toThrow(TypeError);
    expect([...facade.capabilities.supportedAnalyticShapes]).toEqual(['box']);
    expect(facade.capabilities.causticStrategy).toBe('none');
    expect(facade.capabilities.presentationMode).toBe('swapchain-optional');
    expect(facade.capabilities.supportsAccumulatorSeed).toBe(false);
    expect(facade.capabilities.supportsProgressiveSeedSource).toBe(false);
    expect(facade.capabilities.supportsIncrementalScene).toBe(true);
    expect(facade.capabilities.supportsAddRemovePrimitive).toBe(true);
    const activeFeatureIdentity = facade.capabilities.activeFeatures;
    facade.renderFrame(input());
    facade.renderFrame(input());
    expect(facade.backendProfileId).toBe('pt-webgpu');
    expect(facade.profileId).toBe('pt-webgpu');
    expect(facade.capabilities.activeFeatures).toBe(activeFeatureIdentity);
  });

  it('does not advertise fast mutation support without methods or a scene fallback', () => {
    const first = stubEngine();
    const second = stubEngine();
    const caps = composeProgressiveCapabilities(first, second, false);

    expect(caps.supportsIncrementalScene).toBe(false);
    expect(caps.supportsAddRemovePrimitive).toBe(false);
    expect(caps.incrementalPatchSupport).toEqual({
      transform: false,
      positions: false,
      material: false,
      emitter: false,
      topology: false,
    });
  });

  it('routes emitter/environment mutations through coordinator scene authority', () => {
    const realtime = stubEngine();
    const converged = stubEngine();
    const facade = progressiveHandleAsEngine(handleFor(realtime, converged));

    facade.updateEmitter?.('sun', { intensity: 3 });
    expect(realtime.setScene).toHaveBeenCalledTimes(1);
    expect(converged.setScene).toHaveBeenCalledTimes(1);
    expect(facade.getScene?.()?.emitters[0]?.intensity).toBe(3);

    facade.updateEnvironment?.({ kind: 'hdri', hdri: {}, intensity: 2 });
    expect(realtime.setScene).toHaveBeenCalledTimes(2);
    expect(converged.setScene).toHaveBeenCalledTimes(2);
    expect(facade.getScene?.()?.environment.kind).toBe('hdri');
    expect(facade.updateLighting).toBeUndefined();
  });

  it('routes runtime lighting through both phases and advertises the weakest shared row', () => {
    const realtimeUpdateLighting = vi.fn();
    const convergedUpdateLighting = vi.fn();
    const realtime = {
      ...stubEngine(capabilities({
        supportDetails: BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails,
      })),
      updateLighting: realtimeUpdateLighting,
    } as Engine;
    const converged = {
      ...stubEngine(capabilities({
        supportDetails: BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails,
        inverseRendering: {
          methods: { 'finite-difference': 'native', 'path-replay': 'native' },
          pathReplay: {
            failurePolicy: 'error',
            materialFields: new Set(['emissive']),
            emitterFields: new Set(),
            maxBounces: 1,
            supportsSpectral: false,
            supportsBdpt: false,
            supportsRestirPtReuse: false,
            supportsCausticStrategies: false,
          },
        },
      })),
      updateLighting: convergedUpdateLighting,
    } as Engine;
    const facade = progressiveHandleAsEngine(handleFor(realtime, converged));
    const update = { environmentIntensity: 2, sunIntensity: 0.5 };

    facade.updateLighting?.(update);

    expect(realtimeUpdateLighting).toHaveBeenCalledWith(update);
    expect(convergedUpdateLighting).toHaveBeenCalledWith(update);
    expect(facade.capabilities.supportDetails?.mutations.lighting).toBe('native');
    expect(facade.capabilities.inverseRendering).toBeUndefined();
  });

  it('reports composite lifecycle state instead of borrowing realtime state', () => {
    const realtime = mutableEngine(stubCapabilities());
    const converged = mutableEngine(stubCapabilities());
    const facade = progressiveHandleAsEngine(handleFor(realtime.engine, converged.engine));

    expect(facade.state).toBe('ready');
    converged.setState('error');
    expect(facade.state).toBe('error');
    realtime.setState('disposed');
    converged.setState('disposed');
    expect(facade.state).toBe('disposed');
  });

  it('exposes a live composite debug surface and reports both engines memory', () => {
    const realtimePick = vi.fn(() => 'realtime-hit');
    const convergedPick = vi.fn(() => 'converged-hit');
    const realtime = {
      ...stubEngine(capabilities({ debugSurface: true })),
      debug: {
        pickPrimitive: realtimePick,
        estimatedGpuMemoryBytes: () => ({
          total: 10,
          byCategory: { realtime: 10 },
          byTextureFormat: { rgba16float: 6 },
          byBufferUsage: { storage: 4 },
        }),
      },
      renderFrame: vi.fn(() => rendered(1)),
    } as Engine;
    const converged = {
      ...stubEngine(capabilities({ debugSurface: true })),
      debug: {
        pickPrimitive: convergedPick,
        estimatedGpuMemoryBytes: () => ({
          total: 20,
          byCategory: { converged: 20 },
          byTextureFormat: { rgba16float: 8, rgba32float: 4 },
          byBufferUsage: { storage: 12, uniform: 2 },
        }),
      },
      renderFrame: vi.fn(() => rendered(1)),
    } as Engine;
    const facade = progressiveHandleAsEngine(handleFor(realtime, converged));

    expect(facade.capabilities.debugSurface).toBe(true);
    expect(facade.debug?.pickPrimitive?.(1, 2)).toBe('realtime-hit');
    expect(facade.debug?.estimatedGpuMemoryBytes?.()).toEqual({
      total: 30,
      byCategory: { realtime: 10, converged: 20 },
      byTextureFormat: { rgba16float: 14, rgba32float: 4 },
      byBufferUsage: { storage: 16, uniform: 2 },
    });

    facade.renderFrame(input());
    facade.renderFrame(input());
    expect(facade.debug?.pickPrimitive?.(1, 2)).toBe('converged-hit');
    expect(realtimePick).toHaveBeenCalledTimes(1);
    expect(convergedPick).toHaveBeenCalledTimes(1);
  });

  it('captures the displayed phase and validates resize before touching either backend', async () => {
    const realtimeCapture = vi.fn(async (): Promise<CapturedFrame> => ({
      width: 1, height: 1, rgba: new Float32Array(4), colorSpace: 'linear',
    } as CapturedFrame));
    const convergedCapture = vi.fn(async (): Promise<CapturedFrame> => ({
      width: 2, height: 2, rgba: new Float32Array(16), colorSpace: 'linear',
    } as CapturedFrame));
    const realtime = {
      ...stubEngine(),
      renderFrame: vi.fn(() => rendered(1)),
      captureFrame: realtimeCapture,
      setSize: vi.fn(),
    } as Engine;
    const converged = {
      ...stubEngine(),
      renderFrame: vi.fn(() => rendered(1)),
      captureFrame: convergedCapture,
      setSize: vi.fn(),
    } as Engine;
    const facade = progressiveHandleAsEngine(handleFor(realtime, converged));

    expect((await facade.captureFrame?.())?.width).toBe(1);
    facade.renderFrame(input());
    facade.renderFrame(input());
    expect((await facade.captureFrame?.())?.width).toBe(2);
    expect(realtimeCapture).toHaveBeenCalledTimes(1);
    expect(convergedCapture).toHaveBeenCalledTimes(1);

    expect(() => facade.setSize?.(Number.NaN, 10)).toThrow(/positive safe integers/);
    expect(realtime.setSize).not.toHaveBeenCalled();
    expect(converged.setSize).not.toHaveBeenCalled();
    facade.setSize?.(640, 360);
    expect(realtime.setSize).toHaveBeenCalledWith(640, 360);
    expect(converged.setSize).toHaveBeenCalledWith(640, 360);
  });
});
