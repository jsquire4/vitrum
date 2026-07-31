import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import type { DDGI } from '../ddgi/DDGI.js';
import type { PreparedSceneMutation } from '../SceneMutationTransaction.js';
import type { WalkaroundGPUPipeline } from '../pipeline/WalkaroundGPUPipeline.js';

function makeDevice(): GPUDevice {
  const noop = vi.fn();
  return {
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    })),
    queue: { submit: noop, writeBuffer: noop, writeTexture: noop },
    features: new Set<string>(),
    limits: {},
    addEventListener: noop,
    removeEventListener: noop,
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function options(
  overrides: Partial<HybridEngineOptions> = {},
): HybridEngineOptions {
  return {
    device: makeDevice(),
    width: 32,
    height: 32,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
    ...overrides,
  };
}

function meshScene(): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'mesh',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      transform: asMat4(new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ])),
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    }],
    emitters: [{
      kind: 'directional',
      id: 'sun',
      direction: [0, 1, 0],
      color: [1, 0.5, 0.25],
      intensity: 2,
    }],
    environment: { kind: 'none' },
  };
}

interface Internals {
  _lastScene: Scene | null;
  _renderScene: Scene | null;
  _ddgi: DDGI;
  _pipeline: WalkaroundGPUPipeline | null;
  _primaryLightDir: [number, number, number];
  _primaryLightDirOverrideActive: boolean;
  _primaryLightIntensity: number;
  _skyTint: [number, number, number];
  _skyIrradiance: number;
}

function prepared(
  events: string[],
  commitFault = false,
): PreparedSceneMutation {
  return {
    commit: vi.fn(() => {
      events.push('ddgi:commit');
      if (commitFault) throw new Error('DDGI commit fault');
    }),
    rollback: vi.fn(() => {
      events.push('ddgi:rollback');
    }),
    finalize: vi.fn(() => {
      events.push('ddgi:finalize');
    }),
  };
}

describe('HybridEngine.updateLighting transaction', () => {
  it('keeps the prior CPU generation and temporal state when DDGI preparation fails', () => {
    const engine = new HybridEngine(options());
    const state = engine as unknown as Internals;
    const realDdgi = state._ddgi;
    const previousDirection = state._primaryLightDir;
    const previousSky = state._skyTint;
    const requestAccumReset = vi.fn();
    const updateAnalyticLights = vi.fn();
    const prepareDirectionalEnvironment = vi.fn();
    state._pipeline = {
      requestAccumReset,
      updateAnalyticLights,
      prepareDirectionalEnvironment,
    } as unknown as WalkaroundGPUPipeline;
    state._ddgi = {
      prepareRuntimeLightingMutation: vi.fn(() => {
        throw new Error('DDGI prepare fault');
      }),
    } as unknown as DDGI;

    try {
      expect(() => engine.updateLighting({
        primaryLightDir: [1, 2, 3],
        primaryLightIntensity: 4,
        skyTint: [0.2, 0.3, 0.4],
        skyIrradiance: 5,
      })).toThrow('DDGI prepare fault');

      expect(state._primaryLightDir).toBe(previousDirection);
      expect(state._primaryLightDirOverrideActive).toBe(false);
      expect(state._primaryLightIntensity).toBe(1);
      expect(state._skyTint).toBe(previousSky);
      expect(state._skyIrradiance).toBe(1);
      expect(requestAccumReset).not.toHaveBeenCalled();
      expect(updateAnalyticLights).not.toHaveBeenCalled();
      expect(prepareDirectionalEnvironment).not.toHaveBeenCalled();
    } finally {
      state._pipeline = null;
      state._ddgi = realDdgi;
      engine.dispose();
    }
  });

  it('rolls DDGI back and leaves CPU state untouched when publication fails', () => {
    const engine = new HybridEngine(options());
    const state = engine as unknown as Internals;
    const realDdgi = state._ddgi;
    const events: string[] = [];
    const mutation = prepared(events, true);
    const requestAccumReset = vi.fn();
    state._pipeline = {
      requestAccumReset,
      updateAnalyticLights: vi.fn(),
      prepareDirectionalEnvironment: vi.fn(),
    } as unknown as WalkaroundGPUPipeline;
    state._ddgi = {
      prepareRuntimeLightingMutation: vi.fn(() => mutation),
    } as unknown as DDGI;

    try {
      expect(() => engine.updateLighting({
        primaryLightDir: [1, 0, 0],
        primaryLightIntensity: 3,
      })).toThrow('DDGI commit fault');

      expect(events).toEqual(['ddgi:commit', 'ddgi:rollback']);
      expect(state._primaryLightDir).toEqual([0, -1, 0]);
      expect(state._primaryLightDirOverrideActive).toBe(false);
      expect(state._primaryLightIntensity).toBe(1);
      expect(requestAccumReset).not.toHaveBeenCalled();
    } finally {
      state._pipeline = null;
      state._ddgi = realDdgi;
      engine.dispose();
    }
  });

  it('publishes one canonical snapshot without rebuilding analytic lights or environment resources', () => {
    const engine = new HybridEngine(options());
    const state = engine as unknown as Internals;
    const realDdgi = state._ddgi;
    const scene = meshScene();
    state._lastScene = scene;
    state._renderScene = scene;
    const events: string[] = [];
    const mutation = prepared(events);
    const prepareRuntimeLightingMutation = vi.fn(() => mutation);
    const requestAccumReset = vi.fn();
    const updateAnalyticLights = vi.fn();
    const prepareDirectionalEnvironment = vi.fn();
    state._pipeline = {
      requestAccumReset,
      updateAnalyticLights,
      prepareDirectionalEnvironment,
    } as unknown as WalkaroundGPUPipeline;
    state._ddgi = {
      prepareRuntimeLightingMutation,
    } as unknown as DDGI;
    const direction: [number, number, number] = [
      Number.MAX_VALUE,
      Number.MAX_VALUE,
      0,
    ];
    const skyTint: [number, number, number] = [0.1, 1e-46, 0.3];

    try {
      engine.updateLighting({
        primaryLightDir: direction,
        primaryLightIntensity: 0.1,
        skyTint,
        skyIrradiance: 0.2,
      });

      const expectedAxis = Math.fround(1 / Math.sqrt(2));
      expect(state._primaryLightDir).toEqual([expectedAxis, expectedAxis, 0]);
      expect(state._primaryLightDirOverrideActive).toBe(true);
      expect(state._primaryLightIntensity).toBe(Math.fround(0.1));
      expect(state._skyTint).toEqual([Math.fround(0.1), 0, Math.fround(0.3)]);
      expect(state._skyIrradiance).toBe(Math.fround(0.2));
      expect(events).toEqual(['ddgi:commit', 'ddgi:finalize']);
      expect(requestAccumReset).toHaveBeenCalledTimes(1);
      expect(updateAnalyticLights).not.toHaveBeenCalled();
      expect(prepareDirectionalEnvironment).not.toHaveBeenCalled();

      const [candidate] = prepareRuntimeLightingMutation.mock.calls[0] as unknown as [{
        readonly lights: ReadonlyArray<{
          readonly direction?: { readonly x: number; readonly y: number; readonly z: number };
        }>;
        readonly sunIntensityMultiplier: number;
        readonly skyTint: readonly [number, number, number];
        readonly skyIrradiance: number;
      }];
      expect(candidate.sunIntensityMultiplier).toBe(1);
      expect(candidate.skyTint).toEqual(state._skyTint);
      expect(candidate.skyIrradiance).toBe(state._skyIrradiance);
      expect(candidate.lights[0]?.direction).toEqual({
        x: -expectedAxis,
        y: -expectedAxis,
        z: 0,
      });

      direction[0] = 0;
      direction[1] = 0;
      direction[2] = 1;
      skyTint[0] = 1;
      skyTint[1] = 1;
      skyTint[2] = 1;
      expect(state._primaryLightDir).toEqual([expectedAxis, expectedAxis, 0]);
      expect(state._skyTint).toEqual([Math.fround(0.1), 0, Math.fround(0.3)]);
    } finally {
      state._pipeline = null;
      state._ddgi = realDdgi;
      engine.dispose();
    }
  });

  it('deep-snapshots constructor lighting into exact GPU-facing f32 values', () => {
    const direction: [number, number, number] = [2, 3, 4];
    const skyTint: [number, number, number] = [0.1, 0.2, 0.3];
    const light = {
      kind: 'fixture' as const,
      id: 'fixture',
      on: true,
      intensity: 0.2,
      position: { x: 0.1, y: 0.2, z: 0.3 },
      color: { r: 0.4, g: 0.5, b: 0.6 },
    };
    const engine = new HybridEngine(options({
      primaryLightDir: direction,
      primaryLightIntensity: 0.1,
      skyTint,
      skyIrradiance: 0.2,
      lights: [light],
    }));
    const state = engine as unknown as Internals & {
      _ctorLights: readonly typeof light[];
    };

    try {
      direction[0] = 99;
      skyTint[0] = 99;
      light.position.x = 99;
      light.color.r = 99;
      light.intensity = 99;

      expect(state._primaryLightDir).toEqual([
        Math.fround(2 / Math.sqrt(29)),
        Math.fround(3 / Math.sqrt(29)),
        Math.fround(4 / Math.sqrt(29)),
      ]);
      expect(state._primaryLightIntensity).toBe(Math.fround(0.1));
      expect(state._skyTint).toEqual([
        Math.fround(0.1),
        Math.fround(0.2),
        Math.fround(0.3),
      ]);
      expect(state._skyIrradiance).toBe(Math.fround(0.2));
      expect(state._ctorLights[0]?.intensity).toBe(Math.fround(0.2));
      expect(state._ctorLights[0]?.position?.x).toBe(Math.fround(0.1));
      expect(state._ctorLights[0]?.color?.r).toBe(Math.fround(0.4));
    } finally {
      engine.dispose();
    }
  });

  it('stores the resolved environment sky as the exact DDGI/GPU f32 identity', () => {
    const resolverTint: [number, number, number] = [0.1, 1e-46, 0.3];
    const engine = new HybridEngine(options({
      extensions: {
        'walkaround-hybrid': {
          resolveEnvironmentMap: () => ({
            kind: 'scalar-only',
            skyTint: resolverTint,
            skyIrradiance: 0.2,
          }),
        },
      },
    }));
    const state = engine as unknown as Internals & {
      _resolvedEnvironment: {
        readonly skyTint?: readonly [number, number, number];
        readonly skyIrradiance?: number;
      };
    };

    try {
      engine.updateEnvironment({
        kind: 'hdri',
        hdri: { opaque: true },
        intensity: 1,
      });

      const expectedTint: [number, number, number] = [
        Math.fround(0.1),
        0,
        Math.fround(0.3),
      ];
      expect(state._skyTint).toEqual(expectedTint);
      expect(state._skyIrradiance).toBe(Math.fround(0.2));
      expect(state._resolvedEnvironment.skyTint).toEqual(expectedTint);
      expect(state._resolvedEnvironment.skyIrradiance).toBe(Math.fround(0.2));
      expect(
        (state._ddgi as unknown as {
          _configuredSkyTint: [number, number, number];
        })._configuredSkyTint,
      ).toEqual(expectedTint);

      resolverTint[0] = 1;
      resolverTint[1] = 1;
      resolverTint[2] = 1;
      expect(state._skyTint).toEqual(expectedTint);
      expect(state._resolvedEnvironment.skyTint).toEqual(expectedTint);
    } finally {
      engine.dispose();
    }
  });

  it('always rolls the directional provider back when DDGI restore throws', () => {
    const engine = new HybridEngine(options());
    const state = engine as unknown as Internals;
    const oldView = {} as GPUTextureView;
    const oldSampler = {} as GPUSampler;
    const candidateView = {} as GPUTextureView;
    const candidateSampler = {} as GPUSampler;
    const providerRollback = vi.fn();
    const pipeline = {
      getEnvBindings: vi.fn(() => ({
        textureView: oldView,
        sampler: oldSampler,
        rotationY: 0.25,
        intensity: 2,
        hasDirectionalEnvironment: true,
      })),
      prepareDirectionalEnvironment: vi.fn(() => ({
        envBindings: {
          textureView: candidateView,
          sampler: candidateSampler,
          rotationY: 0.5,
          intensity: 3,
          hasDirectionalEnvironment: true,
        },
        commit: vi.fn(),
        rollback: providerRollback,
        finalize: vi.fn(),
      })),
    } as unknown as WalkaroundGPUPipeline;
    const setEnvironment = vi.spyOn(state._ddgi, 'setEnvironment');
    setEnvironment
      .mockImplementationOnce(() => undefined)
      .mockImplementationOnce(() => {
        throw new Error('DDGI restore fault');
      });

    try {
      const mutation = (
        engine as unknown as {
          _prepareDirectionalEnvironmentToPipeline(
            value: WalkaroundGPUPipeline,
            resolved: {
              readonly mode: 'none';
              readonly skyIrradiance: 0;
              readonly warnings: readonly string[];
            },
          ): PreparedSceneMutation;
        }
      )._prepareDirectionalEnvironmentToPipeline(
        pipeline,
        { mode: 'none', skyIrradiance: 0, warnings: [] },
      );

      expect(() => mutation.rollback()).toThrow('DDGI restore fault');
      expect(providerRollback).toHaveBeenCalledTimes(1);
      expect(setEnvironment).toHaveBeenNthCalledWith(
        1,
        candidateView,
        candidateSampler,
        0.5,
        3,
        true,
      );
      expect(setEnvironment).toHaveBeenNthCalledWith(
        2,
        oldView,
        oldSampler,
        0.25,
        2,
        true,
      );
    } finally {
      setEnvironment.mockRestore();
      engine.dispose();
    }
  });
});
