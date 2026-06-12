import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import { asMat4, type EngineWarning, type Scene, type SceneEmitter, type ScenePrimitive } from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import {
  buildReSTIRSceneBVHForCoreScene,
  type ReSTIRBvhMode,
  type SceneBVHBuffers,
} from '../restir/bvhCore.js';

function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn(), writeTexture: vi.fn() },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeOpts(warnings: EngineWarning[] = []): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 2,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
    onWarning: (warning) => warnings.push(warning),
  };
}

function mat4Translate(x: number, y = 0, z = 0) {
  return asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]));
}

function mesh(id: string, x: number, baseColor: [number, number, number]): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor, roughness: 0.5, metallic: 0 },
    transform: mat4Translate(x),
  };
}

function pointEmitter(intensity = 1): SceneEmitter {
  return {
    kind: 'point',
    id: 'lamp',
    position: [0, 2, 0],
    color: [1, 0.8, 0.6],
    intensity,
  };
}

function baseScene(emitters: readonly SceneEmitter[] = []): Scene {
  return {
    primitives: [
      mesh('mesh-a', 0, [0.8, 0.2, 0.2]),
      mesh('mesh-b', 3, [0.2, 0.2, 0.8]),
    ],
    emitters,
    environment: { kind: 'none' },
  };
}

function makePipeline() {
  return {
    dispose: vi.fn(),
    refreshBvhMaterialSlice: vi.fn(),
    refreshBvhNormalsSlice: vi.fn(),
    refreshBvhRefit: vi.fn(),
    refreshBvhFullRebuild: vi.fn(),
    refreshTlasRefit: vi.fn(),
    updateEmitters: vi.fn(),
    updateAnalyticLights: vi.fn(),
    updateDirectionalEnvironment: vi.fn(),
    getEnvBindings: vi.fn(() => null),
    requestAccumReset: vi.fn(),
    resize: vi.fn(),
  };
}

function makeDdgi() {
  return {
    invalidateProbeCache: vi.fn(),
    markInstancesDirty: vi.fn(),
    syncRestirBvhBuffers: vi.fn(),
    setSkyParams: vi.fn(),
    setLights: vi.fn(),
    setSunIntensityMultiplier: vi.fn(),
    setEmitterTris: vi.fn(),
    setEnvironment: vi.fn(),
    dispose: vi.fn(),
  };
}

function makeRc() {
  return {
    refreshMaterialsFromCore: vi.fn(),
    invalidateBindings: vi.fn(),
    dispose: vi.fn(),
    refitCascadeBounds: vi.fn(),
    syncRestirBvhBuffers: vi.fn(),
    setSceneFromCore: vi.fn(),
    refitMergedInstance: vi.fn(() => false),
  };
}

interface SeededEngine {
  engine: HybridEngine;
  buffers: SceneBVHBuffers;
  pipeline: ReturnType<typeof makePipeline>;
  ddgi: ReturnType<typeof makeDdgi>;
  rc: ReturnType<typeof makeRc> | null;
  warnings: EngineWarning[];
}

interface HybridEngineInternals {
  _state: string;
  _lastScene: Scene | null;
  _renderScene: Scene | null;
  _bvhBuffers: SceneBVHBuffers | null;
  _pipeline: unknown;
  _ddgi: unknown;
  _rc: unknown;
}

function seedEngine(
  scene: Scene,
  opts: { bvhMode?: ReSTIRBvhMode; rc?: boolean } = {},
): SeededEngine {
  const warnings: EngineWarning[] = [];
  const engine = new HybridEngine(makeOpts(warnings));
  const pipeline = makePipeline();
  const ddgi = makeDdgi();
  const rc = opts.rc === true ? makeRc() : null;
  const buffers = buildReSTIRSceneBVHForCoreScene(scene, {
    bvhMode: opts.bvhMode ?? 'tlas',
  });
  const internals = engine as unknown as HybridEngineInternals;
  internals._state = 'ready';
  internals._lastScene = scene;
  internals._renderScene = scene;
  internals._bvhBuffers = buffers;
  internals._pipeline = pipeline;
  internals._ddgi = ddgi;
  internals._rc = rc;
  return { engine, buffers, pipeline, ddgi, rc, warnings };
}

function storedScene(engine: HybridEngine): Scene {
  const scene = (engine as unknown as HybridEngineInternals)._lastScene;
  if (scene == null) throw new Error('expected seeded scene');
  return scene;
}

describe('HybridEngine mutation matrix (non-GPU seam)', () => {
  let warnSpy: MockInstance;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('updatePrimitive(transform) refits TLAS, resets accumulation, and re-syncs DDGI BVH without invalidating probes', () => {
    const { engine, pipeline, ddgi } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      const moved = mat4Translate(2, 0.5, 0);

      engine.updatePrimitive('mesh-a', { transform: moved });

      expect(pipeline.refreshTlasRefit).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(ddgi.markInstancesDirty).toHaveBeenCalledTimes(1);
      expect(ddgi.syncRestirBvhBuffers).toHaveBeenCalledTimes(1);
      expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
      expect(Array.from((storedScene(engine).primitives[0] as { transform: Float32Array }).transform))
        .toEqual(Array.from(moved));
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(material) uploads material resources, refreshes RC materials, and skips geometry GI propagation', () => {
    const { engine, pipeline, ddgi, rc } = seedEngine(baseScene(), { bvhMode: 'tlas', rc: true });
    try {
      engine.updatePrimitive('mesh-a', {
        material: { baseColor: [0.1, 0.9, 0.3], roughness: 0.25, metallic: 0 },
      });

      expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(rc?.refreshMaterialsFromCore).toHaveBeenCalledTimes(1);
      expect(ddgi.syncRestirBvhBuffers).not.toHaveBeenCalled();
      expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
      expect(pipeline.updateEmitters).not.toHaveBeenCalled();
      expect((storedScene(engine).primitives[0] as { material: { roughness: number } }).material.roughness)
        .toBe(0.25);
    } finally {
      engine.dispose();
    }
  });

  it('updateEmitter repacks emitters, invalidates GI lighting state, invalidates RC bindings, and resets accumulation', () => {
    const { engine, pipeline, ddgi, rc } = seedEngine(baseScene([pointEmitter()]), {
      bvhMode: 'tlas',
      rc: true,
    });
    try {
      engine.updateEmitter('lamp', { intensity: 4, position: [2, 3, 4] } as Partial<SceneEmitter>);

      expect(pipeline.updateEmitters).toHaveBeenCalledTimes(1);
      expect(rc?.invalidateBindings).toHaveBeenCalledTimes(1);
      expect(ddgi.setSunIntensityMultiplier).toHaveBeenCalledTimes(1);
      expect(ddgi.setLights).toHaveBeenCalledTimes(1);
      expect(ddgi.setEmitterTris).toHaveBeenCalledTimes(1);
      expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
      expect(pipeline.updateAnalyticLights).toHaveBeenCalledTimes(1);
      expect(pipeline.updateDirectionalEnvironment).toHaveBeenCalledWith(null, 0, 0);
      expect(ddgi.setEnvironment).toHaveBeenCalledWith(null, null, 0, 0, false);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect((storedScene(engine).emitters[0] as { intensity: number }).intensity).toBe(4);
    } finally {
      engine.dispose();
    }
  });

  it('updateEnvironment warns for procedural-sky fallback, stores the env, resets directional IBL, invalidates DDGI, and resets accumulation', () => {
    const { engine, pipeline, ddgi, warnings } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      engine.updateEnvironment({
        kind: 'procedural-sky',
        sunDirection: [0, 1, 0],
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.01,
        mieDirectionalG: 0.8,
        intensity: 3,
      });

      expect(warnings.map((w) => w.code)).toContain('walkaround-hybrid.environment-approximation');
      expect(warnings[0]?.method).toBe('updateEnvironment');
      expect(storedScene(engine).environment.kind).toBe('procedural-sky');
      expect(ddgi.setSkyParams).toHaveBeenCalledWith(expect.any(Array), 3);
      expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(pipeline.updateDirectionalEnvironment).toHaveBeenCalledWith(null, 0, 0);
      expect(ddgi.setEnvironment).toHaveBeenCalledWith(null, null, 0, 0, false);
    } finally {
      engine.dispose();
    }
  });

  it('updateLighting warns on unknown keys and invalidates sky lighting without rebuilding resources', () => {
    const { engine, pipeline, ddgi, warnings } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      engine.updateLighting({
        skyTint: [0.4, 0.5, 0.6],
        typoIntensity: 8,
      } as never);

      expect(warnings.some((w) =>
        w.code === 'walkaround-hybrid.unknown-lighting-key' &&
        w.details?.key === 'typoIntensity',
      )).toBe(true);
      expect(ddgi.setSkyParams).toHaveBeenCalledWith([0.4, 0.5, 0.6], 1);
      expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(pipeline.resize).not.toHaveBeenCalled();
    } finally {
      engine.dispose();
    }
  });

  it('setSize resizes frame resources only; zero/same-size calls are no-ops and DDGI is preserved', () => {
    const { engine, pipeline, ddgi } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      engine.setSize(64, 64);
      engine.setSize(0, 128);
      expect(pipeline.resize).not.toHaveBeenCalled();

      engine.setSize(128, 80);

      expect(pipeline.resize).toHaveBeenCalledTimes(1);
      expect(pipeline.resize).toHaveBeenCalledWith(128, 80);
      expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
      expect(pipeline.requestAccumReset).not.toHaveBeenCalled();
    } finally {
      engine.dispose();
    }
  });
});
