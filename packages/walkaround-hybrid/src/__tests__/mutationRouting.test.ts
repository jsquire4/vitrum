// Characterization pin for HybridEngine._routePrimitiveUpdate's (patch shape →
// routed path) matrix (D3 finding 4). The branch order in _routePrimitiveUpdate
// + the wholesale-field interception in updatePrimitive are load-bearing: a
// silent reorder would change which build function runs for a given patch and
// therefore what work the GPU does. Each routed path calls a DISTINCT pipeline
// method, so we pin the routing by asserting exactly which one fired:
//   - transformRefit (TLAS)   → refreshTlasRefit
//   - positionsRefit          → refreshBvhNormalsSlice
//   - topologyRebuild         → refreshBvhFullRebuild
//   - materialPatch           → refreshBvhMaterialSlice (+ optional atlas refresh)
//   - wholesale topology field → intercepted to setScene (none of the refit/
//     rebuild slice methods fire from _routePrimitiveUpdate)
//
// This complements mutationMatrix.test.ts, which pins the material-atlas-rebuild
// PREDICATES (which material fields trigger refreshMaterialTextureAtlas) but does
// not pin the outer routing DECISION itself.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import {
  asMat4,
  type EngineWarning,
  type Scene,
  type ScenePrimitive,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';
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

function identityMat4(): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function mesh(id: string, x: number): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.8, 0.2, 0.2], roughness: 0.5, metallic: 0 },
    transform: mat4Translate(x),
  };
}

function skinnedMesh(id: string, x: number): SkinnedMeshPrimitive {
  const skinIndices = new Uint32Array(12);
  const skinWeights = new Float32Array(12);
  for (let v = 0; v < 3; v += 1) skinWeights[v * 4] = 1;
  return {
    kind: 'skinned-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    skinIndices,
    skinWeights,
    bones: identityMat4(),
    boneInverses: identityMat4(),
    material: { baseColor: [0.8, 0.2, 0.2], roughness: 0.5, metallic: 0 },
    transform: mat4Translate(x),
  };
}

function baseScene(): Scene {
  return {
    primitives: [mesh('mesh-a', 0), mesh('mesh-b', 3)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function skinnedScene(): Scene {
  return {
    primitives: [skinnedMesh('skin-a', 0)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makePipeline() {
  return {
    dispose: vi.fn(),
    refreshBvhMaterialSlice: vi.fn(),
    refreshBvhEmissiveLe: vi.fn(),
    refreshBvhNormalsSlice: vi.fn(),
    refreshBvhRefit: vi.fn(),
    refreshBvhFullRebuild: vi.fn(),
    refreshMaterialTextureAtlas: vi.fn(),
    refreshTlasRefit: vi.fn(),
    updateEmitters: vi.fn(),
    updateAnalyticLights: vi.fn(),
    updateDirectionalEnvironment: vi.fn(),
    getEnvBindings: vi.fn((): { textureView: GPUTextureView; sampler: GPUSampler } | null => null),
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

interface HybridEngineInternals {
  _state: string;
  _lastScene: Scene | null;
  _renderScene: Scene | null;
  _bvhBuffers: SceneBVHBuffers | null;
  _pipeline: unknown;
  _ddgi: unknown;
  _rc: unknown;
}

function seedEngine(scene: Scene, bvhMode: ReSTIRBvhMode = 'tlas') {
  const warnings: EngineWarning[] = [];
  const engine = new HybridEngine(makeOpts(warnings));
  const pipeline = makePipeline();
  const ddgi = makeDdgi();
  const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode });
  const internals = engine as unknown as HybridEngineInternals;
  internals._state = 'ready';
  internals._lastScene = scene;
  internals._renderScene = scene;
  internals._bvhBuffers = buffers;
  internals._pipeline = pipeline;
  internals._ddgi = ddgi;
  internals._rc = null;
  return { engine, pipeline, ddgi, warnings };
}

/** The four routing markers — exactly ONE (or the specified combination) fires
 *  per routed path, so asserting the tuple pins the branch decision. */
function routeMarkers(pipeline: ReturnType<typeof makePipeline>) {
  return {
    transformRefit: pipeline.refreshTlasRefit.mock.calls.length,
    positionsRefit: pipeline.refreshBvhNormalsSlice.mock.calls.length,
    topologyRebuild: pipeline.refreshBvhFullRebuild.mock.calls.length,
    materialPatch: pipeline.refreshBvhMaterialSlice.mock.calls.length,
  };
}

describe('HybridEngine _routePrimitiveUpdate (patch shape → routed path) matrix', () => {
  let warnSpy: MockInstance;
  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('transform only → transformRefit', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      engine.updatePrimitive('mesh-a', { transform: mat4Translate(2, 0, 0) });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 1, positionsRefit: 0, topologyRebuild: 0, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });

  it('positions (+ same-count normals) → positionsRefit, NOT rebuild', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      engine.updatePrimitive('mesh-a', {
        positions: new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 1, positionsRefit: 1, topologyRebuild: 0, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });

  it('normals only (no positions) → topologyRebuild', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      engine.updatePrimitive('mesh-a', {
        normals: new Float32Array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
      });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 0, positionsRefit: 0, topologyRebuild: 1, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });

  it('structural topology field (uvs) → topologyRebuild', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      engine.updatePrimitive('mesh-a', {
        uvs: new Float32Array([0, 0, 0.5, 0, 0, 0.5]),
      });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 0, positionsRefit: 0, topologyRebuild: 1, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });

  it('material only → materialPatch (slice re-pack), NOT rebuild', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      engine.updatePrimitive('mesh-a', {
        material: { baseColor: [0.1, 0.9, 0.2], roughness: 0.3, metallic: 0 },
      });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 0, positionsRefit: 0, topologyRebuild: 0, materialPatch: 1,
      });
    } finally {
      engine.dispose();
    }
  });

  it('transform + material → topologyRebuild (mixed patch beats transform refit)', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      engine.updatePrimitive('mesh-a', {
        transform: mat4Translate(2, 0, 0),
        material: { baseColor: [0.1, 0.9, 0.2], roughness: 0.3, metallic: 0 },
      });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 0, positionsRefit: 0, topologyRebuild: 1, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });

  it('positions + material → topologyRebuild (mixed patch beats positions refit)', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      engine.updatePrimitive('mesh-a', {
        positions: new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0]),
        material: { baseColor: [0.1, 0.9, 0.2], roughness: 0.3, metallic: 0 },
      });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 0, positionsRefit: 0, topologyRebuild: 1, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });

  it('skinned pose (bones) → skinned refit path (positionsRefit), NOT material/rebuild', () => {
    const { engine, pipeline } = seedEngine(skinnedScene());
    try {
      engine.updatePrimitive('skin-a', { bones: identityMat4() });
      const m = routeMarkers(pipeline);
      expect(m.positionsRefit).toBe(1);
      expect(m.topologyRebuild).toBe(0);
      expect(m.materialPatch).toBe(0);
    } finally {
      engine.dispose();
    }
  });

  it('skinned pose + material → topologyRebuild (mixed beats skinned pose refit)', () => {
    const { engine, pipeline } = seedEngine(skinnedScene());
    try {
      engine.updatePrimitive('skin-a', {
        bones: identityMat4(),
        material: { baseColor: [0.1, 0.9, 0.2], roughness: 0.3, metallic: 0 },
      });
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 0, positionsRefit: 0, topologyRebuild: 1, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });

  it('wholesale topology field (kind) → intercepted to setScene, not the refit/rebuild slice methods', () => {
    const { engine, pipeline } = seedEngine(baseScene());
    try {
      // `kind` is a wholesale-replacement field: updatePrimitive intercepts it
      // and routes through setScene BEFORE _routePrimitiveUpdate, so none of the
      // in-place slice methods fire. (setScene itself would fail on the stubbed
      // pipeline; we only assert the routing decision, so guard the throw.)
      try {
        engine.updatePrimitive('mesh-a', { kind: 'mesh' });
      } catch {
        /* setScene needs a real device; the routing decision already happened */
      }
      expect(routeMarkers(pipeline)).toEqual({
        transformRefit: 0, positionsRefit: 0, topologyRebuild: 0, materialPatch: 0,
      });
    } finally {
      engine.dispose();
    }
  });
});
