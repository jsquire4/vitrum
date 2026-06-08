/**
 * Theme T16 (incremental path) — `HybridEngine._syncDdgiLightsFromThreeRoot`
 * radiometric fidelity.
 *
 * The INIT path (HybridEngineLifecycle) already prefers the lossless core
 * projection `coreEmittersToDDGILights` over the lossy THREE walk
 * `collectDDGILightsFromThreeRoot`. This pins that the INCREMENTAL refresh
 * path (driven by `updateEmitter` / `refreshDdgiLightsFromThreeScene`) makes
 * the SAME choice: when the engine holds a core scene supplying meshes, the
 * DDGI lights it re-publishes must carry the emitter's chroma and the true
 * cross-product area `4·|uAxis × vAxis|` — NOT the lossy white + `width·height`
 * the THREE round-trip produced.
 *
 * The discriminator: `vitrumSceneToThree` is mocked to return a THREE root
 * with NO light objects, so the lossy fallback `collectDDGILightsFromThreeRoot`
 * would yield an EMPTY list. The only way a chroma-bearing fixture reaches
 * `DDGI.setLights` is the core-emitter projection — so observing one proves
 * the incremental path took the corrected branch.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

interface IncState {
  pipelineInitDeferreds: Array<{ promise: Promise<void>; resolve: () => void; reject: (e: unknown) => void }>;
  pipelineConstructed: number;
  buildBVHCalls: number;
}

vi.mock('../src/pipeline/WalkaroundGPUPipeline.js', async () => {
  const g = globalThis as unknown as { __HYBRID_INC_STATE__?: IncState };
  if (!g.__HYBRID_INC_STATE__) {
    g.__HYBRID_INC_STATE__ = { pipelineInitDeferreds: [], pipelineConstructed: 0, buildBVHCalls: 0 };
  }
  const state = g.__HYBRID_INC_STATE__;

  class MockWalkaroundGPUPipeline {
    private readonly index: number;
    constructor(_device: GPUDevice, _w: number, _h: number) {
      this.index = state.pipelineInitDeferreds.length;
      let resolve!: () => void;
      let reject!: (e: unknown) => void;
      const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
      state.pipelineInitDeferreds.push({ promise, resolve, reject });
      state.pipelineConstructed += 1;
    }
    async initialize(): Promise<void> {
      await state.pipelineInitDeferreds[this.index]!.promise;
    }
    dispose = vi.fn();
    resize = vi.fn();
    requestAccumReset = vi.fn();
    presentLastFrame = vi.fn();
    setDDGIInputs = vi.fn();
    renderFrame = vi.fn();
    refreshBvhRefit = vi.fn();
    refreshBvhNodesOnly = vi.fn();
    refreshBvhMaterialSlice = vi.fn();
    refreshBvhFullRebuild = vi.fn();
    updateEmitters = vi.fn();
  }

  return {
    WalkaroundGPUPipeline: MockWalkaroundGPUPipeline,
    HYBRID_WEBGPU_REQUIRED_LIMITS: {},
    HYBRID_WEBGPU_REQUIRED_FEATURES: [],
  };
});

vi.mock('../src/restir/bvhCore.js', async () => {
  const g = globalThis as unknown as { __HYBRID_INC_STATE__?: IncState };
  if (!g.__HYBRID_INC_STATE__) {
    g.__HYBRID_INC_STATE__ = { pipelineInitDeferreds: [], pipelineConstructed: 0, buildBVHCalls: 0 };
  }
  const state = g.__HYBRID_INC_STATE__;

  function makeFakeBuffers(): unknown {
    const nodeBuf = new ArrayBuffer(32);
    const nodeU32 = new Uint32Array(nodeBuf);
    nodeU32[6] = 0;
    nodeU32[7] = 0xffff0001;
    return {
      bvhNodes:        { cpuData: nodeBuf, count: 1, byteLength: 32 },
      bvhIndex:        { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      bvhBeerColors:   { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      bvhPositions:    { cpuData: new ArrayBuffer(64), count: 4, byteLength: 64 },
      emitters:        { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      emitterCdf:      { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      emitterCount:    1,
      totalEmissivePower: 1,
      mergedGeometry:  new THREE.BufferGeometry(),
      meshVertexRanges: [],
      bvhIndicesStride3: new Uint32Array([0, 1, 2]),
      triangleMaterialIds: { cpuData: new Uint32Array(1).buffer, byteLength: 4, count: 1 },
      buildMaterials: [new THREE.MeshStandardMaterial()],
      coreMaterials: [{ baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 }],
      emitterNormals: new Float32Array(16),
      bvhMode: 'merged' as const,
      primitiveTlasBindings: [],
    };
  }

  const buildFn = vi.fn(() => {
    state.buildBVHCalls += 1;
    return makeFakeBuffers();
  });

  return {
    buildReSTIRSceneBVHForCoreScene: buildFn,
    rebuildEmitterBuffersFromCoreScene: vi.fn(() => ({
      emitters: { cpuData: new ArrayBuffer(80), byteLength: 80, count: 1 },
      emitterCdf: { cpuData: new Float32Array(1).buffer, byteLength: 4, count: 1 },
      emitterCount: 1,
      totalEmissivePower: 2,
    })),
    disposeSceneBVH: vi.fn(),
  };
});

// `vitrumSceneToThree` returns a root with a mesh but NO light objects — so
// the lossy `collectDDGILightsFromThreeRoot` fallback would yield [].
vi.mock('@vitrum/three-bindings', async () => {
  function makeSyntheticScene(): THREE.Scene {
    const scene = new THREE.Scene();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3),
    );
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array([0, 1, 2]), 1));
    const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial());
    mesh.name = 'mesh-a';
    scene.add(mesh);
    return scene;
  }
  return {
    vitrumSceneToThree: vi.fn(() => makeSyntheticScene()),
    disposeVitrumThreeSceneRoot: vi.fn(),
    applyVitrumMaterialToMesh: vi.fn(),
    solveSkin: vi.fn(),
    findMeshByPrimitiveId: vi.fn(),
  };
});

import { HybridEngine } from '../src/HybridEngine.js';
import type { Scene, RectAreaEmitter } from '@vitrum/core';
import { collectDDGILightsFromThreeRoot } from '../src/HybridEngineLifecycle.js';
import type { DDGI } from '../src/ddgi/DDGI.js';
import type { DDGILight } from '../src/ddgi/types.js';

function getState(): IncState {
  return (globalThis as unknown as { __HYBRID_INC_STATE__: IncState }).__HYBRID_INC_STATE__;
}

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeEngine(): HybridEngine {
  return new HybridEngine({
    device:                makeMockDevice(),
    width:                 64,
    height:                64,
    primaryLightDir:       [0, -1, 0],
    primaryLightIntensity: 1.0,
    skyTint:               [1, 1, 1],
    skyIrradiance:         1.0,
  });
}

/** Green rect-area emitter with NON-orthogonal half-axes — the case where the
 *  THREE round-trip's `width·height` metric diverges from the true area. */
const GREEN_RECT_SHEARED: RectAreaEmitter = {
  id: 'rect-green',
  kind: 'rect-area',
  color: [0, 1, 0],
  intensity: 3,
  position: [1, 1, 1],
  uAxis: [1, 0, 0],
  vAxis: [1, 1, 0], // not perpendicular to uAxis → u×v = (0,0,1), area = 4
};

const SCENE_WITH_RECT: Scene = {
  primitives: [
    {
      id: 'mesh-a',
      kind: 'mesh',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      indices: new Uint32Array([0, 1, 2]),
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    },
  ],
  emitters: [GREEN_RECT_SHEARED],
  environment: { kind: 'none' },
};

async function drainMicrotasks(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
  await new Promise((r) => setTimeout(r, 50));
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

async function waitForPipelineCount(n: number, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (getState().pipelineInitDeferreds.length < n) {
    if (Date.now() > deadline) throw new Error(`waitForPipelineCount(${n}) timed out`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

beforeEach(() => {
  const s = getState();
  s.pipelineInitDeferreds.length = 0;
  s.pipelineConstructed = 0;
  s.buildBVHCalls = 0;
});

describe('HybridEngine incremental DDGI-light refresh — T16 radiometric fidelity', () => {
  it('updateEmitter re-publishes DDGI lights via the lossless core projection', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_RECT);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    // Spy on the live DDGI instance's setLights so we capture exactly what the
    // incremental refresh publishes.
    const ddgi = (engine as unknown as { _ddgi: DDGI })._ddgi;
    const setLightsSpy = vi.spyOn(ddgi, 'setLights');

    engine.updateEmitter('rect-green', { intensity: 6 });

    expect(setLightsSpy).toHaveBeenCalledTimes(1);
    const published = setLightsSpy.mock.calls[0]![0] as DDGILight[];

    // Exactly one fixture (the rect-area emitter); chroma preserved.
    expect(published).toHaveLength(1);
    const light = published[0]!;
    expect(light.kind).toBe('fixture');
    expect(light.id).toBe('rect-green');
    expect(light.color).toEqual({ r: 0, g: 1, b: 0 });

    // True cross-product area: |u×v| = |(0,0,1)| = 1 → area = 4. The patched
    // intensity is 6, so flux-equivalent = 6 · 4 = 24.
    expect(light.intensity).toBeCloseTo(6 * 4, 6);
  });

  it('the lossy THREE-walk fallback would have produced NO lights here', () => {
    // Confirms the discriminator: the mocked THREE root carries no light
    // objects, so the old `collectDDGILightsFromThreeRoot` path yields [].
    // Any chroma-bearing fixture observed above therefore had to come from
    // the core-emitter projection, not the THREE walk.
    const root = new THREE.Scene();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    root.add(mesh);
    expect(collectDDGILightsFromThreeRoot(root)).toHaveLength(0);
  });

  it('refreshDdgiLightsFromThreeScene also uses the core projection', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_RECT);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    const ddgi = (engine as unknown as { _ddgi: DDGI })._ddgi;
    const setLightsSpy = vi.spyOn(ddgi, 'setLights');

    engine.refreshDdgiLightsFromThreeScene();

    expect(setLightsSpy).toHaveBeenCalledTimes(1);
    const published = setLightsSpy.mock.calls[0]![0] as DDGILight[];
    expect(published).toHaveLength(1);
    expect(published[0]!.color).toEqual({ r: 0, g: 1, b: 0 });
    // Unpatched intensity here = 3, area = 4 → 12.
    expect(published[0]!.intensity).toBeCloseTo(3 * 4, 6);
  });
});
