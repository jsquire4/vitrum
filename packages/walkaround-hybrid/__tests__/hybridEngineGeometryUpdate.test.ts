/**
 * Items A3 follow-up — `HybridEngine.updatePrimitive` with geometry-change
 * BVH leaf rebuild (`feat/a3-geometry-change-bvh-leaf-rebuild`).
 *
 * Two paths are exercised:
 *
 *  (c) Transform-only fast path — refit the BVH bounds in place + write
 *      the affected primitive's vertex slice via `device.queue.writeBuffer`.
 *      No new pipeline is constructed (pipeline-compile counter stays
 *      flat). No full BVH rebuild (buildReSTIRSceneBVH call counter
 *      stays flat). Accumulator reset is requested so the next 1–2
 *      frames recompute fresh history.
 *
 *  (a) Topology-change full rebuild — call `buildReSTIRSceneBVH` again,
 *      destroy + reupload the four BVH GPU buffers. Pipeline-compile
 *      counter STILL stays flat (no setScene path runs). Accumulator
 *      reset is requested.
 *
 * Pre-fix: `HybridEngine.updatePrimitive` was typed `never`; the only way
 * to push a geometry change was `setScene()` which calls `_initPipeline()`
 * and burns ~50–500 ms on shader recompile.
 *
 * Tests assert (no real WebGPU needed — pipeline is mocked):
 *  - Transform-only patches do NOT increment pipelineConstructed.length
 *    (the "pipeline-compile count" stand-in).
 *  - Transform-only patches do NOT call buildReSTIRSceneBVH a second time.
 *  - Transform-only patches call refreshBvhRefit + requestAccumReset on
 *    the live pipeline (the convergence-within-2-frames guarantee — the
 *    accumulator is reset so any subsequent renderFrame call starts
 *    fresh).
 *  - Topology patches DO call buildReSTIRSceneBVH (a second time) and
 *    refreshBvhFullRebuild, but DO NOT recreate the pipeline.
 *  - The capability flag flips to supportsIncrementalScene = true.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

interface GeoUpdateState {
  pipelineInitDeferreds: Array<{ promise: Promise<void>; resolve: () => void; reject: (e: unknown) => void }>;
  pipelineConstructed: MockPipeline[];
  buildBVHCalls: Array<{ idx: number }>;
}

interface MockPipeline {
  index: number;
  initialized: boolean;
  disposed: boolean;
  width: number;
  height: number;
  dispose: ReturnType<typeof vi.fn>;
  resize: ReturnType<typeof vi.fn>;
  requestAccumReset: ReturnType<typeof vi.fn>;
  presentLastFrame: ReturnType<typeof vi.fn>;
  setDDGIInputs: ReturnType<typeof vi.fn>;
  renderFrame: ReturnType<typeof vi.fn>;
  refreshBvhRefit: ReturnType<typeof vi.fn>;
  refreshBvhNodesOnly: ReturnType<typeof vi.fn>;
  refreshBvhMaterialSlice: ReturnType<typeof vi.fn>;
  refreshBvhFullRebuild: ReturnType<typeof vi.fn>;
  updateEmitters: ReturnType<typeof vi.fn>;
}

vi.mock('../src/pipeline/WalkaroundGPUPipeline.js', async () => {
  const g = globalThis as unknown as { __HYBRID_GEO_STATE__?: GeoUpdateState };
  if (!g.__HYBRID_GEO_STATE__) {
    g.__HYBRID_GEO_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
    };
  }
  const state = g.__HYBRID_GEO_STATE__;

  class MockWalkaroundGPUPipeline implements MockPipeline {
    public readonly index: number;
    public initialized = false;
    public disposed = false;
    public width: number;
    public height: number;
    public dispose: ReturnType<typeof vi.fn>;
    public resize: ReturnType<typeof vi.fn>;
    public requestAccumReset = vi.fn();
    public presentLastFrame = vi.fn();
    public setDDGIInputs = vi.fn();
    public renderFrame = vi.fn();
    public refreshBvhRefit = vi.fn();
    public refreshBvhNodesOnly = vi.fn();
    public refreshBvhMaterialSlice = vi.fn();
    public refreshBvhFullRebuild = vi.fn();
    public updateEmitters = vi.fn();

    constructor(_device: GPUDevice, w: number, h: number) {
      this.index = state.pipelineInitDeferreds.length;
      this.width = w;
      this.height = h;
      let resolve!: () => void;
      let reject!: (e: unknown) => void;
      const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
      state.pipelineInitDeferreds.push({ promise, resolve, reject });
      this.dispose = vi.fn(() => { this.disposed = true; }) as ReturnType<typeof vi.fn>;
      this.resize = vi.fn();
      state.pipelineConstructed.push(this);
    }

    async initialize(): Promise<void> {
      await state.pipelineInitDeferreds[this.index]!.promise;
      this.initialized = true;
    }
  }

  return {
    WalkaroundGPUPipeline: MockWalkaroundGPUPipeline,
    HYBRID_WEBGPU_REQUIRED_LIMITS: {},
    HYBRID_WEBGPU_REQUIRED_FEATURES: [],
  };
});

vi.mock('../src/restir/bvhCompute.js', async () => {
  const g = globalThis as unknown as { __HYBRID_GEO_STATE__?: GeoUpdateState };
  if (!g.__HYBRID_GEO_STATE__) {
    g.__HYBRID_GEO_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
    };
  }
  const state = g.__HYBRID_GEO_STATE__;

  function makeFakeBuffers(): unknown {
    // Build a single-leaf BVH node buffer (1 node × 32 B = 32 B). The
    // refit helper detects leaves via `(splitOrCount >>> 16) === 0xffff`,
    // so we pack a LEAFNODE flag with `triCount = 1` into u32[7] and
    // `triOffset = 0` into u32[6]. Bounds (slots 0..5) are dont-care:
    // refit overwrites them.
    const nodeBuf = new ArrayBuffer(32);
    const nodeU32 = new Uint32Array(nodeBuf);
    nodeU32[6] = 0;          // triOffset
    nodeU32[7] = 0xffff0001; // LEAFNODE_FLAG | triCount=1

    return {
      bvhNodes:        { cpuData: nodeBuf, count: 1, byteLength: 32 },
      bvhIndex:        { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      bvhBeerColors:   { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      // 4 vertices × stride-4 × 4 B = 64 B (room for the affected
      // primitive's vertex slice + headroom).
      bvhPositions:    { cpuData: new ArrayBuffer(64), count: 4, byteLength: 64 },
      emitters:        { cpuData: new ArrayBuffer(16), count: 0, byteLength: 16 },
      emitterCdf:      { cpuData: new ArrayBuffer(16), count: 0, byteLength: 16 },
      emitterCount:    0,
      totalEmissivePower: 0,
      mergedGeometry:  new THREE.BufferGeometry(),
      meshVertexRanges: [
        { name: 'mesh-a', vertexStart: 0, vertexCount: 3, triStart: 0, triCount: 1,
          matrixWorldAtBuild: new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
          ]) },
      ],
      bvhIndicesStride3: new Uint32Array([0, 1, 2]),
      triangleMaterialIds: { cpuData: new Uint32Array(1).buffer, byteLength: 4, count: 1 },
      buildMaterials: [new THREE.MeshStandardMaterial()],
      coreMaterials: [],
      emitterNormals: new Float32Array(16),
      bvhMode: 'merged' as const,
      primitiveTlasBindings: [],
    };
  }

  const buildFn = vi.fn(() => {
    const idx = state.buildBVHCalls.length;
    state.buildBVHCalls.push({ idx });
    return makeFakeBuffers();
  });

  return {
    buildReSTIRSceneBVH: buildFn,
    buildReSTIRSceneBVHForScene: buildFn,
    rebuildEmitterBuffersFromCoreScene: vi.fn(() => ({
      emitters: { cpuData: new ArrayBuffer(80), byteLength: 80, count: 1 },
      emitterCdf: { cpuData: new Float32Array(1).buffer, byteLength: 4, count: 1 },
      emitterCount: 1,
      totalEmissivePower: 2,
    })),
    rebuildEmitterBuffersFromSceneRoots: vi.fn(() => ({
      emitters: { cpuData: new ArrayBuffer(80), byteLength: 80, count: 1 },
      emitterCdf: { cpuData: new Float32Array(1).buffer, byteLength: 4, count: 1 },
      emitterCount: 1,
      totalEmissivePower: 2,
    })),
    disposeSceneBVH: vi.fn(),
  };
});

// vitrumSceneToThree is mocked to return a THREE.Scene containing a
// mesh whose `.name === 'mesh-a'` so `updatePrimitive` can locate it
// via the standard `traverseVisible` walk.
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
    findMeshByPrimitiveId: vi.fn((root: THREE.Object3D, id: string): THREE.Mesh | null => {
      let found: THREE.Mesh | null = null;
      root.traverseVisible((obj) => {
        if (found != null) return;
        if (!(obj as THREE.Mesh).isMesh) return;
        if (obj.uuid === id || obj.name === id) found = obj as THREE.Mesh;
      });
      return found;
    }),
  };
});

// Theme-B characterization spy — count `propagateBvhToGiSubsystems` calls so a
// test can pin WHICH updatePrimitive branches apply the GI subsystems
// (transform / positions / topology / skinned) vs which deliberately do NOT
// (material-only). The real implementation is otherwise a no-op against the
// mocked subsystems, so swapping in a spy preserves observable behaviour.
interface GiPropagationState {
  calls: Array<{ rcRefitBounds: unknown }>;
}
vi.mock('../src/HybridEngineGiPropagation.js', async () => {
  const g = globalThis as unknown as { __HYBRID_GIPROP_STATE__?: GiPropagationState };
  if (!g.__HYBRID_GIPROP_STATE__) g.__HYBRID_GIPROP_STATE__ = { calls: [] };
  const state = g.__HYBRID_GIPROP_STATE__;
  return {
    propagateBvhToGiSubsystems: vi.fn((deps: { rcRefitBounds?: unknown }) => {
      state.calls.push({ rcRefitBounds: deps.rcRefitBounds });
    }),
  };
});

import { HybridEngine } from '../src/HybridEngine.js';
import { asMat4, type Scene } from '@vitrum/core';
import {
  rebuildEmitterBuffersFromCoreScene,
  rebuildEmitterBuffersFromSceneRoots,
} from '../src/restir/bvhCompute.js';

function getGiPropState(): GiPropagationState {
  return (globalThis as unknown as { __HYBRID_GIPROP_STATE__: GiPropagationState }).__HYBRID_GIPROP_STATE__;
}

function getState(): GeoUpdateState {
  return (globalThis as unknown as { __HYBRID_GEO_STATE__: GeoUpdateState }).__HYBRID_GEO_STATE__;
}

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeEngine(w = 64, h = 64): HybridEngine {
  return new HybridEngine({
    device:                makeMockDevice(),
    width:                 w,
    height:                h,
    primaryLightDir:       [0, -1, 0],
    primaryLightIntensity: 1.0,
    skyTint:               [1, 1, 1],
    skyIrradiance:         1.0,
  });
}

const SCENE_WITH_MESH: Scene = {
  primitives: [
    {
      id: 'mesh-a',
      kind: 'mesh',
      mesh: {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        indices:   new Uint32Array([0, 1, 2]),
      },
      material: { kind: 'lambertian', albedo: [1, 1, 1] },
    } as unknown as Scene['primitives'][number],
  ],
  emitters: [],
  environment: { kind: 'none' },
};

// Uses a `point` emitter because this test patches position-bearing fixture
// lights. Scene-supplied `directional` emitters are supported too (mapped to
// the DDGI sun path), but they do not exercise the point/spot fixture update
// branch this test is pinning.
const SCENE_WITH_EMITTER: Scene = {
  ...SCENE_WITH_MESH,
  emitters: [
    {
      id: 'point-a',
      kind: 'point',
      color: [1, 1, 1],
      intensity: 1,
      position: [0, 5, 0],
    },
  ],
};

const SCENE_WITH_INSTANCED_MESH: Scene = {
  primitives: [
    {
      id: 'inst-a',
      kind: 'instanced-mesh',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      instances: [
        asMat4([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
        ]),
      ],
    },
  ],
  emitters: [],
  environment: { kind: 'none' },
};

beforeEach(() => {
  const s = getState();
  s.pipelineInitDeferreds.length = 0;
  s.pipelineConstructed.length = 0;
  s.buildBVHCalls.length = 0;
  getGiPropState().calls.length = 0;
});

async function waitForPipelineCount(n: number, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (getState().pipelineInitDeferreds.length < n) {
    if (Date.now() > deadline) throw new Error(`waitForPipelineCount(${n}) timed out`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

async function drainMicrotasks(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
  await new Promise((r) => setTimeout(r, 50));
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

async function bootReadyEngine(): Promise<{ engine: HybridEngine; s: GeoUpdateState }> {
  const engine = makeEngine();
  const s = getState();
  engine.setScene(SCENE_WITH_MESH);
  await waitForPipelineCount(1);
  s.pipelineInitDeferreds[0]!.resolve();
  await drainMicrotasks();
  return { engine, s };
}

describe('HybridEngine.updatePrimitive — geometry change (A3 follow-up)', () => {
  it('flips supportsIncrementalScene to true', () => {
    const engine = makeEngine();
    expect(engine.capabilities.supportsIncrementalScene).toBe(true);
  });

  it('transform-only patch does NOT recompile the pipeline (no new pipeline construction)', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();
    expect(s.pipelineConstructed.length).toBe(1);

    const pipelineCountBefore = s.pipelineConstructed.length;
    const buildCountBefore    = s.buildBVHCalls.length;

    // Push a transform-only patch (translate the mesh +5 along X).
    engine.updatePrimitive!('mesh-a', {
      transform: asMat4([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        5, 0, 0, 1,
      ]),
    });

    // Convergence within 2 frames is satisfied by:
    //   (a) no full pipeline recompile (pipelineConstructed.length unchanged);
    //   (b) no full BVH rebuild (buildBVHCalls.length unchanged);
    //   (c) accumulator reset requested so the next frame starts at α=1.
    expect(s.pipelineConstructed.length).toBe(pipelineCountBefore);
    expect(s.buildBVHCalls.length).toBe(buildCountBefore);

    const livePipeline = s.pipelineConstructed[0]!;
    expect(livePipeline.refreshBvhRefit).toHaveBeenCalledTimes(1);
    expect(livePipeline.refreshBvhFullRebuild).not.toHaveBeenCalled();
    expect(livePipeline.requestAccumReset).toHaveBeenCalled();
  });

  it('transform-only refit writes both bvhNodes + the affected position slice', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    engine.updatePrimitive!('mesh-a', {
      transform: asMat4([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 2, 0, 1,
      ]),
    });

    const pipeline = s.pipelineConstructed[0]!;
    expect(pipeline.refreshBvhRefit).toHaveBeenCalledTimes(1);
    const call = pipeline.refreshBvhRefit.mock.calls[0]!;
    const nodeBytes = call[0] as ArrayBuffer;
    const slice = call[1] as { byteOffset: number; data: ArrayBuffer };
    // Single-leaf BVH (1 node × 32 B = 32 B); the refit helper rewrites
    // the bounds in place, so the buffer size is preserved.
    expect(nodeBytes.byteLength).toBe(32);
    // The fake meshVertexRanges puts mesh-a at vertexStart=0, count=3.
    // stride-4 float positions → 3 × 4 × 4 = 48 bytes starting at offset 0.
    expect(slice.byteOffset).toBe(0);
    expect(slice.data.byteLength).toBe(48);
  });

  it('applyGpuSkinnedRefit uploads BVH nodes only (no position slice)', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    const buildCountBefore = s.buildBVHCalls.length;
    engine.applyGpuSkinnedRefit(
      'mesh-a',
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0]),
      new Float32Array([0, 1, 0, 0, 1, 0, 0, 0, 1]),
    );

    expect(s.buildBVHCalls.length).toBe(buildCountBefore);
    const pipeline = s.pipelineConstructed[0]!;
    expect(pipeline.refreshBvhNodesOnly).toHaveBeenCalledTimes(1);
    expect(pipeline.refreshBvhRefit).not.toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
  });

  it('positions-only patch (A3 fast path) refits BVH bounds — no full rebuild, no pipeline recompile', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    const pipelineCountBefore = s.pipelineConstructed.length;
    const buildCountBefore    = s.buildBVHCalls.length;

    // Same vertex count (3 verts × 3 floats = 9) as the original mesh-a
    // geometry — the BVH topology is preserved; only the AABB bounds
    // need to refit.
    engine.updatePrimitive!('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0]), // resized triangle
    });

    // A3 fast path — neither pipeline recompile NOR full BVH rebuild.
    expect(s.pipelineConstructed.length).toBe(pipelineCountBefore);
    expect(s.buildBVHCalls.length).toBe(buildCountBefore);

    const pipeline = s.pipelineConstructed[0]!;
    expect(pipeline.refreshBvhRefit).toHaveBeenCalledTimes(1);
    expect(pipeline.refreshBvhFullRebuild).not.toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
  });

  it('true topology patch (indices) triggers full BVH rebuild but NOT a pipeline recompile', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    const pipelineCountBefore = s.pipelineConstructed.length;
    const buildCountBefore    = s.buildBVHCalls.length;

    // Index buffer change — true topology change. The A3 fast path
    // does NOT trigger; this routes through topologyRebuild (Option (a)).
    engine.updatePrimitive!('mesh-a', {
      indices: new Uint32Array([0, 2, 1]), // winding flipped
    });

    expect(s.pipelineConstructed.length).toBe(pipelineCountBefore);
    expect(s.buildBVHCalls.length).toBe(buildCountBefore + 1);

    const pipeline = s.pipelineConstructed[0]!;
    expect(pipeline.refreshBvhFullRebuild).toHaveBeenCalledTimes(1);
    expect(pipeline.refreshBvhRefit).not.toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
  });

  it('positions patch with mismatched vertex count falls through to topology rebuild', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    const buildCountBefore = s.buildBVHCalls.length;

    // 4 vertices supplied vs cached 3 → vertex count mismatch.
    // positionsRefit detects this and routes to topologyRebuild.
    engine.updatePrimitive!('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
    });

    // Full rebuild ran, not refit.
    expect(s.buildBVHCalls.length).toBe(buildCountBefore + 1);
    const pipeline = s.pipelineConstructed[0]!;
    expect(pipeline.refreshBvhFullRebuild).toHaveBeenCalledTimes(1);
    expect(pipeline.refreshBvhRefit).not.toHaveBeenCalled();
  });

  it('material-only patch uses material fast path (no full BVH rebuild)', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    const buildCountBefore = s.buildBVHCalls.length;
    const pipeline = s.pipelineConstructed[0]!;
    engine.updatePrimitive!('mesh-a', {
      material: { baseColor: [0.5, 0.2, 0.7], roughness: 0.5, metallic: 0 },
    });
    expect(s.buildBVHCalls.length).toBe(buildCountBefore);
    expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
  });

  it('unknown primitive id throws', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    expect(() => engine.updatePrimitive!('mesh-not-found', {
      transform: asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    })).toThrow(/not found/);
  });

  it('call before setScene throws (no scene state)', () => {
    const engine = makeEngine();
    expect(() => engine.updatePrimitive!('mesh-a', {
      transform: asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    })).toThrow(/no scene set/);
  });

  it('call during setScene initialization throws instead of racing BVH/pipeline publish', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    const buildCountBefore = s.buildBVHCalls.length;

    expect(() => engine.updatePrimitive!('mesh-a', {
      transform: asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]),
    })).toThrow(/initializing/);
    expect(() => engine.applyGpuSkinnedRefit(
      'mesh-a',
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0]),
      new Float32Array([0, 1, 0, 0, 1, 0, 0, 0, 1]),
    )).toThrow(/initializing/);
    expect(s.buildBVHCalls.length).toBe(buildCountBefore);

    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();
  });
});

describe('HybridEngine lifecycle mutation guards', () => {
  it('blocks scene and primitive mutations after dispose', async () => {
    const { engine, s } = await bootReadyEngine();
    engine.dispose();

    const buildCountBefore = s.buildBVHCalls.length;
    expect(() => engine.setScene(SCENE_WITH_MESH)).toThrow(/disposed/);
    expect(() => engine.setSize(128, 128)).toThrow(/disposed/);
    expect(() => engine.updatePrimitive!('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0]),
    })).toThrow(/disposed/);
    expect(() => engine.applyGpuSkinnedRefit(
      'mesh-a',
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0]),
      new Float32Array([0, 1, 0, 0, 1, 0, 0, 0, 1]),
    )).toThrow(/disposed/);
    expect(() => engine.reset()).toThrow(/disposed/);
    expect(s.buildBVHCalls.length).toBe(buildCountBefore);
  });

  it('pause/resume throw after dispose and no-op while initializing', async () => {
    const engine = makeEngine();
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);

    expect(() => engine.pause()).not.toThrow();
    expect(engine.state).toBe('initializing');
    expect(() => engine.resume()).not.toThrow();
    expect(engine.state).toBe('initializing');

    getState().pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();
    expect(engine.state).toBe('ready');

    engine.dispose();
    expect(() => engine.pause()).toThrow(/disposed/);
    expect(() => engine.resume()).toThrow(/disposed/);
  });
});

describe('HybridEngine.updateEmitter', () => {
  it('refreshes emitter buffers without setScene when emitter patch is applied', async () => {
    const engine = makeEngine();
    const s = getState();
    engine.setScene(SCENE_WITH_EMITTER);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();
    const buildCountBefore = s.buildBVHCalls.length;
    const pipeline = s.pipelineConstructed[0]!;
    const coreRebuildsBefore = vi.mocked(rebuildEmitterBuffersFromCoreScene).mock.calls.length;
    const legacyRebuildsBefore = vi.mocked(rebuildEmitterBuffersFromSceneRoots).mock.calls.length;
    engine.updateEmitter!('point-a', { intensity: 2 });
    expect(s.buildBVHCalls.length).toBe(buildCountBefore);
    expect(vi.mocked(rebuildEmitterBuffersFromCoreScene).mock.calls.length).toBe(coreRebuildsBefore + 1);
    expect(vi.mocked(rebuildEmitterBuffersFromSceneRoots).mock.calls.length).toBe(legacyRebuildsBefore);
    expect(pipeline.updateEmitters).toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
  });

  it('throws on unknown emitter id', async () => {
    const engine = makeEngine();
    const s = getState();
    engine.setScene(SCENE_WITH_EMITTER);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();
    expect(() => engine.updateEmitter!('missing', { intensity: 2 })).toThrow(/not found/);
  });

  it('throws when called before setScene', () => {
    const engine = makeEngine();
    expect(() => engine.updateEmitter!('point-a', { intensity: 2 })).toThrow(/no scene set/);
  });
});

// ── Theme-B characterization — updatePrimitive routing epilogue ───────────────
//
// Pins the observable epilogue of each updatePrimitive branch so the routing
// collapse (uniform result shape + one `_applyUpdateResult`) is verified to be
// behaviour-preserving:
//   - WHICH branch ran (refit vs full rebuild vs material slice),
//   - whether `_bvhBuffers` / `_lastScene` were re-assigned (observed via the
//     downstream re-upload calls + getBvhMode staying live),
//   - whether the GI subsystems were applied (transform / positions / topology /
//     skinned DO; material-only deliberately does NOT),
//   - that a wholesale topology field (`shape` / `instances` / `params`) routes
//     through a full setScene rebuild (P5: honors incrementalPatchSupport.topology
//     instead of throwing — geometry edits that need a primitive replacement do a
//     rebuild, not a fast path).
describe('HybridEngine.updatePrimitive — routing epilogue characterization (Theme B)', () => {
  async function bootEngine(scene: Scene = SCENE_WITH_MESH): Promise<{ engine: HybridEngine; s: GeoUpdateState }> {
    const engine = makeEngine();
    const s = getState();
    engine.setScene(scene);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();
    // setScene's own publishBvh propagation runs during init; clear so the
    // assertions below count only the updatePrimitive-triggered call.
    getGiPropState().calls.length = 0;
    return { engine, s };
  }

  it('transform-only patch APPLIES GI subsystems exactly once', async () => {
    const { engine } = await bootEngine();
    engine.updatePrimitive!('mesh-a', {
      transform: asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]),
    });
    expect(getGiPropState().calls.length).toBe(1);
  });

  it('positions-only patch APPLIES GI subsystems exactly once', async () => {
    const { engine } = await bootEngine();
    engine.updatePrimitive!('mesh-a', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0]),
    });
    expect(getGiPropState().calls.length).toBe(1);
  });

  it('topology patch (indices) APPLIES GI subsystems exactly once', async () => {
    const { engine } = await bootEngine();
    engine.updatePrimitive!('mesh-a', { indices: new Uint32Array([0, 2, 1]) });
    expect(getGiPropState().calls.length).toBe(1);
  });

  it('material-only patch does NOT apply GI subsystems (epilogue distinction)', async () => {
    const { engine, s } = await bootEngine();
    const pipeline = s.pipelineConstructed[0]!;
    engine.updatePrimitive!('mesh-a', {
      material: { baseColor: [0.5, 0.2, 0.7], roughness: 0.5, metallic: 0 },
    });
    // Material fast path ran (slice upload + accum reset) ...
    expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
    // ... but deliberately skips the GI-subsystem propagation.
    expect(getGiPropState().calls.length).toBe(0);
  });

  it('skinned-mesh refit APPLIES GI subsystems exactly once', async () => {
    const { engine } = await bootEngine();
    engine.applyGpuSkinnedRefit(
      'mesh-a',
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 2, 0]),
      new Float32Array([0, 1, 0, 0, 1, 0, 0, 0, 1]),
    );
    expect(getGiPropState().calls.length).toBe(1);
  });

  it('valid wholesale instance-count patch routes through a full setScene rebuild — no throw (P5)', async () => {
    const { engine, s } = await bootEngine(SCENE_WITH_INSTANCED_MESH);
    // `instances` can't be an in-place THREE.Mesh attribute edit, so
    // updatePrimitive routes it through the canonical patchScene -> setScene
    // spine instead of throwing "call setScene()" — honoring
    // incrementalPatchSupport.topology.
    const nextInstances = [
      asMat4([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ]),
      asMat4([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        4, 0, 0, 1,
      ]),
    ];
    expect(() => engine.updatePrimitive!('inst-a', {
      instances: nextInstances,
    } as unknown as Partial<Scene['primitives'][number]>)).not.toThrow();
    // setScene tore down + rebuilt the pipeline (a fresh construction + init).
    await waitForPipelineCount(2);
    s.pipelineInitDeferreds[1]!.resolve();
    await drainMicrotasks();
    expect(s.pipelineConstructed.length).toBe(2);

    const stored = (engine as unknown as { _lastScene: Scene | null })._lastScene;
    const primitive = stored?.primitives[0];
    expect(primitive?.kind).toBe('instanced-mesh');
    if (primitive?.kind !== 'instanced-mesh') throw new Error('expected instanced-mesh');
    expect(primitive.instances).toHaveLength(2);
  });

  it('invalid wholesale fields still use core invariants and do not start a rebuild', async () => {
    const { engine, s } = await bootEngine();
    expect(() => engine.updatePrimitive!('mesh-a', {
      shape: 'sphere',
    } as never)).toThrow(/cannot accept analytic "shape"/);
    expect(() => engine.updatePrimitive!('mesh-a', {
      kind: 'analytic',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
    } as never)).toThrow(/kind cannot change/);
    expect(s.pipelineConstructed.length).toBe(1);
  });

  it('keeps `_bvhBuffers` live after a transform refit (getBvhMode still resolves)', async () => {
    const { engine } = await bootEngine();
    expect(engine.getBvhMode()).toBe('merged');
    engine.updatePrimitive!('mesh-a', {
      transform: asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1]),
    });
    // The epilogue re-assigned `_bvhBuffers` to the refit result; the mode
    // is preserved (refit keeps the same buffer in 'merged' mode).
    expect(engine.getBvhMode()).toBe('merged');
  });

  it('no-recognised-field patch is a no-op (no branch, no subsystems, no throw)', async () => {
    const { engine, s } = await bootEngine();
    const buildBefore = s.buildBVHCalls.length;
    expect(() => engine.updatePrimitive!('mesh-a', {})).not.toThrow();
    expect(s.buildBVHCalls.length).toBe(buildBefore);
    expect(getGiPropState().calls.length).toBe(0);
  });
});
