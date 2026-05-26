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
      emitterNormals: new Float32Array(16),
    };
  }

  return {
    buildReSTIRSceneBVH: vi.fn(() => {
      const idx = state.buildBVHCalls.length;
      state.buildBVHCalls.push({ idx });
      return makeFakeBuffers();
    }),
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
  };
});

import { HybridEngine } from '../src/HybridEngine.js';
import { asMat4, type Scene } from '@vitrum/core';

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

const SCENE_WITH_EMITTER: Scene = {
  ...SCENE_WITH_MESH,
  emitters: [
    {
      id: 'sun-a',
      kind: 'directional',
      color: [1, 1, 1],
      intensity: 1,
      direction: [0, -1, 0],
    },
  ],
};

beforeEach(() => {
  const s = getState();
  s.pipelineInitDeferreds.length = 0;
  s.pipelineConstructed.length = 0;
  s.buildBVHCalls.length = 0;
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
    engine.updateEmitter!('sun-a', { intensity: 2 });
    expect(s.buildBVHCalls.length).toBe(buildCountBefore);
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
    expect(() => engine.updateEmitter!('sun-a', { intensity: 2 })).toThrow(/no scene set/);
  });
});
