/**
 * Fix 2 — `dispose()` is honest about late-resolving init chains.
 *
 * The bug: `dispose()` was synchronous and tore down only what was already
 * published (`_pipeline`, `_bvhBuffers`, `_ddgiTraversalScene`). If an
 * `_initPipeline()` chain was mid-flight (waiting on `pipeline.initialize()`),
 * its post-await publish would still write the just-built pipeline into
 * `_pipeline` AFTER `dispose()` had run — the engine was disposed but the
 * GPU resources were live and orphaned (~1 GB leak on 4K).
 *
 * The fix: `dispose()` checks `_initRunning`. If an init is in flight, it
 * sets `_pendingTeardown` and `_disposed`, transitions state to
 * `'disposed'`, and DEFERS the actual `_teardownPipeline()` + `_ddgi.dispose()`
 * to the in-flight chain's `finally` block. The chain checks
 * `_pendingTeardown` at every shared-state-write checkpoint and bails
 * without publishing; its finally then finalises teardown.
 *
 * dispose() remains synchronous (no `Promise<void>` API ripple into hosts).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

interface RaceState {
  pipelineInitDeferreds: Array<{ promise: Promise<void>; resolve: () => void; reject: (e: unknown) => void }>;
  pipelineDisposeSpies: Array<ReturnType<typeof vi.fn>>;
  pipelineConstructed: Array<{ index: number; initialized: boolean; disposed: boolean }>;
  buildBVHCalls: Array<{ idx: number }>;
  disposeBVHCalls: unknown[];
  sceneToThreeCalls: THREE.Object3D[];
  disposeSceneRootCalls: THREE.Object3D[];
}

vi.mock('../src/pipeline/WalkaroundGPUPipeline.js', async () => {
  const g = globalThis as unknown as { __HYBRID_DISPOSE_STATE__?: RaceState };
  if (!g.__HYBRID_DISPOSE_STATE__) {
    g.__HYBRID_DISPOSE_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineDisposeSpies: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
      disposeBVHCalls: [],
      sceneToThreeCalls: [],
      disposeSceneRootCalls: [],
    };
  }
  const state = g.__HYBRID_DISPOSE_STATE__;

  class MockWalkaroundGPUPipeline {
    public readonly index: number;
    public initialized = false;
    public disposed = false;
    public dispose: ReturnType<typeof vi.fn>;

    constructor(_device: GPUDevice, _w: number, _h: number) {
      this.index = state.pipelineInitDeferreds.length;
      let resolve!: () => void;
      let reject!: (e: unknown) => void;
      const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
      state.pipelineInitDeferreds.push({ promise, resolve, reject });
      this.dispose = vi.fn(() => { this.disposed = true; }) as ReturnType<typeof vi.fn>;
      state.pipelineDisposeSpies.push(this.dispose);
      state.pipelineConstructed.push(this);
    }

    async initialize(): Promise<void> {
      await state.pipelineInitDeferreds[this.index]!.promise;
      this.initialized = true;
    }

    requestAccumReset = vi.fn();
    presentLastFrame = vi.fn();
    setDDGIInputs = vi.fn();
    renderFrame = vi.fn();
  }

  return {
    WalkaroundGPUPipeline: MockWalkaroundGPUPipeline,
    HYBRID_WEBGPU_REQUIRED_LIMITS: {},
    HYBRID_WEBGPU_REQUIRED_FEATURES: [],
  };
});

vi.mock('../src/restir/bvhCore.js', async () => {
  const g = globalThis as unknown as { __HYBRID_DISPOSE_STATE__?: RaceState };
  if (!g.__HYBRID_DISPOSE_STATE__) {
    g.__HYBRID_DISPOSE_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineDisposeSpies: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
      disposeBVHCalls: [],
      sceneToThreeCalls: [],
      disposeSceneRootCalls: [],
    };
  }
  const state = g.__HYBRID_DISPOSE_STATE__;
  const buildFn = vi.fn(() => {
    const idx = state.buildBVHCalls.length;
    state.buildBVHCalls.push({ idx });
    return {
      bvhMode: 'merged' as const,
      primitiveTlasBindings: [],
      bvhNodes:        { cpuData: new ArrayBuffer(32), count: 1 },
      bvhIndex:        { cpuData: new ArrayBuffer(16), count: 1 },
      bvhBeerColors:   { cpuData: new ArrayBuffer(16), count: 1 },
      bvhPositions:    { cpuData: new ArrayBuffer(16), count: 1 },
      emitters:        { cpuData: new ArrayBuffer(16), count: 0 },
      emitterCdf:      { cpuData: new ArrayBuffer(16), count: 0 },
      emitterCount:    0,
      totalEmissivePower: 0,
      mergedGeometry:  new THREE.BufferGeometry(),
      meshVertexRanges: [],
      bvhIndicesStride3: new Uint32Array(0),
      triangleMaterialIds: { cpuData: new ArrayBuffer(4), count: 0, byteLength: 4 },
      buildMaterials: [],
      emitterNormals: new Float32Array(0),
      __testIdx: idx,
    };
  });
  return {
    buildReSTIRSceneBVHForCoreScene: buildFn,
    disposeSceneBVH: vi.fn((b: unknown) => {
      state.disposeBVHCalls.push(b);
    }),
  };
});

vi.mock('@vitrum/three-bindings', async () => {
  const g = globalThis as unknown as { __HYBRID_DISPOSE_STATE__?: RaceState };
  if (!g.__HYBRID_DISPOSE_STATE__) {
    g.__HYBRID_DISPOSE_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineDisposeSpies: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
      disposeBVHCalls: [],
      sceneToThreeCalls: [],
      disposeSceneRootCalls: [],
    };
  }
  const state = g.__HYBRID_DISPOSE_STATE__;
  return {
    vitrumSceneToThree: vi.fn(() => {
      const root = new THREE.Scene();
      state.sceneToThreeCalls.push(root);
      return root;
    }),
    disposeVitrumThreeSceneRoot: vi.fn((r: THREE.Object3D) => {
      state.disposeSceneRootCalls.push(r);
    }),
  };
});

import { HybridEngine } from '../src/HybridEngine.js';
import type { Scene } from '@vitrum/core';

function getState(): RaceState {
  return (globalThis as unknown as { __HYBRID_DISPOSE_STATE__: RaceState }).__HYBRID_DISPOSE_STATE__;
}

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeEngine(): HybridEngine {
  const scene = new THREE.Scene();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  scene.add(new THREE.Mesh(geom, new THREE.MeshBasicMaterial()));
  return new HybridEngine({
    device:                makeMockDevice(),
    width:                 64,
    height:                64,
    threeScene:            scene,
    primaryLightDir:       [0, -1, 0],
    primaryLightIntensity: 1.0,
    skyTint:               [1, 1, 1],
    skyIrradiance:         1.0,
  });
}

function makeEngineWithoutThreeScene(): HybridEngine {
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

beforeEach(() => {
  const s = getState();
  s.pipelineInitDeferreds.length = 0;
  s.pipelineDisposeSpies.length = 0;
  s.pipelineConstructed.length = 0;
  s.buildBVHCalls.length = 0;
  s.disposeBVHCalls.length = 0;
  s.sceneToThreeCalls.length = 0;
  s.disposeSceneRootCalls.length = 0;
});

async function waitForPipelineCount(n: number, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (getState().pipelineInitDeferreds.length < n) {
    if (Date.now() > deadline) throw new Error(`waitForPipelineCount(${n}) timed out (have ${getState().pipelineInitDeferreds.length})`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

async function drainMicrotasks(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
  await new Promise((r) => setTimeout(r, 50));
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

describe('HybridEngine — dispose() honours in-flight init (Fix 2)', () => {
  it('dispose() before init finishes still disposes the late pipeline (no leaked GPU resources)', async () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;
    const s = getState();

    // Fire setScene → init starts.
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    expect(s.pipelineInitDeferreds.length).toBeGreaterThanOrEqual(1);
    const inFlightIdx = s.pipelineInitDeferreds.length - 1;

    // Dispose the engine while the init is still awaiting initialize().
    expect(e['_initRunning']).toBe(true);
    engine.dispose();

    // dispose() should have transitioned state to 'disposed' and set the
    // pending-teardown flag — but NOT torn down the in-flight pipeline
    // yet (it's still being built).
    expect(engine.state).toBe('disposed');
    expect(e['_pendingTeardown']).toBe(true);
    // The in-flight pipeline hasn't been disposed by dispose() itself.
    expect(s.pipelineDisposeSpies[inFlightIdx]).not.toHaveBeenCalled();

    // Now resolve the in-flight pipeline.initialize(). The chain's finally
    // block sees _pendingTeardown and disposes the late-built pipeline.
    s.pipelineInitDeferreds[inFlightIdx]!.resolve();
    await drainMicrotasks();

    // The in-flight pipeline must now be disposed (not leaked).
    expect(s.pipelineDisposeSpies[inFlightIdx]).toHaveBeenCalled();
    // _pipeline must be null — nothing should have been published.
    expect(e['_pipeline']).toBeNull();
    expect(e['_bvhBuffers']).toBeNull();
    expect(s.sceneToThreeCalls.length).toBe(0);
    expect(s.disposeSceneRootCalls.length).toBe(0);
  });

  it('dispose() with no in-flight init runs synchronously (no _pendingTeardown)', async () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;
    const s = getState();

    // Drive an init to completion so there's a published pipeline + BVH to
    // tear down.
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    // No init in flight now.
    expect(e['_initRunning']).toBe(false);

    engine.dispose();

    // Synchronous teardown — no deferred work.
    expect(engine.state).toBe('disposed');
    expect(e['_pendingTeardown']).toBe(false);
  });

  it('dispose() is idempotent when called twice', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    s.pipelineInitDeferreds[0]!.resolve();
    await drainMicrotasks();

    engine.dispose();
    expect(engine.state).toBe('disposed');

    // Second dispose is a no-op.
    expect(() => engine.dispose()).not.toThrow();
    expect(engine.state).toBe('disposed');
  });

  it('dispose() during in-flight init also disposes whatever the in-flight chain published mid-checkpoint', async () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;
    const s = getState();

    // Fire setScene; pipeline-A is mid-flight.
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);

    // Dispose now. Pipeline mid-flight; nothing published yet from this
    // chain (its first checkpoint hasn't been hit since it's blocked on
    // initialize()). _pendingTeardown should be set.
    engine.dispose();
    expect(e['_pendingTeardown']).toBe(true);

    // Resolve the in-flight pipeline. Chain's finally disposes its own
    // pipeline local + finalises teardown of any prior published state.
    s.pipelineInitDeferreds[s.pipelineInitDeferreds.length - 1]!.resolve();
    await drainMicrotasks();

    // The pipeline that was mid-build is disposed.
    const lastIdx = s.pipelineInitDeferreds.length - 1;
    expect(s.pipelineDisposeSpies[lastIdx]).toHaveBeenCalled();
  });

  it('dispose() during in-flight init still transitions state to "disposed" immediately', async () => {
    const engine = makeEngine();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);

    expect(engine.state).not.toBe('disposed');
    engine.dispose();

    // Immediately reflects 'disposed' even though teardown is deferred.
    expect(engine.state).toBe('disposed');
  });
});
