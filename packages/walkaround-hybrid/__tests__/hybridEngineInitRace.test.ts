/**
 * Fix 1 — In-flight init guard for HybridEngine.
 *
 * The bug: `_initPipeline()` was fire-and-forget with no in-flight guard.
 * When the host (or vitrum's own factory bootstrap) called `setScene()` while
 * a previous `_initPipeline()` was still running, both async chains wrote the
 * shared state (`_bvhBuffers`, `_pipeline`, `_ddgiTraversalScene`) at multiple
 * points. The loser leaked ~1 GB of GPU resources (18 rgba16float full-res
 * textures + BVH buffers + DDGI atlases). React StrictMode + window resizes
 * compounded this.
 *
 * The fix: `_initSeq` is incremented at the start of every `_initPipeline()`.
 * Each in-flight init captures `mySeq` at entry, then re-checks
 * `mySeq === this._initSeq` before each shared-state write. If the value
 * drifted, the loser disposes its locals and bails without mutating shared
 * state.
 *
 * These tests exercise the guard by:
 *   - Mocking `WalkaroundGPUPipeline` so `initialize()` resolves on a
 *     test-controlled deferred — that's how we hold one chain mid-flight.
 *   - Mocking `buildReSTIRSceneBVH` / `disposeSceneBVH` /
 *     `vitrumSceneToThree` / `disposeVitrumThreeSceneRoot` so we can spy on
 *     dispose calls without needing a real GPU device.
 *   - Firing N concurrent setScene() / reset() calls and asserting only the
 *     last chain wins; the losers' pipelines + BVHs are disposed.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

// ── Module-level deferred infrastructure ────────────────────────────────────
//
// vi.mock factories are hoisted, so they cannot reference top-level test
// variables directly. We register module-level state on `globalThis` from
// inside the factory, then read it back in the test bodies.

interface RaceState {
  pipelineInitDeferreds: Array<{
    promise: Promise<void>;
    resolve: () => void;
    reject: (e: unknown) => void;
  }>;
  pipelineDisposeSpies: Array<ReturnType<typeof vi.fn>>;
  pipelineConstructed: Array<{ index: number; initialized: boolean }>;
  buildBVHCalls: Array<{ idx: number }>;
  disposeBVHCalls: unknown[];
  sceneToThreeCalls: THREE.Object3D[];
  disposeSceneRootCalls: THREE.Object3D[];
}

vi.mock('../src/pipeline/WalkaroundGPUPipeline.js', async () => {
  // Initialise per-test-run state on globalThis. beforeEach clears it.
  const g = globalThis as unknown as { __HYBRID_RACE_STATE__?: RaceState };
  if (!g.__HYBRID_RACE_STATE__) {
    g.__HYBRID_RACE_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineDisposeSpies: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
      disposeBVHCalls: [],
      sceneToThreeCalls: [],
      disposeSceneRootCalls: [],
    };
  }
  const state = g.__HYBRID_RACE_STATE__;

  class MockWalkaroundGPUPipeline {
    public readonly index: number;
    public initialized = false;
    public disposed = false;
    public dispose: ReturnType<typeof vi.fn>;

    constructor(_device: GPUDevice, _w: number, _h: number) {
      this.index = state.pipelineInitDeferreds.length;
      let resolve!: () => void;
      let reject!: (e: unknown) => void;
      const promise = new Promise<void>((res, rej) => {
        resolve = res;
        reject = rej;
      });
      state.pipelineInitDeferreds.push({ promise, resolve, reject });
      this.dispose = vi.fn(() => {
        this.disposed = true;
      });
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

vi.mock('../src/restir/bvhCompute.js', async () => {
  const g = globalThis as unknown as { __HYBRID_RACE_STATE__?: RaceState };
  if (!g.__HYBRID_RACE_STATE__) {
    g.__HYBRID_RACE_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineDisposeSpies: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
      disposeBVHCalls: [],
      sceneToThreeCalls: [],
      disposeSceneRootCalls: [],
    };
  }
  const state = g.__HYBRID_RACE_STATE__;
  return {
    buildReSTIRSceneBVH: vi.fn(() => {
      const idx = state.buildBVHCalls.length;
      state.buildBVHCalls.push({ idx });
      return {
        bvhNodes: { cpuData: new ArrayBuffer(32), count: 1 },
        bvhIndex: { cpuData: new ArrayBuffer(16), count: 1 },
        bvhBeerColors: { cpuData: new ArrayBuffer(16), count: 1 },
        bvhPositions: { cpuData: new ArrayBuffer(16), count: 1 },
        emitters: { cpuData: new ArrayBuffer(16), count: 0 },
        emitterCdf: { cpuData: new ArrayBuffer(16), count: 0 },
        emitterCount: 0,
        totalEmissivePower: 0,
        __testIdx: idx,
      };
    }),
    disposeSceneBVH: vi.fn((b: unknown) => {
      state.disposeBVHCalls.push(b);
    }),
  };
});

vi.mock('@vitrum/three-bindings', async () => {
  const g = globalThis as unknown as { __HYBRID_RACE_STATE__?: RaceState };
  if (!g.__HYBRID_RACE_STATE__) {
    g.__HYBRID_RACE_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineDisposeSpies: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
      disposeBVHCalls: [],
      sceneToThreeCalls: [],
      disposeSceneRootCalls: [],
    };
  }
  const state = g.__HYBRID_RACE_STATE__;
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

// Now import after mocks are registered.
import { HybridEngine } from '../src/HybridEngine.js';
import type { Scene } from '@vitrum/core';

function getState(): RaceState {
  return (globalThis as unknown as { __HYBRID_RACE_STATE__: RaceState }).__HYBRID_RACE_STATE__;
}

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeEngine(): HybridEngine {
  // threeScene supplies the BVH source so scene-readiness passes fast and
  // the path goes through buildReSTIRSceneBVH directly. defaultIsSceneReady
  // requires at least one triangle; we add a single-triangle mesh.
  const scene = new THREE.Scene();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3),
  );
  scene.add(new THREE.Mesh(geom, new THREE.MeshBasicMaterial()));
  return new HybridEngine({
    device: makeMockDevice(),
    width: 64,
    height: 64,
    threeScene: scene,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1.0,
    skyTint: [1, 1, 1],
    skyIrradiance: 1.0,
  });
}

const SCENE_WITH_MESH: Scene = {
  primitives: [
    {
      id: 'mesh-a',
      kind: 'mesh',
      mesh: {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 1, 2]),
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

// Helper: wait until N pipeline mocks have been constructed.
async function waitForPipelineCount(n: number, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (getState().pipelineInitDeferreds.length < n) {
    if (Date.now() > deadline)
      throw new Error(
        `waitForPipelineCount(${n}) timed out (have ${getState().pipelineInitDeferreds.length})`,
      );
    await new Promise((r) => setTimeout(r, 10));
  }
}

// Drain microtasks + setTimeout(0) repeatedly so pipeline.initialize() chains
// can traverse their finally blocks after the deferreds resolve.
async function drainMicrotasks(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
  await new Promise((r) => setTimeout(r, 50));
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

describe('HybridEngine — in-flight init guard (Fix 1)', () => {
  it('three concurrent setScene() calls — only the last one wins; first two dispose their locals', async () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;
    const s = getState();

    // First setScene fires init-A. Wait until init-A's pipeline is constructed
    // and blocking on initialize() (its deferred is unresolved).
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);

    // Second setScene fires init-B. This bumps _initSeq; init-A is now stale.
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(2);

    // Third setScene fires init-C. Both A and B are now stale.
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(3);

    // Resolve all three pipeline.initialize() calls in order. After the
    // microtask queue drains, only init-C should have published _pipeline.
    s.pipelineInitDeferreds[0]!.resolve();
    s.pipelineInitDeferreds[1]!.resolve();
    s.pipelineInitDeferreds[2]!.resolve();
    await drainMicrotasks();

    // Pipeline 0 (init-A) and pipeline 1 (init-B) lost the race — both should
    // be disposed by the finally block. Pipeline 2 (init-C) is the live one.
    expect(s.pipelineDisposeSpies[0]).toHaveBeenCalled();
    expect(s.pipelineDisposeSpies[1]).toHaveBeenCalled();
    expect(s.pipelineDisposeSpies[2]).not.toHaveBeenCalled();

    // _pipeline should be the last (third post-bootstrap) constructed pipeline.
    const livePipeline = e['_pipeline'] as { index: number } | null;
    expect(livePipeline).not.toBeNull();
    expect(livePipeline!.index).toBe(s.pipelineInitDeferreds.length - 1);
  });

  it('losers dispose their BVH locals (no leaked SceneBVHBuffers)', async () => {
    const engine = makeEngine();
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(2);

    s.pipelineInitDeferreds[0]!.resolve();
    s.pipelineInitDeferreds[1]!.resolve();
    await drainMicrotasks();

    // We built BVHs for init-A and init-B + the bootstrap setScene from
    // construction (which also goes through _initPipeline). Each loser BVH
    // should be disposed.
    expect(s.disposeBVHCalls.length).toBeGreaterThan(0);
  });

  it('losers dispose their synthesized THREE.Scene (no leaked traversal-scene root)', async () => {
    const engine = makeEngine();
    const s = getState();
    // Use vitrum-mesh path so vitrumSceneToThree is invoked → synthesized
    // scene roots are owned by the engine and must be disposed on race.
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);
    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(2);

    s.pipelineInitDeferreds[0]!.resolve();
    s.pipelineInitDeferreds[1]!.resolve();
    await drainMicrotasks();

    // The loser's synthesized scene should be disposed (or replaced).
    expect(s.disposeSceneRootCalls.length).toBeGreaterThan(0);
  });

  it('reset() during in-flight init invalidates the in-flight chain', async () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;
    const s = getState();

    engine.setScene(SCENE_WITH_MESH);
    await waitForPipelineCount(1);

    // reset() calls _teardownPipeline + _initPipeline → bumps _initSeq.
    engine.reset();
    await waitForPipelineCount(2);

    s.pipelineInitDeferreds[0]!.resolve();
    s.pipelineInitDeferreds[1]!.resolve();
    await drainMicrotasks();

    // First (stale) pipeline should be disposed; second is live.
    expect(s.pipelineDisposeSpies[0]).toHaveBeenCalled();
    const livePipeline = e['_pipeline'] as { index: number } | null;
    expect(livePipeline).not.toBeNull();
    expect(livePipeline!.index).toBe(s.pipelineInitDeferreds.length - 1);
  });

  it('_initSeq increments monotonically', async () => {
    const engine = makeEngine();
    const e = engine as unknown as Record<string, unknown>;

    // Initial bootstrap from construction may or may not have bumped seq;
    // capture starting value and assert subsequent setScene calls bump.
    const startSeq = e['_initSeq'] as number;
    engine.setScene(SCENE_WITH_MESH);
    expect(e['_initSeq']).toBe(startSeq + 1);
    engine.setScene(SCENE_WITH_MESH);
    expect(e['_initSeq']).toBe(startSeq + 2);
    engine.reset();
    expect(e['_initSeq']).toBe(startSeq + 3);
  });
});
