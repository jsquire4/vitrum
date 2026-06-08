/**
 * HybridEngine.addPrimitive / removePrimitive (walkaround-hybrid).
 *
 * Pins the explicit whole-primitive add/remove contract on the realtime
 * stack. Both route a fresh mutated `Scene` copy through the engine's existing
 * `setScene` spine: `partitionSceneBySupport` → `_teardownPipeline` →
 * `_initCoordinator.startInit()` (the same full BVH / DDGI / ReSTIR rebuild +
 * temporal-accumulator reset the initial scene build runs). There is no
 * cheaper-than-rebuild path here — a geometry change invalidates every cached
 * GI signal, so the value is API consistency with pt-webgl / pt-webgpu, not a
 * perf win.
 *
 * Test seams (same shape as the pt-webgl add/remove test + the existing
 * capabilitiesPartition test):
 *   - `@vitrum/three-bindings` is stubbed so `vitrumSceneToThree` returns a
 *     real empty THREE.Scene (valid for the CPU BVH builder) without pulling in
 *     real texture disposal on the background chain / dispose().
 *   - `./restir/bvhCore.js` is wrapped so `buildReSTIRSceneBVHForCoreScene`
 *     RECORDS the vitrum `Scene` it was handed. The async init chain hands the
 *     BVH builder exactly the mutated primitive list (the new primitive on add,
 *     the survivors-only list on remove) — that same scene is what DDGI's probe
 *     rays + ReSTIR's BVH index off, so recording it proves the mutation
 *     reaches the GI scene + BVH. The builder returns a real (empty) BVH so the
 *     CPU publishBvh + DDGI sync phase completes; the WebGPU `pipeline.initialize`
 *     phase fails afterwards in this non-WebGPU env (expected, orthogonal — its
 *     console.error is silenced).
 *
 * Temporal/accumulation reset is asserted via the SYNCHRONOUS pipeline teardown
 * the mutation triggers: `_pipeline` is torn down to null and engine state goes
 * back to `'initializing'` (a fresh pipeline = blank temporal accumulator +
 * reservoirs + DDGI/ReSTIR history) — exactly what `setScene` does.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import * as THREE from 'three';
import type { Scene, ScenePrimitive } from '@vitrum/core';
import { HybridEngine } from '../HybridEngine.js';
import type { HybridEngineOptions } from '../HybridEngine.js';

vi.mock('@vitrum/three-bindings', () => ({
  vitrumSceneToThree: () => new THREE.Scene(),
  disposeVitrumThreeSceneRoot: () => undefined,
  solveSkin: () => ({ positions: new Float32Array(0), normals: new Float32Array(0) }),
  applyVitrumMaterialToMesh: () => undefined,
  findMeshByPrimitiveId: () => null,
}));

/** The vitrum scenes handed to `buildReSTIRSceneBVHForCoreScene`, in call order.
 *  Each entry is the scene a (re)build saw — proving WHICH primitive list
 *  reached the ReSTIR BVH + DDGI traversal scene. */
const bvhBuildScenes: Scene[] = [];

vi.mock('../restir/bvhCore.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../restir/bvhCore.js')>();
  return {
    ...actual,
    buildReSTIRSceneBVHForCoreScene: vi.fn((scene: Scene, ...rest: unknown[]) => {
      bvhBuildScenes.push(scene);
      return (actual.buildReSTIRSceneBVHForCoreScene as unknown as (...a: unknown[]) => unknown)(
        scene,
        ...rest,
      );
    }),
  };
});

function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(),
    createTexture: vi.fn(),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn() },
  } as unknown as GPUDevice;
}

function makeOpts(): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 3,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

function tri(id: string, x: number): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([x, 0, 0, x + 1, 0, 0, x, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

function baseScene(): Scene {
  return {
    primitives: [tri('mesh-a', 0), tri('mesh-b', 10)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** Engine-internal scene snapshot (the partition-filtered `_lastScene`). */
function storedIds(engine: HybridEngine): string[] {
  const stored = (engine as unknown as { _lastScene: Scene | null })._lastScene;
  return (stored?.primitives ?? []).map((p) => String(p.id));
}

/** Primitive ids of the most-recent ReSTIR-BVH (re)build. */
function lastBvhBuildIds(): string[] {
  const s = bvhBuildScenes[bvhBuildScenes.length - 1];
  return (s?.primitives ?? []).map((p) => String(p.id));
}

describe('walkaround-hybrid HybridEngine.addPrimitive / removePrimitive', () => {
  let errorSpy: MockInstance | undefined;
  let warnSpy: MockInstance | undefined;

  beforeEach(() => {
    bvhBuildScenes.length = 0;
    // The WebGPU pipeline.initialize phase fails in this non-WebGPU env after
    // the (CPU) BVH-build phase we assert on; silence its expected console.error.
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    errorSpy?.mockRestore();
    warnSpy?.mockRestore();
  });

  it('exposes addPrimitive / removePrimitive as functions (capability + ledger agree)', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      expect(typeof engine.addPrimitive).toBe('function');
      expect(typeof engine.removePrimitive).toBe('function');
      expect(engine.capabilities.supportsAddRemovePrimitive).toBe(true);
    } finally {
      engine.dispose();
    }
  });

  it('addPrimitive routes the new primitive (alongside survivors) into the ReSTIR BVH + DDGI scene', async () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(baseScene());
      // Synchronously, the partition-filtered scene already holds the survivors.
      expect(storedIds(engine)).toEqual(['mesh-a', 'mesh-b']);

      bvhBuildScenes.length = 0;
      engine.addPrimitive(tri('mesh-c', 20));

      // Synchronous post-add scene snapshot includes the appended primitive.
      expect(storedIds(engine)).toEqual(['mesh-a', 'mesh-b', 'mesh-c']);

      // The mutated primitive list reaches the ReSTIR BVH builder (= the DDGI
      // probe + ReSTIR traversal source), appended after the survivors.
      await vi.waitFor(() => {
        expect(lastBvhBuildIds()).toEqual(['mesh-a', 'mesh-b', 'mesh-c']);
      });
    } finally {
      engine.dispose();
    }
  });

  it('removePrimitive evicts the primitive from the BVH while survivors stay intact', async () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(baseScene());

      bvhBuildScenes.length = 0;
      engine.removePrimitive('mesh-a');

      expect(storedIds(engine)).toEqual(['mesh-b']);
      await vi.waitFor(() => {
        const ids = lastBvhBuildIds();
        expect(ids).toEqual(['mesh-b']);
        expect(ids).not.toContain('mesh-a');
      });
    } finally {
      engine.dispose();
    }
  });

  it('removing the last primitive yields a renderable (empty / sky-only) scene', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene({
        primitives: [tri('only', 0)],
        emitters: [],
        environment: { kind: 'none' },
      });
      expect(() => engine.removePrimitive('only')).not.toThrow();
      expect(storedIds(engine)).toEqual([]);
      // The engine is mid-(re)init for the empty scene, not errored/disposed.
      expect(engine.state).toBe('initializing');
    } finally {
      engine.dispose();
    }
  });

  it('resets temporal/accumulation state on add and on remove (pipeline torn down → re-init)', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(baseScene());

      // Simulate a live, ready pipeline so the teardown is observable.
      const fakePipeline = { dispose: vi.fn() };
      (engine as unknown as { _pipeline: unknown })._pipeline = fakePipeline;
      (engine as unknown as { _state: string })._state = 'ready';

      engine.addPrimitive(tri('mesh-c', 20));
      // setScene's _teardownPipeline disposed the old pipeline (blank
      // accumulator + reservoirs on the rebuild) and kicked a fresh init.
      expect(fakePipeline.dispose).toHaveBeenCalledTimes(1);
      expect((engine as unknown as { _pipeline: unknown })._pipeline).toBeNull();
      expect(engine.state).toBe('initializing');

      // Same teardown on remove.
      const fakePipeline2 = { dispose: vi.fn() };
      (engine as unknown as { _pipeline: unknown })._pipeline = fakePipeline2;
      (engine as unknown as { _state: string })._state = 'ready';
      engine.removePrimitive('mesh-c');
      expect(fakePipeline2.dispose).toHaveBeenCalledTimes(1);
      expect((engine as unknown as { _pipeline: unknown })._pipeline).toBeNull();
      expect(engine.state).toBe('initializing');
    } finally {
      engine.dispose();
    }
  });

  it('accepts an analytic addition: authored kind kept, generated mesh fallback rendered', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(baseScene());
      const analytic = {
        kind: 'analytic',
        id: 'sphere-x',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: { baseColor: [1, 0, 0], roughness: 0.3, metallic: 0 },
      } as unknown as ScenePrimitive;
      // analytic is a SUPPORTED kind now (generated-MeshPrimitive fallback):
      // partitionSceneBySupport keeps it — no warn-skip, no throw.
      expect(() => engine.addPrimitive(analytic)).not.toThrow();
      const warned = warnSpy.mock.calls.flat().map(String);
      expect(warned.some((m) => m.includes('sphere-x') && m.includes('not supported'))).toBe(false);
      // The authored snapshot keeps the analytic primitive as-authored…
      expect(storedIds(engine)).toEqual(['mesh-a', 'mesh-b', 'sphere-x']);
      const stored = (engine as unknown as { _lastScene: Scene })._lastScene;
      expect(stored.primitives.find((p) => p.id === 'sphere-x')?.kind).toBe('analytic');
      // …while the render-ingestion view replaces it with a generated mesh.
      const render = (engine as unknown as { _renderScene: Scene })._renderScene;
      const generated = render.primitives.find((p) => p.id === 'sphere-x');
      expect(generated?.kind).toBe('mesh');
    } finally {
      engine.dispose();
    }
  });

  it('addPrimitive throws on a duplicate id and leaves the scene unchanged', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(baseScene());
      bvhBuildScenes.length = 0;
      expect(() => engine.addPrimitive(tri('mesh-a', 5))).toThrow(/already exists/);
      // No rebuild ran — the scene was not touched on throw.
      expect(bvhBuildScenes.length).toBe(0);
      expect(storedIds(engine)).toEqual(['mesh-a', 'mesh-b']);
    } finally {
      engine.dispose();
    }
  });

  it('removePrimitive throws on a missing id and leaves the scene unchanged', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(baseScene());
      bvhBuildScenes.length = 0;
      expect(() => engine.removePrimitive('does-not-exist')).toThrow(/no primitive with id/);
      expect(bvhBuildScenes.length).toBe(0);
      expect(storedIds(engine)).toEqual(['mesh-a', 'mesh-b']);
    } finally {
      engine.dispose();
    }
  });

  it('addPrimitive before setScene throws', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      expect(() => engine.addPrimitive(tri('x', 0))).toThrow(/no scene set/);
    } finally {
      engine.dispose();
    }
  });

  it('removePrimitive before setScene throws', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      expect(() => engine.removePrimitive('mesh-a')).toThrow(/no scene set/);
    } finally {
      engine.dispose();
    }
  });

  it('addPrimitive after dispose throws', () => {
    const engine = new HybridEngine(makeOpts());
    engine.setScene(baseScene());
    engine.dispose();
    expect(() => engine.addPrimitive(tri('x', 0))).toThrow(/disposed/);
  });

  it('removePrimitive after dispose throws', () => {
    const engine = new HybridEngine(makeOpts());
    engine.setScene(baseScene());
    engine.dispose();
    expect(() => engine.removePrimitive('mesh-a')).toThrow(/disposed/);
  });

  it('addPrimitive then removePrimitive round-trips back to the original BVH primitive set', async () => {
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(baseScene());
      expect(() => engine.addPrimitive(tri('mesh-c', 20))).not.toThrow();
      expect(storedIds(engine)).toEqual(['mesh-a', 'mesh-b', 'mesh-c']);
      expect(() => engine.removePrimitive('mesh-c')).not.toThrow();
      expect(storedIds(engine)).toEqual(['mesh-a', 'mesh-b']);
      await vi.waitFor(() => {
        expect(lastBvhBuildIds()).toEqual(['mesh-a', 'mesh-b']);
      });
    } finally {
      engine.dispose();
    }
  });
});
