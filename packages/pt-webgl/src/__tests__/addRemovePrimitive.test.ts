import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import type { MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

// ─────────────────────────────────────────────────────────────────────────────
// Fork + three-bindings stubs.
//
// add/remove route through `setScene`, whose spine is
// `vitrumSceneToThree(scene)` → `expandInstancedMeshesInScene` →
// `tracer.setScene(threeRoot, camera)`. We stub `vitrumSceneToThree` to RECORD
// the vitrum scene it was handed (proving the mutated primitive list — the new
// primitive on add, the survivors-only list on remove — reaches the THREE
// conversion + the fork) and to return a per-primitive THREE-mesh root (so the
// material on each primitive is what would be packed into the MaterialsTexture).
// `tracer.setScene` / `tracer.reset` are spies so we can assert the full-repack
// path ran and the accumulator was cleared.
// ─────────────────────────────────────────────────────────────────────────────

const setScene = vi.fn();
const reset = vi.fn();
const updateMaterials = vi.fn();
const disposeGeom = vi.fn();
const disposeMat = vi.fn();

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

/** The vitrum scenes handed to `vitrumSceneToThree` across the test, in order.
 *  Index 0 is the initial setScene; each subsequent entry is an add/remove. */
const sceneToThreeCalls: Scene[] = [];

/** Build a stub THREE root that surfaces one stub Mesh per primitive in the
 *  vitrum scene. Each mesh carries `name`/`uuid` = primitive id and a
 *  `material` tag derived from the primitive's material baseColor so a test can
 *  assert WHICH materials are present in the converted root (i.e. that the new
 *  primitive's material reaches the fork-bound scene, and a removed primitive's
 *  does not). The root exposes the Object3D surface the engine touches:
 *  `traverse` (disposeObject3DTree + expandInstancedMeshesInScene) and
 *  `traverseVisible` (findAllMeshesByPrimitiveId). */
function makeStubRootFor(scene: Scene) {
  const meshes = scene.primitives.map((p) => {
    const mat = (p as MeshPrimitive).material;
    return {
      isMesh: true as const,
      name: String(p.id),
      uuid: String(p.id),
      geometry: { dispose: disposeGeom },
      material: { baseColor: mat?.baseColor, dispose: disposeMat },
    };
  });
  return {
    _primitiveIds: scene.primitives.map((p) => String(p.id)),
    updateMatrixWorld: vi.fn(),
    traverse: (cb: (o: unknown) => void) => meshes.forEach(cb),
    traverseVisible: (cb: (o: unknown) => void) => meshes.forEach(cb),
  };
}

vi.mock('@vitrum/three-bindings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/three-bindings')>();
  return {
    ...actual,
    vitrumSceneToThree: vi.fn((scene: Scene) => {
      sceneToThreeCalls.push(scene);
      return makeStubRootFor(scene);
    }),
    applyVitrumMaterialToMesh: vi.fn(),
  };
});

vi.mock('three-gpu-pathtracer', () => {
  class WebGLPathTracer {
    readonly target = { texture: {} };
    samples = 0;
    bounces = 0;
    transmissiveBounces = 0;
    filterGlossyFactor = 0;
    renderDelay = 0;
    minSamples = 0;
    dynamicLowRes = false;
    multipleImportanceSampling = false;
    tileRepeatFactors: Uint8Array | null = null;
    configureAdditiveAccumulation = vi.fn();
    readonly tiles = { set: vi.fn() };
    readonly _pathTracer = { material: { uniforms: {} } };

    // `tracer.setScene` clears the fork accumulator in production; the stub
    // mirrors that so a test can assert the reset side effect on add/remove.
    setScene = (...args: unknown[]): void => {
      setScene(...args);
      this.samples = 0;
    };
    setCamera(): void {}
    renderSample(): void {
      this.samples += 1;
    }
    reset = reset;
    dispose(): void {}
    updateEnvironment(): void {}
    updateMaterials = updateMaterials;
  }

  return { WebGLPathTracer };
});

/** A 1-triangle mesh primitive at the given x offset with a tagged material. */
function tri(id: string, x: number, baseColor: [number, number, number]): MeshPrimitive {
  return {
    id,
    kind: 'mesh',
    positions: new Float32Array([x, 0, 0, x + 1, 0, 0, x, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor, roughness: 0.5, metallic: 0 },
  };
}

function baseScene(): Scene {
  return {
    primitives: [tri('mesh-a', 0, [0.9, 0.1, 0.1]), tri('mesh-b', 10, [0.1, 0.9, 0.1])],
    emitters: [],
    environment: { kind: 'none' },
  };
}

/** The primitive ids in the most-recently-converted THREE root. */
function lastConvertedIds(): string[] {
  return sceneToThreeCalls[sceneToThreeCalls.length - 1]!.primitives.map((p) => String(p.id));
}

/** The baseColors present in the most-recently-converted THREE root. */
function lastConvertedColors(): Array<readonly [number, number, number] | undefined> {
  return sceneToThreeCalls[sceneToThreeCalls.length - 1]!.primitives.map(
    (p) => (p as MeshPrimitive).material?.baseColor,
  );
}

describe('PTEngineWebGL2.addPrimitive / removePrimitive (pt-webgl)', () => {
  let teardownGlobalStub: (() => void) | null = null;

  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  function freshEngine() {
    sceneToThreeCalls.length = 0;
    setScene.mockClear();
    reset.mockClear();
    updateMaterials.mockClear();
    disposeGeom.mockClear();
    disposeMat.mockClear();
    return createPTEngine_WebGL2({ device: makeRendererStub() as never });
  }

  it('exposes addPrimitive / removePrimitive as functions (capability + ledger agree)', async () => {
    const engine = await freshEngine();
    expect(typeof engine.addPrimitive).toBe('function');
    expect(typeof engine.removePrimitive).toBe('function');
    expect(engine.capabilities.supportsAddRemovePrimitive).toBe(true);
  });

  it('addPrimitive converts + binds the new primitive (geometry + material) alongside survivors', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());
    expect(lastConvertedIds()).toEqual(['mesh-a', 'mesh-b']);

    setScene.mockClear();
    engine.addPrimitive!(tri('mesh-c', 20, [0.2, 0.3, 0.9]));

    // The new primitive reached vitrumSceneToThree (the fork's geometry path)
    // appended after the survivors, with its own material.
    expect(lastConvertedIds()).toEqual(['mesh-a', 'mesh-b', 'mesh-c']);
    expect(lastConvertedColors()).toContainEqual([0.2, 0.3, 0.9]);
    // A full repack ran (tracer.setScene, NOT the geometry-only regen).
    expect(setScene).toHaveBeenCalledTimes(1);
  });

  it('removePrimitive evicts the primitive (gone from the bound scene) while survivors stay intact', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());

    setScene.mockClear();
    engine.removePrimitive!('mesh-a');

    expect(lastConvertedIds()).toEqual(['mesh-b']);
    // The evicted primitive's material is no longer in the bound root.
    expect(lastConvertedColors()).not.toContainEqual([0.9, 0.1, 0.1]);
    // The survivor IS still present.
    expect(lastConvertedColors()).toContainEqual([0.1, 0.9, 0.1]);
    expect(setScene).toHaveBeenCalledTimes(1);
  });

  it('removing the last primitive yields a renderable empty scene', async () => {
    const engine = await freshEngine();
    engine.setScene({
      primitives: [tri('only', 0, [0.5, 0.5, 0.5])],
      emitters: [],
      environment: { kind: 'none' },
    });

    expect(() => engine.removePrimitive!('only')).not.toThrow();
    expect(lastConvertedIds()).toEqual([]);
    expect(engine.state).toBe('ready');
  });

  it('resets sample accumulation on add and on remove', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());

    // Accumulate a few samples so the reset is observable.
    const input = {
      viewMatrix: new Float32Array(16),
      projMatrix: new Float32Array(16),
      cameraPosition: new Float32Array(3),
      viewport: { width: 4, height: 4, devicePixelRatio: 1 },
      quality: { samplesTarget: 8 },
    } as never;
    engine.renderFrame(input);
    engine.renderFrame(input);
    const out1 = engine.renderFrame(input);
    expect((out1 as { samplesAccumulated: number }).samplesAccumulated).toBeGreaterThan(0);

    // add → full repack → tracer.setScene clears the fork accumulator.
    engine.addPrimitive!(tri('mesh-c', 20, [0.2, 0.3, 0.9]));
    const outAfterAdd = engine.renderFrame(input);
    // First frame after the add restarts from sample 0 (then advances by the
    // per-frame batch); the running count is well below the pre-add total.
    expect((outAfterAdd as { samplesAccumulated: number }).samplesAccumulated).toBeLessThanOrEqual(
      (out1 as { samplesAccumulated: number }).samplesAccumulated,
    );

    // remove → full repack → accumulator clears again.
    engine.removePrimitive!('mesh-c');
    const outAfterRemove = engine.renderFrame(input);
    expect(
      (outAfterRemove as { samplesAccumulated: number }).samplesAccumulated,
    ).toBeLessThanOrEqual((out1 as { samplesAccumulated: number }).samplesAccumulated);
  });

  it('addPrimitive throws on a duplicate id and leaves the scene unchanged', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());

    sceneToThreeCalls.length = 0;
    setScene.mockClear();
    expect(() => engine.addPrimitive!(tri('mesh-a', 5, [0, 0, 0]))).toThrow(/already exists/);
    // No conversion / repack happened — the scene was not touched on throw.
    expect(sceneToThreeCalls.length).toBe(0);
    expect(setScene).not.toHaveBeenCalled();
  });

  it('removePrimitive throws on a missing id and leaves the scene unchanged', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());

    sceneToThreeCalls.length = 0;
    setScene.mockClear();
    expect(() => engine.removePrimitive!('does-not-exist')).toThrow(/no primitive with id/);
    expect(sceneToThreeCalls.length).toBe(0);
    expect(setScene).not.toHaveBeenCalled();
  });

  it('addPrimitive before setScene throws (engine not yet initialized)', async () => {
    const engine = await freshEngine();
    expect(() => engine.addPrimitive!(tri('x', 0, [0, 0, 0]))).toThrow(/setScene/);
  });

  it('removePrimitive before setScene throws (engine not yet initialized)', async () => {
    const engine = await freshEngine();
    expect(() => engine.removePrimitive!('mesh-a')).toThrow(/setScene/);
  });

  it('addPrimitive after dispose throws', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());
    engine.dispose();
    expect(() => engine.addPrimitive!(tri('x', 0, [0, 0, 0]))).toThrow(/disposed/);
  });

  it('removePrimitive after dispose throws', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());
    engine.dispose();
    expect(() => engine.removePrimitive!('mesh-a')).toThrow(/disposed/);
  });

  it('addPrimitive then removePrimitive round-trips and leaves the engine ready', async () => {
    const engine = await freshEngine();
    engine.setScene(baseScene());
    expect(() => engine.addPrimitive!(tri('mesh-c', 20, [0.2, 0.3, 0.9]))).not.toThrow();
    expect(lastConvertedIds()).toEqual(['mesh-a', 'mesh-b', 'mesh-c']);
    expect(() => engine.removePrimitive!('mesh-c')).not.toThrow();
    expect(lastConvertedIds()).toEqual(['mesh-a', 'mesh-b']);
    expect(engine.state).toBe('ready');
  });
});
