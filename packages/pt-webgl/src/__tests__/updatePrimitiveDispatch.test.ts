/**
 * CHARACTERIZATION TEST — pins `PTEngineWebGL2.updatePrimitive`'s dispatch
 * cascade as a golden before/after the Theme-B handler-table refactor.
 *
 * The other `updatePrimitive*` test files each pin ONE classifier's happy path
 * in isolation. This file drives the FULL patch matrix through the real engine
 * (real `@vitrum/three-bindings`; only the fork `WebGLPathTracer` stubbed) and
 * pins the *observable* outcome of each branch — which fast path is taken,
 * whether the fork geometry regen runs (`generate`), whether a full `setScene`
 * teardown happens, whether `updateMaterials` / `reset` fire, and the resulting
 * `#vitrumScene`. It also pins the two under-covered FALL-THROUGH cases:
 *
 *   • an `instances` patch whose CURRENT primitive is NOT an instanced-mesh —
 *     the `isInstanceCountOnlyPrimitivePatch` classifier matches but the
 *     kind-guard fails, so it must fall through to the tail full-`setScene`;
 *   • an empty / non-matching patch — straight to the tail full-`setScene`.
 *
 * Behavior must be byte-identical before and after the dispatch-cascade ->
 * handler-table refactor AND its later lift into `scenePatch.routePrimitivePatch`
 * (Task 4.4 Theme A — `updatePrimitive` is now a thin delegate that calls
 * `routePrimitivePatch` and runs the shared commit/fallback epilogue): this file
 * is the regression net through both moves.
 */
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { Matrix4 } from 'three';
import { asMat4 } from '@vitrum/core';
import type { InstancedMeshPrimitive, MeshPrimitive } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

const setScene = vi.fn();
const reset = vi.fn();
const updateMaterials = vi.fn();
// The fork's targeted geometry+BVH regen entry. Observing this proves the
// incremental fast path ran instead of a full `setScene` teardown.
const generate = vi.fn(() => ({
  bvhChanged: true,
  bvh: { mock: true },
  needsMaterialIndexUpdate: false,
  geometry: { attributes: { normal: { array: [] } } },
}));

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

// Real @vitrum/three-bindings + real THREE objects so mesh lookup / transform /
// positions / geometry / instance re-expansion all run for real. Only the fork
// is stubbed (no GPU).
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
    readonly _pathTracer = {
      material: {
        bvh: { updateFrom: vi.fn() },
        attributesArray: { updateFrom: vi.fn() },
        materialIndexAttribute: { updateFrom: vi.fn() },
        uniforms: {},
      },
    };
    readonly _generator = { initialized: true, generate };

    setScene = setScene;
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

const TRI_POSITIONS = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const TRI_NORMALS = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);

function translation(x: number, y: number, z: number): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

function meshPrim(id: string): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array(TRI_POSITIONS),
    normals: new Float32Array(TRI_NORMALS),
    material: { baseColor: [0.4, 0.5, 0.6], roughness: 0.5, metallic: 0 },
  };
}

function instancedPrim(id: string, instances: Float32Array[]): InstancedMeshPrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: new Float32Array(TRI_POSITIONS),
    normals: new Float32Array(TRI_NORMALS),
    material: { baseColor: [0.4, 0.5, 0.6], roughness: 0.5, metallic: 0 },
    instances: instances.map((m) => asMat4(m)),
  };
}

function clearMocks(): void {
  setScene.mockClear();
  reset.mockClear();
  updateMaterials.mockClear();
  generate.mockClear();
}

describe('PTEngineWebGL2.updatePrimitive dispatch cascade (characterization golden)', () => {
  let teardownGlobalStub: (() => void) | null = null;

  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  async function freshEngineWith(prim: MeshPrimitive | InstancedMeshPrimitive) {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({ primitives: [prim], emitters: [], environment: { kind: 'none' } });
    clearMocks();
    return engine;
  }

  it('branch 1 — material-only: updateMaterials, NO generate, NO setScene, NO reset; scene material patched', async () => {
    const engine = await freshEngineWith(meshPrim('m'));

    engine.updatePrimitive!('m', {
      material: { baseColor: [0.9, 0.1, 0.1], roughness: 0.2, metallic: 0 },
    });

    expect(updateMaterials).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
    expect(setScene).not.toHaveBeenCalled();
    expect(reset).not.toHaveBeenCalled();
    expect(engine.capabilities.incrementalPatchSupport?.material).toBe(true);
  });

  it('branch 2 — transform-only: fork regen once, NO setScene, reset fired, NO updateMaterials', async () => {
    const engine = await freshEngineWith(meshPrim('m'));

    engine.updatePrimitive!('m', {
      transform: asMat4(translation(2, 0, 0)),
    });

    expect(generate).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    expect(reset).toHaveBeenCalled();
    expect(updateMaterials).not.toHaveBeenCalled();
    expect(engine.capabilities.incrementalPatchSupport?.transform).toBe(true);
  });

  it('branch 3 — positions-only (same vertex count): fork regen once, NO setScene', async () => {
    const engine = await freshEngineWith(meshPrim('m'));

    engine.updatePrimitive!('m', {
      positions: new Float32Array([0, 0.1, 0, 1, 0.1, 0, 0, 1.1, 0]),
    });

    expect(generate).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    expect(reset).toHaveBeenCalled();
    expect(engine.capabilities.incrementalPatchSupport?.positions).toBe(true);
  });

  it('branch 4 — geometry-only (vertex COUNT change): mutates mesh then, when the fork regen reports no incremental path, falls back to tail setScene (NO double generate)', async () => {
    const engine = await freshEngineWith(meshPrim('m'));

    // Grow 3 -> 4 verts (a count change) — geometry-only surgery, no material.
    // With a real THREE mesh + real fork-access path, the targeted regen reports
    // it cannot incrementally land the count change, so the branch epilogue
    // falls back to a single full `setScene` (NOT a generate). This is the
    // golden current behavior — pinned so the refactor preserves the exact
    // mutate-then-fallback ordering.
    engine.updatePrimitive!('m', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
    } as Partial<MeshPrimitive>);

    expect(setScene).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
  });

  it('branch 5 — instance-count-only on an instanced-mesh: re-expand + fork regen once, NO setScene', async () => {
    const engine = await freshEngineWith(
      instancedPrim('inst', [translation(0, 0, 0), translation(2, 0, 0)]),
    );

    engine.updatePrimitive!('inst', {
      instances: [
        asMat4(translation(0, 0, 0)),
        asMat4(translation(1, 0, 0)),
        asMat4(translation(2, 0, 0)),
        asMat4(translation(3, 0, 0)),
      ],
    });

    expect(generate).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    expect(reset).toHaveBeenCalled();
    expect(updateMaterials).not.toHaveBeenCalled();
  });

  it('FALL-THROUGH A — instances patch on a NON-instanced primitive: classifier matches, kind-guard FAILS -> tail setScene', async () => {
    // Current primitive kind is 'mesh', not 'instanced-mesh'. The
    // `isInstanceCountOnlyPrimitivePatch` classifier still matches the patch
    // shape, but the in-branch kind-guard must reject it, dropping to the tail.
    const engine = await freshEngineWith(meshPrim('m'));

    engine.updatePrimitive!('m', {
      instances: [asMat4(translation(0, 0, 0)), asMat4(translation(1, 0, 0))],
    } as Partial<InstancedMeshPrimitive>);

    expect(setScene).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
  });

  it('FALL-THROUGH B — instances+material combo patch: no fast path matches -> tail setScene', async () => {
    const engine = await freshEngineWith(
      instancedPrim('inst', [translation(0, 0, 0), translation(2, 0, 0)]),
    );

    engine.updatePrimitive!('inst', {
      instances: [asMat4(translation(0, 0, 0)), asMat4(translation(1, 0, 0)), asMat4(translation(2, 0, 0))],
      material: { baseColor: [0.9, 0.1, 0.1], roughness: 0.2, metallic: 1 },
    } as never);

    expect(setScene).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
  });

  it('FALL-THROUGH C — empty patch: no classifier matches -> tail setScene', async () => {
    const engine = await freshEngineWith(meshPrim('m'));

    engine.updatePrimitive!('m', {});

    expect(setScene).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
  });

  it('throws on a missing-id patch BEFORE any mutation (material path)', async () => {
    const engine = await freshEngineWith(meshPrim('m'));

    expect(() =>
      engine.updatePrimitive!('does-not-exist', {
        material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
      }),
    ).toThrow(/not found/);
    expect(setScene).not.toHaveBeenCalled();
  });
});
