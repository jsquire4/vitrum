/**
 * CHARACTERIZATION TEST — pins `routePrimitivePatch` (Task 4.4 Theme A) at the
 * free-function boundary in `scenePatch.ts`.
 *
 * The engine-level `updatePrimitiveDispatch.test.ts` golden drives the full
 * matrix through the real `PTEngineWebGL2.updatePrimitive` (which now delegates
 * to this function). This file pins the function DIRECTLY: for the same patch
 * matrix it asserts the returned `'commit' | 'fallback'` outcome, the THREE-side
 * mutation side effects (mesh material / transform / geometry), and the fork
 * regen / updateMaterials calls — independent of the engine epilogue. The two
 * together prove the extraction preserved both the routing decisions AND the
 * engine-side commit/setScene wiring.
 */
import { describe, it, expect, vi } from 'vitest';
import { Scene as ThreeScene, Mesh, BufferGeometry, BufferAttribute, MeshStandardMaterial } from 'three';
/* MeshStandardMaterial is the placeholder pre-patch material; the route REPLACES
 * it with a MeshPhysicalMaterial via applyVitrumMaterialToMesh. */
import { asMat4 } from '@vitrum/core';
import type { Scene, MeshPrimitive, InstancedMeshPrimitive } from '@vitrum/core';
import {
  routePrimitivePatch,
  type PrimitivePatchContext,
} from '../scenePatch.js';

// A minimal fork-tracer stub exposing only what the route's mutators touch:
// the `_generator` (for refreshPathTracerSceneGeometry via ForkAccess) +
// `updateMaterials` / `reset`. `refreshPathTracerSceneGeometry` ultimately
// calls ForkAccess.regenerateSceneGeometry, which reads `_generator.generate`.
function makeTracerStub(generate: () => unknown) {
  return {
    updateMaterials: vi.fn(),
    reset: vi.fn(),
    _generator: { initialized: true, generate },
    _pathTracer: {
      material: {
        bvh: { updateFrom: vi.fn() },
        attributesArray: { updateFrom: vi.fn() },
        materialIndexAttribute: { updateFrom: vi.fn() },
        uniforms: {},
      },
    },
  };
}

const TRI = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const TRI_N = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);

function meshPrimRecord(id: string): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array(TRI),
    normals: new Float32Array(TRI_N),
    material: { baseColor: [0.4, 0.5, 0.6], roughness: 0.5, metallic: 0 },
  };
}

/** Build a THREE scene root containing one mesh whose `name` is the primitive
 *  id, mirroring what `vitrumSceneToThree` produces (`findMeshByPrimitiveId`
 *  matches on `name`/`uuid`). */
function makeThreeRootWithMesh(id: string): { root: ThreeScene; mesh: Mesh } {
  const geom = new BufferGeometry();
  geom.setAttribute('position', new BufferAttribute(new Float32Array(TRI), 3));
  geom.setAttribute('normal', new BufferAttribute(new Float32Array(TRI_N), 3));
  const mesh = new Mesh(geom, new MeshStandardMaterial());
  mesh.name = id;
  const root = new ThreeScene();
  root.add(mesh);
  return { root, mesh };
}

/** Valid fork-regen result (the generator landed a refit). */
function okGenerateResult() {
  return {
    bvhChanged: true,
    bvh: { mock: true },
    needsMaterialIndexUpdate: false,
    geometry: { attributes: { normal: { array: [] } } },
  };
}

function vitrumScene(prim: MeshPrimitive | InstancedMeshPrimitive): Scene {
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } };
}

describe('routePrimitivePatch — outcome + side effects (characterization)', () => {
  it('material-only → "commit", calls updateMaterials, no fork regen', () => {
    const generate = vi.fn();
    const tracer = makeTracerStub(generate);
    const { root, mesh } = makeThreeRootWithMesh('m');
    const ctx = {
      pathTracer: tracer as never,
      threeSceneRoot: root,
      vitrumScene: vitrumScene(meshPrimRecord('m')),
    } satisfies PrimitivePatchContext;

    const originalMaterial = mesh.material;
    const outcome = routePrimitivePatch(ctx, 'm', {
      material: { baseColor: [0.9, 0.1, 0.1], roughness: 0.2, metallic: 0 },
    });

    expect(outcome).toBe('commit');
    expect(tracer.updateMaterials).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
    // material was re-pointed on the mesh (applyVitrumMaterialToMesh REPLACES it)
    expect(mesh.material).not.toBe(originalMaterial);
  });

  it('transform-only → "commit" when fork regen lands, copies matrix to mesh', () => {
    const generate = vi.fn(okGenerateResult);
    const tracer = makeTracerStub(generate);
    const { root, mesh } = makeThreeRootWithMesh('m');
    const ctx = {
      pathTracer: tracer as never,
      threeSceneRoot: root,
      vitrumScene: vitrumScene(meshPrimRecord('m')),
    } satisfies PrimitivePatchContext;

    const tf = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1]);
    const outcome = routePrimitivePatch(ctx, 'm', { transform: asMat4(tf) });

    expect(outcome).toBe('commit');
    expect(generate).toHaveBeenCalledTimes(1);
    expect(mesh.matrixAutoUpdate).toBe(false);
    expect(mesh.matrix.elements[12]).toBe(2);
    expect(mesh.matrix.elements[13]).toBe(3);
  });

  it('positions-only same-count → "commit", calls fork regen once', () => {
    const generate = vi.fn(okGenerateResult);
    const tracer = makeTracerStub(generate);
    const { root, mesh } = makeThreeRootWithMesh('m');
    const ctx = {
      pathTracer: tracer as never,
      threeSceneRoot: root,
      vitrumScene: vitrumScene(meshPrimRecord('m')),
    } satisfies PrimitivePatchContext;

    const beforeAttr = mesh.geometry.getAttribute('position');
    const beforeArray = beforeAttr.array;
    const outcome = routePrimitivePatch(ctx, 'm', {
      positions: new Float32Array([0, 0.1, 0, 1, 0.1, 0, 0, 1.1, 0]),
    });

    expect(outcome).toBe('commit');
    expect(generate).toHaveBeenCalledTimes(1);
    expect(tracer.reset).toHaveBeenCalled();
    const afterAttr = mesh.geometry.getAttribute('position');
    expect(afterAttr).toBe(beforeAttr);
    expect(afterAttr.array).toBe(beforeArray);
    expect(afterAttr.count).toBe(3);
    expect(Array.from(afterAttr.array.slice(0, 9))).toEqual(
      Array.from(new Float32Array([0, 0.1, 0, 1, 0.1, 0, 0, 1.1, 0])),
    );
  });

  it('positions vertex-COUNT change → "fallback" (mutator bails BEFORE fork regen)', () => {
    // isPositionsOnlyPrimitivePatch still matches (positions/normals only), but
    // applyPositionsPatchToMesh returns false on a count mismatch → 'fallback'
    // WITHOUT ever calling the fork regen. Golden branch-4 behavior.
    const generate = vi.fn(okGenerateResult);
    const tracer = makeTracerStub(generate);
    const { root } = makeThreeRootWithMesh('m');
    const ctx = {
      pathTracer: tracer as never,
      threeSceneRoot: root,
      vitrumScene: vitrumScene(meshPrimRecord('m')),
    } satisfies PrimitivePatchContext;

    const outcome = routePrimitivePatch(ctx, 'm', {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]), // 3 → 4 verts
    } as Partial<MeshPrimitive>);

    expect(outcome).toBe('fallback');
    expect(generate).not.toHaveBeenCalled();
  });

  it('instances patch on a NON-instanced primitive → passthrough → "fallback"', () => {
    const generate = vi.fn();
    const tracer = makeTracerStub(generate);
    const { root } = makeThreeRootWithMesh('m');
    const ctx = {
      pathTracer: tracer as never,
      threeSceneRoot: root,
      vitrumScene: vitrumScene(meshPrimRecord('m')), // kind 'mesh', not instanced
    } satisfies PrimitivePatchContext;

    const outcome = routePrimitivePatch(ctx, 'm', {
      instances: [asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))],
    } as Partial<InstancedMeshPrimitive>);

    expect(outcome).toBe('fallback');
    expect(generate).not.toHaveBeenCalled();
  });

  it('empty patch → no classifier matches → "fallback"', () => {
    const generate = vi.fn();
    const tracer = makeTracerStub(generate);
    const { root } = makeThreeRootWithMesh('m');
    const ctx = {
      pathTracer: tracer as never,
      threeSceneRoot: root,
      vitrumScene: vitrumScene(meshPrimRecord('m')),
    } satisfies PrimitivePatchContext;

    expect(routePrimitivePatch(ctx, 'm', {})).toBe('fallback');
    expect(generate).not.toHaveBeenCalled();
  });

  it('throws on a missing-id mesh lookup BEFORE returning (material path)', () => {
    const generate = vi.fn();
    const tracer = makeTracerStub(generate);
    const { root } = makeThreeRootWithMesh('m');
    const ctx = {
      pathTracer: tracer as never,
      threeSceneRoot: root,
      vitrumScene: vitrumScene(meshPrimRecord('m')),
    } satisfies PrimitivePatchContext;

    expect(() =>
      routePrimitivePatch(ctx, 'nope', {
        material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
      }),
    ).toThrow(/not found/);
  });
});
