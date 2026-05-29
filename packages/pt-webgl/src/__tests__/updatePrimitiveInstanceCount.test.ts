/**
 * pt-webgl incremental INSTANCE-COUNT topology path.
 *
 * `vitrumSceneToThree` builds ONE THREE.InstancedMesh per `instanced-mesh`
 * primitive; pt-webgl's `setScene` then expands it into N baked THREE.Mesh
 * children (the fork bakes only `mesh.matrixWorld`, ignoring `instanceMatrix`).
 * An `instances`-only patch that GROWS or SHRINKS the instance COUNT must
 * re-expand ONLY that primitive's children in the live scene root + take the
 * fork's targeted geometry+BVH regen — NOT a full `setScene` teardown. These
 * tests pin:
 *   1. the `isInstanceCountOnlyPrimitivePatch` classifier;
 *   2. `reexpandInstancedMeshInScene` swaps N children for N' on a real THREE
 *      scene, reusing the shared geometry/material + preserving the dispose-dedup
 *      `vitrumExpandedInstanceOf` tag;
 *   3. the engine takes the incremental regen path on an instance-count change
 *      (generate once, no setScene) and lands the new count;
 *   4. an instance-count change that ALSO changes material falls back to a full
 *      setScene.
 */
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { InstancedMesh as TInstancedMesh, Matrix4, Mesh, Scene } from 'three';
import type { Mesh as TMesh } from 'three';
import { asMat4 } from '@vitrum/core';
import type { InstancedMeshPrimitive, Scene as VitrumScene } from '@vitrum/core';
import { vitrumSceneToThree } from '@vitrum/three-bindings';
import {
  expandInstancedMeshesInScene,
  findAllMeshesByPrimitiveId,
  reexpandInstancedMeshInScene,
} from '../expandInstancedMeshes.js';
import {
  isInstanceCountOnlyPrimitivePatch,
  isGeometryOnlyPrimitivePatch,
  isMaterialOnlyPrimitivePatch,
} from '../scenePatch.js';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

const setScene = vi.fn();
const reset = vi.fn();
const updateMaterials = vi.fn();
// The fork's targeted geometry+BVH regen entry. Observing this proves the
// incremental path ran instead of a full setScene teardown.
const generate = vi.fn(() => ({
  bvhChanged: true,
  bvh: { mock: true },
  needsMaterialIndexUpdate: false,
  geometry: { attributes: { normal: { array: [] } } },
}));

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

// Use the REAL @vitrum/three-bindings + REAL THREE objects so the instanced-mesh
// expansion + re-expansion actually mutate a real scene graph (no GPU). Only the
// fork (WebGLPathTracer) is stubbed.
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

function instancedPrim(id: string, instances: Float32Array[]): InstancedMeshPrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: TRI_POSITIONS,
    normals: TRI_NORMALS,
    material: { baseColor: [0.4, 0.5, 0.6], roughness: 0.5, metallic: 0 },
    instances: instances.map((m) => asMat4(m)),
  };
}

describe('isInstanceCountOnlyPrimitivePatch classifier', () => {
  it('classifies an instances-only patch (count grow) as instance-count-only', () => {
    expect(
      isInstanceCountOnlyPrimitivePatch({
        instances: [asMat4(translation(0, 0, 0)), asMat4(translation(1, 0, 0))],
      }),
    ).toBe(true);
  });

  it('classifies an instances-only patch (count shrink) as instance-count-only', () => {
    expect(isInstanceCountOnlyPrimitivePatch({ instances: [asMat4(translation(0, 0, 0))] })).toBe(true);
  });

  it('does NOT classify an instances+material patch as instance-count-only', () => {
    const patch = {
      instances: [asMat4(translation(0, 0, 0))],
      material: { baseColor: [1, 0, 0] as [number, number, number], roughness: 0.5, metallic: 0 },
    };
    expect(isInstanceCountOnlyPrimitivePatch(patch)).toBe(false);
    // ...nor material-only nor geometry-only — routes to full rebuild.
    expect(isMaterialOnlyPrimitivePatch(patch)).toBe(false);
    expect(isGeometryOnlyPrimitivePatch(patch)).toBe(false);
  });

  it('does NOT classify an empty / id-only / geometry patch as instance-count-only', () => {
    expect(isInstanceCountOnlyPrimitivePatch({})).toBe(false);
    expect(isInstanceCountOnlyPrimitivePatch({ id: 'x' })).toBe(false);
    expect(isInstanceCountOnlyPrimitivePatch({ positions: new Float32Array(12) })).toBe(false);
  });
});

describe('reexpandInstancedMeshInScene (unit, real THREE, no GPU)', () => {
  it('swaps N expanded children for N′ reusing shared geometry + material', () => {
    const vscene: VitrumScene = {
      primitives: [instancedPrim('inst', [translation(0, 0, 0), translation(2, 0, 0)])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const threeScene = vitrumSceneToThree(vscene);
    expandInstancedMeshesInScene(threeScene);
    expect(findAllMeshesByPrimitiveId(threeScene, 'inst')).toHaveLength(2);

    const before = findAllMeshesByPrimitiveId(threeScene, 'inst');
    const sharedGeo = before[0]!.geometry;
    const sharedMat = before[0]!.material;
    const tag = before[0]!.userData['vitrumExpandedInstanceOf'];

    // Grow 2 -> 4.
    const ok = reexpandInstancedMeshInScene(threeScene, 'inst', [
      asMat4(translation(0, 0, 0)),
      asMat4(translation(1, 0, 0)),
      asMat4(translation(2, 0, 0)),
      asMat4(translation(3, 0, 0)),
    ]);
    expect(ok).toBe(true);

    const after = findAllMeshesByPrimitiveId(threeScene, 'inst');
    expect(after).toHaveLength(4);
    // No THREE.InstancedMesh leaked into the scene (the throwaway temp is never added).
    let instanced = 0;
    threeScene.traverse((o) => {
      if ((o as TInstancedMesh).isInstancedMesh === true) instanced += 1;
    });
    expect(instanced).toBe(0);
    // Shared geometry + material reused (material slot untouched -> safe to skip updateMaterials).
    for (const c of after) {
      expect(c.geometry).toBe(sharedGeo);
      expect(c.material).toBe(sharedMat);
      // Dispose-dedup tag preserved from the original expansion.
      expect(c.userData['vitrumExpandedInstanceOf']).toBe(tag);
      expect(c.name).toBe('inst');
    }
    // World transforms reflect the 4 new instance matrices.
    const xs = after
      .map((c: TMesh) => new Matrix4().copy(c.matrixWorld).elements[12]!)
      .sort((a, b) => a - b);
    expect(xs.map((x) => Math.round(x))).toEqual([0, 1, 2, 3]);
  });

  it('shrinks N expanded children for a smaller instance count', () => {
    const vscene: VitrumScene = {
      primitives: [
        instancedPrim('inst', [translation(0, 0, 0), translation(1, 0, 0), translation(2, 0, 0)]),
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const threeScene = vitrumSceneToThree(vscene);
    expandInstancedMeshesInScene(threeScene);
    expect(findAllMeshesByPrimitiveId(threeScene, 'inst')).toHaveLength(3);

    const ok = reexpandInstancedMeshInScene(threeScene, 'inst', [asMat4(translation(5, 0, 0))]);
    expect(ok).toBe(true);
    expect(findAllMeshesByPrimitiveId(threeScene, 'inst')).toHaveLength(1);
  });

  it('returns false when the primitive has no expanded children (caller full-rebuilds)', () => {
    const scene = new Scene();
    const plain = new Mesh();
    plain.name = 'plain';
    scene.add(plain);
    expect(reexpandInstancedMeshInScene(scene, 'nonexistent', [asMat4(translation(0, 0, 0))])).toBe(
      false,
    );
  });
});

describe('PTEngineWebGL2.updatePrimitive instance-COUNT topology', () => {
  let teardownGlobalStub: (() => void) | null = null;

  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('re-expands only the changed mesh + runs the fork regen once for a same-material instance-COUNT change', async () => {
    setScene.mockClear();
    generate.mockClear();
    reset.mockClear();
    updateMaterials.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({
      primitives: [instancedPrim('inst-a', [translation(0, 0, 0), translation(2, 0, 0)])],
      emitters: [],
      environment: { kind: 'none' },
    });

    // Capture the live THREE scene root the engine handed the fork at setScene —
    // the engine mutates THIS object in place on the incremental path, so we can
    // assert the new instance count landed on it.
    const threeRoot = setScene.mock.calls[0]![0] as { traverseVisible: (cb: (o: unknown) => void) => void };
    expect(findAllMeshesByPrimitiveId(threeRoot as never, 'inst-a')).toHaveLength(2);

    setScene.mockClear();
    generate.mockClear();
    reset.mockClear();
    updateMaterials.mockClear();

    // Grow 2 -> 4 instances.
    engine.updatePrimitive!('inst-a', {
      instances: [
        asMat4(translation(0, 0, 0)),
        asMat4(translation(1, 0, 0)),
        asMat4(translation(2, 0, 0)),
        asMat4(translation(3, 0, 0)),
      ],
    });

    // Incremental fork regen ran exactly once (NOT a full setScene teardown).
    expect(generate).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    // Accumulator cleared via the regen's reset callback.
    expect(reset).toHaveBeenCalled();
    // No material re-pack on the instances-only path.
    expect(updateMaterials).not.toHaveBeenCalled();
    // The new instance count landed on the live THREE scene root (4 expanded
    // children sharing the primitive id) — only that mesh's children were
    // re-expanded.
    expect(findAllMeshesByPrimitiveId(threeRoot as never, 'inst-a')).toHaveLength(4);
  });

  it('falls back to full setScene when the patch ALSO changes the material', async () => {
    setScene.mockClear();
    generate.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({
      primitives: [instancedPrim('inst-a', [translation(0, 0, 0), translation(2, 0, 0)])],
      emitters: [],
      environment: { kind: 'none' },
    });

    setScene.mockClear();
    generate.mockClear();

    // Instance-COUNT change + a NEW material in the same patch: the instances-only
    // regen skips updateMaterials(), so this MUST full-rebuild.
    engine.updatePrimitive!('inst-a', {
      instances: [asMat4(translation(0, 0, 0)), asMat4(translation(1, 0, 0)), asMat4(translation(2, 0, 0))],
      material: { baseColor: [0.9, 0.1, 0.1], roughness: 0.2, metallic: 1 },
    } as never);

    expect(setScene).toHaveBeenCalledTimes(1);
    expect(generate).not.toHaveBeenCalled();
  });
});
