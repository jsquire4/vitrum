import {
  asMat4,
  type MeshPrimitive,
  type Scene,
} from '@vitrum/core';
import { describe, expect, it, vi } from 'vitest';
import { HybridEngine } from '../HybridEngine.js';
import {
  materialPatchAffectsDisplacementGeometry,
  SKIN_POSE_PATCH_FIELDS,
  SKIN_REST_STREAM_PATCH_FIELDS,
} from '../HybridEnginePrimitiveUpdates.js';
import { WALKAROUND_FULL_CERTIFIED_SUPPORT_MANIFEST } from '../supportManifest.js';

function mesh(): MeshPrimitive {
  return {
    kind: 'mesh',
    id: 'mesh-a',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    material: {
      baseColor: [0.8, 0.2, 0.2],
      roughness: 0.5,
      metallic: 0,
    },
    transform: asMat4(new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ])),
  };
}

function makeIdentityPatchEngine(): {
  readonly engine: HybridEngine;
  readonly scene: Scene;
  readonly routePrimitiveUpdate: ReturnType<typeof vi.fn>;
  readonly setScene: ReturnType<typeof vi.fn>;
} {
  const scene: Scene = {
    primitives: [mesh()],
    emitters: [],
    environment: { kind: 'none' },
  };
  const engine = Object.create(HybridEngine.prototype) as HybridEngine;
  const routePrimitiveUpdate = vi.fn();
  const setScene = vi.fn();
  Object.assign(engine as unknown as Record<string, unknown>, {
    _state: 'ready',
    _lastScene: scene,
    _supportManifest: WALKAROUND_FULL_CERTIFIED_SUPPORT_MANIFEST,
    _routePrimitiveUpdate: routePrimitiveUpdate,
    setScene,
  });
  return { engine, scene, routePrimitiveUpdate, setScene };
}

describe('primitive patch routing closure', () => {
  it.each([
    ['id', { id: 'mesh-a' }, /id cannot be changed or supplied/],
    ['kind', { kind: 'mesh' }, /kind cannot change or be supplied/],
  ] satisfies ReadonlyArray<
    readonly [string, Record<string, unknown>, RegExp]
  >)(
    'rejects even an equal %s discriminant before any rebuild',
    (_field, patch, expected) => {
      const { engine, scene, routePrimitiveUpdate, setScene } =
        makeIdentityPatchEngine();

      expect(() => engine.updatePrimitive('mesh-a', patch as never)).toThrow(expected);
      expect(setScene).not.toHaveBeenCalled();
      expect(routePrimitiveUpdate).not.toHaveBeenCalled();
      expect(
        (engine as unknown as { readonly _lastScene: Scene })._lastScene,
      ).toBe(scene);
    },
  );

  it.each([
    ['id', { id: 'mesh-b' }, /id cannot be changed or supplied/],
    ['kind', { kind: 'analytic' }, /kind cannot change or be supplied/],
  ] satisfies ReadonlyArray<
    readonly [string, Record<string, unknown>, RegExp]
  >)(
    'rejects a changed %s discriminant before any rebuild',
    (_field, patch, expected) => {
      const { engine, scene, routePrimitiveUpdate, setScene } =
        makeIdentityPatchEngine();

      expect(() => engine.updatePrimitive('mesh-a', patch as never)).toThrow(expected);
      expect(setScene).not.toHaveBeenCalled();
      expect(routePrimitiveUpdate).not.toHaveBeenCalled();
      expect(
        (engine as unknown as { readonly _lastScene: Scene })._lastScene,
      ).toBe(scene);
    },
  );

  it('routes every authored skin/morph definition field through solveSkin', () => {
    expect(SKIN_POSE_PATCH_FIELDS).toEqual(expect.arrayContaining([
      'skinIndices',
      'skinWeights',
      'skinInfluencesPerVertex',
      'bones',
      'boneInverses',
      'bindMatrix',
      'bindMatrixInverse',
      'morphTargets',
      'morphTargetNormals',
      'morphTargetTangents',
      'morphTargetUvs',
      'morphTargetUv1s',
      'morphTargetUvSets',
      'morphTargetColors',
      'morphTargetColorSets',
      'morphWeights',
    ]));
    expect(SKIN_REST_STREAM_PATCH_FIELDS).toEqual([
      'positions',
      'normals',
      'tangents',
      'uvs',
      'uv1',
      'uvSets',
      'colors',
      'colorSets',
    ]);
  });

  it('treats omitted displacement mip filtering as the public linear default', () => {
    const handle = { name: 'height' };
    const previous = {
      baseColor: [1, 1, 1] as const,
      roughness: 0.5,
      metallic: 0,
      displacementMap: { handle },
    };
    expect(materialPatchAffectsDisplacementGeometry(previous, {
      displacementMap: { handle, mipFilter: 'linear' },
    })).toBe(false);
    expect(materialPatchAffectsDisplacementGeometry(previous, {
      displacementMap: { handle, mipFilter: 'none' },
    })).toBe(true);
  });
});
