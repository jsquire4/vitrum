import { describe, it, expect } from 'vitest';
import { computeSceneAABB } from '../src/sceneAABB.js';
import type {
  Scene,
  MeshPrimitive,
  InstancedMeshPrimitive,
  SkinnedMeshPrimitive,
  AnalyticPrimitive,
  AnalyticShape,
  MaterialSpec,
  Mat4,
} from '@vitrum/core';
import { asMat4 } from '@vitrum/core';

const MAT: MaterialSpec = {
  baseColor: [0.5, 0.5, 0.5],
  metallic: 0,
  roughness: 0.5,
};

function unitCube(id: string, transform?: Mat4): MeshPrimitive {
  // 8 corners of [-0.5, 0.5]³
  // prettier-ignore
  const positions = new Float32Array([
    -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, 0.5, -0.5,   -0.5, 0.5, -0.5,
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5, 0.5,  0.5,   -0.5, 0.5,  0.5,
  ]);
  const indices = new Uint32Array([
    0,1,2, 0,2,3, 4,6,5, 4,7,6,
    0,4,5, 0,5,1, 1,5,6, 1,6,2,
    2,6,7, 2,7,3, 3,7,4, 3,4,0,
  ]);
  return {
    kind: 'mesh',
    id,
    positions,
    normals: new Float32Array(positions.length),
    indices,
    material: MAT,
    ...(transform ? { transform } : {}),
  };
}

function emptyScene(prims: Scene['primitives']): Scene {
  return {
    primitives: prims,
    emitters: [],
    environment: { kind: 'none' },
  };
}

function analytic(
  id: string,
  shape: AnalyticShape,
  params: readonly number[],
  transform?: Mat4,
): AnalyticPrimitive {
  return {
    kind: 'analytic',
    id,
    shape,
    params: Float32Array.from(params),
    material: MAT,
    ...(transform != null ? { transform } : {}),
  };
}

describe('computeSceneAABB', () => {
  it('returns identity-fallback for an empty scene', () => {
    const aabb = computeSceneAABB(emptyScene([]));
    expect(aabb.diagonal).toBe(1.0);
    expect(aabb.triangleCount).toBe(0);
    expect(aabb.center).toEqual([0, 0, 0]);
  });

  it('measures a single unit cube at the origin', () => {
    const aabb = computeSceneAABB(emptyScene([unitCube('a')]));
    expect(aabb.min).toEqual([-0.5, -0.5, -0.5]);
    expect(aabb.max).toEqual([0.5, 0.5, 0.5]);
    expect(aabb.center).toEqual([0, 0, 0]);
    // Diagonal = sqrt(3) ≈ 1.732
    expect(aabb.diagonal).toBeCloseTo(Math.sqrt(3), 5);
    // 12 triangles in a cube
    expect(aabb.triangleCount).toBe(12);
  });

  it('preserves the physical diagonal of a sub-meter scene', () => {
    const tiny = unitCube('tiny');
    const scale = 0.02;
    for (let index = 0; index < tiny.positions.length; index += 1) {
      tiny.positions[index] = tiny.positions[index]! * scale;
    }
    const aabb = computeSceneAABB(emptyScene([tiny]));
    expect(aabb.diagonal).toBeCloseTo(Math.sqrt(3) * scale, 7);
    expect(aabb.diagonal).toBeLessThan(1);
  });

  it('uses the scale fallback for nonempty degenerate point geometry', () => {
    const point = unitCube('point');
    point.positions.fill(0);
    const aabb = computeSceneAABB(emptyScene([point]));
    expect(aabb.diagonal).toBe(1);
    expect(aabb.extent).toEqual([0, 0, 0]);
  });

  it('honours an affine transform on a mesh primitive', () => {
    // translate +10 on X via column-major identity * translate(10, 0, 0)
    // prettier-ignore
    const xform = asMat4([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      10, 0, 0, 1,
    ]);
    const aabb = computeSceneAABB(emptyScene([unitCube('a', xform)]));
    expect(aabb.min[0]).toBeCloseTo(9.5);
    expect(aabb.max[0]).toBeCloseTo(10.5);
    expect(aabb.center[0]).toBeCloseTo(10);
  });

  it('grows the AABB across multiple primitives', () => {
    // prettier-ignore
    const right = asMat4([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      5, 0, 0, 1,
    ]);
    const aabb = computeSceneAABB(
      emptyScene([unitCube('a'), unitCube('b', right)]),
    );
    expect(aabb.min[0]).toBeCloseTo(-0.5);
    expect(aabb.max[0]).toBeCloseTo(5.5);
    expect(aabb.triangleCount).toBe(24);
  });

  it('aggregates instanced-mesh transforms', () => {
    const cube = unitCube('inst');
    const instances: ReadonlyArray<Mat4> = [
      // identity
      // prettier-ignore
      asMat4([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      // translate +3 X
      // prettier-ignore
      asMat4([1,0,0,0, 0,1,0,0, 0,0,1,0, 3,0,0,1]),
    ];
    const inst: InstancedMeshPrimitive = {
      kind: 'instanced-mesh',
      id: 'i',
      positions: cube.positions,
      normals: cube.normals,
      indices: cube.indices!,
      material: cube.material,
      instances,
    };
    const aabb = computeSceneAABB(emptyScene([inst]));
    expect(aabb.min[0]).toBeCloseTo(-0.5);
    expect(aabb.max[0]).toBeCloseTo(3.5);
    // 12 tris × 2 instances
    expect(aabb.triangleCount).toBe(24);
  });

  it('ignores empty instanced-mesh entries and keeps fallback bounds stable', () => {
    const cube = unitCube('inst-empty');
    const inst: InstancedMeshPrimitive = {
      kind: 'instanced-mesh',
      id: 'i-empty',
      positions: cube.positions,
      normals: cube.normals,
      indices: cube.indices!,
      material: cube.material,
      instances: [],
    };
    const aabb = computeSceneAABB(emptyScene([inst]));
    expect(aabb.diagonal).toBe(1.0);
    expect(aabb.triangleCount).toBe(0);
    expect(aabb.center).toEqual([0, 0, 0]);
  });

  it('measures a skinned mesh using its rest-pose positions (C1, 2026-05-19)', () => {
    const cube = unitCube('rest');
    const skinned: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'skin',
      positions: cube.positions,
      normals: cube.normals,
      indices: cube.indices!,
      material: cube.material,
      // 8 verts × 4 bones-per-vert
      skinIndices: new Uint32Array(8 * 4),
      // All weight on bone 0.
      skinWeights: (() => {
        const w = new Float32Array(8 * 4);
        for (let i = 0; i < 8; i++) w[i * 4] = 1.0;
        return w;
      })(),
      // One identity bone + one identity bone-inverse.
      bones: new Float32Array([
        1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
      ]),
      boneInverses: new Float32Array([
        1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
      ]),
    };
    const aabb = computeSceneAABB(emptyScene([skinned]));
    expect(aabb.min).toEqual([-0.5, -0.5, -0.5]);
    expect(aabb.max).toEqual([0.5, 0.5, 0.5]);
    expect(aabb.triangleCount).toBe(12);
  });

  it.each([
    {
      shape: 'sphere',
      params: [2, -1, 3, 0.5],
      min: [1.5, -1.5, 2.5],
      max: [2.5, -0.5, 3.5],
    },
    {
      shape: 'box',
      params: [-1, 2, 0, 1, 2, 3],
      min: [-2, 0, -3],
      max: [0, 4, 3],
    },
    {
      shape: 'capsule',
      params: [-2, 1, 0, 3, -4, 5, 0.25],
      min: [-2.25, -4.25, -0.25],
      max: [3.25, 1.25, 5.25],
    },
    {
      shape: 'cylinder',
      params: [1, 2, 3, 2, 0.5],
      min: [-1, 1.5, 1],
      max: [3, 2.5, 5],
    },
    {
      shape: 'h-channel-came',
      params: [4, 2, 6, 0.5],
      min: [-2, -3, -1],
      max: [2, 3, 1],
    },
  ] as const)(
    'measures a native $shape analytic without requiring fallback geometry',
    ({ shape, params, min, max }) => {
      const aabb = computeSceneAABB(
        emptyScene([analytic(`analytic-${shape}`, shape, params)]),
      );
      expect(aabb.min).toEqual(min);
      expect(aabb.max).toEqual(max);
      expect(aabb.triangleCount).toBe(0);
    },
  );

  it('transforms native analytic bounds into world space', () => {
    // Scale sphere bounds by (2,3,4), then translate by (10,-5,2).
    // prettier-ignore
    const transform = asMat4([
      2,0,0,0,
      0,3,0,0,
      0,0,4,0,
      10,-5,2,1,
    ]);
    const aabb = computeSceneAABB(
      emptyScene([analytic('scaled-sphere', 'sphere', [0, 0, 0, 1], transform)]),
    );
    expect(aabb.min).toEqual([8, -8, -2]);
    expect(aabb.max).toEqual([12, -2, 6]);
  });
});
