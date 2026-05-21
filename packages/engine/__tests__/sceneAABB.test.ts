import { describe, it, expect } from 'vitest';
import { computeSceneAABB } from '../src/sceneAABB.js';
import type { Scene, MeshPrimitive, InstancedMeshPrimitive, Material, Mat4, AnalyticPrimitive } from '@vitrum/core';

const MAT: Material = {
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

  it('honours an affine transform on a mesh primitive', () => {
    // translate +10 on X via column-major identity * translate(10, 0, 0)
    // prettier-ignore
    const xform = new Float32Array([
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
    const right = new Float32Array([
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
      new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      // translate +3 X
      // prettier-ignore
      new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 3,0,0,1]),
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

  it('ignores empty instanced-mesh instance arrays instead of returning infinities', () => {
    const cube = unitCube('inst-empty');
    const inst: InstancedMeshPrimitive = {
      kind: 'instanced-mesh',
      id: 'ie',
      positions: cube.positions,
      normals: cube.normals,
      indices: cube.indices!,
      material: cube.material,
      instances: [],
    };
    const aabb = computeSceneAABB(emptyScene([inst]));
    expect(aabb.min).toEqual([-0.5, -0.5, -0.5]);
    expect(aabb.max).toEqual([0.5, 0.5, 0.5]);
    expect(Number.isFinite(aabb.diagonal)).toBe(true);
  });

  it('uses analytic primitive params for bounds when no fallback mesh is present', () => {
    const sphere: AnalyticPrimitive = {
      kind: 'analytic',
      id: 's',
      shape: 'sphere',
      params: new Float32Array([2, 3, 4, 1]),
      material: MAT,
    };
    const aabb = computeSceneAABB(emptyScene([sphere]));
    expect(aabb.min).toEqual([1, 2, 3]);
    expect(aabb.max).toEqual([3, 4, 5]);
  });

  it('includes non-directional emitters in scene bounds', () => {
    const scene: Scene = {
      primitives: [],
      emitters: [
        { kind: 'point', id: 'p', color: [1, 1, 1], intensity: 1, position: [10, 0, 0] },
      ],
      environment: { kind: 'none' },
    };
    const aabb = computeSceneAABB(scene);
    expect(aabb.center[0]).toBe(10);
    expect(aabb.diagonal).toBe(1);
  });
});
