import { describe, it, expect } from 'vitest';
import { summarizeScene } from '../scene/flattenScene.js';
import type {
  Scene,
  MaterialSpec,
  MeshPrimitive,
  InstancedMeshPrimitive,
  AnalyticPrimitive,
  SkinnedMeshPrimitive,
} from '@vitrum/core';
import { asMat4 } from '@vitrum/core';

const MAT: MaterialSpec = {
  baseColor: [0.5, 0.5, 0.5],
  metallic: 0,
  roughness: 0.5,
};

const POS_TRIPLE = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const NORM_TRIPLE = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);

function mesh(id: string): MeshPrimitive {
  return { kind: 'mesh', id, positions: POS_TRIPLE, normals: NORM_TRIPLE, material: MAT };
}
function instanced(id: string, count: number): InstancedMeshPrimitive {
  const instances = Array.from({ length: count }, () =>
    asMat4([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]),
  );
  return { kind: 'instanced-mesh', id, positions: POS_TRIPLE, normals: NORM_TRIPLE, material: MAT, instances };
}
function analytic(id: string): AnalyticPrimitive {
  return { kind: 'analytic', id, shape: 'sphere', params: new Float32Array([0,0,0,1]), material: MAT };
}
function skinned(id: string): SkinnedMeshPrimitive {
  return {
    kind: 'skinned-mesh',
    id,
    positions: POS_TRIPLE,
    normals: NORM_TRIPLE,
    material: MAT,
    skinIndices: new Uint32Array(3 * 4),
    skinWeights: (() => { const w = new Float32Array(3 * 4); for (let i=0;i<3;i++) w[i*4]=1; return w; })(),
    bones: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
    boneInverses: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  };
}
function scene(primitives: Scene['primitives']): Scene {
  return { primitives, emitters: [], environment: { kind: 'none' } };
}

describe('summarizeScene', () => {
  it('zero primitives → all counts zero', () => {
    const s = summarizeScene(scene([]));
    expect(s.primitiveCount).toBe(0);
    expect(s.meshPrimitiveCount).toBe(0);
    expect(s.instancedMeshPrimitiveCount).toBe(0);
    expect(s.analyticPrimitiveCount).toBe(0);
    expect(s.skinnedMeshPrimitiveCount).toBe(0);
    expect(s.vertexCountEstimate).toBe(0);
    expect(s.instanceCountEstimate).toBe(0);
  });

  it('mesh / instanced-mesh / analytic / skinned-mesh classified into the correct count fields', () => {
    const s = summarizeScene(scene([
      mesh('m'),
      instanced('i', 5),
      analytic('a'),
      skinned('s'),
    ]));
    expect(s.primitiveCount).toBe(4);
    expect(s.meshPrimitiveCount).toBe(1);
    expect(s.instancedMeshPrimitiveCount).toBe(1);
    expect(s.analyticPrimitiveCount).toBe(1);
    expect(s.skinnedMeshPrimitiveCount).toBe(1);
    expect(s.instanceCountEstimate).toBe(5);     // only the instanced-mesh contributes
  });

  it('skinned-mesh vertex count contributes to vertexCountEstimate (C1)', () => {
    // POS_TRIPLE is 3 vertices.
    const s = summarizeScene(scene([skinned('s'), mesh('m')]));
    expect(s.vertexCountEstimate).toBe(6);       // 3 (mesh) + 3 (skinned)
  });
});
