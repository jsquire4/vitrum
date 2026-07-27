import { describe, expect, it } from 'vitest';
import { asMat4, type Scene, type SkinnedMeshPrimitive } from '@vitrum/core';
import { packGpuUvSets } from '../scene/gpuUvPacking.js';
import { solveSkinnedPrimitive } from '../scene/solveSkinnedPrimitive.js';

const IDENTITY = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);
const IDENTITY_MAT4 = asMat4(IDENTITY);

const TRI_POSITIONS = new Float32Array([
  0, 0, 0,
  1, 0, 0,
  0, 1, 0,
]);
const TRI_NORMALS = new Float32Array([
  0, 0, 1,
  0, 0, 1,
  0, 0, 1,
]);
const MATERIAL = {
  baseColor: [1, 1, 1] as [number, number, number],
  roughness: 0.5,
  metallic: 0,
};

function sceneWithPrimitives(primitives: Scene['primitives']): Scene {
  return {
    primitives,
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('pt-webgpu compact arbitrary UV packing', () => {
  it('stores one tail plane for a sparse authored texCoord and zero-fills missing streams', () => {
    const uv3 = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const scene = sceneWithPrimitives([
      {
        kind: 'mesh',
        id: 'a',
        positions: TRI_POSITIONS,
        normals: TRI_NORMALS,
        uvSets: [undefined, undefined, undefined, uv3],
        material: MATERIAL,
      },
      {
        kind: 'mesh',
        id: 'b',
        positions: TRI_POSITIONS,
        normals: TRI_NORMALS,
        material: MATERIAL,
      },
    ]);
    const primary = new Float32Array(6 * 4);
    const packed = packGpuUvSets(
      scene,
      primary,
      [
        { primitiveId: 'a', vertexStart: 0, vertexCount: 3 },
        { primitiveId: 'b', vertexStart: 3, vertexCount: 3 },
      ],
      [0, 1, 3],
    );

    expect(packed).toHaveLength(primary.length * 2);
    const tail = packed.slice(primary.length);
    expect(Array.from(tail.slice(0, 12))).toEqual([
      expect.closeTo(0.1), expect.closeTo(0.2), 0, 0,
      expect.closeTo(0.3), expect.closeTo(0.4), 0, 0,
      expect.closeTo(0.5), expect.closeTo(0.6), 0, 0,
    ]);
    expect(Array.from(tail.slice(12))).toEqual(new Array(12).fill(0));
  });

  it('duplicates an authored UV set across merged instance ranges', () => {
    const uv3 = new Float32Array([0, 0, 1, 0, 0, 1]);
    const scene = sceneWithPrimitives([{
      kind: 'instanced-mesh',
      id: 'instanced',
      positions: TRI_POSITIONS,
      normals: TRI_NORMALS,
      uvSets: [undefined, undefined, undefined, uv3],
      instances: [IDENTITY_MAT4, IDENTITY_MAT4],
      material: MATERIAL,
    }]);
    const primary = new Float32Array(6 * 4);
    const packed = packGpuUvSets(
      scene,
      primary,
      [
        { sourcePrimitiveId: 'instanced', vertexStart: 0, vertexCount: 3 },
        { sourcePrimitiveId: 'instanced', vertexStart: 3, vertexCount: 3 },
      ],
      [0, 1, 3],
    );
    const tail = packed.slice(primary.length);
    expect(Array.from(tail.slice(0, 12))).toEqual(Array.from(tail.slice(12)));
  });

  it('compacts UV ids at and above the native array-index ceiling into adjacent GPU planes', () => {
    const nativeCeilingIndex = 0xffff_fffe;
    const ordinaryPropertyIndex = 0x1_0000_0001;
    const nativeUv = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const ordinaryUv = new Float32Array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
    const uvSets: Array<Float32Array | undefined> = [];
    uvSets[nativeCeilingIndex] = nativeUv;
    uvSets[ordinaryPropertyIndex] = ordinaryUv;
    const scene = sceneWithPrimitives([{
      kind: 'mesh',
      id: 'array-boundary',
      positions: TRI_POSITIONS,
      normals: TRI_NORMALS,
      uvSets,
      material: MATERIAL,
    }]);
    const primary = new Float32Array(3 * 4);

    const packed = packGpuUvSets(
      scene,
      primary,
      [{ primitiveId: 'array-boundary', vertexStart: 0, vertexCount: 3 }],
      [0, 1, nativeCeilingIndex, ordinaryPropertyIndex],
    );

    const nativePlane = packed.slice(primary.length, primary.length * 2);
    const ordinaryPlane = packed.slice(primary.length * 2);
    expect(Array.from(nativePlane)).toEqual([
      expect.closeTo(0.1), expect.closeTo(0.2), 0, 0,
      expect.closeTo(0.3), expect.closeTo(0.4), 0, 0,
      expect.closeTo(0.5), expect.closeTo(0.6), 0, 0,
    ]);
    expect(Array.from(ordinaryPlane)).toEqual([
      expect.closeTo(0.6), expect.closeTo(0.5), 0, 0,
      expect.closeTo(0.4), expect.closeTo(0.3), 0, 0,
      expect.closeTo(0.2), expect.closeTo(0.1), 0, 0,
    ]);
  });

  it('CPU-solves eight skin influences and arbitrary morph UV sets before packing', () => {
    const influenceWidth = 8;
    const skinIndices = new Uint32Array(3 * influenceWidth);
    const skinWeights = new Float32Array(3 * influenceWidth);
    for (let vertex = 0; vertex < 3; vertex += 1) {
      skinWeights[vertex * influenceWidth] = 1;
    }
    const uv3 = new Float32Array([0, 0, 1, 0, 0, 1]);
    const uv3Delta = new Float32Array([0.25, 0.5, 0, 0, 0, 0]);
    const primitive: SkinnedMeshPrimitive = {
      kind: 'skinned-mesh',
      id: 'skin',
      positions: TRI_POSITIONS,
      normals: TRI_NORMALS,
      uvSets: [undefined, undefined, undefined, uv3],
      skinIndices,
      skinWeights,
      skinInfluencesPerVertex: influenceWidth,
      bones: IDENTITY,
      boneInverses: IDENTITY,
      morphTargets: [new Float32Array(TRI_POSITIONS.length)],
      morphTargetUvSets: [undefined, undefined, undefined, [uv3Delta]],
      morphWeights: new Float32Array([1]),
      material: MATERIAL,
    };

    const solved = solveSkinnedPrimitive(primitive);
    expect(solved.uvSets?.[3]?.[0]).toBeCloseTo(0.25);
    expect(solved.uvSets?.[3]?.[1]).toBeCloseTo(0.5);
    const solvedScene = sceneWithPrimitives([{ ...primitive, ...solved }]);
    const packed = packGpuUvSets(
      solvedScene,
      new Float32Array(3 * 4),
      [{ primitiveId: 'skin', vertexStart: 0, vertexCount: 3 }],
      [0, 1, 3],
    );
    expect(packed[3 * 4]).toBeCloseTo(0.25);
    expect(packed[3 * 4 + 1]).toBeCloseTo(0.5);
  });
});
