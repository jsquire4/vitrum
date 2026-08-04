import { describe, expect, it } from 'vitest';
import { asMat4, type MaterialSpec, type MeshPrimitive, type Scene } from '@vitrum/core';
import {
  analyzeOpticalMediumTopology,
  lowerTransmissiveAnalyticPrimitives,
} from '../opticalMediumTopology.js';
import {
  decodeOpticalMediumBoundaryId,
  encodeOpticalMediumBoundaryId,
  OPTICAL_MEDIUM_MAX_BOUNDARY_ID,
  packMergedOpticalMediumBoundaryIds,
  packOpticalMediumBoundaryIds,
  resolvePackedOpticalMediumEncodedBoundaryId,
} from '../opticalMediumBoundaryPacking.js';
import { packSceneFromCore } from '../scenePack.js';
import { mergeWorldSpaceFromCore } from '../worldSpaceMerge.js';

const BULK: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 0,
  metallic: 0,
  transmission: 1,
  thickness: 1,
};

const POSITIONS = new Float32Array([
  -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
  -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
]);
const INDICES = new Uint32Array([
  0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
  0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
  0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
]);

function translation(x: number) {
  return asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, 0, 0, 1,
  ]));
}

function mesh(id: string): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: POSITIONS,
    normals: new Float32Array(POSITIONS.length).fill(1),
    indices: INDICES,
    material: BULK,
  };
}

function makeScene(primitives: Scene['primitives']): Scene {
  return { primitives, emitters: [], environment: { kind: 'none' } };
}

describe('packed optical medium boundary identity', () => {
  it('keeps component zero valid and gives shared-BLAS instances distinct IDs', () => {
    const base = mesh('instanced');
    const scene = makeScene([{
      ...base,
      kind: 'instanced-mesh',
      instances: [translation(-3), translation(3)],
    }]);
    const analysis = analyzeOpticalMediumTopology(scene);
    const pack = packSceneFromCore(scene, { tlas: true, resolveMaterialId: () => 0 });
    const ids = packOpticalMediumBoundaryIds(scene, pack, analysis);

    expect(new Set(ids.triangleComponentIndexPlusOne)).toEqual(new Set([1]));
    expect(new Set(ids.triangleRepresentedPrimitiveInstanceIds)).toEqual(new Set([1]));
    expect([...ids.instanceBoundaryIdBasePlusOne]).toEqual([1, 2]);
    expect(resolvePackedOpticalMediumEncodedBoundaryId(ids, 0, 0)).toBe(1);
    expect(resolvePackedOpticalMediumEncodedBoundaryId(ids, 0, 1)).toBe(2);
    expect(decodeOpticalMediumBoundaryId(1)).toBe(0);
  });

  it('maps BVH-reordered source triangles to disconnected component ordinals', () => {
    const positions = new Float32Array(POSITIONS.length * 2);
    positions.set(POSITIONS, 0);
    positions.set(POSITIONS, POSITIONS.length);
    for (let vertex = POSITIONS.length / 3; vertex < positions.length / 3; vertex += 1) {
      positions[vertex * 3] = positions[vertex * 3]! + 5;
    }
    const indices = new Uint32Array(INDICES.length * 2);
    indices.set(INDICES, 0);
    for (let index = 0; index < INDICES.length; index += 1) {
      indices[INDICES.length + index] = INDICES[index]! + POSITIONS.length / 3;
    }
    const scene = makeScene([{
      ...mesh('two-shells'),
      positions,
      normals: new Float32Array(positions.length).fill(1),
      indices,
    }]);
    const analysis = analyzeOpticalMediumTopology(scene);
    expect(analysis.components.map((component) => component.boundaryId)).toEqual([0, 1]);
    const pack = packSceneFromCore(scene, { tlas: false, resolveMaterialId: () => 0 });
    const ids = packOpticalMediumBoundaryIds(scene, pack, analysis);

    expect(new Set(ids.triangleComponentIndexPlusOne)).toEqual(new Set([1, 2]));
    expect([...ids.instanceBoundaryIdBasePlusOne]).toEqual([1]);
    for (let triangle = 0; triangle < pack.triangleCount; triangle += 1) {
      const source = pack.triangleSourceIndices[triangle]!;
      expect(ids.triangleComponentIndexPlusOne[triangle]).toBe(source < 12 ? 1 : 2);
    }
  });

  it('lowers bulk analytics before topology, pack, and identity generation', () => {
    const original = makeScene([{
      kind: 'analytic',
      id: 'bulk-sphere',
      shape: 'sphere',
      params: new Float32Array([0, 0, 0, 1]),
      material: BULK,
    }]);
    const represented = lowerTransmissiveAnalyticPrimitives(original);
    const analysis = analyzeOpticalMediumTopology(represented);
    const pack = packSceneFromCore(represented, {
      tlas: false,
      resolveMaterialId: () => 0,
    });
    const ids = packOpticalMediumBoundaryIds(represented, pack, analysis);
    expect(represented.primitives[0]?.kind).toBe('mesh');
    expect(new Set(ids.triangleComponentIndexPlusOne)).toEqual(new Set([1]));
    expect([...ids.instanceBoundaryIdBasePlusOne]).toEqual([1]);
  });

  it('round-trips the first and maximum encodable boundary IDs with zero invalid', () => {
    expect(decodeOpticalMediumBoundaryId(0)).toBeNull();
    expect(encodeOpticalMediumBoundaryId(0)).toBe(1);
    expect(encodeOpticalMediumBoundaryId(OPTICAL_MEDIUM_MAX_BOUNDARY_ID)).toBe(0xffff_ffff);
    expect(decodeOpticalMediumBoundaryId(0xffff_ffff)).toBe(OPTICAL_MEDIUM_MAX_BOUNDARY_ID);
    expect(() => encodeOpticalMediumBoundaryId(0xffff_ffff)).toThrow(RangeError);
  });

  it('packs final boundary IDs and range identity for flattened world-space instances', () => {
    const sceneValue = makeScene([
      mesh('bulk'),
      {
        ...mesh('thin'),
        positions: new Float32Array(POSITIONS).map((value, index) =>
          index % 3 === 0 ? value + 5 : value),
        material: { ...BULK, thickness: 0 },
      },
    ]);
    const analysis = analyzeOpticalMediumTopology(sceneValue);
    const merged = mergeWorldSpaceFromCore(sceneValue, { positionStride: 4 });
    const ids = packMergedOpticalMediumBoundaryIds(sceneValue, merged, analysis);

    expect([...ids.instanceBoundaryIdBasePlusOne]).toEqual([1]);
    expect(new Set(ids.triangleComponentIndexPlusOne)).toEqual(new Set([0, 1]));
    expect(new Set(ids.triangleRepresentedPrimitiveInstanceIds)).toEqual(new Set([1, 2]));
    for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
      const source = merged.bvhTriToMergedTri[triangle]!;
      const isBulk = source < INDICES.length / 3;
      expect(ids.triangleComponentIndexPlusOne[triangle]).toBe(isBulk ? 1 : 0);
      expect(ids.triangleRepresentedPrimitiveInstanceIds[triangle]).toBe(isBulk ? 1 : 2);
    }
  });
});
