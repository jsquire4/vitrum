import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';

import { buildReSTIRSceneBVHForCoreScene } from '../restir/bvhCore.js';

function tangentScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'tangent-tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        tangents: new Float32Array([
          1, 0, 0, -1,
          1, 0, 0, -1,
          1, 0, 0, -1,
        ]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          normalMap: { handle: { width: 1, height: 1, data: new Float32Array([0.5, 0.5, 1, 1]) } },
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('walkaround BVH tangent payload', () => {
  it('merged mode exposes authored tangent.xyzw beside positions/normals for material-map TBN', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(tangentScene(), { bvhMode: 'merged' });
    const tangents = new Float32Array(buffers.bvhTangents.cpuData);

    expect(buffers.bvhTangents.count).toBe(3);
    expect(tangents.length).toBe(12);
    expect(Array.from(tangents.slice(0, 4))).toEqual([1, 0, 0, -1]);
  });

  it('TLAS mode forwards packSceneFromCore tangent.xyzw without dropping the stream', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(tangentScene(), { bvhMode: 'tlas' });
    const tangents = new Float32Array(buffers.bvhTangents.cpuData);

    expect(buffers.bvhTangents.count).toBe(3);
    expect(tangents.length).toBe(12);
    expect(Array.from(tangents.slice(0, 4))).toEqual([1, 0, 0, -1]);
    expect(buffers.scenePack?.tangents.length).toBe(12);
  });
});
