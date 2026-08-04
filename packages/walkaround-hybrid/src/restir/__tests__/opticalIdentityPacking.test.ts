import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../bvhCore.js';

const THIN: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 0,
  metallic: 0,
  transmission: 1,
};

const BULK: MaterialSpec = {
  ...THIN,
  thickness: 1,
};

function thinTriangle(): MeshPrimitive {
  return {
    kind: 'mesh',
    id: 'thin-zero',
    positions: new Float32Array([
      -4, -1, 0,
      -2, -1, 0,
      -3, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    indices: new Uint32Array([0, 1, 2]),
    material: THIN,
  };
}

function bulkCube(): MeshPrimitive {
  return {
    kind: 'mesh',
    id: 'bulk-one',
    positions: new Float32Array([
      -1, -1, -1,
       1, -1, -1,
       1,  1, -1,
      -1,  1, -1,
      -1, -1,  1,
       1, -1,  1,
       1,  1,  1,
      -1,  1,  1,
    ]),
    normals: new Float32Array(24).fill(1),
    indices: new Uint32Array([
      0, 2, 1, 0, 3, 2,
      4, 5, 6, 4, 6, 7,
      0, 1, 5, 0, 5, 4,
      3, 7, 6, 3, 6, 2,
      0, 4, 7, 0, 7, 3,
      1, 2, 6, 1, 6, 5,
    ]),
    material: BULK,
  };
}

function scene(): Scene {
  return {
    primitives: [thinTriangle(), bulkCube()],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function triangleIdentityWords(
  buffers: ReturnType<typeof buildReSTIRSceneBVHForCoreScene>,
): Uint32Array {
  return new Uint32Array(buffers.opticalTriangleIdentity.cpuData);
}

describe('walkaround optical identity packing', () => {
  it('keeps primitive/range zero as an encoded nonzero thin-sheet identity in merged mode', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(scene(), {
      bvhMode: 'merged',
    });
    const words = triangleIdentityWords(buffers);
    const seen = new Set<string>();
    for (let triangle = 0; triangle < buffers.opticalTriangleIdentity.count; triangle += 1) {
      seen.add(`${words[triangle * 2]!}:${words[triangle * 2 + 1]!}`);
    }

    // boundary zero is invalid, while represented range ordinal zero encodes
    // as one. The bulk component's final encoded boundary is also one but its
    // represented range remains independently encoded as two.
    expect(seen).toEqual(new Set(['0:1', '1:2']));
    expect(new Uint32Array(
      buffers.opticalInstanceBoundaryIdBasePlusOne.cpuData,
    )).toEqual(new Uint32Array([1]));
  });

  it('packs component-relative triangle IDs and per-instance bases in TLAS mode', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(scene(), {
      bvhMode: 'tlas',
    });
    const words = triangleIdentityWords(buffers);
    const seen = new Set<string>();
    for (let triangle = 0; triangle < buffers.opticalTriangleIdentity.count; triangle += 1) {
      seen.add(`${words[triangle * 2]!}:${words[triangle * 2 + 1]!}`);
    }

    expect(seen).toEqual(new Set(['0:1', '1:2']));
    expect(new Uint32Array(
      buffers.opticalInstanceBoundaryIdBasePlusOne.cpuData,
    )).toEqual(new Uint32Array([0, 1]));
  });
});
