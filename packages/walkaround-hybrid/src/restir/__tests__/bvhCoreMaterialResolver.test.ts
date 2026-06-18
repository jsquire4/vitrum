import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../bvhCore.js';

function material(baseColor: readonly [number, number, number]): MaterialSpec {
  return { baseColor, roughness: 1, metallic: 0 };
}

function triangle(id: string, xOffset: number, mat: MaterialSpec): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([
      xOffset, 0, 0,
      xOffset + 1, 0, 0,
      xOffset, 1, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    indices: new Uint32Array([0, 1, 2]),
    material: mat,
  };
}

function scene(primitives: MeshPrimitive[]): Scene {
  return {
    primitives,
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('ReSTIR bvhCore material resolver', () => {
  it('packs one material slot per unique mesh-like primitive in TLAS mode', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(scene([
      triangle('red-panel', 0, material([1, 0, 0])),
      triangle('green-panel', 2, material([0, 1, 0])),
    ]), { bvhMode: 'tlas' });

    expect([...new Uint32Array(buffers.triangleMaterialIds.cpuData)]).toEqual([0, 1]);
    expect(buffers.coreMaterials.map((m) => m.baseColor)).toEqual([
      [1, 0, 0],
      [0, 1, 0],
    ]);
  });

  it('rejects duplicate mesh-like primitive ids instead of reusing the first material slot', () => {
    const duplicateScene = scene([
      triangle('panel', 0, material([1, 0, 0])),
      triangle('panel', 2, material([0, 1, 0])),
    ]);

    expect(() => buildReSTIRSceneBVHForCoreScene(duplicateScene, { bvhMode: 'tlas' }))
      .toThrow(/duplicate mesh-like primitive id\(s\): panel/);
  });

  it('packs TLAS bvhIndex.xyz from stride-4 scene indices, not tri*3 offsets', () => {
    const buffers = buildReSTIRSceneBVHForCoreScene(scene([
      triangle('red-panel', 0, material([1, 0, 0])),
      triangle('green-panel', 2, material([0, 1, 0])),
    ]), { bvhMode: 'tlas' });

    const source = buffers.scenePack!.indices;
    const packed = new Uint32Array(buffers.bvhIndex.cpuData);
    expect(buffers.scenePack!.triangleCount).toBe(2);
    expect([...source.slice(0, 8)]).toEqual([0, 1, 2, 0, 3, 4, 5, 0]);

    for (let t = 0; t < buffers.scenePack!.triangleCount; t += 1) {
      expect([...packed.slice(t * 4, t * 4 + 3)]).toEqual([
        source[t * 4]!,
        source[t * 4 + 1]!,
        source[t * 4 + 2]!,
      ]);
    }

    // Regression guard for the old F-TLAS1 class: tri 1 used to read
    // [source[3], source[4], source[5]] = [0, 3, 4].
    expect([...packed.slice(4, 7)]).toEqual([3, 4, 5]);
  });
});
