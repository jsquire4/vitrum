import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '../worldSpaceMerge.js';
import { packSceneFromCore } from '../scenePack.js';

function selectedColorScene(vertexColorSet: number | null): Scene {
  const colorSets: Array<Float32Array | undefined> = [];
  colorSets[0] = new Float32Array([
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
  ]);
  colorSets[3] = new Float32Array([
    0, 1, 0, 0.2,
    0, 0.5, 0, 0.4,
    0, 0.25, 0, 0.6,
  ]);
  return {
    primitives: [{
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      colors: colorSets[0],
      colorSets,
      vertexColorSet,
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('selected COLOR_n backend packing', () => {
  it('feeds the selected high color lane into both canonical BVH packs', () => {
    const scene = selectedColorScene(3);
    const local = packSceneFromCore(scene, {
      tlas: false,
      resolveMaterialId: () => 0,
    });
    const merged = mergeWorldSpaceFromCore(scene);
    const expected = [...new Float32Array([
      0, 1, 0, 0.2,
      0, 0.5, 0, 0.4,
      0, 0.25, 0, 0.6,
    ])];
    expect([...local.colors]).toEqual(expected);
    expect([...merged.colors]).toEqual(expected);
  });

  it('packs identity white when vertex colors are explicitly disabled', () => {
    const scene = selectedColorScene(null);
    const packed = packSceneFromCore(scene, {
      tlas: false,
      resolveMaterialId: () => 0,
    });
    expect([...packed.colors]).toEqual([
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
    ]);
  });
});
