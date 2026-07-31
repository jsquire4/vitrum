import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec, Scene } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '../worldSpaceMerge.js';

const MATERIAL: MaterialSpec = { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 };

function validNormals(vertexCount: number): Float32Array {
  const normals = new Float32Array(vertexCount * 3);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) normals[vertex * 3 + 2] = 1;
  return normals;
}

describe('mergeWorldSpaceFromCore malformed triangle rejection', () => {
  it('supports caller-keyed material ownership without mutating shared materials', () => {
    const makePrimitive = (id: string, x: number): Scene['primitives'][number] => ({
      kind: 'mesh',
      id,
      positions: new Float32Array([
        x, 0, 0,
        x + 1, 0, 0,
        x, 1, 0,
      ]),
      normals: validNormals(3),
      indices: new Uint32Array([0, 1, 2]),
      material: MATERIAL,
    });
    const scene: Scene = {
      primitives: [makePrimitive('owner', 0), makePrimitive('ordinary', 2)],
      emitters: [],
      environment: { kind: 'none' },
    };

    expect(mergeWorldSpaceFromCore(scene).materials).toHaveLength(1);
    const owned = mergeWorldSpaceFromCore(scene, {
      materialDedupKey: (primitive) =>
        primitive.id === 'owner' ? 'mesh-area-owner:owner' : '',
    });
    expect(owned.materials).toHaveLength(2);
    expect([...owned.mergedTriMaterialId]).toEqual([0, 1]);
    expect(scene.primitives[0]!.material).toBe(MATERIAL);
    expect(() => mergeWorldSpaceFromCore(scene, {
      materialDedupKey: (() => 42) as never,
    })).toThrow(/materialDedupKey must return a string/);
  });

  it('rejects out-of-range indexed triangles before building a partial stream', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'indexed-oob',
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
          2, 0, 0,
          0, 2, 0,
          2, 2, 0,
        ]),
        normals: validNormals(6),
        indices: new Uint32Array([
          0, 1, 2,
          3, 999, 5,
        ]),
        material: MATERIAL,
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => mergeWorldSpaceFromCore(scene, { positionStride: 4 })).toThrow(
        /indices\[4\].*reference a vertex/,
      );
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('does not invoke onWarning for invalid input that must throw', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'indexed-oob-callback',
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
          2, 0, 0,
          0, 2, 0,
          2, 2, 0,
        ]),
        normals: validNormals(6),
        indices: new Uint32Array([
          0, 1, 2,
          3, 999, 5,
        ]),
        material: MATERIAL,
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const onWarning = vi.fn(() => {
      throw new Error('host warning callback failed');
    });

    expect(() => mergeWorldSpaceFromCore(scene, { positionStride: 4, onWarning })).toThrow(
      /indices\[4\].*reference a vertex/,
    );
    expect(onWarning).not.toHaveBeenCalled();
  });

  it('rejects non-finite triangles instead of returning a repaired bounding box', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'nan-triangle',
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
          2, 0, 0,
          Number.NaN, 2, 0,
          2, 2, 0,
        ]),
        normals: validNormals(6),
        indices: new Uint32Array([
          0, 1, 2,
          3, 4, 5,
        ]),
        material: MATERIAL,
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => mergeWorldSpaceFromCore(scene, { positionStride: 4 })).toThrow(
        /positions\[12\].*finite/,
      );
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});
