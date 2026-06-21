import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec, Scene } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '../worldSpaceMerge.js';

const MATERIAL: MaterialSpec = { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 };

describe('mergeWorldSpaceFromCore malformed triangle filtering', () => {
  it('filters out-of-range indexed triangles before they enter the merged stream', () => {
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
        normals: new Float32Array(18),
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
      const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });

      expect(merged.triangleCount).toBe(1);
      expect(merged.indices.length).toBe(3);
      expect(merged.triMaterialId.length).toBe(1);
      expect(merged.bvhTriToMergedTri.length).toBe(1);
      expect(merged.mergedIndices.length).toBe(3);
      expect(merged.mergedTriMaterialId.length).toBe(1);
      expect(merged.meshVertexRanges[0]?.triCount).toBe(1);
      for (const index of merged.indices) {
        expect(index).toBeLessThan(merged.vertexCount);
      }
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('out-of-range vertex index'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('keeps returned warnings authoritative when onWarning throws', () => {
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
        normals: new Float32Array(18),
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

    const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4, onWarning });

    expect(onWarning).toHaveBeenCalled();
    expect(merged.triangleCount).toBe(1);
    expect(merged.warnings).toEqual(expect.arrayContaining([
      expect.stringContaining('out-of-range vertex index'),
    ]));
  });

  it('filters non-finite triangles without poisoning the merged bounding box', () => {
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
        normals: new Float32Array(18),
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
      const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });

      expect(merged.triangleCount).toBe(1);
      expect(Array.from(merged.mergedIndices)).toEqual([0, 1, 2]);
      expect(merged.meshVertexRanges[0]?.triCount).toBe(1);
      for (const value of [...merged.boundingBox.min, ...merged.boundingBox.max]) {
        expect(Number.isFinite(value)).toBe(true);
      }
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('non-finite transformed vertex coordinate'));
    } finally {
      warnSpy.mockRestore();
    }
  });
});
