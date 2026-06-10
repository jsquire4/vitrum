/**
 * Out-of-range vertex-index filtering — unit tests.
 *
 * A triangle index that references a vertex beyond the `positions` buffer is
 * malformed mesh data. Before the guard, `getPosition`'s `?? 0` silently
 * yielded (0,0,0) — a FINITE coordinate that slips past the NaN/Inf filter and
 * collapses the triangle toward the origin, corrupting the BVH with no
 * diagnostic. The guard filters such triangles (warn + skip) like the NaN path.
 */

import { describe, expect, it, vi } from 'vitest';
import { buildArrayBvh } from '../buildArrayBvh.js';

describe('out-of-range vertex-index filtering', () => {
  it('index 999 into a 6-vertex buffer → that triangle filtered, BVH still builds', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      // 6 vertices (stride-4). vertexCount = 24 floats / 4 = 6, valid indices 0..5.
      const positions = new Float32Array([
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        2, 0, 0, 0,
        0, 2, 0, 0,
        2, 2, 0, 0,
      ]);
      // tri 0 = {0,1,2} (valid); tri 1 = {3,999,5} (999 is out of range).
      const indices = new Uint32Array([
        0, 1, 2, 0,
        3, 999, 5, 0,
      ]);
      const matIds = new Uint32Array([0, 0]);

      const result = buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
      });

      // Root AABB finite, no origin-collapse contamination.
      const f32 = new Float32Array(result.bvhNodes.buffer);
      for (let i = 0; i < 6; i += 1) {
        expect(isFinite(f32[i]!)).toBe(true);
      }
      // Only the 1 valid triangle survives → leaf with count 1.
      const u32 = new Uint32Array(result.bvhNodes.buffer);
      expect((u32[7]! >>> 0) >>> 16).toBe(0xffff);
      expect(u32[7]! & 0xffff).toBe(1);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('out-of-range vertex'),
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('an index exactly == vertexCount is out of range (0-based) → filtered', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      // 3 vertices (stride-4) → vertexCount=3, valid indices 0..2; index 3 is OOB.
      const positions = new Float32Array([
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
      ]);
      const indices = new Uint32Array([0, 1, 3, 0]);
      const matIds = new Uint32Array([0]);
      const result = buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
      });
      // All triangles filtered → empty-BVH shape (single zeroed node).
      expect(result.bvhNodes.length).toBe(8);
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('out-of-range vertex'),
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('does NOT filter when all indices are in range', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const positions = new Float32Array([
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 0,
      ]);
      const indices = new Uint32Array([0, 1, 2, 0]);
      const matIds = new Uint32Array([0]);
      const result = buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
      });
      const u32 = new Uint32Array(result.bvhNodes.buffer);
      expect((u32[7]! >>> 0) >>> 16).toBe(0xffff);
      expect(u32[7]! & 0xffff).toBe(1);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});
