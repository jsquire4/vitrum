/**
 * H34-b unit tests — non-finite triangle filtering
 *
 * Verifies that a triangle with at least one NaN/Inf vertex coordinate is
 * filtered BEFORE the BVH build (so the root AABB stays finite) and that the
 * BVH still builds correctly for the remaining valid triangles.
 */

import { describe, expect, it, vi } from 'vitest';
import { buildArrayBvh } from '../buildArrayBvh.js';

describe('H34-b: NaN/Inf triangle filtering', () => {
  it('one NaN vertex → that triangle filtered, BVH builds, root AABB finite', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      // Two triangles: first is valid, second has a NaN vertex coordinate.
      // stride-4 positions: 3 valid verts + 3 bad verts (one NaN).
      const positions = new Float32Array([
        // vertex 0 — valid
        0, 0, 0, 0,
        // vertex 1 — valid
        1, 0, 0, 0,
        // vertex 2 — valid
        0, 1, 0, 0,
        // vertex 3 — NaN coordinate
        NaN, 0, 0, 0,
        // vertex 4 — valid (but part of the bad triangle)
        2, 0, 0, 0,
        // vertex 5 — valid
        0, 2, 0, 0,
      ]);
      // stride-4 indices: tri 0 = {0,1,2} (valid), tri 1 = {3,4,5} (bad — v3 is NaN)
      const indices = new Uint32Array([
        0, 1, 2, 0,
        3, 4, 5, 0,
      ]);
      const matIds = new Uint32Array([0, 0]);

      const result = buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
      });

      // Root AABB must be finite (not poisoned by NaN).
      const f32 = new Float32Array(result.bvhNodes.buffer);
      for (let i = 0; i < 6; i += 1) {
        expect(isFinite(f32[i]!)).toBe(true);
      }

      // Root node leaf count word: 0xFFFF0000 | count.
      // After filtering triangle 1, only 1 valid triangle remains.
      // The node's splitAxisOrTriCount word (u32[7]) must be 0xFFFF0001.
      const u32 = new Uint32Array(result.bvhNodes.buffer);
      expect((u32[7]! >>> 0) >>> 16).toBe(0xffff);           // leaf flag
      expect(u32[7]! & 0xffff).toBe(1);                      // 1 triangle

      // A console.warn must have been emitted for triangle 1.
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Triangle 1'),
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('all-Inf vertex triangle is also filtered', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const positions = new Float32Array([
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        Infinity, 0, 0, 0,
        2, 0, 0, 0,
        0, 2, 0, 0,
      ]);
      const indices = new Uint32Array([
        0, 1, 2, 0,
        3, 4, 5, 0,
      ]);
      const matIds = new Uint32Array([0, 0]);
      const result = buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
      });
      // Root node must be a leaf with 1 triangle (the Inf one filtered).
      const u32 = new Uint32Array(result.bvhNodes.buffer);
      expect((u32[7]! >>> 0) >>> 16).toBe(0xffff);
      expect(u32[7]! & 0xffff).toBe(1);
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('non-finite'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('does NOT filter valid triangles (no NaN/Inf)', () => {
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
      // Root node must have 1 triangle (no filtering occurred).
      const u32 = new Uint32Array(result.bvhNodes.buffer);
      expect((u32[7]! >>> 0) >>> 16).toBe(0xffff);
      expect(u32[7]! & 0xffff).toBe(1);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('H60: ALL triangles non-finite', () => {
  it('returns the empty-BVH shape instead of building from zero records', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      // Two triangles, every one carries a NaN vertex — the triCount===0 guard
      // does NOT fire, so without the H60 post-filter guard the recursive build
      // ran on an empty subset (degenerate ±Infinity root).
      const positions = new Float32Array([
        NaN, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, NaN, 0,
        2, 0, 0, 0,
        0, 2, 0, 0,
      ]);
      const indices = new Uint32Array([0, 1, 2, 0, 3, 4, 5, 0]);
      const matIds = new Uint32Array([0, 0]);

      const result = buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
      });

      // Same shape as the zero-input path: a single zeroed 8-float node.
      expect(result.bvhNodes.length).toBe(8);
      const f32 = new Float32Array(result.bvhNodes.buffer);
      for (let i = 0; i < 8; i += 1) {
        expect(Number.isFinite(f32[i]!)).toBe(true);
      }
      expect(
        warnSpy.mock.calls.some(c => String(c[0]).includes('every input triangle was non-finite')),
      ).toBe(true);
    } finally {
      warnSpy.mockRestore();
    }
  });
});
