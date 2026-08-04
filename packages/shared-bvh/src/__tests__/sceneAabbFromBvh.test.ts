import { describe, expect, it } from 'vitest';
import { deriveSceneAABBFromBvhPositions, expandIndicesToStride4 } from '../index.js';

/**
 * Behaviour-pin for the deduped scene-AABB scan. The expected values are the
 * EXACT output of the three former inline copies (vec4f stride-4 scan, pad
 * `(max-min)*0.01 + 1e-3`, ±10 fallback). Re-derived here from first principles
 * so the helper can never silently drift from the pinned math.
 */
describe('deriveSceneAABBFromBvhPositions', () => {
  function bvh(floats: number[]): { bvhPositions: { cpuData: ArrayBuffer } } {
    return { bvhPositions: { cpuData: new Float32Array(floats).buffer } };
  }

  it('returns the padded min/max over vec4f xyz lanes', () => {
    // Two vertices: (1,2,3,_) and (4,6,8,_). w lanes (99, 77) must be ignored.
    const aabb = deriveSceneAABBFromBvhPositions(bvh([1, 2, 3, 99, 4, 6, 8, 77]));
    const padX = (4 - 1) * 0.01 + 1e-3;
    const padY = (6 - 2) * 0.01 + 1e-3;
    const padZ = (8 - 3) * 0.01 + 1e-3;
    expect(aabb.min).toEqual([1 - padX, 2 - padY, 3 - padZ]);
    expect(aabb.max).toEqual([4 + padX, 6 + padY, 8 + padZ]);
  });

  it('single vertex: min==max before padding, pad is the +1e-3 floor', () => {
    const aabb = deriveSceneAABBFromBvhPositions(bvh([5, -5, 0, 0]));
    expect(aabb.min).toEqual([5 - 1e-3, -5 - 1e-3, 0 - 1e-3]);
    expect(aabb.max).toEqual([5 + 1e-3, -5 + 1e-3, 0 + 1e-3]);
  });

  it('empty buffer → ±10 fallback', () => {
    expect(deriveSceneAABBFromBvhPositions(bvh([]))).toEqual({
      min: [-10, -10, -10],
      max: [10, 10, 10],
    });
  });

  it('sub-vec4 buffer (length < 4) → ±10 fallback', () => {
    expect(deriveSceneAABBFromBvhPositions(bvh([1, 2, 3]))).toEqual({
      min: [-10, -10, -10],
      max: [10, 10, 10],
    });
  });

  it('all-NaN positions (non-finite extent) → ±10 fallback', () => {
    expect(deriveSceneAABBFromBvhPositions(bvh([NaN, NaN, NaN, 0]))).toEqual({
      min: [-10, -10, -10],
      max: [10, 10, 10],
    });
  });

  // REGRESSION: in `tlas` mode the position buffer holds per-BLAS LOCAL-space
  // vertices (traversal transforms the ray into instance space), so scanning it
  // yields bounds in the wrong coordinate space. `resolveReSTIRBvhMode` selects
  // tlas for ANY scene with >1 mesh-like primitive, so NRC / PPG / ReGIR were
  // handed a local-space AABB for essentially every multi-mesh scene. TLAS node
  // 0 stores the world-space AABB of the whole scene by construction.
  describe('tlas mode uses world-space TLAS root bounds, not local BLAS positions', () => {
    /** One TLAS node: f32[0..2]=boundsMin, f32[3..5]=boundsMax, u32[6..7]=payload. */
    function tlasNodes(
      min: [number, number, number],
      max: [number, number, number],
    ): { cpuData: ArrayBuffer } {
      const words = new Float32Array(8);
      words.set(min, 0);
      words.set(max, 3);
      return { cpuData: words.buffer };
    }

    it('prefers TLAS root bounds over the local-space position scan', () => {
      const aabb = deriveSceneAABBFromBvhPositions({
        bvhMode: 'tlas',
        // Local-space BLAS vertices — a unit cube at the origin.
        bvhPositions: { cpuData: new Float32Array([-1, -1, -1, 0, 1, 1, 1, 0]).buffer },
        // The instances are actually placed far from the origin in world space.
        tlas: { nodes: tlasNodes([10, 20, 30], [40, 60, 80]), nodeCount: 1 },
      });
      const padX = (40 - 10) * 0.01 + 1e-3;
      const padY = (60 - 20) * 0.01 + 1e-3;
      const padZ = (80 - 30) * 0.01 + 1e-3;
      expect(aabb.min).toEqual([10 - padX, 20 - padY, 30 - padZ]);
      expect(aabb.max).toEqual([40 + padX, 60 + padY, 80 + padZ]);
    });

    it('merged mode still scans positions (they are already world-space)', () => {
      const aabb = deriveSceneAABBFromBvhPositions({
        bvhMode: 'merged',
        bvhPositions: { cpuData: new Float32Array([1, 2, 3, 0, 4, 6, 8, 0]).buffer },
        tlas: { nodes: tlasNodes([10, 20, 30], [40, 60, 80]), nodeCount: 1 },
      });
      expect(aabb.min[0]).toBeCloseTo(1 - ((4 - 1) * 0.01 + 1e-3), 9);
      expect(aabb.max[0]).toBeCloseTo(4 + ((4 - 1) * 0.01 + 1e-3), 9);
    });

    it('falls back to the position scan when TLAS bounds are absent or degenerate', () => {
      const noTlas = deriveSceneAABBFromBvhPositions({
        bvhMode: 'tlas',
        bvhPositions: { cpuData: new Float32Array([1, 2, 3, 0, 4, 6, 8, 0]).buffer },
        tlas: undefined,
      });
      expect(noTlas.min[0]).toBeCloseTo(1 - ((4 - 1) * 0.01 + 1e-3), 9);

      const nanRoot = deriveSceneAABBFromBvhPositions({
        bvhMode: 'tlas',
        bvhPositions: { cpuData: new Float32Array([1, 2, 3, 0, 4, 6, 8, 0]).buffer },
        tlas: { nodes: tlasNodes([NaN, NaN, NaN], [NaN, NaN, NaN]), nodeCount: 1 },
      });
      expect(nanRoot.min[0]).toBeCloseTo(1 - ((4 - 1) * 0.01 + 1e-3), 9);

      const zeroCount = deriveSceneAABBFromBvhPositions({
        bvhMode: 'tlas',
        bvhPositions: { cpuData: new Float32Array([1, 2, 3, 0, 4, 6, 8, 0]).buffer },
        tlas: { nodes: tlasNodes([10, 20, 30], [40, 60, 80]), nodeCount: 0 },
      });
      expect(zeroCount.min[0]).toBeCloseTo(1 - ((4 - 1) * 0.01 + 1e-3), 9);
    });

    it('legacy shapes (raw buffer / positions-only) keep scanning', () => {
      const raw = deriveSceneAABBFromBvhPositions(new Float32Array([1, 2, 3, 0]));
      expect(raw.min).toEqual([1 - 1e-3, 2 - 1e-3, 3 - 1e-3]);
    });
  });
});

describe('expandIndicesToStride4', () => {
  it('expands stride-3 to stride-4 with zero-filled .w by default', () => {
    const out = expandIndicesToStride4(new Uint32Array([0, 1, 2, 3, 4, 5]));
    expect(Array.from(out)).toEqual([0, 1, 2, 0, 3, 4, 5, 0]);
  });

  it('packs a caller payload into the .w lane', () => {
    const out = expandIndicesToStride4(new Uint32Array([7, 8, 9]), (t) => 100 + t);
    expect(Array.from(out)).toEqual([7, 8, 9, 100]);
  });

  it('truncates a ragged tail (length not a multiple of 3)', () => {
    const out = expandIndicesToStride4(new Uint32Array([1, 2, 3, 4]));
    expect(Array.from(out)).toEqual([1, 2, 3, 0]);
  });
});
