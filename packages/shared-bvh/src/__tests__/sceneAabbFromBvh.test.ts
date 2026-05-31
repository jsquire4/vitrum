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
