import { describe, expect, it, vi } from 'vitest';
import { buildArrayBvh } from '../buildArrayBvh.js';

const positions = new Float32Array([
  0, 0, 0, 0,
  1, 0, 0, 0,
  0, 1, 0, 0,
  2, 0, 0, 0,
  0, 2, 0, 0,
  2, 2, 0, 0,
]);

describe('out-of-range vertex indices fail closed', () => {
  it('rejects one malformed triangle instead of publishing a partial BVH', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => buildArrayBvh(
        positions,
        new Uint32Array([0, 1, 2, 0, 3, 999, 5, 0]),
        new Uint32Array([0, 0]),
        { positionStride: 4, indexStride: 4 },
      )).toThrow(/triangle 1.*out-of-range vertex.*i1=999/);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects an index exactly equal to vertexCount', () => {
    expect(() => buildArrayBvh(
      positions.subarray(0, 12),
      new Uint32Array([0, 1, 3, 0]),
      new Uint32Array([0]),
      { positionStride: 4, indexStride: 4 },
    )).toThrow(/vertexCount=3/);
  });

  it('still builds valid in-range geometry without warnings', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const result = buildArrayBvh(
        positions,
        new Uint32Array([0, 1, 2, 0]),
        new Uint32Array([0]),
        { positionStride: 4, indexStride: 4 },
      );
      expect(new Uint32Array(result.bvhNodes.buffer)[7]).toBe(0xffff0001);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });
});
