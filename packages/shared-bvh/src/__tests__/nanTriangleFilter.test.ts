import { describe, expect, it, vi } from 'vitest';
import { buildArrayBvh } from '../buildArrayBvh.js';

function build(positions: Float32Array, triangleCount: number) {
  const indices = new Uint32Array(triangleCount * 4);
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    indices[triangle * 4] = triangle * 3;
    indices[triangle * 4 + 1] = triangle * 3 + 1;
    indices[triangle * 4 + 2] = triangle * 3 + 2;
  }
  return buildArrayBvh(positions, indices, new Uint32Array(triangleCount), {
    positionStride: 4,
    indexStride: 4,
  });
}

describe('non-finite BVH coordinates fail closed', () => {
  it.each([
    ['NaN', Number.NaN],
    ['Infinity', Number.POSITIVE_INFINITY],
  ])('rejects %s before publishing a partial BVH', (_label, bad) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => build(new Float32Array([
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        bad, 0, 0, 0,
        2, 0, 0, 0,
        0, 2, 0, 0,
      ]), 2)).toThrow(/positions\[12\].*finite/);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('accepts finite geometry without warnings', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const result = build(new Float32Array([
        0, 0, 0, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
      ]), 1);
      expect(new Uint32Array(result.bvhNodes.buffer)[7]).toBe(0xffff0001);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('rejects all-invalid input instead of returning a fabricated empty BVH', () => {
    expect(() => build(new Float32Array([
      Number.NaN, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
    ]), 1)).toThrow(/positions\[0\].*finite/);
  });
});
