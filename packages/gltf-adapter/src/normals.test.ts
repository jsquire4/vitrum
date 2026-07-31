import { describe, expect, it } from 'vitest';
import { generateVertexNormals } from './normals.js';

describe('generateVertexNormals', () => {
  it('area-weights shared indexed vertices instead of last-face overwriting them', () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const indices = new Uint32Array([
      0, 1, 2, // +Z
      0, 1, 3, // -Y
    ]);

    const normals = generateVertexNormals(positions, indices);
    const invSqrt2 = Math.fround(1 / Math.sqrt(2));

    expect(Array.from(normals.slice(0, 3))).toEqual([
      0,
      -invSqrt2,
      invSqrt2,
    ]);
    expect(Array.from(normals.slice(3, 6))).toEqual([
      0,
      -invSqrt2,
      invSqrt2,
    ]);
    expect(Array.from(normals.slice(6, 9))).toEqual([0, 0, 1]);
    expect(Array.from(normals.slice(9, 12))).toEqual([0, -1, 0]);
  });

  it('keeps non-indexed triangle-list normals flat', () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);

    expect(Array.from(generateVertexNormals(positions, undefined))).toEqual([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      1, 0, 0,
      1, 0, 0,
      1, 0, 0,
    ]);
  });

  it('does not erase a tiny isolated face beside ordinary-sized geometry', () => {
    const s = 1e-30;
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      0, 0, 0,
      0, s, 0,
      0, 0, s,
    ]);

    expect(Array.from(generateVertexNormals(positions, undefined).slice(9))).toEqual([
      1, 0, 0,
      1, 0, 0,
      1, 0, 0,
    ]);
  });

  it('rejects indices outside the position accessor', () => {
    expect(() => generateVertexNormals(
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      new Uint32Array([0, 1, 3]),
    )).toThrow(/outside/);
  });
});
