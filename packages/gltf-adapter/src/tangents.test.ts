import { describe, expect, it } from 'vitest';
import { generateVertexNormals } from './normals.js';
import { generateTangents } from './tangents.js';

describe('generateTangents scale-independent UV rank', () => {
  it('keeps a valid tangent frame on a uniformly tiny UV island', () => {
    const tangents = generateTangents(
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      new Float32Array([0, 0, 1e-8, 0, 0, 1e-8]),
      new Uint32Array([0, 1, 2]),
    );

    expect(tangents).toBeDefined();
    for (let vertex = 0; vertex < 3; vertex += 1) {
      const offset = vertex * 4;
      expect(tangents?.[offset]).toBeCloseTo(1, 6);
      expect(tangents?.[offset + 1]).toBeCloseTo(0, 6);
      expect(tangents?.[offset + 2]).toBeCloseTo(0, 6);
      expect(tangents?.[offset + 3]).toBe(1);
    }
  });

  it('still rejects an exactly rank-deficient UV triangle', () => {
    expect(generateTangents(
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      new Float32Array([0, 0, 1e-8, 0, 2e-8, 0]),
      new Uint32Array([0, 1, 2]),
    )).toBeUndefined();
  });

  it('preserves a tiny geometry-derived tangent instead of replacing its direction', () => {
    const s = 1e-30;
    const tangents = generateTangents(
      new Float32Array([0, 0, 0, 0, s, 0, -s, 0, 0]),
      new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      new Float32Array([0, 0, 1, 0, 0, 1]),
      new Uint32Array([0, 1, 2]),
    );

    expect(tangents).toBeDefined();
    for (let vertex = 0; vertex < 3; vertex += 1) {
      expect(tangents?.[vertex * 4]).toBeCloseTo(0, 6);
      expect(tangents?.[vertex * 4 + 1]).toBeCloseTo(1, 6);
      expect(tangents?.[vertex * 4 + 2]).toBeCloseTo(0, 6);
    }
  });
});

describe('generateVertexNormals scale-independent geometry', () => {
  it('keeps the direction of a tiny but non-degenerate face', () => {
    const normals = generateVertexNormals(
      new Float32Array([0, 0, 0, 1e-20, 0, 0, 0, 1e-20, 0]),
      new Uint32Array([0, 1, 2]),
    );

    for (let vertex = 0; vertex < 3; vertex += 1) {
      expect(normals[vertex * 3]).toBeCloseTo(0, 6);
      expect(normals[vertex * 3 + 1]).toBeCloseTo(0, 6);
      expect(normals[vertex * 3 + 2]).toBeCloseTo(1, 6);
    }
  });
});
