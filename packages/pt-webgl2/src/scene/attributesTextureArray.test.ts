import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '@vitrum/shared-bvh';
import {
  packAttributesArray,
  ATTR_LAYER_NORMAL,
  ATTR_LAYER_TANGENT,
  ATTR_LAYER_COLOR,
  ATTR_LAYER_COUNT,
} from './attributesTextureArray.js';

// ─────────────────────────────────────────────────────────────────────────────
// GPU-FREE attribute-array gate. Pack a 2-triangle merged stream and verify the
// 4-layer payload: layer count, the normal layer round-trips the merged normals,
// and the DERIVED tangents are unit-length and orthogonal-ish to the per-vertex
// normal (the load-bearing property for normal mapping). The merge ships no UVs,
// so the tangents come from the Frisvad orthonormal-basis fallback — which must
// still satisfy |t| = 1 and t·n ≈ 0.
// ─────────────────────────────────────────────────────────────────────────────

const GREY: MaterialSpec = { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0 };

/** One quad (2 triangles, 4 vertices) with +Z face normals. */
function quadScene(): Scene {
  const prim: MeshPrimitive = {
    kind: 'mesh',
    id: 'quad',
    positions: new Float32Array([
      -0.5, -0.5, 0,
      0.5, -0.5, 0,
      0.5, 0.5, 0,
      -0.5, 0.5, 0,
    ]),
    normals: new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: GREY,
  };
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } } as Scene;
}

describe('packAttributesArray — 4-layer normal/tangent/uv/color array', () => {
  const merged = mergeWorldSpaceFromCore(quadScene(), { positionStride: 4 });
  const grid = packAttributesArray(merged);
  const floatsPerLayer = grid.dim * grid.dim * 4;

  it('emits exactly 4 layers, dim = ceil(sqrt(vertexCount)), data sized for all layers', () => {
    expect(grid.layers).toBe(ATTR_LAYER_COUNT);
    expect(grid.layers).toBe(4);
    expect(grid.dim).toBe(Math.ceil(Math.sqrt(merged.vertexCount)));
    expect(grid.data.length).toBe(floatsPerLayer * 4);
  });

  it('layer 0 (normal) matches the merged per-vertex normals (vec3 in RGBA, .a = 0)', () => {
    const base = ATTR_LAYER_NORMAL * floatsPerLayer;
    const stride = merged.positionStrideFloats;
    for (let v = 0; v < merged.vertexCount; v += 1) {
      const o = base + v * 4;
      expect(grid.data[o]).toBeCloseTo(merged.normals[v * stride]!, 6);
      expect(grid.data[o + 1]).toBeCloseTo(merged.normals[v * stride + 1]!, 6);
      expect(grid.data[o + 2]).toBeCloseTo(merged.normals[v * stride + 2]!, 6);
      expect(grid.data[o + 3]).toBe(0); // vec3 → RGBA promotion, .a = 0
    }
  });

  it('layer 1 (tangent) tangents are unit-length and orthogonal-ish to the normal', () => {
    const tBase = ATTR_LAYER_TANGENT * floatsPerLayer;
    const stride = merged.positionStrideFloats;
    for (let v = 0; v < merged.vertexCount; v += 1) {
      const to = tBase + v * 4;
      const tx = grid.data[to]!;
      const ty = grid.data[to + 1]!;
      const tz = grid.data[to + 2]!;
      const len = Math.hypot(tx, ty, tz);
      expect(len).toBeCloseTo(1, 5); // unit-length

      const nx = merged.normals[v * stride]!;
      const ny = merged.normals[v * stride + 1]!;
      const nz = merged.normals[v * stride + 2]!;
      const ndt = nx * tx + ny * ty + nz * tz;
      expect(Math.abs(ndt)).toBeLessThan(1e-4); // orthogonal to the normal

      expect(grid.data[to + 3]).toBe(0); // .a = 0
    }
  });

  it('layer 3 (color) defaults to opaque white when the merge has no vertex colors', () => {
    const cBase = ATTR_LAYER_COLOR * floatsPerLayer;
    for (let v = 0; v < merged.vertexCount; v += 1) {
      const o = cBase + v * 4;
      expect(grid.data[o]).toBe(1);
      expect(grid.data[o + 1]).toBe(1);
      expect(grid.data[o + 2]).toBe(1);
      expect(grid.data[o + 3]).toBe(1);
    }
  });
});
