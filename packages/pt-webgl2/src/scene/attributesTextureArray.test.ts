import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { mergeWorldSpaceFromCore } from '@vitrum/shared-bvh';
import {
  packAttributesArray,
  ATTR_LAYER_NORMAL,
  ATTR_LAYER_TANGENT,
  ATTR_LAYER_UV,
  ATTR_LAYER_COLOR,
  ATTR_LAYER_UV1,
  ATTR_LAYER_COUNT,
} from './attributesTextureArray.js';

// ─────────────────────────────────────────────────────────────────────────────
// GPU-FREE attribute-array gate. Pack a 2-triangle merged stream and verify the
// 5-layer payload: layer count, the normal layer round-trips the merged normals,
// the DERIVED tangents are unit-length and orthogonal-ish to the per-vertex
// normal (the load-bearing property for normal mapping), and layer 4 (uv1) carries
// correct data when present (falling back to uv0 when absent). The merge ships no
// UVs by default, so the tangents come from the Frisvad orthonormal-basis fallback
// — which must still satisfy |t| = 1 and t·n ≈ 0.
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
  return { primitives: [prim], emitters: [], environment: { kind: 'none' } };
}

describe('packAttributesArray — 5-layer normal/tangent/uv0/color/uv1 array', () => {
  const merged = mergeWorldSpaceFromCore(quadScene(), { positionStride: 4 });
  const grid = packAttributesArray(merged);
  const floatsPerLayer = grid.dim * grid.dim * 4;

  it('emits exactly 5 layers, dim = ceil(sqrt(vertexCount)), data sized for all layers', () => {
    expect(grid.layers).toBe(ATTR_LAYER_COUNT);
    expect(grid.layers).toBe(5);
    expect(grid.dim).toBe(Math.ceil(Math.sqrt(merged.vertexCount)));
    expect(grid.data.length).toBe(floatsPerLayer * 5);
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

      expect(grid.data[to + 3]).toBe(1); // handedness defaults to +1
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

  it('layer 3 (color) packs RGB and RGBA vertex-color streams with opaque alpha fallback', () => {
    const rgb = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      0.25, 0.5, 0.75,
    ]);
    const rgbGrid = packAttributesArray({ ...merged, colors: rgb });
    const rgbFpl = rgbGrid.dim * rgbGrid.dim * 4;
    const rgbBase = ATTR_LAYER_COLOR * rgbFpl;
    expect(Array.from(rgbGrid.data.slice(rgbBase, rgbBase + 16))).toEqual([
      1, 0, 0, 1,
      0, 1, 0, 1,
      0, 0, 1, 1,
      0.25, 0.5, 0.75, 1,
    ]);

    const rgba = new Float32Array([
      1, 0, 0, 0.2,
      0, 1, 0, 0.4,
      0, 0, 1, 0.6,
      0.25, 0.5, 0.75, 0.8,
    ]);
    const rgbaGrid = packAttributesArray({ ...merged, colors: rgba });
    const rgbaFpl = rgbaGrid.dim * rgbaGrid.dim * 4;
    const rgbaBase = ATTR_LAYER_COLOR * rgbaFpl;
    const expectedRgba = [
      1, 0, 0, 0.2,
      0, 1, 0, 0.4,
      0, 0, 1, 0.6,
      0.25, 0.5, 0.75, 0.8,
    ];
    const actualRgba = Array.from(rgbaGrid.data.slice(rgbaBase, rgbaBase + 16));
    for (let i = 0; i < expectedRgba.length; i += 1) {
      expect(actualRgba[i]).toBeCloseTo(expectedRgba[i]!, 6);
    }
  });

  it('layer 4 (uv1) falls back to uv0 when no uv1 supplied', () => {
    // No uv1 on the quad scene — layer 4 must match layer 2 (uv0) per vertex.
    const uv0Base = ATTR_LAYER_UV * floatsPerLayer;
    const uv1Base = ATTR_LAYER_UV1 * floatsPerLayer;
    for (let v = 0; v < merged.vertexCount; v += 1) {
      const o = v * 4;
      expect(grid.data[uv1Base + o]).toBe(grid.data[uv0Base + o]);
      expect(grid.data[uv1Base + o + 1]).toBe(grid.data[uv0Base + o + 1]);
      expect(grid.data[uv1Base + o + 2]).toBe(0);
      expect(grid.data[uv1Base + o + 3]).toBe(0);
    }
  });
});

describe('packAttributesArray — tangent handedness', () => {
  it('uses authored tangents when supplied by the caller', () => {
    const merged2 = mergeWorldSpaceFromCore(quadScene(), { positionStride: 4 });
    const tangents = new Float32Array(merged2.vertexCount * 4);
    for (let v = 0; v < merged2.vertexCount; v += 1) {
      tangents[v * 4] = 0;
      tangents[v * 4 + 1] = 1;
      tangents[v * 4 + 2] = 0;
      tangents[v * 4 + 3] = -1;
    }

    const gridWithTangents = packAttributesArray({ ...merged2, tangents });
    const fpl = gridWithTangents.dim * gridWithTangents.dim * 4;
    const tangentBase = ATTR_LAYER_TANGENT * fpl;
    for (let v = 0; v < merged2.vertexCount; v += 1) {
      const o = tangentBase + v * 4;
      expect(gridWithTangents.data[o]).toBeCloseTo(0, 6);
      expect(gridWithTangents.data[o + 1]).toBeCloseTo(1, 6);
      expect(gridWithTangents.data[o + 2]).toBeCloseTo(0, 6);
      expect(gridWithTangents.data[o + 3]).toBe(-1);
    }
  });

  it('derives tangents for a uniformly tiny but full-rank UV island', () => {
    const prim: MeshPrimitive = {
      kind: 'mesh',
      id: 'tiny-uv-island',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array([0, 0, 1e-8, 0, 0, 1e-8]),
      indices: new Uint32Array([0, 1, 2]),
      material: GREY,
    };
    const mergedTiny = mergeWorldSpaceFromCore(
      { primitives: [prim], emitters: [], environment: { kind: 'none' } },
      { positionStride: 4 },
    );
    const gridTiny = packAttributesArray(mergedTiny);
    const floatsPerLayer = gridTiny.dim * gridTiny.dim * 4;
    const tangentBase = ATTR_LAYER_TANGENT * floatsPerLayer;

    for (let vertex = 0; vertex < 3; vertex += 1) {
      const offset = tangentBase + vertex * 4;
      expect(gridTiny.data[offset]).toBeCloseTo(1, 6);
      expect(gridTiny.data[offset + 1]).toBeCloseTo(0, 6);
      expect(gridTiny.data[offset + 2]).toBeCloseTo(0, 6);
      expect(gridTiny.data[offset + 3]).toBe(1);
    }
  });

  it.each([1e-30, 1e30])(
    'preserves a non-fallback tangent direction for a finite geometry scale of %s',
    geometryScale => {
      const prim: MeshPrimitive = {
        kind: 'mesh',
        id: `scaled-${geometryScale}`,
        positions: new Float32Array([
          0, 0, 0,
          0, geometryScale, 0,
          geometryScale, 0, 0,
        ]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        indices: new Uint32Array([0, 1, 2]),
        material: GREY,
      };
      const scaledMerged = mergeWorldSpaceFromCore(
        { primitives: [prim], emitters: [], environment: { kind: 'none' } },
        { positionStride: 4 },
      );
      const scaledGrid = packAttributesArray(scaledMerged);
      const scaledFloatsPerLayer = scaledGrid.dim * scaledGrid.dim * 4;
      const tangentBase = ATTR_LAYER_TANGENT * scaledFloatsPerLayer;

      for (let vertex = 0; vertex < 3; vertex += 1) {
        const offset = tangentBase + vertex * 4;
        expect(scaledGrid.data[offset]).toBeCloseTo(0, 5);
        expect(scaledGrid.data[offset + 1]).toBeCloseTo(1, 5);
        expect(scaledGrid.data[offset + 2]).toBeCloseTo(0, 5);
      }
    },
  );

  it('derives the conditioned tangent for a finite near-rank UV parameterization', () => {
    const prim: MeshPrimitive = {
      kind: 'mesh',
      id: 'near-rank-uv',
      positions: new Float32Array([0, 0, 0, 0, 1, 0, 1, 0, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uvs: new Float32Array([0, 0, 1, 0, 1, 1e-12]),
      indices: new Uint32Array([0, 1, 2]),
      material: GREY,
    };
    const nearRankMerged = mergeWorldSpaceFromCore(
      { primitives: [prim], emitters: [], environment: { kind: 'none' } },
      { positionStride: 4 },
    );
    const nearRankGrid = packAttributesArray(nearRankMerged);
    const nearRankFloatsPerLayer = nearRankGrid.dim * nearRankGrid.dim * 4;
    const tangentBase = ATTR_LAYER_TANGENT * nearRankFloatsPerLayer;

    expect(nearRankGrid.data[tangentBase]).toBeCloseTo(0, 5);
    expect(nearRankGrid.data[tangentBase + 1]).toBeCloseTo(1, 5);
    expect(nearRankGrid.data[tangentBase + 2]).toBeCloseTo(0, 5);
  });

  it('honors a finite authored tangent below the former absolute length cutoff', () => {
    const merged2 = mergeWorldSpaceFromCore(quadScene(), { positionStride: 4 });
    const tangents = new Float32Array(merged2.vertexCount * 4);
    for (let vertex = 0; vertex < merged2.vertexCount; vertex += 1) {
      tangents[vertex * 4 + 1] = 1e-30;
      tangents[vertex * 4 + 3] = -1;
    }

    const authoredGrid = packAttributesArray({ ...merged2, tangents });
    const authoredFloatsPerLayer = authoredGrid.dim * authoredGrid.dim * 4;
    const tangentBase = ATTR_LAYER_TANGENT * authoredFloatsPerLayer;
    for (let vertex = 0; vertex < merged2.vertexCount; vertex += 1) {
      const offset = tangentBase + vertex * 4;
      expect(authoredGrid.data[offset]).toBeCloseTo(0, 6);
      expect(authoredGrid.data[offset + 1]).toBeCloseTo(1, 6);
      expect(authoredGrid.data[offset + 2]).toBeCloseTo(0, 6);
      expect(authoredGrid.data[offset + 3]).toBe(-1);
    }
  });

  it('derives negative handedness for mirrored UVs', () => {
    const prim: MeshPrimitive = {
      kind: 'mesh',
      id: 'mirrored',
      positions: new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      uvs: new Float32Array([
        0, 0,
        0, 1,
        1, 0,
      ]),
      indices: new Uint32Array([0, 1, 2]),
      material: GREY,
    };
    const mergedMirrored = mergeWorldSpaceFromCore(
      { primitives: [prim], emitters: [], environment: { kind: 'none' } },
      { positionStride: 4 },
    );
    const gridMirrored = packAttributesArray(mergedMirrored);
    const fpl = gridMirrored.dim * gridMirrored.dim * 4;
    const tangentBase = ATTR_LAYER_TANGENT * fpl;
    for (let v = 0; v < mergedMirrored.vertexCount; v += 1) {
      expect(gridMirrored.data[tangentBase + v * 4 + 3]).toBe(-1);
    }
  });
});

describe('packAttributesArray — uv1 data from caller-supplied Float32Array', () => {
  it('layer 4 (uv1) carries distinct values when mergedUv1 is supplied', () => {
    const merged2 = mergeWorldSpaceFromCore(quadScene(), { positionStride: 4 });
    // Build a synthetic uv1 array with values that differ from uv0 (all 0).
    const uv1Data = new Float32Array(merged2.vertexCount * 2);
    for (let v = 0; v < merged2.vertexCount; v += 1) {
      uv1Data[v * 2] = 0.1 + v * 0.1;     // u1 distinct from uv0 (0)
      uv1Data[v * 2 + 1] = 0.2 + v * 0.1; // v1 distinct from uv0 (0)
    }
    const gridWithUv1 = packAttributesArray({ ...merged2, uv1: uv1Data });
    const fpl = gridWithUv1.dim * gridWithUv1.dim * 4;
    const uv1Base = ATTR_LAYER_UV1 * fpl;
    for (let v = 0; v < merged2.vertexCount; v += 1) {
      const o = v * 4;
      expect(gridWithUv1.data[uv1Base + o]).toBeCloseTo(0.1 + v * 0.1, 6);
      expect(gridWithUv1.data[uv1Base + o + 1]).toBeCloseTo(0.2 + v * 0.1, 6);
      expect(gridWithUv1.data[uv1Base + o + 2]).toBe(0);
      expect(gridWithUv1.data[uv1Base + o + 3]).toBe(0);
    }
  });
});
