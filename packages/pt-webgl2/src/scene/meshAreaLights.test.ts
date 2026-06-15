import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { packMeshAreaLights, TRI_AREA_LIGHT_TYPE, TRI_LIGHT_PIXELS } from './meshAreaLights.js';

// B4 — mesh-area triangle-light packer. Pure-CPU: builds a fake merged geometry
// stream + scene and checks triangle extraction, area, radiance, and the layout.

function fakeMerged(overrides: Partial<WorldSpaceMergeResult> = {}): WorldSpaceMergeResult {
  // One unit quad in the y=0 plane (2 tris, total area 1), merge-order indices.
  const positions = new Float32Array([
    0, 0, 0, 0, // v0 (stride 4)
    1, 0, 0, 0, // v1
    1, 0, 1, 0, // v2
    0, 0, 1, 0, // v3
  ]);
  const mergedIndices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return {
    bvhNodes: new Float32Array(),
    positions,
    positionStrideFloats: 4,
    indices: mergedIndices,
    bvhIndexStride: 3,
    triMaterialId: new Uint32Array([0, 0]),
    normals: new Float32Array(positions.length),
    tangents: new Float32Array(positions.length),
    uvs: new Float32Array(8),
    mergedIndices,
    mergedTriMaterialId: new Uint32Array([0, 0]),
    materials: [],
    boundingBox: { min: [0, 0, 0], max: [1, 0, 1] },
    meshVertexRanges: [
      { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 2 },
    ],
    vertexCount: 4,
    triangleCount: 2,
    ...overrides,
  };
}

function sceneWith(emitters: Scene['emitters']): Scene {
  return { primitives: [], emitters, environment: { kind: 'none' } };
}

describe('packMeshAreaLights (B4)', () => {
  it('returns null/empty when there are no mesh-area emitters', () => {
    const out = packMeshAreaLights(sceneWith([]), fakeMerged());
    expect(out.data).toBeNull();
    expect(out.triLightCount).toBe(0);
    expect(out.totalEmissiveArea).toBe(0);
  });

  it('extracts every triangle of the referenced mesh with correct area + radiance', () => {
    const out = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [0.2, 0.4, 0.8], intensity: 10 }]),
      fakeMerged(),
    );
    expect(out.triLightCount).toBe(2);
    // Two right triangles of a unit quad → total area 1.
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data).not.toBeNull();
    const d = out.data!;
    // First triangle: s0 = (v0, type), s1 = (radiance, 0), s3.a = triArea.
    expect(d[3]).toBe(TRI_AREA_LIGHT_TYPE); // type id
    expect(d[4]).toBeCloseTo(0.2 * 10, 5); // radiance.r = color*intensity
    expect(d[5]).toBeCloseTo(0.4 * 10, 5);
    expect(d[6]).toBeCloseTo(0.8 * 10, 5);
    expect(d[15]).toBeCloseTo(0.5, 6); // each tri area = 0.5
  });

  it('skips meshes with no matching emitter (only emissive meshes contribute)', () => {
    const merged = fakeMerged({
      meshVertexRanges: [
        { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 1 },
        { name: 'wall', vertexStart: 0, vertexCount: 4, triStart: 1, triCount: 1 },
      ],
    });
    const out = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 }]),
      merged,
    );
    expect(out.triLightCount).toBe(1); // only 'panel' emits
  });

  it('uses the 6-texel stride per triangle light', () => {
    const out = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 }]),
      fakeMerged(),
    );
    expect(TRI_LIGHT_PIXELS).toBe(6);
    // 2 lights × 6 texels = 12 texels → square dim ≥ ceil(sqrt(12)) = 4.
    expect(out.dim).toBeGreaterThanOrEqual(4);
    expect(out.data!.length).toBe(out.dim * out.dim * 4);
  });

  it('packs mesh-area emitter castShadow:false into the shared s5.g shadow-disable lane', () => {
    const out = packMeshAreaLights(
      sceneWith([
        {
          kind: 'mesh-area',
          id: 'm',
          meshId: 'panel',
          color: [1, 1, 1],
          intensity: 1,
          castShadow: false,
        },
      ]),
      fakeMerged(),
    );
    expect(out.triLightCount).toBe(2);
    const d = out.data!;
    expect(d[21]).toBe(1);
    expect(d[TRI_LIGHT_PIXELS * 4 + 21]).toBe(1);

    const defaultOut = packMeshAreaLights(
      sceneWith([{ kind: 'mesh-area', id: 'm', meshId: 'panel', color: [1, 1, 1], intensity: 1 }]),
      fakeMerged(),
    );
    expect(defaultOut.data![21]).toBe(0);
    expect(defaultOut.data![TRI_LIGHT_PIXELS * 4 + 21]).toBe(0);
  });
});
