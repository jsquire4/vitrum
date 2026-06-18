import { describe, expect, it } from 'vitest';
import type { MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import {
  hasMeshAreaLightForPrimitive,
  packMeshAreaLights,
  TRI_AREA_LIGHT_TYPE,
  TRI_LIGHT_PIXELS,
} from './meshAreaLights.js';

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
    bvhTriToMergedTri: new Uint32Array([0, 1]),
    normals: new Float32Array(positions.length),
    tangents: new Float32Array(positions.length),
    colors: new Float32Array([
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
      1, 1, 1, 1,
    ]),
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

function material(overrides: Partial<MaterialSpec>): MaterialSpec {
  return { baseColor: [0.5, 0.5, 0.5], roughness: 1, metallic: 0, ...overrides };
}

function luminance(rgb: readonly [number, number, number]): number {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

function panelPrimitive(mat: MaterialSpec, castShadow = true): MeshPrimitive {
  return {
    kind: 'mesh',
    id: 'panel',
    positions: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 0, 1,
      0, 0, 1,
    ]),
    normals: new Float32Array([
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
    ]),
    uvs: new Float32Array(8),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    material: mat,
    castShadow,
  };
}

function sceneWithPrimitive(primitive: MeshPrimitive, emitters: Scene['emitters'] = []): Scene {
  return { primitives: [primitive], emitters, environment: { kind: 'none' } };
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
    expect(d[16]).toBeCloseTo(luminance([2, 4, 8]) * 0.5, 6);
    expect(out.totalEmissivePower).toBeCloseTo(luminance([2, 4, 8]), 6);
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

  it('synthesizes triangle-light NEE for emissive mesh materials without explicit emitters', () => {
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({ emissive: [0.25, 0.5, 1], emissiveIntensity: 4 }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(2);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data![4]).toBeCloseTo(1, 6);
    expect(out.data![5]).toBeCloseTo(2, 6);
    expect(out.data![6]).toBeCloseTo(4, 6);
    expect(hasMeshAreaLightForPrimitive(
      sceneWithPrimitive(panelPrimitive(material({ emissive: [0.25, 0.5, 1], emissiveIntensity: 4 }))),
      'panel',
    )).toBe(true);
  });

  it('subdivides CPU-readable emissiveMap implicit triangle lights with UV-local radiance', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
    };
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveIntensity: 3,
        emissiveMap,
      }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(8);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    // This fixture's fake merged UVs are all (0,0), so every sub-triangle samples
    // the red texel; power is luminance([6,0,0]) over total area 1.
    expect(out.totalEmissivePower).toBeCloseTo(luminance([6, 0, 0]), 6);
    expect(out.data![4]).toBeCloseTo(6, 6);
    expect(out.data![5]).toBeCloseTo(0, 6);
    expect(out.data![6]).toBeCloseTo(0, 6);
    expect(out.warnings).toEqual([]);
  });

  it('clips CPU-readable emissiveMap UV footprints to exact texel cells', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
    };
    const oneTri = new Uint32Array([0, 1, 3]);
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [2, 2, 2],
        emissiveMap,
      }))),
      fakeMerged({
        indices: oneTri,
        mergedIndices: oneTri,
        bvhTriToMergedTri: new Uint32Array([0]),
        triMaterialId: new Uint32Array([0]),
        mergedTriMaterialId: new Uint32Array([0]),
        meshVertexRanges: [
          { name: 'panel', vertexStart: 0, vertexCount: 4, triStart: 0, triCount: 1 },
        ],
        triangleCount: 1,
        uvs: new Float32Array([
          0, 0,
          1, 0,
          0, 0,
          0, 1,
        ]),
      }),
    );

    expect(out.triLightCount).toBe(3);
    expect(out.totalEmissiveArea).toBeCloseTo(0.5, 6);
    expect(out.totalEmissivePower).toBeGreaterThan(0);
    expect(out.data![4]).toBeCloseTo(2, 6);
    expect(out.data![5]).toBeCloseTo(0, 6);
    expect(out.data![6]).toBeCloseTo(0, 6);
    const radianceRecords = Array.from({ length: out.triLightCount }, (_, i) => {
      const base = i * TRI_LIGHT_PIXELS * 4;
      return [out.data![base + 4], out.data![base + 5], out.data![base + 6]].join(',');
    });
    expect(radianceRecords).toContain('0,2,0');
  });

  it('subdivides explicit mesh-area triangle lights through the referenced material emissiveMap', () => {
    const emissiveMap = {
      handle: {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
        ]),
      },
    };
    const scene: Scene = {
      primitives: [panelPrimitive(material({
        emissiveMap,
      }))],
      emitters: [{
        kind: 'mesh-area',
        id: 'mapped-explicit',
        meshId: 'panel',
        color: [2, 2, 2],
        intensity: 1,
      }],
      environment: { kind: 'none' },
    };
    const out = packMeshAreaLights(
      scene,
      fakeMerged({ uvs: new Float32Array([0.75, 0, 0.75, 0, 0.75, 0, 0.75, 0]) }),
    );

    expect(out.triLightCount).toBe(8);
    expect(out.totalEmissiveArea).toBeCloseTo(1, 6);
    expect(out.data![4]).toBeCloseTo(0, 6);
    expect(out.data![5]).toBeCloseTo(2, 6);
    expect(out.data![6]).toBeCloseTo(0, 6);
    expect(out.warnings).toEqual([]);
  });

  it('suppresses implicit mesh-light synthesis when a readable emissiveMap is black', () => {
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [10, 10, 10],
        emissiveIntensity: 5,
        emissiveMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Uint8Array([0, 0, 0, 255]),
          },
        },
      }))),
      fakeMerged(),
    );

    expect(out.data).toBeNull();
    expect(out.triLightCount).toBe(0);
    expect(out.totalEmissiveArea).toBe(0);
    expect(hasMeshAreaLightForPrimitive(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [10, 10, 10],
        emissiveIntensity: 5,
        emissiveMap: {
          handle: {
            width: 1,
            height: 1,
            data: new Uint8Array([0, 0, 0, 255]),
          },
        },
      }))),
      'panel',
    )).toBe(false);
  });

  it('warns and falls back to scalar emissive radiance for unreadable emissiveMap handles', () => {
    const out = packMeshAreaLights(
      sceneWithPrimitive(panelPrimitive(material({
        emissive: [0.5, 0.25, 0.125],
        emissiveIntensity: 8,
        emissiveMap: { handle: { id: 'gpu-only-texture' } },
      }))),
      fakeMerged(),
    );

    expect(out.triLightCount).toBe(2);
    expect(out.data![4]).toBeCloseTo(4, 6);
    expect(out.data![5]).toBeCloseTo(2, 6);
    expect(out.data![6]).toBeCloseTo(1, 6);
    expect(out.warnings).toEqual([
      '@vitrum/pt-webgl2: primitive "panel" has an emissiveMap without CPU-readable texels; ' +
        'implicit mesh-area NEE uses scalar emissive radiance only.',
    ]);
  });
});
