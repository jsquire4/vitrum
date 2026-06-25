import { describe, expect, it } from 'vitest';
import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  gltfToScene,
  type GltfJson,
} from './index.js';
import type { MeshPrimitive, SkinnedMeshPrimitive } from '@vitrum/core';

const POINT_LINE_MODES = [
  { mode: 0, name: 'POINTS' },
  { mode: 1, name: 'LINES' },
  { mode: 2, name: 'LINE_LOOP' },
  { mode: 3, name: 'LINE_STRIP' },
] as const;

const POSITIONS = [
  0, 0, 0,
  1, 0, 0,
  1, 1, 0,
  0, 1, 0,
];
const NORMALS = [
  0, 0, 1,
  0, 0, 1,
  0, 0, 1,
  0, 0, 1,
];

function f32Buffer(values: readonly number[]): ArrayBuffer {
  const arr = new Float32Array(values);
  return arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength);
}

function makePointLineBuffers(): Map<number, ArrayBuffer> {
  return new Map([
    [0, f32Buffer(POSITIONS)],
    [1, f32Buffer(NORMALS)],
  ]);
}

function makeAttributeRemapBuffers(): Map<number, ArrayBuffer> {
  return new Map([
    [0, f32Buffer([0, 0, 0, 1, 0, 0])],
    [1, f32Buffer([0, 0, 1, 0, 0, 1])],
    [2, f32Buffer([0.25, 0.5, 0.75, 1])],
    [3, f32Buffer([1, 0, 0, 1, 0, 1, 0, 1])],
    [4, f32Buffer([0.1, 0, 0, 0, 0.2, 0])],
  ]);
}

function makePointAttributeRemapGltf(mode = 0): GltfJson {
  const buffers = makeAttributeRemapBuffers();
  return {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      name: 'points-with-streams',
      primitives: [{
        attributes: {
          POSITION: 0,
          NORMAL: 1,
          TEXCOORD_0: 2,
          COLOR_0: 3,
        },
        mode,
        targets: [{ POSITION: 4 }],
      }],
    }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 2, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 2, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 2, type: 'VEC2' },
      { bufferView: 3, componentType: 5126, count: 2, type: 'VEC4' },
      { bufferView: 4, componentType: 5126, count: 2, type: 'VEC3' },
    ],
    bufferViews: [...buffers.entries()].map(([buffer, data]) => ({
      buffer,
      byteLength: data.byteLength,
    })),
    buffers: [...buffers.values()].map((data) => ({ byteLength: data.byteLength })),
  };
}

function makePolylineAttributeRemapBuffers(): Map<number, ArrayBuffer> {
  return new Map([
    [0, f32Buffer([0, 0, 0, 1, 0, 0, 1, 1, 0])],
    [1, f32Buffer([0, 0, 1, 0, 0, 1, 0, 0, 1])],
    [2, f32Buffer([0.125, 0.25, 0.5, 0.625, 0.75, 1])],
    [3, f32Buffer([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1])],
    [4, f32Buffer([0.125, 0, 0, 0, 0.25, 0, 0, 0, 0.5])],
  ]);
}

function makePolylineAttributeRemapGltf(mode: 2 | 3): GltfJson {
  const buffers = makePolylineAttributeRemapBuffers();
  return {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      name: 'polyline-with-streams',
      primitives: [{
        attributes: {
          POSITION: 0,
          NORMAL: 1,
          TEXCOORD_0: 2,
          COLOR_0: 3,
        },
        mode,
        targets: [{ POSITION: 4 }],
      }],
    }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 3, componentType: 5126, count: 3, type: 'VEC4' },
      { bufferView: 4, componentType: 5126, count: 3, type: 'VEC3' },
    ],
    bufferViews: [...buffers.entries()].map(([buffer, data]) => ({
      buffer,
      byteLength: data.byteLength,
    })),
    buffers: [...buffers.values()].map((data) => ({ byteLength: data.byteLength })),
  };
}

function makePointLineModeGltf(): GltfJson {
  const posBuf = f32Buffer(POSITIONS);
  const normBuf = f32Buffer(NORMALS);
  return {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{
      name: 'unsupported-topologies',
      primitives: POINT_LINE_MODES.map(({ mode }) => ({
        attributes: { POSITION: 0, NORMAL: 1 },
        mode,
      })),
    }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: POSITIONS.length / 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: NORMALS.length / 3, type: 'VEC3' },
    ],
    bufferViews: [
      { buffer: 0, byteLength: posBuf.byteLength },
      { buffer: 1, byteLength: normBuf.byteLength },
    ],
    buffers: [{ byteLength: posBuf.byteLength }, { byteLength: normBuf.byteLength }],
  };
}

describe('POINTS / line primitive policy', () => {
  it('reports each point/line topology as fallback-generated mesh compatibility', () => {
    const report = analyzeGltfAsset(makePointLineModeGltf());
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(report.primitives.byMode).toEqual({
      '0': 1,
      '1': 1,
      '2': 1,
      '3': 1,
    });
    expect(report.primitives.unsupportedModes).toEqual([]);
    expect(report.primitives.fallbackGeneratedModes).toEqual(['0', '1', '2', '3']);

    for (const { mode } of POINT_LINE_MODES) {
      expect(compatibility.issues).toContainEqual(expect.objectContaining({
        category: 'primitive',
        name: `mode:${mode}`,
        support: 'fallback-generated-mesh',
      }));
    }

    expect(compatibility.unsupportedCount).toBe(0);
    expect(compatibility.approximateCount).toBeGreaterThanOrEqual(POINT_LINE_MODES.length);
    expect(compatibility.isCompatible).toBe(true);
  });

  it('warns once per fallback topology, emits diagnostics, and imports generated meshes', async () => {
    const { scene, warnings, diagnostics } = await gltfToScene(makePointLineModeGltf(), {
      buffers: makePointLineBuffers(),
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(POINT_LINE_MODES.length);
    for (const [primitiveIndex, { mode, name }] of POINT_LINE_MODES.entries()) {
      expect(warnings.some((warning) =>
        warning.includes(`mode ${mode} (${name})`) &&
        warning.includes('fallback-generated mesh geometry'),
      )).toBe(true);
      expect(diagnostics).toContainEqual(expect.objectContaining({
        severity: 'warning',
        code: 'fallback-generated-primitive-mode',
        path: `meshes[0].primitives[${primitiveIndex}].mode`,
      }));
      const primitive = scene.primitives[primitiveIndex]! as MeshPrimitive;
      expect(primitive.kind).toBe('mesh');
      expect(primitive.positions.length).toBeGreaterThan(0);
      expect(primitive.normals.length).toBe(primitive.positions.length);
      expect(primitive.indices?.length).toBeGreaterThan(0);
    }
    expect(warnings).toHaveLength(POINT_LINE_MODES.length);
    expect(diagnostics).toHaveLength(POINT_LINE_MODES.length);
  });

  it('does not report discarded normal diagnostics for generated point/line fallback meshes', async () => {
    const withoutNormals = makePointLineModeGltf();
    for (const primitive of withoutNormals.meshes![0]!.primitives) {
      delete primitive.attributes.NORMAL;
    }

    const missingNormals = await gltfToScene(withoutNormals, {
      buffers: makePointLineBuffers(),
      pointLineFallbackRadius: 0.05,
    });
    const missingCodes = missingNormals.diagnostics.map((diagnostic) => diagnostic.code);
    expect(missingNormals.scene.primitives).toHaveLength(POINT_LINE_MODES.length);
    expect(missingCodes.filter((code) => code === 'fallback-generated-primitive-mode'))
      .toHaveLength(POINT_LINE_MODES.length);
    expect(missingCodes).not.toContain('generated-flat-normals');
    expect(missingCodes).not.toContain('unreadable-normal');
    expect(missingNormals.warnings).toHaveLength(POINT_LINE_MODES.length);
    expect(missingNormals.warnings.some((warning) =>
      warning.includes('Generating flat normals') || warning.includes('NORMAL unreadable'),
    )).toBe(false);

    const unreadableNormals = await gltfToScene(makePointLineModeGltf(), {
      buffers: new Map([[0, f32Buffer(POSITIONS)]]),
      pointLineFallbackRadius: 0.05,
    });
    const unreadableCodes = unreadableNormals.diagnostics.map((diagnostic) => diagnostic.code);
    expect(unreadableNormals.scene.primitives).toHaveLength(POINT_LINE_MODES.length);
    expect(unreadableCodes.filter((code) => code === 'fallback-generated-primitive-mode'))
      .toHaveLength(POINT_LINE_MODES.length);
    expect(unreadableCodes).not.toContain('generated-flat-normals');
    expect(unreadableCodes).not.toContain('unreadable-normal');
    expect(unreadableNormals.warnings).toHaveLength(POINT_LINE_MODES.length);
    expect(unreadableNormals.warnings.some((warning) =>
      warning.includes('Generating flat normals') || warning.includes('NORMAL unreadable'),
    )).toBe(false);
  });

  it('replicates UVs, vertex colors, identity skin, and morph deltas onto generated point meshes', async () => {
    const { scene } = await gltfToScene(makePointAttributeRemapGltf(), {
      buffers: makeAttributeRemapBuffers(),
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(1);
    const primitive = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(primitive.kind).toBe('skinned-mesh');
    expect(primitive.positions.length).toBe(2 * 24 * 3);
    expect(primitive.uvs?.length).toBe(2 * 24 * 2);
    expect(primitive.colors?.length).toBe(2 * 24 * 4);
    expect(primitive.skinWeights.length).toBe(2 * 24 * 4);
    expect(primitive.morphTargets?.[0]?.length).toBe(primitive.positions.length);

    expect(Array.from(primitive.uvs!.slice(0, 2))).toEqual([0.25, 0.5]);
    expect(Array.from(primitive.colors!.slice(0, 4))).toEqual([1, 0, 0, 1]);
    expect(primitive.morphTargets![0]![0]).toBeCloseTo(0.1);
    expect(primitive.morphTargets![0]![1]).toBeCloseTo(0);
    expect(primitive.morphTargets![0]![2]).toBeCloseTo(0);

    const secondPointGeneratedVertex = 24;
    expect(Array.from(primitive.uvs!.slice(secondPointGeneratedVertex * 2, secondPointGeneratedVertex * 2 + 2))).toEqual([0.75, 1]);
    expect(Array.from(primitive.colors!.slice(secondPointGeneratedVertex * 4, secondPointGeneratedVertex * 4 + 4))).toEqual([0, 1, 0, 1]);
    expect(primitive.morphTargets![0]![secondPointGeneratedVertex * 3 + 0]).toBeCloseTo(0);
    expect(primitive.morphTargets![0]![secondPointGeneratedVertex * 3 + 1]).toBeCloseTo(0.2);
    expect(primitive.morphTargets![0]![secondPointGeneratedVertex * 3 + 2]).toBeCloseTo(0);
  });

  it('replicates endpoint UVs, vertex colors, identity skin, and morph deltas onto generated line meshes', async () => {
    const { scene, diagnostics } = await gltfToScene(makePointAttributeRemapGltf(1), {
      buffers: makeAttributeRemapBuffers(),
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(1);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'fallback-generated-primitive-mode',
      path: 'meshes[0].primitives[0].mode',
    }));
    const primitive = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(primitive.kind).toBe('skinned-mesh');
    expect(primitive.positions.length).toBe(24 * 3);
    expect(primitive.uvs?.length).toBe(24 * 2);
    expect(primitive.colors?.length).toBe(24 * 4);
    expect(primitive.skinWeights.length).toBe(24 * 4);
    expect(primitive.morphTargets?.[0]?.length).toBe(primitive.positions.length);

    const firstEndpointVertices = [0, 3, 4, 7, 8, 11, 12, 15, 16, 17, 18, 19];
    const secondEndpointVertices = [1, 2, 5, 6, 9, 10, 13, 14, 20, 21, 22, 23];
    for (const vertex of firstEndpointVertices) {
      expect(Array.from(primitive.uvs!.slice(vertex * 2, vertex * 2 + 2))).toEqual([0.25, 0.5]);
      expect(Array.from(primitive.colors!.slice(vertex * 4, vertex * 4 + 4))).toEqual([1, 0, 0, 1]);
      expect(primitive.morphTargets![0]![vertex * 3 + 0]).toBeCloseTo(0.1);
      expect(primitive.morphTargets![0]![vertex * 3 + 1]).toBeCloseTo(0);
      expect(primitive.morphTargets![0]![vertex * 3 + 2]).toBeCloseTo(0);
    }
    for (const vertex of secondEndpointVertices) {
      expect(Array.from(primitive.uvs!.slice(vertex * 2, vertex * 2 + 2))).toEqual([0.75, 1]);
      expect(Array.from(primitive.colors!.slice(vertex * 4, vertex * 4 + 4))).toEqual([0, 1, 0, 1]);
      expect(primitive.morphTargets![0]![vertex * 3 + 0]).toBeCloseTo(0);
      expect(primitive.morphTargets![0]![vertex * 3 + 1]).toBeCloseTo(0.2);
      expect(primitive.morphTargets![0]![vertex * 3 + 2]).toBeCloseTo(0);
    }
  });

  it.each([
    { mode: 2 as const, name: 'LINE_LOOP', expectedCounts: [24, 24, 24] },
    { mode: 3 as const, name: 'LINE_STRIP', expectedCounts: [12, 24, 12] },
  ])('replicates endpoint streams across generated $name fallback meshes', async ({
    mode,
    expectedCounts,
  }) => {
    const buffers = makePolylineAttributeRemapBuffers();
    const { scene, diagnostics } = await gltfToScene(makePolylineAttributeRemapGltf(mode), {
      buffers,
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(1);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'fallback-generated-primitive-mode',
      path: 'meshes[0].primitives[0].mode',
    }));

    const primitive = scene.primitives[0] as SkinnedMeshPrimitive;
    expect(primitive.kind).toBe('skinned-mesh');
    expect(primitive.positions.length).toBe(expectedCounts.reduce((sum, count) => sum + count, 0) * 3);
    expect(primitive.uvs?.length).toBe(expectedCounts.reduce((sum, count) => sum + count, 0) * 2);
    expect(primitive.colors?.length).toBe(expectedCounts.reduce((sum, count) => sum + count, 0) * 4);
    expect(primitive.morphTargets?.[0]?.length).toBe(primitive.positions.length);

    const counts: [number, number, number] = [0, 0, 0];
    for (let vertex = 0; vertex < primitive.positions.length / 3; vertex += 1) {
      const color = Array.from(primitive.colors!.slice(vertex * 4, vertex * 4 + 4));
      const uv = Array.from(primitive.uvs!.slice(vertex * 2, vertex * 2 + 2));
      const morph = primitive.morphTargets![0]!.slice(vertex * 3, vertex * 3 + 3);

      if (color[0] === 1) {
        counts[0] += 1;
        expect(uv).toEqual([0.125, 0.25]);
        expect(Array.from(morph)).toEqual([0.125, 0, 0]);
      } else if (color[1] === 1) {
        counts[1] += 1;
        expect(uv).toEqual([0.5, 0.625]);
        expect(Array.from(morph)).toEqual([0, 0.25, 0]);
      } else {
        counts[2] += 1;
        expect(color).toEqual([0, 0, 1, 1]);
        expect(uv).toEqual([0.75, 1]);
        expect(Array.from(morph)).toEqual([0, 0, 0.5]);
      }
    }
    expect(counts).toEqual(expectedCounts);
  });
});
