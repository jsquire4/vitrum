import { describe, expect, it } from 'vitest';
import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  gltfToScene,
  type GltfJson,
} from './index.js';
import type { AnalyticPrimitive } from '@vitrum/core';

const POINT_LINE_MODES = [
  { mode: 0, name: 'POINTS', count: 4, shape: 'sphere' as const },
  { mode: 1, name: 'LINES', count: 2, shape: 'capsule' as const },
  { mode: 2, name: 'LINE_LOOP', count: 4, shape: 'capsule' as const },
  { mode: 3, name: 'LINE_STRIP', count: 3, shape: 'capsule' as const },
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
  it('reports point/line topologies as tessellated analytics on triangle-only backends', () => {
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

  it('reports point/line analytics as native on full pt-webgpu', () => {
    const report = analyzeGltfAsset(makePointLineModeGltf());
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgpu');

    for (const { mode } of POINT_LINE_MODES) {
      expect(compatibility.issues.filter((issue) => issue.name === `mode:${mode}`)).toEqual([]);
    }
  });

  it('warns once per topology and imports analytic spheres/capsules', async () => {
    const { scene, warnings, diagnostics } = await gltfToScene(makePointLineModeGltf(), {
      buffers: makePointLineBuffers(),
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(
      POINT_LINE_MODES.reduce((sum, mode) => sum + mode.count, 0),
    );
    let offset = 0;
    for (const [primitiveIndex, { mode, name, count, shape }] of POINT_LINE_MODES.entries()) {
      expect(warnings.some((warning) =>
        warning.includes(`mode ${mode} (${name})`) &&
        warning.includes('analytic sphere/capsule geometry'),
      )).toBe(true);
      expect(diagnostics).toContainEqual(expect.objectContaining({
        severity: 'warning',
        code: 'fallback-generated-primitive-mode',
        path: `meshes[0].primitives[${primitiveIndex}].mode`,
      }));
      const imported = scene.primitives.slice(offset, offset + count);
      expect(imported).toHaveLength(count);
      for (const primitive of imported) {
        const analytic = primitive as AnalyticPrimitive;
        expect(analytic.kind).toBe('analytic');
        expect(analytic.shape).toBe(shape);
        expect(analytic.params[analytic.params.length - 1]).toBeCloseTo(0.05);
      }
      offset += count;
    }
    expect(warnings).toHaveLength(POINT_LINE_MODES.length);
    expect(diagnostics).toHaveLength(POINT_LINE_MODES.length);
  });

  it('does not report discarded normal diagnostics for analytic point/line import', async () => {
    const withoutNormals = makePointLineModeGltf();
    for (const primitive of withoutNormals.meshes![0]!.primitives) {
      delete primitive.attributes.NORMAL;
    }

    const missingNormals = await gltfToScene(withoutNormals, {
      buffers: makePointLineBuffers(),
      pointLineFallbackRadius: 0.05,
    });
    const missingCodes = missingNormals.diagnostics.map((diagnostic) => diagnostic.code);
    expect(missingNormals.scene.primitives.length).toBeGreaterThan(0);
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
    expect(unreadableNormals.scene.primitives.length).toBeGreaterThan(0);
    expect(unreadableCodes.filter((code) => code === 'fallback-generated-primitive-mode'))
      .toHaveLength(POINT_LINE_MODES.length);
    expect(unreadableCodes).not.toContain('generated-flat-normals');
    expect(unreadableCodes).not.toContain('unreadable-normal');
    expect(unreadableNormals.warnings).toHaveLength(POINT_LINE_MODES.length);
    expect(unreadableNormals.warnings.some((warning) =>
      warning.includes('Generating flat normals') || warning.includes('NORMAL unreadable'),
    )).toBe(false);
  });

  it('imports POINTS as rest-pose spheres and drops vertex streams', async () => {
    const { scene, diagnostics } = await gltfToScene(makePointAttributeRemapGltf(), {
      buffers: makeAttributeRemapBuffers(),
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(2);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'fallback-generated-primitive-mode',
      path: 'meshes[0].primitives[0].mode',
    }));
    expect(diagnostics.some((diagnostic) =>
      diagnostic.message.includes('skin and morph streams are not applied'),
    )).toBe(true);
    const first = scene.primitives[0] as AnalyticPrimitive;
    const second = scene.primitives[1] as AnalyticPrimitive;
    expect(first.kind).toBe('analytic');
    expect(first.shape).toBe('sphere');
    expect(first.params[0]).toBeCloseTo(0);
    expect(first.params[1]).toBeCloseTo(0);
    expect(first.params[2]).toBeCloseTo(0);
    expect(first.params[3]).toBeCloseTo(0.05);
    expect(second.params[0]).toBeCloseTo(1);
    expect(second.params[1]).toBeCloseTo(0);
    expect(second.params[2]).toBeCloseTo(0);
    expect(second.params[3]).toBeCloseTo(0.05);
    expect('uvs' in first).toBe(false);
    expect('colors' in first).toBe(false);
    expect('morphTargets' in first).toBe(false);
  });

  it('imports LINES as a rest-pose capsule between endpoints', async () => {
    const { scene, diagnostics } = await gltfToScene(makePointAttributeRemapGltf(1), {
      buffers: makeAttributeRemapBuffers(),
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(1);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'fallback-generated-primitive-mode',
      path: 'meshes[0].primitives[0].mode',
    }));
    const primitive = scene.primitives[0] as AnalyticPrimitive;
    expect(primitive.kind).toBe('analytic');
    expect(primitive.shape).toBe('capsule');
    expect(primitive.params[0]).toBeCloseTo(0);
    expect(primitive.params[1]).toBeCloseTo(0);
    expect(primitive.params[2]).toBeCloseTo(0);
    expect(primitive.params[3]).toBeCloseTo(1);
    expect(primitive.params[4]).toBeCloseTo(0);
    expect(primitive.params[5]).toBeCloseTo(0);
    expect(primitive.params[6]).toBeCloseTo(0.05);
  });

  it.each([
    { mode: 2 as const, name: 'LINE_LOOP', expected: 3 },
    { mode: 3 as const, name: 'LINE_STRIP', expected: 2 },
  ])('imports $name as rest-pose capsules without vertex streams', async ({
    mode,
    expected,
  }) => {
    const buffers = makePolylineAttributeRemapBuffers();
    const { scene, diagnostics } = await gltfToScene(makePolylineAttributeRemapGltf(mode), {
      buffers,
      pointLineFallbackRadius: 0.05,
    });

    expect(scene.primitives).toHaveLength(expected);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'fallback-generated-primitive-mode',
      path: 'meshes[0].primitives[0].mode',
    }));
    for (const primitive of scene.primitives) {
      const analytic = primitive as AnalyticPrimitive;
      expect(analytic.kind).toBe('analytic');
      expect(analytic.shape).toBe('capsule');
      expect(analytic.params).toHaveLength(7);
      expect(analytic.params[6]).toBeCloseTo(0.05);
    }
  });

  it('honors extras.vitrum.radius over the host override', async () => {
    const gltf = makePointAttributeRemapGltf();
    gltf.meshes![0]!.primitives[0]!.extras = { vitrum: { radius: 0.4 } };

    const { scene } = await gltfToScene(gltf, {
      buffers: makeAttributeRemapBuffers(),
      pointLineFallbackRadius: 0.05,
    });

    const primitive = scene.primitives[0] as AnalyticPrimitive;
    expect(primitive.params[3]).toBeCloseTo(0.4);
  });
});
