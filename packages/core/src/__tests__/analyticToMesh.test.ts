import { describe, expect, it } from 'vitest';
import { asMat4 } from '../scene/math.js';
import {
  analyticPrimitiveToMesh,
  type AnalyticPrimitiveToMeshOptions,
} from '../scene/analyticToMesh.js';
import type { AnalyticPrimitive, MeshPrimitive } from '../scene/primitives.js';

const MAT = { baseColor: [0.4, 0.5, 0.6] as [number, number, number], roughness: 0.45, metallic: 0.1 };

function analytic(
  shape: AnalyticPrimitive['shape'],
  params: readonly number[],
  extra: Partial<Omit<AnalyticPrimitive, 'kind' | 'id' | 'shape' | 'params' | 'material'>> = {},
): AnalyticPrimitive {
  return {
    kind: 'analytic',
    id: `${shape}-a`,
    shape,
    params: Float32Array.from(params),
    material: MAT,
    ...extra,
  };
}

function bounds(positions: Float32Array): { min: [number, number, number]; max: [number, number, number] } {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    const z = positions[i + 2]!;
    min[0] = Math.min(min[0], x);
    min[1] = Math.min(min[1], y);
    min[2] = Math.min(min[2], z);
    max[0] = Math.max(max[0], x);
    max[1] = Math.max(max[1], y);
    max[2] = Math.max(max[2], z);
  }
  return { min, max };
}

function expectBounds(mesh: MeshPrimitive, min: [number, number, number], max: [number, number, number]): void {
  const b = bounds(mesh.positions);
  for (let i = 0; i < 3; i++) {
    expect(b.min[i]).toBeCloseTo(min[i]!, 6);
    expect(b.max[i]).toBeCloseTo(max[i]!, 6);
  }
}

function expectUnitNormals(mesh: MeshPrimitive): void {
  expect(mesh.normals.length).toBe(mesh.positions.length);
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const x = mesh.normals[i]!;
    const y = mesh.normals[i + 1]!;
    const z = mesh.normals[i + 2]!;
    expect(Math.hypot(x, y, z)).toBeCloseTo(1, 5);
  }
}

/** Assert that a generated mesh has UVs present and all values in [0, 1]. */
function expectUVsInRange(mesh: MeshPrimitive): void {
  const vertexCount = mesh.positions.length / 3;
  expect(mesh.uvs).toBeDefined();
  expect(mesh.uvs!.length).toBe(vertexCount * 2);
  for (let i = 0; i < mesh.uvs!.length; i++) {
    const val = mesh.uvs![i]!;
    expect(val).toBeGreaterThanOrEqual(0);
    expect(val).toBeLessThanOrEqual(1);
  }
}

function expectNoIndexedUvWrap(mesh: MeshPrimitive): void {
  const uvs = mesh.uvs!;
  for (let triangle = 0; triangle < mesh.indices!.length; triangle += 3) {
    const u0 = uvs[mesh.indices![triangle]! * 2]!;
    const u1 = uvs[mesh.indices![triangle + 1]! * 2]!;
    const u2 = uvs[mesh.indices![triangle + 2]! * 2]!;
    expect(Math.max(u0, u1, u2) - Math.min(u0, u1, u2)).toBeLessThanOrEqual(0.5);
  }
}

function meshOutputBytes(mesh: MeshPrimitive): number {
  let bytes = mesh.positions.byteLength + mesh.normals.byteLength;
  if (mesh.uvs != null) bytes += mesh.uvs.byteLength;
  if (mesh.uv1 != null) bytes += mesh.uv1.byteLength;
  if (mesh.uvSets != null) {
    for (const stream of Object.values(mesh.uvSets)) {
      if (stream != null) bytes += stream.byteLength;
    }
  }
  if (mesh.tangents != null) bytes += mesh.tangents.byteLength;
  if (mesh.colors != null) bytes += mesh.colors.byteLength;
  if (mesh.colorSets != null) {
    for (const stream of Object.values(mesh.colorSets)) {
      if (stream != null) bytes += stream.byteLength;
    }
  }
  if (mesh.indices != null) bytes += mesh.indices.byteLength;
  return bytes;
}

function meshFor(shape: AnalyticPrimitive['shape'], params: readonly number[], options?: AnalyticPrimitiveToMeshOptions): MeshPrimitive {
  return analyticPrimitiveToMesh(analytic(shape, params), options);
}

describe('analyticPrimitiveToMesh', () => {
  it('preserves id, material, transform, castShadow, and hard-edged box extents', () => {
    const transform = asMat4([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      10, 20, 30, 1,
    ]);
    const prim = analytic('box', [1, 2, 3, 4, 5, 6], {
      transform,
      castShadow: false,
    });
    const mesh = analyticPrimitiveToMesh(prim);

    expect(mesh.kind).toBe('mesh');
    expect(mesh.id).toBe(prim.id);
    expect(mesh.material).toBe(prim.material);
    expect(mesh.transform).toBe(transform);
    expect(mesh.castShadow).toBe(false);
    expect(mesh.positions.length / 3).toBe(24);
    expect(mesh.indices).toHaveLength(36);
    expectBounds(mesh, [-3, -3, -3], [5, 7, 9]);
    expectUnitNormals(mesh);
    expectUVsInRange(mesh);
  });

  it('generates deterministic smooth sphere geometry from options', () => {
    const prim = analytic('sphere', [1, 2, 3, 2]);
    const a = analyticPrimitiveToMesh(prim, { segments: 8, rings: 4 });
    const b = analyticPrimitiveToMesh(prim, { segments: 8, rings: 4 });

    expect(a.positions.length / 3).toBe(29);
    expect(a.indices).toHaveLength(144);
    expect(Array.from(a.positions)).toEqual(Array.from(b.positions));
    expect(Array.from(a.normals)).toEqual(Array.from(b.normals));
    expect(Array.from(a.indices!)).toEqual(Array.from(b.indices!));
    expectBounds(a, [-1, 0, 1], [3, 4, 5]);
    expectUnitNormals(a);
    expectUVsInRange(a);
    expectNoIndexedUvWrap(a);
  });

  it('generates cylinder geometry along the local Y axis', () => {
    const mesh = meshFor('cylinder', [0, 1, 0, 2, 3], { segments: 8 });

    expect(mesh.positions.length / 3).toBe(36);
    expect(mesh.indices).toHaveLength(96);
    expectBounds(mesh, [-2, -2, -2], [2, 4, 2]);
    expectUnitNormals(mesh);
    expectUVsInRange(mesh);
    expectNoIndexedUvWrap(mesh);
  });

  it('generates capsule geometry between endpoint params', () => {
    const mesh = meshFor('capsule', [0, 0, 0, 0, 2, 0, 0.5], { segments: 8, rings: 4 });

    expect(mesh.positions.length / 3).toBe(38);
    expect(mesh.indices).toHaveLength(192);
    expectBounds(mesh, [-0.5, -0.5, -0.5], [0.5, 2.5, 0.5]);
    expectUnitNormals(mesh);
    expectUVsInRange(mesh);
    expectNoIndexedUvWrap(mesh);
  });

  it('generates an H-channel came extrusion matching the core param layout', () => {
    const mesh = meshFor('h-channel-came', [10, 2, 4, 0.5]);

    expect(mesh.positions.length / 3).toBe(72);
    expect(mesh.indices).toHaveLength(132);
    expectBounds(mesh, [-5, -2, -1], [5, 2, 1]);
    expectUnitNormals(mesh);
    expectUVsInRange(mesh);
  });

  it('preserves every positive authored f32 dimension below the former tessellation floor', () => {
    const tiny = 1e-9;
    const cases = [
      meshFor('sphere', [0, 0, 0, tiny], { segments: 8, rings: 4 }),
      meshFor('box', [0, 0, 0, tiny, tiny * 2, tiny * 3]),
      meshFor('cylinder', [0, 0, 0, tiny, tiny * 2], { segments: 8 }),
      meshFor('h-channel-came', [tiny * 4, tiny, tiny * 2, tiny * 0.4]),
    ];

    for (const mesh of cases) {
      const maxCoordinate = Math.max(...mesh.positions.map(Math.abs));
      expect(maxCoordinate).toBeGreaterThan(0);
      expect(maxCoordinate).toBeLessThan(1e-7);
      expectUnitNormals(mesh);
    }

    const shortCapsule = meshFor(
      'capsule',
      [0, 0, 0, 0, tiny * 10, 0, tiny],
      { segments: 8, rings: 4 },
    );
    expect(shortCapsule.positions.length / 3).toBe(38);
    expect(Math.max(...shortCapsule.positions)).toBeGreaterThan(tiny * 9);
    expect(Math.max(...shortCapsule.positions.map(Math.abs))).toBeLessThan(1e-7);
    expectUnitNormals(shortCapsule);
  });

  it('uses and clones a supplied fallbackMesh by default', () => {
    const fallbackPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const fallbackNormals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const fallbackIndices = new Uint16Array([0, 1, 2]);
    const prim = analytic('box', [0, 0, 0, 1, 1, 1], {
      fallbackMesh: {
        positions: fallbackPositions,
        normals: fallbackNormals,
        indices: fallbackIndices,
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        castShadow: false,
      },
    });

    const mesh = analyticPrimitiveToMesh(prim);

    expect(Array.from(mesh.positions)).toEqual(Array.from(fallbackPositions));
    expect(Array.from(mesh.normals)).toEqual(Array.from(fallbackNormals));
    expect(Array.from(mesh.indices ?? [])).toEqual([0, 1, 2]);
    expect(mesh.positions).not.toBe(fallbackPositions);
    expect(mesh.normals).not.toBe(fallbackNormals);
    expect(mesh.indices).not.toBe(fallbackIndices);
    expect(mesh.uvs).not.toBe(prim.fallbackMesh?.uvs);
    expect(mesh.castShadow).toBe(false);
  });

  it('gives primitive-level castShadow precedence over fallback metadata', () => {
    const prim = analytic('box', [0, 0, 0, 1, 1, 1], {
      castShadow: true,
      fallbackMesh: {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint16Array([0, 1, 2]),
        castShadow: false,
      },
    });

    const mesh = analyticPrimitiveToMesh(prim);

    expect(mesh.castShadow).toBe(true);
  });

  it('sparse-clones high-index fallback UV/color lanes without scanning array length', () => {
    const highIndex = 4_294_967_295;
    const uvSets = [] as Array<Float32Array | undefined>;
    const colorSets = [] as Array<Float32Array | undefined>;
    const highUvs = new Float32Array([0, 0, 1, 0, 0, 1]);
    const highColors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    uvSets[highIndex] = highUvs;
    colorSets[highIndex] = highColors;
    const prim = analytic('box', [0, 0, 0, 1, 1, 1], {
      fallbackMesh: {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvSets,
        colorSets,
      },
    });

    const mesh = analyticPrimitiveToMesh(prim);

    expect(Object.keys(mesh.uvSets ?? [])).toEqual([String(highIndex)]);
    expect(Object.keys(mesh.colorSets ?? [])).toEqual([String(highIndex)]);
    expect(mesh.uvSets?.[highIndex]).toEqual(highUvs);
    expect(mesh.colorSets?.[highIndex]).toEqual(highColors);
    expect(mesh.uvSets?.[highIndex]).not.toBe(highUvs);
    expect(mesh.colorSets?.[highIndex]).not.toBe(highColors);
  });

  it('can force regenerated analytic geometry over a supplied fallbackMesh', () => {
    const prim = analytic('box', [0, 0, 0, 1, 1, 1], {
      fallbackMesh: {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        indices: new Uint16Array([0, 1, 2]),
      },
    });

    const mesh = analyticPrimitiveToMesh(prim, { preferFallbackMesh: false });

    expect(mesh.positions.length / 3).toBe(24);
    expect(mesh.indices).toHaveLength(36);
    expectBounds(mesh, [-1, -1, -1], [1, 1, 1]);
  });

  it('enforces an output-allocation budget before generating geometry', () => {
    const prim = analytic('sphere', [0, 0, 0, 1]);
    const options = {
      segments: 8,
      rings: 4,
      maxOutputBytes: 1_503,
    } as AnalyticPrimitiveToMeshOptions;

    expect(() => analyticPrimitiveToMesh(prim, options)).toThrow(/1,504 bytes.*output allocation.*budget.*1,503 bytes/i);
  });

  it('preflights exact generated output sizes for every analytic shape formula', () => {
    const cases: ReadonlyArray<{
      shape: AnalyticPrimitive['shape'];
      params: readonly number[];
      options: AnalyticPrimitiveToMeshOptions;
    }> = [
      { shape: 'sphere', params: [0, 0, 0, 1], options: { segments: 3, rings: 2 } },
      { shape: 'sphere', params: [0, 0, 0, 1], options: { segments: 7, rings: 5 } },
      { shape: 'box', params: [0, 0, 0, 1, 1, 1], options: { segments: 91, rings: 73 } },
      { shape: 'capsule', params: [0, 0, 0, 0, 1, 0, 0.5], options: { segments: 3, rings: 2 } },
      { shape: 'capsule', params: [0, 0, 0, 0, 1, 0, 0.5], options: { segments: 7, rings: 5 } },
      { shape: 'capsule', params: [0, 0, 0, 0, 0, 0, 0.5], options: { segments: 3, rings: 2 } },
      { shape: 'capsule', params: [0, 0, 0, 0, 0, 0, 0.5], options: { segments: 7, rings: 5 } },
      { shape: 'cylinder', params: [0, 0, 0, 1, 1], options: { segments: 3, rings: 99 } },
      { shape: 'h-channel-came', params: [2, 1, 1, 0.25], options: { segments: 91, rings: 73 } },
    ];

    for (const { shape, params, options } of cases) {
      const mesh = meshFor(shape, params, options);
      const exactBytes = meshOutputBytes(mesh);
      expect(() => meshFor(shape, params, { ...options, maxOutputBytes: exactBytes }))
        .not.toThrow();
      expect(() => meshFor(shape, params, { ...options, maxOutputBytes: exactBytes - 1 }))
        .toThrow(/output allocation/);
    }
  });

  it('rejects huge segment/ring products before entering tessellation loops', () => {
    expect(() => meshFor(
      'sphere',
      [0, 0, 0, 1],
      { segments: Number.MAX_SAFE_INTEGER, rings: Number.MAX_SAFE_INTEGER },
    )).toThrow(/output allocation.*budget/);
    expect(() => meshFor(
      'cylinder',
      [0, 0, 0, 1, 1],
      { segments: Number.MAX_VALUE },
    )).toThrow(/output allocation.*budget/);
  });

  it('budgets every typed-array clone in an authored fallback mesh', () => {
    const positions = new Float32Array(9);
    const normals = new Float32Array(9);
    const uvs = new Float32Array(6);
    const uv1 = new Float32Array(6);
    const tangents = new Float32Array(12);
    const colors = new Float32Array(9);
    const color1 = new Float32Array(12);
    const indices = new Uint16Array([0, 1, 2]);
    const prim = analytic('box', [0, 0, 0, 1, 1, 1], {
      fallbackMesh: {
        positions,
        normals,
        uvs,
        uv1,
        uvSets: [uvs, uv1],
        tangents,
        colors,
        colorSets: [colors, color1],
        indices,
      },
    });
    const exactBytes = 342;

    expect(() => analyticPrimitiveToMesh(prim, { maxOutputBytes: exactBytes - 1 }))
      .toThrow(/342 bytes.*output allocation.*341 bytes/i);
    expect(meshOutputBytes(analyticPrimitiveToMesh(prim, { maxOutputBytes: exactBytes })))
      .toBe(exactBytes);
  });

  it('rejects invalid maxOutputBytes overrides', () => {
    for (const maxOutputBytes of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => meshFor('box', [0, 0, 0, 1, 1, 1], { maxOutputBytes }))
        .toThrow(/maxOutputBytes must be a non-negative safe integer/);
    }
  });

  it('rejects analytic params with the wrong shape length', () => {
    expect(() => analyticPrimitiveToMesh(analytic('sphere', [0, 0, 0]))).toThrow(/expects 4/);
  });
});
