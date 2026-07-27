import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import type { Scene, Vec3 } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { packSceneFromCore, rebuildPrimitiveBlas, type PrimitiveTlasBinding } from '../scenePack.js';
import { tlasIntersect, type TlasBufferView } from '../tlas.js';

/**
 * BYTE-IDENTITY behaviour pin for `packSceneFromCore` + the BLAS-splice paths.
 *
 * `scenePack.golden.json` was captured from the PRE-refactor implementation
 * (the inline `packSceneFromCore` loop, the resize splice, and the same-size
 * splice). Theme-F dedup makes the loop delegate to `packOneMeshLikePrimitive`
 * and factors out the splice helpers — every packed buffer (positions, normals,
 * indices, bvhNodes, triMaterialIds, all TLAS buffers, bindings) must stay
 * byte-for-byte identical. If this test fails, the refactor changed output.
 */
const here = dirname(fileURLToPath(import.meta.url));

interface SerializedScenePack {
  readonly positions: readonly number[];
  readonly normals: readonly number[];
  readonly indices: readonly number[];
  readonly triMaterialIds: readonly number[];
  readonly bvhNodes: readonly number[];
  readonly triangleCount: number;
  readonly tlasNodes: readonly number[];
  readonly tlasInstanceIndices: readonly number[];
  readonly tlasBlasRoots: readonly number[];
  readonly tlasInstanceWorldToLocal: readonly number[];
  readonly tlasInstanceLocalToWorld: readonly number[];
  readonly tlasNodeCount: number;
  readonly primitiveTlasBindings: readonly PrimitiveTlasBinding[];
  readonly warnings: readonly string[];
}

const golden = JSON.parse(readFileSync(join(here, 'scenePack.golden.json'), 'utf8')) as {
  readonly multiMesh: SerializedScenePack;
  readonly resizeSplice: SerializedScenePack;
  readonly sameSizeSplice: SerializedScenePack;
};

function boxMesh(id: string, min: Vec3, max: Vec3, transform?: number[]): Scene['primitives'][number] {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([
      x0, y0, z0, x1, y0, z0, x0, y1, z0, x0, y0, z1,
      x1, y1, z1, x1, y0, z1, x0, y1, z1, x1, y1, z1,
    ]),
    normals: new Float32Array(24).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
    indices: new Uint32Array([
      0, 1, 2, 4, 1, 2, 1, 5, 6, 5, 4, 6, 0, 2, 3, 2, 6, 7,
      0, 1, 3, 1, 5, 3, 3, 5, 7, 5, 6, 7, 0, 4, 3, 4, 6, 7,
    ]),
    material: { baseColor: [0.6, 0.6, 0.6], roughness: 0.5, metallic: 0 },
    ...(transform != null ? { transform: asMat4(new Float32Array(transform)) } : {}),
  };
}

function unitTriMesh(id: string): Scene['primitives'][number] {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.4, metallic: 0 },
  };
}

function serializePack(p: ReturnType<typeof packSceneFromCore>) {
  return {
    positions: Array.from(p.positions),
    normals: Array.from(p.normals),
    indices: Array.from(p.indices),
    triMaterialIds: Array.from(p.triMaterialIds),
    bvhNodes: Array.from(new Uint32Array(p.bvhNodes.buffer, p.bvhNodes.byteOffset, p.bvhNodes.length)),
    triangleCount: p.triangleCount,
    tlasNodes: Array.from(p.tlasNodes),
    tlasInstanceIndices: Array.from(p.tlasInstanceIndices),
    tlasBlasRoots: Array.from(p.tlasBlasRoots),
    tlasInstanceWorldToLocal: Array.from(p.tlasInstanceWorldToLocal),
    tlasInstanceLocalToWorld: Array.from(p.tlasInstanceLocalToWorld),
    tlasNodeCount: p.tlasNodeCount,
    primitiveTlasBindings: p.primitiveTlasBindings,
    warnings: p.warnings,
  };
}

function expectCloseVec3(actual: readonly number[], expected: readonly number[], label: string): void {
  expect(actual.length, `${label} length`).toBe(3);
  for (let i = 0; i < 3; i += 1) {
    expect(actual[i], `${label}[${i}]`).toBeCloseTo(expected[i]!, 5);
  }
}

function localAabbFromVertices(pack: SerializedScenePack, binding: PrimitiveTlasBinding): {
  min: [number, number, number];
  max: [number, number, number];
} {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let vi = binding.vertexStart; vi < binding.vertexStart + binding.vertexCount; vi += 1) {
    const base = vi * 4;
    for (let axis = 0; axis < 3; axis += 1) {
      const value = pack.positions[base + axis]!;
      min[axis] = Math.min(min[axis]!, value);
      max[axis] = Math.max(max[axis]!, value);
    }
  }
  return { min, max };
}

function multiplyColumnMajor4(a: readonly number[], b: readonly number[]): number[] {
  const out = new Array<number>(16).fill(0);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += a[k * 4 + row]! * b[col * 4 + k]!;
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

function expectIdentityMatrix(m: readonly number[], label: string): void {
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      const expected = row === col ? 1 : 0;
      expect(m[col * 4 + row], `${label}[${row},${col}]`).toBeCloseTo(expected, 5);
    }
  }
}

function expectSerializedPackInvariants(pack: SerializedScenePack): void {
  expect(pack.positions.length % 4).toBe(0);
  expect(pack.normals.length).toBe(pack.positions.length);
  expect(pack.indices.length).toBe(pack.triangleCount * 4);
  expect(pack.triMaterialIds.length).toBe(pack.triangleCount);
  expect(pack.bvhNodes.length % 8).toBe(0);
  expect(pack.tlasNodes.length).toBe(pack.tlasNodeCount * 8);

  const expectedInstanceCount = pack.primitiveTlasBindings.reduce(
    (sum, binding) => sum + binding.instanceCount,
    0,
  );
  expect(pack.tlasBlasRoots.length).toBe(expectedInstanceCount);
  expect(pack.tlasInstanceIndices.length).toBe(expectedInstanceCount);
  expect(pack.tlasInstanceWorldToLocal.length).toBe(expectedInstanceCount * 16);
  expect(pack.tlasInstanceLocalToWorld.length).toBe(expectedInstanceCount * 16);

  let instanceOffset = 0;
  for (const binding of pack.primitiveTlasBindings) {
    const aabb = localAabbFromVertices(pack, binding);
    expectCloseVec3(binding.localAabbMin, aabb.min, `${binding.primitiveId}.localAabbMin`);
    expectCloseVec3(binding.localAabbMax, aabb.max, `${binding.primitiveId}.localAabbMax`);

    for (let tri = binding.triStart; tri < binding.triStart + binding.triCount; tri += 1) {
      const triBase = tri * 4;
      for (let lane = 0; lane < 3; lane += 1) {
        const index = pack.indices[triBase + lane]!;
        expect(index, `${binding.primitiveId}.tri[${tri}].idx[${lane}] lower`).toBeGreaterThanOrEqual(binding.vertexStart);
        expect(index, `${binding.primitiveId}.tri[${tri}].idx[${lane}] upper`).toBeLessThan(binding.vertexStart + binding.vertexCount);
      }
    }

    for (let i = 0; i < binding.instanceCount; i += 1) {
      expect(pack.tlasBlasRoots[instanceOffset + i]).toBe(binding.blasRoot);
    }
    instanceOffset += binding.instanceCount;
  }

  for (let i = 0; i < expectedInstanceCount; i += 1) {
    const w2l = pack.tlasInstanceWorldToLocal.slice(i * 16, i * 16 + 16);
    const l2w = pack.tlasInstanceLocalToWorld.slice(i * 16, i * 16 + 16);
    expectIdentityMatrix(multiplyColumnMajor4(l2w, w2l), `instance ${i} localToWorld*worldToLocal`);
    expectIdentityMatrix(multiplyColumnMajor4(w2l, l2w), `instance ${i} worldToLocal*localToWorld`);
  }
}

function tlasDataFromSerialized(pack: SerializedScenePack): TlasBufferView {
  return {
    nodes: Uint32Array.from(pack.tlasNodes),
    nodeCount: pack.tlasNodeCount,
    instanceIndices: Uint32Array.from(pack.tlasInstanceIndices),
    blasRoots: Uint32Array.from(pack.tlasBlasRoots),
    instanceTransforms: Float32Array.from(pack.tlasInstanceWorldToLocal),
  };
}

function expectTlasHits(
  pack: SerializedScenePack,
  rays: ReadonlyArray<{
    readonly origin: readonly [number, number, number];
    readonly direction: readonly [number, number, number];
    readonly tMax: number;
    readonly expectedInstance: number;
  }>,
): void {
  const tlas = tlasDataFromSerialized(pack);
  for (const ray of rays) {
    const hits = tlasIntersect(tlas, ray.origin, ray.direction, ray.tMax);
    expect(
      hits,
      `ray ${ray.origin.join(',')} -> ${ray.direction.join(',')} should hit instance ${ray.expectedInstance}`,
    ).toContain(ray.expectedInstance);
  }
}

const opts = { tlas: true, resolveMaterialId: (id: string) => id.charCodeAt(0) % 7 };

function multiMeshScene(): Scene {
  return {
    primitives: [
      boxMesh('box-a', [0, 0, 0], [1, 1, 1]),
      boxMesh('box-b', [0, 0, 0], [1, 1, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]),
      {
        kind: 'instanced-mesh',
        id: 'inst',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        instances: [
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1])),
          asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 7, 0, 0, 1])),
        ],
      },
      boxMesh('box-c', [10, 0, 0], [11, 1, 1]),
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('scenePack byte-identity golden', () => {
  it('packSceneFromCore on a multi-mesh + TLAS scene matches the golden buffers', () => {
    const pack = packSceneFromCore(multiMeshScene(), opts);
    expect(serializePack(pack)).toEqual(golden.multiMesh);
  });

  it('rebuildPrimitiveBlas resize-splice matches the golden buffers', () => {
    const base = packSceneFromCore(
      { primitives: [unitTriMesh('shape-a'), boxMesh('box-b', [5, 0, 0], [6, 1, 1])], emitters: [], environment: { kind: 'none' } },
      opts,
    );
    const next: Scene = {
      primitives: [boxMesh('shape-a', [0, 0, 0], [1, 1, 1]), boxMesh('box-b', [5, 0, 0], [6, 1, 1])],
      emitters: [],
      environment: { kind: 'none' },
    };
    const resized = rebuildPrimitiveBlas(next, 'shape-a', base, opts);
    expect(resized.ok).toBe(true);
    if (!resized.ok) return;
    expect(serializePack(resized.pack)).toEqual(golden.resizeSplice);
  });

  it('rebuildPrimitiveBlas same-size splice matches the golden buffers', () => {
    const scene = multiMeshScene();
    const base = packSceneFromCore(scene, opts);
    const next: Scene = {
      ...scene,
      primitives: [
        scene.primitives[0]!,
        boxMesh('box-b', [0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]),
        scene.primitives[2]!,
        scene.primitives[3]!,
      ],
    };
    const sameSize = rebuildPrimitiveBlas(next, 'box-b', base, opts);
    expect(sameSize.ok).toBe(true);
    if (!sameSize.ok) return;
    expect(serializePack(sameSize.pack)).toEqual(golden.sameSizeSplice);
  });
});

describe('scenePack golden independent structural oracle (H55)', () => {
  it('multiMesh golden has self-consistent bindings, matrices, and TLAS routes', () => {
    expectSerializedPackInvariants(golden.multiMesh);
    expectTlasHits(golden.multiMesh, [
      { origin: [-1, 0.5, 0.5], direction: [1, 0, 0], tMax: 2, expectedInstance: 0 },
      { origin: [2, 0.5, 0.5], direction: [1, 0, 0], tMax: 2.5, expectedInstance: 1 },
      { origin: [5.2, 0.2, 1], direction: [0, 0, -1], tMax: 2, expectedInstance: 2 },
      { origin: [7.2, 0.2, 1], direction: [0, 0, -1], tMax: 2, expectedInstance: 3 },
      { origin: [9.5, 0.5, 0.5], direction: [1, 0, 0], tMax: 2, expectedInstance: 4 },
    ]);
  });

  it('resize-splice golden has self-consistent rebased geometry and TLAS routes', () => {
    expectSerializedPackInvariants(golden.resizeSplice);
    expectTlasHits(golden.resizeSplice, [
      { origin: [0.5, 0.5, 5], direction: [0, 0, -1], tMax: 10, expectedInstance: 0 },
      { origin: [5.5, 0.5, 5], direction: [0, 0, -1], tMax: 10, expectedInstance: 1 },
    ]);
  });

  it('same-size-splice golden has self-consistent updated AABBs and TLAS routes', () => {
    expectSerializedPackInvariants(golden.sameSizeSplice);
    expectTlasHits(golden.sameSizeSplice, [
      { origin: [-1, 0.5, 0.5], direction: [1, 0, 0], tMax: 2, expectedInstance: 0 },
      { origin: [2.5, 0.5, 0.5], direction: [1, 0, 0], tMax: 2, expectedInstance: 1 },
      { origin: [5.2, 0.2, 1], direction: [0, 0, -1], tMax: 2, expectedInstance: 2 },
      { origin: [7.2, 0.2, 1], direction: [0, 0, -1], tMax: 2, expectedInstance: 3 },
      { origin: [9.5, 0.5, 0.5], direction: [1, 0, 0], tMax: 2, expectedInstance: 4 },
    ]);
  });
});
