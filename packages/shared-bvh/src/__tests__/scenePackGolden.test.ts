import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import type { Scene, Vec3 } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { packSceneFromCore, rebuildPrimitiveBlas } from '../scenePack.js';

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
const golden = JSON.parse(readFileSync(join(here, 'scenePack.golden.json'), 'utf8'));

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
