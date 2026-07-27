import { describe, expect, it } from 'vitest';
import { ImportResourceLedger } from './importResourceBudget.js';
import { generateVertexNormals } from './normals.js';
import { generateTangents } from './tangents.js';
import {
  buildPointLineFallbackGeometry,
  GLTF_MODE_POINTS,
} from './primitiveModeFallback.js';
import {
  GLTF_MODE_TRIANGLE_STRIP,
  sequentialIndices,
  triangulateTopology,
} from './triangulation.js';
import {
  buildWorldTransforms,
  composeTrsMat4,
} from './transforms.js';
import { collectSceneCameras } from './cameraMetadata.js';
import type { GltfJson } from './gltfTypes.js';

describe('derived geometry resource accounting', () => {
  it('preflights generated normals and tangent working/output arrays', () => {
    const positions = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
    ]);
    const normalsLedger = new ImportResourceLedger({
      maxDecodedGeometryBytes: positions.byteLength - 1,
    });
    expect(() => generateVertexNormals(
      positions,
      undefined,
      normalsLedger,
      'normals',
    )).toThrow(/decoded-geometry-bytes/);
    expect(normalsLedger.decodedGeometryBytes).toBe(0);

    const normals = new Float32Array([
      0, 0, 1,
      0, 0, 1,
      0, 0, 1,
    ]);
    const uvs = new Float32Array([0, 0, 1, 0, 0, 1]);
    const tangentBytes = 3 * 10 * Float32Array.BYTES_PER_ELEMENT;
    const tangentLedger = new ImportResourceLedger({
      maxDecodedGeometryBytes: tangentBytes,
    });
    expect(generateTangents(
      positions,
      normals,
      uvs,
      undefined,
      tangentLedger,
      'tangents',
    )).toHaveLength(12);
    expect(tangentLedger.decodedGeometryBytes).toBe(tangentBytes);
  });

  it('accounts sequential and triangulated index allocations cumulatively', () => {
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 40,
    });
    const sequential = sequentialIndices(4, ledger, 'sequential');
    const triangles = triangulateTopology(
      sequential,
      GLTF_MODE_TRIANGLE_STRIP,
      ledger,
      'strip',
    );

    expect(Array.from(triangles)).toEqual([0, 1, 2, 2, 1, 3]);
    expect(ledger.decodedGeometryBytes).toBe(40);
  });

  it('preallocates point fallback outputs exactly before construction', () => {
    const positions = new Float32Array([0, 0, 0]);
    // One generated cube: 24 xyz positions + normals, 36 indices, and
    // 24 source-vertex entries = 816 bytes.
    const failingLedger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 815,
    });
    expect(() => buildPointLineFallbackGeometry(
      positions,
      undefined,
      GLTF_MODE_POINTS,
      undefined,
      failingLedger,
      'point fallback',
    )).toThrow(/decoded-geometry-bytes/);
    expect(failingLedger.decodedGeometryBytes).toBe(0);

    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 816,
    });
    const fallback = buildPointLineFallbackGeometry(
      positions,
      undefined,
      GLTF_MODE_POINTS,
      undefined,
      ledger,
      'point fallback',
    );
    expect(fallback?.positions).toHaveLength(72);
    expect(fallback?.normals).toHaveLength(72);
    expect(fallback?.indices).toHaveLength(36);
    expect(fallback?.sourceVertices).toHaveLength(24);
    expect(ledger.decodedGeometryBytes).toBe(816);
  });
});

describe('transform and camera matrix resource accounting', () => {
  it('charges the direct one-matrix TRS composition before allocation', () => {
    const failingLedger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 63,
    });
    expect(() => composeTrsMat4(
      [1, 2, 3],
      [0, 0, 0, 1],
      [2, 3, 4],
      failingLedger,
      'trs',
    )).toThrow(/decoded-geometry-bytes/);

    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 64,
    });
    const matrix = composeTrsMat4(
      [1, 2, 3],
      [0, 0, 0, 1],
      [2, 3, 4],
      ledger,
      'trs',
    );
    expect(Array.from(matrix)).toEqual([
      2, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 4, 0,
      1, 2, 3, 1,
    ]);
    expect(ledger.decodedGeometryBytes).toBe(64);
  });

  it('charges local/world matrices per reachable node and camera copies', () => {
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      scenes: [{ nodes: [0] }],
      nodes: [
        { children: [1] },
        { translation: [1, 0, 0], camera: 0 },
      ],
      cameras: [{
        type: 'perspective',
        perspective: { yfov: 1, znear: 0.1 },
      }],
    };
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 320,
    });
    const world = buildWorldTransforms(
      gltf,
      [0],
      ledger,
      'nodes',
    );
    expect(world.size).toBe(2);
    expect(ledger.decodedGeometryBytes).toBe(256);

    const cameras = collectSceneCameras(
      gltf,
      world,
      ledger,
      'cameras',
    );
    expect(cameras).toHaveLength(1);
    expect(cameras[0]?.worldMatrix[12]).toBe(1);
    expect(ledger.decodedGeometryBytes).toBe(320);
  });
});
