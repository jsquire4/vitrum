import { describe, expect, it } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import {
  isRestirTlasOnlyRefit,
  makeRestirBvhSnapshot,
} from '../src/restir/restirBvhSnapshot.js';
import type { SceneBVHBuffers } from '../src/restir/bvhTypes.js';
import type { RestirMergedGeometryLike } from '../src/restir/bvhTypes.js';
import { packMaterialTextureAtlas } from '../src/pipeline/materialTextureAtlas.js';

function boundsGeometry(max = 2): RestirMergedGeometryLike {
  return {
    boundingBox: {
      min: { x: 0, y: 0, z: 0 },
      max: { x: max, y: max, z: max },
    },
    computeBoundingBox: () => undefined,
    dispose: () => undefined,
  };
}

function minimalSceneBVH(overrides: Partial<SceneBVHBuffers> = {}): SceneBVHBuffers {
  return {
    bvhMode: 'merged',
    bvhNodes: { cpuData: new ArrayBuffer(32), byteLength: 32, count: 1 },
    bvhIndex: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    bvhPositions: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    triangleMaterialIds: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhBeerColors: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhEmissiveLe: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    materialTextureAtlas: packMaterialTextureAtlas([], new Uint32Array([0]), 1),
    bvhRoughMetal: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhNormals: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    emitters: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 0 },
    emitterCdf: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 0 },
    emitterCount: 0,
    totalEmissivePower: 0,
    lightTree: { cpuData: new ArrayBuffer(48), byteLength: 48, count: 1 },
    lightTreeNodeCount: 0,
    lightTreeEnabled: false,
    mergedGeometry: boundsGeometry(),
    meshVertexRanges: [],
    bvhIndicesStride3: new Uint32Array(3),
    buildMaterials: [],
    coreMaterials: [],
    emitterNormals: new Float32Array(12),
    primitiveTlasBindings: [],
    ...overrides,
  };
}

describe('makeRestirBvhSnapshot (PR-5.1)', () => {
  it('uses merged geometry bounds when bvhMode is merged', () => {
    const snap = makeRestirBvhSnapshot(minimalSceneBVH());
    expect(snap.bvhMode).toBe('merged');
    expect(snap.tlasNodeCount).toBe(0);
    expect(snap.boundingBox.max.x).toBeCloseTo(2);
  });

  it('bumps contentVersion when TLAS nodes change', () => {
    const a = makeRestirBvhSnapshot(minimalSceneBVH({
      bvhMode: 'tlas',
      tlas: {
        nodes: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 2 },
        instanceIndices: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        blasRoots: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        worldToLocal: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        localToWorld: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        nodeCount: 2,
      },
    }));
    const b = makeRestirBvhSnapshot(minimalSceneBVH({
      bvhMode: 'tlas',
      tlas: {
        nodes: { cpuData: new ArrayBuffer(128), byteLength: 128, count: 4 },
        instanceIndices: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        blasRoots: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        worldToLocal: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        localToWorld: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        nodeCount: 4,
      },
    }));
    expect(a.contentVersion).not.toBe(b.contentVersion);
  });

  it('bumps contentVersion when TLAS node payload changes at fixed length', () => {
    const nodesA = new Uint32Array(16);
    nodesA[0] = 1;
    const nodesB = new Uint32Array(16);
    nodesB[0] = 99;
    const base = minimalSceneBVH({
      bvhMode: 'tlas',
      tlas: {
        nodes: { cpuData: nodesA.buffer, byteLength: nodesA.byteLength, count: 2 },
        instanceIndices: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        blasRoots: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        worldToLocal: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        localToWorld: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        nodeCount: 2,
      },
    });
    const snapA = makeRestirBvhSnapshot(base);
    const snapB = makeRestirBvhSnapshot(minimalSceneBVH({
      bvhMode: 'tlas',
      tlas: {
        nodes: { cpuData: nodesB.buffer, byteLength: nodesB.byteLength, count: 2 },
        instanceIndices: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        blasRoots: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
        worldToLocal: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        localToWorld: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
        nodeCount: 2,
      },
    }));
    expect(snapA.contentVersion).not.toBe(snapB.contentVersion);
  });

  it('splits blas vs tlas content versions on transform-only TLAS refit', () => {
    const w2lA = new Float32Array(16);
    w2lA[12] = 0;
    const w2lB = new Float32Array(16);
    w2lB[12] = 5;
    const tlasBase = {
      nodes: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 2 },
      instanceIndices: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
      blasRoots: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
      worldToLocal: { cpuData: w2lA.buffer, byteLength: w2lA.byteLength, count: 1 },
      localToWorld: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
      nodeCount: 2,
    };
    const snapA = makeRestirBvhSnapshot(minimalSceneBVH({ bvhMode: 'tlas', tlas: tlasBase }));
    const snapB = makeRestirBvhSnapshot(minimalSceneBVH({
      bvhMode: 'tlas',
      tlas: { ...tlasBase, worldToLocal: { cpuData: w2lB.buffer, byteLength: w2lB.byteLength, count: 1 } },
    }));
    expect(snapA.blasContentVersion).toBe(snapB.blasContentVersion);
    expect(snapA.tlasContentVersion).not.toBe(snapB.tlasContentVersion);
    expect(snapA.contentVersion).not.toBe(snapB.contentVersion);
    expect(
      isRestirTlasOnlyRefit(snapB, {
        blasContentVersion: snapA.blasContentVersion,
        tlasContentVersion: snapA.tlasContentVersion,
      }),
    ).toBe(true);
    expect(
      isRestirTlasOnlyRefit(snapB, {
        blasContentVersion: snapB.blasContentVersion,
        tlasContentVersion: snapB.tlasContentVersion,
      }),
    ).toBe(false);
  });

  it('uses world AABB for TLAS when scene is provided', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'mesh',
        id: 'box',
        positions: new Float32Array([
          0, 0, 0, 1, 0, 0, 0, 1, 0,
        ]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        transform: asMat4(new Float32Array([
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          10, 0, 0, 1,
        ])),
      }],
      emitters: [],
      environment: { kind: 'none' },
    };
    const snap = makeRestirBvhSnapshot(minimalSceneBVH({
      bvhMode: 'tlas',
      primitiveTlasBindings: [{
        primitiveId: 'box',
        primitiveKind: 'mesh',
        blasRoot: 0,
        instanceCount: 1,
        vertexStart: 0,
        vertexCount: 3,
        triStart: 0,
        triCount: 1,
        localAabbMin: [0, 0, 0],
        localAabbMax: [1, 1, 0],
      }],
    }), scene);
    expect(snap.boundingBox.min.x).toBeGreaterThanOrEqual(9);
  });
});
