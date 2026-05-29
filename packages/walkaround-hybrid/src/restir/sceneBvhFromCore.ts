/**
 * ReSTIR BVH build via `@vitrum/shared-bvh` `packSceneFromCore` (per-primitive BLAS + TLAS).
 */

import type { Scene } from '@vitrum/core';
import {
  buildSceneBVH as buildSharedBVH,
  packSceneFromCore,
  rebuildPrimitiveBlas,
  type ScenePackResult,
} from '@vitrum/shared-bvh';
import * as THREE from 'three';
import {
  packUVIntoPositionW,
  packBVHIndexW,
  packBVHBeerColors,
} from './packingHelpers.js';
import { buildEmitterList, buildLightTreeBuffer } from './emitterList.js';
import type { SceneBVHBuffers } from './bvhCompute.js';
import {
  collectRectAreaLightEmitterTris,
  enrichMeshVertexRangesWithMatrix,
} from './bvhSceneHelpers.js';

export type ReSTIRBvhMode = 'merged' | 'tlas';

export function resolveReSTIRBvhMode(
  scene: Scene,
  override?: ReSTIRBvhMode,
): ReSTIRBvhMode {
  if (override != null) return override;
  const meshLike = scene.primitives.filter(
    (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
  );
  if (meshLike.some((p) => p.kind === 'instanced-mesh')) return 'tlas';
  if (meshLike.length > 1) return 'tlas';
  return 'merged';
}

function buildMaterialResolver(sceneRoots: readonly THREE.Object3D[]): {
  materials: THREE.Material[];
  resolveMaterialId: (primitiveId: string) => number;
} {
  const materials: THREE.Material[] = [];
  const byKey = new Map<string, number>();
  const registerMaterial = (obj: THREE.Mesh | THREE.InstancedMesh): void => {
    const raw = obj.material;
    const mat = (Array.isArray(raw) ? raw[0] : raw) as THREE.Material | undefined;
    if (mat == null) return;
    let idx = materials.indexOf(mat);
    if (idx < 0) {
      idx = materials.length;
      materials.push(mat);
    }
    const keys = [obj.uuid, obj.name].filter((k) => k.length > 0);
    for (const key of keys) {
      if (!byKey.has(key)) byKey.set(key, idx);
    }
  };
  for (const root of sceneRoots) {
    root.traverseVisible((obj) => {
      if (obj instanceof THREE.InstancedMesh) {
        registerMaterial(obj);
        return;
      }
      if (!(obj instanceof THREE.Mesh)) return;
      registerMaterial(obj);
    });
  }
  return {
    materials,
    resolveMaterialId: (id) => byKey.get(id) ?? 0,
  };
}

function makeStorageHandle(
  data: ArrayBufferView,
  elementBytes: number,
): { cpuData: ArrayBuffer; byteLength: number; count: number } {
  return {
    cpuData: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    byteLength: data.byteLength,
    count: Math.floor(data.byteLength / elementBytes),
  };
}

function buffersFromScenePack(
  scene: Scene,
  sceneRoots: readonly THREE.Object3D[],
  geo: ScenePackResult,
  materials: THREE.Material[],
  options: {
    primaryLightDir?: THREE.Vector3;
    primaryLightIntensity?: number;
    proxyMeshNames?: Set<string>;
  },
): SceneBVHBuffers {
  const triCount = geo.triangleCount;
  const vertCount = geo.positions.length / 4;

  const positionsWithUV = packUVIntoPositionW(geo.positions, undefined, vertCount);
  const indexBuf = packBVHIndexW(geo.indices, geo.triMaterialIds, materials, triCount);
  const beerBuf = packBVHBeerColors(geo.triMaterialIds, materials, triCount);

  const sharedWorld = buildSharedBVH(sceneRoots as THREE.Object3D[], {
    positionStride: 4,
    proxyMeshNames: options.proxyMeshNames ?? new Set<string>(),
    // InstancedMesh extends Mesh; exclude it — geometry lives in packSceneFromCore TLAS.
    filter: (obj: THREE.Object3D) =>
      obj instanceof THREE.Mesh && (obj as THREE.InstancedMesh).isInstancedMesh !== true,
  });
  const extraEmitters = collectRectAreaLightEmitterTris(sceneRoots as THREE.Object3D[]);
  const { emitterFloats, cdfArray, totalEmissivePower, treeInput } = buildEmitterList(
    sharedWorld.indices,
    sharedWorld.positions,
    sharedWorld.normals,
    sharedWorld.triMaterialId,
    sharedWorld.materials,
    { ...options, extraEmitters },
  );
  const emitterCount = cdfArray.length;
  const lightTreeBuf = buildLightTreeBuffer(treeInput);

  const meshVertexRanges = enrichMeshVertexRangesWithMatrix(
    sceneRoots as THREE.Object3D[],
    geo.primitiveTlasBindings.map((b) => ({
      name: b.primitiveId,
      vertexStart: b.vertexStart,
      vertexCount: b.vertexCount,
      triStart: b.triStart,
      triCount: b.triCount,
    })),
  );

  const bvhIndicesStride3 = new Uint32Array(triCount * 3);
  for (let t = 0; t < triCount; t += 1) {
    bvhIndicesStride3[t * 3 + 0] = geo.indices[t * 4 + 0]!;
    bvhIndicesStride3[t * 3 + 1] = geo.indices[t * 4 + 1]!;
    bvhIndicesStride3[t * 3 + 2] = geo.indices[t * 4 + 2]!;
  }

  return {
    bvhMode: 'tlas',
    bvhNodes: makeStorageHandle(geo.bvhNodes, 32),
    bvhIndex: makeStorageHandle(indexBuf, 16),
    bvhPositions: makeStorageHandle(positionsWithUV, 16),
    triangleMaterialIds: makeStorageHandle(geo.triMaterialIds, 4),
    bvhBeerColors: makeStorageHandle(beerBuf, 4),
    emitters: {
      cpuData: emitterFloats.buffer as ArrayBuffer,
      byteLength: emitterFloats.byteLength,
      count: emitterCount,
    },
    emitterCdf: {
      cpuData: cdfArray.buffer as ArrayBuffer,
      byteLength: cdfArray.byteLength,
      count: emitterCount,
    },
    emitterCount,
    totalEmissivePower,
    lightTree: {
      cpuData: lightTreeBuf.nodes.buffer as ArrayBuffer,
      byteLength: lightTreeBuf.nodes.byteLength,
      count: Math.max(1, lightTreeBuf.nodeCount),
    },
    lightTreeNodeCount: lightTreeBuf.nodeCount,
    lightTreeEnabled: lightTreeBuf.enabled,
    mergedGeometry: sharedWorld.bvh.geometry,
    meshVertexRanges,
    bvhIndicesStride3,
    buildMaterials: materials,
    emitterNormals: geo.normals,
    tlas: {
      nodes: makeStorageHandle(geo.tlasNodes, 32),
      instanceIndices: makeStorageHandle(geo.tlasInstanceIndices, 4),
      blasRoots: makeStorageHandle(geo.tlasBlasRoots, 4),
      worldToLocal: makeStorageHandle(geo.tlasInstanceWorldToLocal, 64),
      localToWorld: makeStorageHandle(geo.tlasInstanceLocalToWorld, 64),
      nodeCount: geo.tlasNodeCount,
    },
    primitiveTlasBindings: geo.primitiveTlasBindings,
    scenePack: geo,
    warnings: geo.warnings,
  };
}

/** Build ReSTIR buffers from a vitrum scene (local BLAS concat + TLAS). */
export function buildReSTIRSceneBVHFromVitrumScene(
  scene: Scene,
  sceneRoots: readonly THREE.Object3D[],
  options: {
    primaryLightDir?: THREE.Vector3;
    primaryLightIntensity?: number;
    proxyMeshNames?: Set<string>;
  } = {},
): SceneBVHBuffers {
  const { materials, resolveMaterialId } = buildMaterialResolver(sceneRoots);
  const geo = packSceneFromCore(scene, { tlas: true, resolveMaterialId });
  return buffersFromScenePack(scene, sceneRoots, geo, materials, options);
}

/** PR-4.3 — topology rebuild via `rebuildPrimitiveBlas` (in-place splice or full repack). */
export function rebuildReSTIRSceneBVHPrimitive(
  scene: Scene,
  primitiveId: string,
  sceneRoots: readonly THREE.Object3D[],
  prev: SceneBVHBuffers,
  options: {
    primaryLightDir?: THREE.Vector3;
    primaryLightIntensity?: number;
    proxyMeshNames?: Set<string>;
  } = {},
): SceneBVHBuffers | { ok: false; reason: string } {
  if (prev.scenePack == null) {
    return { ok: false, reason: 'previous buffers have no scenePack snapshot' };
  }
  const { materials, resolveMaterialId } = buildMaterialResolver(sceneRoots);
  const rebuilt = rebuildPrimitiveBlas(scene, primitiveId, prev.scenePack, {
    tlas: true,
    resolveMaterialId,
  });
  if (!rebuilt.ok) {
    return { ok: false, reason: rebuilt.reason };
  }
  return buffersFromScenePack(scene, sceneRoots, rebuilt.pack, materials, options);
}
