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
  packBVHEmissiveLe,
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
  // Camera-visible emitters: per-triangle HDR emissive Le, indexed by the SAME
  // `geo` triangle order as beerBuf/bvhIndex (NOT the sharedWorld emitter-list
  // build) so a primary-hit triangle index addresses the right texel in shade.
  const emissiveLeBuf = packBVHEmissiveLe(geo.triMaterialIds, materials, triCount);

  // This SECOND build is load-bearing, NOT a redundant duplicate of `geo`.
  // `buildEmitterList` needs WORLD-space geometry: it derives triangle area,
  // face normal (world-space sun-dot in classifyTriangleEmitter), centroids and
  // AABBs, and appends world-space RectAreaLight tris. `geo` (packSceneFromCore)
  // stores per-primitive BLAS positions in LOCAL/object space — world transforms
  // live separately in the TLAS instance matrices — so feeding `geo` would place
  // every emitter at the wrong world location (changing the CDF, light tree, RNG
  // stratification, and image) for any transformed mesh. `buildSceneBVH` bakes
  // each mesh's matrixWorld into the vertices via StaticGeometryGenerator's
  // applyWorldTransforms. The two builds also produce different triangle
  // orderings (per-primitive BLAS-concat SAH vs one unified merged SAH). See
  // __tests__/emitterListWorldSpace.test.ts for the pinning test.
  // InstancedMesh extends Mesh; exclude it — its geometry lives in
  // packSceneFromCore's TLAS (one local BLAS + N instance matrices), which the
  // emitter list does not consume per-instance.
  const sharedWorld = buildSharedBVH(sceneRoots as THREE.Object3D[], {
    positionStride: 4,
    proxyMeshNames: options.proxyMeshNames ?? new Set<string>(),
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
    bvhEmissiveLe: makeStorageHandle(emissiveLeBuf, 16),
    // WS1 — per-vertex normals (stride-4). In TLAS mode these are the LOCAL-
    // space BLAS normals (geo.normals), indexed by the BLAS-local hit.indices.
    // The smooth-normal blend is DEFERRED for TLAS (the shaders gate on
    // ubo.bvhMode and keep the geometric normal) because the per-instance
    // world transform is not carried out of traceTlasFirstHit — a local-space
    // smooth normal would be wrong for any transformed instance. The buffer is
    // still bound (the layout requires it); it is simply not consumed in TLAS.
    bvhNormals: makeStorageHandle(geo.normals, 16),
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
