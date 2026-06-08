/**
 * Core-scene ReSTIR BVH build path.
 *
 * This file is deliberately free of runtime `three` imports. The historical
 * raw-THREE scene graph path lives under `legacy/three`; the concrete
 * HybridEngine imports from here so core-scene rendering cannot pull a Three
 * dependency through a mixed module.
 */

import type { MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import {
  mergeWorldSpaceFromCore,
  packSceneFromCore,
  rebuildPrimitiveBlas,
  type ScenePackResult,
} from '@vitrum/shared-bvh';
import {
  packUVIntoPositionW,
  packBVHIndexWFromCore,
  packBVHBeerColorsFromCore,
  packBVHEmissiveLeFromCore,
} from './packingHelpers.js';
import { buildEmitterListFromCore, buildLightTreeBuffer } from './emitterList.js';
import {
  collectRectAreaEmitterTrisFromCore,
  enrichMeshVertexRangesWithCoreMatrix,
} from './bvhSceneHelpers.js';
import type {
  RebuiltEmitterBuffers,
  ReSTIRBvhMode,
  SceneBVHBuffers,
} from './bvhTypes.js';

export type { RebuiltEmitterBuffers, ReSTIRBvhMode, SceneBVHBuffers };

interface Vector3Like {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface CoreBvhBuildOptions {
  bvhMode?: ReSTIRBvhMode;
  primaryLightDir?: Vector3Like;
  primaryLightIntensity?: number;
  proxyMeshNames?: Set<string>;
}

function sceneHasCoreMeshes(scene: Scene): boolean {
  return scene.primitives.some(
    (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
  );
}

export function resolveReSTIRBvhMode(scene: Scene, override?: ReSTIRBvhMode): ReSTIRBvhMode {
  if (override != null) return override;
  const meshLike = scene.primitives.filter(
    (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
  );
  if (meshLike.some((p) => p.kind === 'instanced-mesh')) return 'tlas';
  if (meshLike.length > 1) return 'tlas';
  return 'merged';
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

function toProductionEmissiveRadiance(m: MaterialSpec): MaterialSpec {
  if (m.emissive === undefined) return m;
  if (m.emissiveIntensity === 1) return m;
  return { ...m, emissiveIntensity: 1 };
}

function materialResolver(scene: Scene): {
  coreMaterials: MaterialSpec[];
  resolveMaterialId: (primitiveId: string) => number;
} {
  const coreMaterials: MaterialSpec[] = [];
  const byKey = new Map<string, number>();
  for (const p of scene.primitives) {
    if (p.kind === 'mesh' || p.kind === 'instanced-mesh' || p.kind === 'skinned-mesh') {
      if (!byKey.has(String(p.id))) byKey.set(String(p.id), coreMaterials.length);
      coreMaterials.push(p.material);
    }
  }
  return {
    coreMaterials,
    resolveMaterialId: (id) => byKey.get(id) ?? 0,
  };
}

function makeMergedGeometry(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): SceneBVHBuffers['mergedGeometry'] {
  const boundingBox = {
    min: { x: min[0], y: min[1], z: min[2] },
    max: { x: max[0], y: max[1], z: max[2] },
  };
  return {
    boundingBox,
    computeBoundingBox() {},
    dispose() {},
  } as unknown as SceneBVHBuffers['mergedGeometry'];
}

function coreEmitterBuffers(
  scene: Scene,
  options: {
    primaryLightDir?: Vector3Like;
    primaryLightIntensity?: number;
  } = {},
): RebuiltEmitterBuffers {
  const merged = mergeWorldSpaceFromCore(scene, {
    positionStride: 4,
    filter: (p: ScenePrimitive) => p.kind !== 'instanced-mesh',
  });
  const extraEmitters = collectRectAreaEmitterTrisFromCore(scene);
  const { emitterFloats, cdfArray, totalEmissivePower, treeInput } = buildEmitterListFromCore(
    merged.indices,
    merged.positions,
    merged.normals,
    merged.triMaterialId,
    merged.materials.map(toProductionEmissiveRadiance),
    { ...options, extraEmitters },
  );
  const emitterCount = cdfArray.length;
  const lightTreeBuf = buildLightTreeBuffer(treeInput);
  return {
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
  };
}

function buffersFromCoreScenePack(
  scene: Scene,
  geo: ScenePackResult,
  coreMaterials: readonly MaterialSpec[],
  options: CoreBvhBuildOptions,
): SceneBVHBuffers {
  const triCount = geo.triangleCount;
  const vertCount = geo.positions.length / 4;

  const positionsWithUV = packUVIntoPositionW(geo.positions, undefined, vertCount);
  const triIndices3 = new Uint32Array(triCount * 3);
  for (let t = 0; t < triCount; t += 1) {
    triIndices3[t * 3 + 0] = geo.indices[t * 4 + 0]!;
    triIndices3[t * 3 + 1] = geo.indices[t * 4 + 1]!;
    triIndices3[t * 3 + 2] = geo.indices[t * 4 + 2]!;
  }

  const indexBuf = packBVHIndexWFromCore(triIndices3, geo.triMaterialIds, coreMaterials, triCount);
  const beerBuf = packBVHBeerColorsFromCore(geo.triMaterialIds, coreMaterials, triCount);
  const emissiveLeBuf = packBVHEmissiveLeFromCore(geo.triMaterialIds, coreMaterials, triCount);

  const emitterSlice = coreEmitterBuffers(scene, options);
  const merged = mergeWorldSpaceFromCore(scene, {
    positionStride: 4,
    filter: (p: ScenePrimitive) => p.kind !== 'instanced-mesh',
  });
  const rawMeshVertexRanges = geo.primitiveTlasBindings.map((b) => ({
    name: b.primitiveId,
    vertexStart: b.vertexStart,
    vertexCount: b.vertexCount,
    triStart: b.triStart,
    triCount: b.triCount,
  }));

  return {
    bvhMode: 'tlas',
    bvhNodes: makeStorageHandle(geo.bvhNodes, 32),
    bvhIndex: makeStorageHandle(indexBuf, 16),
    bvhPositions: makeStorageHandle(positionsWithUV, 16),
    triangleMaterialIds: makeStorageHandle(geo.triMaterialIds, 4),
    bvhBeerColors: makeStorageHandle(beerBuf, 4),
    bvhEmissiveLe: makeStorageHandle(emissiveLeBuf, 16),
    bvhNormals: makeStorageHandle(geo.normals, 16),
    emitters: emitterSlice.emitters,
    emitterCdf: emitterSlice.emitterCdf,
    emitterCount: emitterSlice.emitterCount,
    totalEmissivePower: emitterSlice.totalEmissivePower,
    lightTree: emitterSlice.lightTree,
    lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
    lightTreeEnabled: emitterSlice.lightTreeEnabled,
    mergedGeometry: makeMergedGeometry(merged.boundingBox.min, merged.boundingBox.max),
    meshVertexRanges: enrichMeshVertexRangesWithCoreMatrix(scene, rawMeshVertexRanges),
    bvhIndicesStride3: triIndices3,
    buildMaterials: [],
    coreMaterials,
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

function buildReSTIRSceneBVHFromCoreTlas(
  scene: Scene,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers {
  const { coreMaterials, resolveMaterialId } = materialResolver(scene);
  const geo = packSceneFromCore(scene, { tlas: true, resolveMaterialId });
  return buffersFromCoreScenePack(scene, geo, coreMaterials, options);
}

function buildReSTIRSceneBVHFromCoreMerged(
  scene: Scene,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers {
  const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
  const triCount = merged.indices.length / 3;
  const vertCount = merged.positions.length / 4;
  const positionsWithUV = packUVIntoPositionW(merged.positions, undefined, vertCount);
  const indexBuf = packBVHIndexWFromCore(
    merged.indices,
    merged.triMaterialId,
    merged.materials,
    triCount,
  );
  const beerBuf = packBVHBeerColorsFromCore(merged.triMaterialId, merged.materials, triCount);
  const emissiveLeBuf = packBVHEmissiveLeFromCore(merged.triMaterialId, merged.materials, triCount);
  const emitterSlice = coreEmitterBuffers(scene, options);

  return {
    bvhMode: 'merged',
    bvhNodes: makeStorageHandle(merged.bvhNodes, 32),
    bvhIndex: makeStorageHandle(indexBuf, 16),
    bvhPositions: makeStorageHandle(positionsWithUV, 16),
    triangleMaterialIds: makeStorageHandle(merged.triMaterialId, 4),
    bvhBeerColors: makeStorageHandle(beerBuf, 4),
    bvhEmissiveLe: makeStorageHandle(emissiveLeBuf, 16),
    bvhNormals: makeStorageHandle(merged.normals, 16),
    emitters: emitterSlice.emitters,
    emitterCdf: emitterSlice.emitterCdf,
    emitterCount: emitterSlice.emitterCount,
    totalEmissivePower: emitterSlice.totalEmissivePower,
    lightTree: emitterSlice.lightTree,
    lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
    lightTreeEnabled: emitterSlice.lightTreeEnabled,
    mergedGeometry: makeMergedGeometry(merged.boundingBox.min, merged.boundingBox.max),
    meshVertexRanges: enrichMeshVertexRangesWithCoreMatrix(scene, merged.meshVertexRanges),
    bvhIndicesStride3: merged.indices,
    buildMaterials: [],
    coreMaterials: merged.materials,
    emitterNormals: merged.normals,
    primitiveTlasBindings: [],
  };
}

export function buildReSTIRSceneBVHForCoreScene(
  scene: Scene,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers {
  if (!sceneHasCoreMeshes(scene)) {
    throw new Error(
      '[HybridEngine] BVH source unavailable: concrete walkaround-hybrid requires a core Scene with mesh primitives.',
    );
  }
  const mode = resolveReSTIRBvhMode(scene, options.bvhMode);
  return mode === 'tlas'
    ? buildReSTIRSceneBVHFromCoreTlas(scene, options)
    : buildReSTIRSceneBVHFromCoreMerged(scene, options);
}

export function rebuildReSTIRSceneBVHPrimitiveCore(
  scene: Scene,
  primitiveId: string,
  prev: SceneBVHBuffers,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers | { ok: false; reason: string } {
  if (prev.scenePack == null) {
    return { ok: false, reason: 'previous buffers have no scenePack snapshot' };
  }
  const { coreMaterials, resolveMaterialId } = materialResolver(scene);
  const rebuilt = rebuildPrimitiveBlas(scene, primitiveId, prev.scenePack, {
    tlas: true,
    resolveMaterialId,
  });
  if (!rebuilt.ok) {
    return { ok: false, reason: rebuilt.reason };
  }
  return buffersFromCoreScenePack(scene, rebuilt.pack, coreMaterials, options);
}

export function rebuildEmitterBuffersFromCoreScene(
  scene: Scene,
  options: {
    primaryLightDir?: Vector3Like;
    primaryLightIntensity?: number;
  } = {},
): RebuiltEmitterBuffers {
  return coreEmitterBuffers(scene, options);
}

export function disposeSceneBVH(buffers: SceneBVHBuffers): void {
  buffers.mergedGeometry.dispose();
}
