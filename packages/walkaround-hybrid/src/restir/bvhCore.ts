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
    resolveMaterialId: (id) => {
      const idx = byKey.get(id);
      if (idx === undefined) {
        // H24-A — warn on unknown primitive id so material-0 fallbacks are visible
        // in the console rather than silently producing incorrect shading. Duplicate
        // ids (two primitives share the same id) would also land here for the second
        // occurrence; both cases warrant investigation.
        console.warn(
          `[ReSTIR bvhCore] unknown primitive id "${id}" — falling back to material 0. ` +
          `This may produce incorrect shading. Check scene.primitives for duplicate or missing ids.`,
        );
        return 0;
      }
      return idx;
    },
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
  // This stream is only for ReSTIR light selection. Expanding instanced meshes
  // here keeps emissive instances visible to direct lighting while the render
  // BVH can still use the TLAS/BLAS path for traversal.
  const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
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

/**
 * H15 — extract the uv0 layer from a stride-4 `ScenePackResult.uvs` array
 * (layout: [u0, v0, u1, v1] per vertex) into a stride-2 Float32Array that
 * `packUVIntoPositionW` can consume via its `{ array }` BufferAttributeLike.
 * The TLAS vertex ordering is inherited from `packSceneFromCore` → same
 * primitive-concat order as `geo.positions`, so vertex indices align 1:1.
 */
function stride4UvsToStride2Uv0(uvs4: Float32Array, vertCount: number): Float32Array {
  const out = new Float32Array(vertCount * 2);
  for (let i = 0; i < vertCount; i++) {
    out[i * 2] = uvs4[i * 4] ?? 0;
    out[i * 2 + 1] = uvs4[i * 4 + 1] ?? 0;
  }
  return out;
}

function buffersFromCoreScenePack(
  scene: Scene,
  geo: ScenePackResult,
  coreMaterials: readonly MaterialSpec[],
  options: CoreBvhBuildOptions,
): SceneBVHBuffers {
  const triCount = geo.triangleCount;
  const vertCount = geo.positions.length / 4;

  // H15 — pass the real UVs from ScenePackResult.uvs (stride-4 vec4f → extract
  // uv0 as stride-2) so every vertex's .w lane carries the packed UV pair instead
  // of (0,0). Vertex ordering: packSceneFromCore emits positions and uvs in the
  // same primitive-concat order, so indices align 1:1.
  const uv0Stride2 = stride4UvsToStride2Uv0(geo.uvs, vertCount);
  const positionsWithUV = packUVIntoPositionW(geo.positions, { array: uv0Stride2 }, vertCount);
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
  // H15 — pass merged.uvs (stride-2, same vertex order as merged.positions) so
  // every vertex's .w lane carries the packed UV pair.  WorldSpaceMergeResult.uvs
  // is stride-2 (u0, v0 per vertex), which is exactly what packUVIntoPositionW's
  // { array } path expects (reads array[i*2] / array[i*2+1]).
  const positionsWithUV = packUVIntoPositionW(merged.positions, { array: merged.uvs }, vertCount);
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
