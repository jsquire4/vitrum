/**
 * ReSTIR BVH build via `@vitrum/shared-bvh` `packSceneFromCore` (per-primitive BLAS + TLAS).
 */

import type { MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import {
  packSceneFromCore,
  rebuildPrimitiveBlas,
  mergeWorldSpaceFromCore,
  type ScenePackResult,
} from '@vitrum/shared-bvh';
import * as THREE from 'three';
import {
  packUVIntoPositionW,
  packBVHIndexW,
  packBVHBeerColors,
  packBVHEmissiveLe,
} from './packingHelpers.js';
import { buildEmitterListFromCore, buildLightTreeBuffer } from './emitterList.js';
import type { SceneBVHBuffers } from './bvhCompute.js';
import {
  collectRectAreaEmitterTrisFromCore,
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

/**
 * Apply the `vitrumSceneToThree` emissive convention to a core material for the
 * emitter-list build: treat `emissive` as the FINAL radiance-space colour and
 * force `emissiveIntensity = 1`, so `materialSpecEmissiveLe`/
 * `classifyTriangleEmitterCore` yield `Le = emissive · 1` — exactly what the
 * THREE round-trip (and the camera-glow packer, which reads those `ei = 1` THREE
 * materials) produce. See the call site for the full rationale.
 *
 * Crucially, `vitrumSceneToThree` forces `emissiveIntensity = 1` for EVERY core
 * material it converts (vitrumSceneToThree.ts:211), regardless of the core
 * material's own `emissiveIntensity`. So we must set `ei = 1` whenever `emissive`
 * is PRESENT — including when the core `emissiveIntensity` is `undefined`: the
 * THREE round-trip still emits `Le = emissive · 1` for such a material (THREE's
 * `materialEmissiveLe` sees `ei = 1 > 0`), whereas a raw core read
 * (`materialSpecEmissiveLe`) would reject `ei === undefined` as "not emissive".
 * Forcing `ei = 1` reproduces the production decision in both cases. A material
 * with NO `emissive` is returned unchanged (it's not an emitter either way).
 */
function toProductionEmissiveRadiance(m: MaterialSpec): MaterialSpec {
  if (m.emissive === undefined) return m;
  if (m.emissiveIntensity === 1) return m; // already the production convention
  return { ...m, emissiveIntensity: 1 };
}

/** Drop the padding `.w` lane: stride-4 (vec3f-aligned) positions → tightly
 *  packed xyz triples for a THREE `position` BufferAttribute (itemSize 3). */
function extractXYZFromStride4(stride4: Float32Array, vertexCount: number): Float32Array {
  const out = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i += 1) {
    out[i * 3 + 0] = stride4[i * 4 + 0]!;
    out[i * 3 + 1] = stride4[i * 4 + 1]!;
    out[i * 3 + 2] = stride4[i * 4 + 2]!;
  }
  return out;
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

  // CORE-FIRST emitter-list build (THREE-decouple, increment of
  // `plan/three-decouple-analysis-2026-06-03.md`). The emitter list needs a
  // WORLD-space triangle stream: it derives triangle area, face normal
  // (world-space sun-dot in classifyTriangleEmitterCore), centroids and AABBs,
  // and appends world-space rect-area emitter tris. `geo` (packSceneFromCore)
  // stores per-primitive BLAS positions in LOCAL/object space — world transforms
  // live separately in the TLAS instance matrices — so feeding `geo` would place
  // every emitter at the wrong world location for any transformed mesh.
  //
  // `mergeWorldSpaceFromCore` is the THREE-free analogue of the former
  // `buildSceneBVH(sceneRoots)` world-bake: it bakes each primitive's core
  // `transform` into the vertices (= the matrixWorld `vitrumSceneToThree` would
  // synthesize) and emits the merged world-space stream + deduped `MaterialSpec[]`
  // — no `StaticGeometryGenerator`, no THREE materials. The emitter SET it
  // produces is identical to the THREE path (CPU-pinned by
  // __tests__/emitterListCoreEquivalence.test.ts), though the triangle ORDER
  // (hence CDF indexing + per-sample RIS selection) differs because the SAH
  // builders differ — so the CONVERGED render matches while a low-spp A/B does
  // not (validated via a high-spp GPU A/B, not pixel-identity).
  //
  // InstancedMesh: the THREE emitter path EXCLUDED instanced meshes (their
  // geometry lives in packSceneFromCore's TLAS — one local BLAS + N instance
  // matrices — which the emitter list does not consume per-instance). The default
  // `mergeWorldSpaceFromCore` filter INCLUDES them, so we pass a filter that
  // rejects `kind === 'instanced-mesh'` to preserve that exclusion exactly.
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
    // RADIOMETRY PARITY (load-bearing): the THREE emitter path this replaces read
    // materials produced by `vitrumSceneToThree`, which treats `MaterialSpec.emissive`
    // as the FINAL radiance-space colour and forces `emissiveIntensity = 1`
    // (vitrumSceneToThree.ts:201-211 — "avoid accidental double-scaling on round-
    // trip"). So the production emitter Le for an emissive mesh was `emissive · 1`,
    // NOT `emissive · emissiveIntensity`. The camera-visible glow packer
    // (`packBVHEmissiveLe` above) STILL reads those `ei = 1` THREE materials, so the
    // NEE radiance and the camera glow must BOTH be `emissive · 1` to stay
    // consistent (no drift). `classifyTriangleEmitterCore` faithfully computes
    // `emissive · emissiveIntensity` (it mirrors `materialEmissiveLe` on a raw core
    // material), so we apply the SAME `vitrumSceneToThree` convention here by
    // forcing `emissiveIntensity = 1` before the emitter build. This keeps the
    // core-first path radiometrically identical to the THREE round-trip it replaces
    // (GPU-validated by the converged emitter-core-ab A/B). The TRANSMISSIVE
    // secondary-emitter branch is unaffected (it never reads emissiveIntensity).
    merged.materials.map(toProductionEmissiveRadiance),
    { ...options, extraEmitters },
  );
  const emitterCount = cdfArray.length;
  const lightTreeBuf = buildLightTreeBuffer(treeInput);

  // `mergedGeometry` (SceneBVHBuffers field, THREE.BufferGeometry) is consumed by
  // the snapshot ONLY for its world-space `boundingBox` — and only as a FALLBACK
  // when `computeWorldAabbForBindings(scene, tlasBindings)` returns empty (the
  // normal TLAS path never reaches that fallback; see restirBvhSnapshot.ts:57).
  // It is also disposed on teardown (bvhCompute.ts). We therefore build a minimal
  // position-only geometry over the core merged world stream and stamp its
  // bounding box from `merged.boundingBox` — NO THREE BVH, no `MeshBVH` build.
  const mergedGeometry = new THREE.BufferGeometry();
  mergedGeometry.setAttribute(
    'position',
    new THREE.BufferAttribute(extractXYZFromStride4(merged.positions, merged.vertexCount), 3),
  );
  mergedGeometry.boundingBox = new THREE.Box3(
    new THREE.Vector3(merged.boundingBox.min[0], merged.boundingBox.min[1], merged.boundingBox.min[2]),
    new THREE.Vector3(merged.boundingBox.max[0], merged.boundingBox.max[1], merged.boundingBox.max[2]),
  );

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
    // V21 — the smooth-normal blend now applies in TLAS too: the shaders read
    // these LOCAL-space normals and transform the blend to world via the hit
    // instance's inverse-transpose. The buffer is bound (the layout requires it)
    // and consumed in both merged and TLAS modes.
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
    mergedGeometry,
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
