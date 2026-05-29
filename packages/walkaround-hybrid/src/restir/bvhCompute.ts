/**
 * bvhCompute.ts — Build scene BVH + emitter list storage buffers for the
 * ReSTIR compute pipeline.
 *
 * This is the JavaScript side; the WGSL shaders consume the buffers via
 * `ptr<storage, array<BVHNode>, read>` etc.
 *
 * Node packing format matches common.wgsl.ts BVHNode struct:
 *   struct BVHNode {
 *     bounds: BVHBoundingBox,                // 6 × f32  = 24 bytes (min xyz, max xyz)
 *     rightChildOrTriangleOffset: u32,       //            4 bytes
 *     splitAxisOrTriangleCount:   u32,       //            4 bytes
 *   }  // = 32 bytes per node
 *
 * Layout matches three-mesh-bvh's internal 8×u32 = 32-byte node format so we
 * can DMA the raw buffer without re-packing.
 *
 * This module wraps the shared BVH-build core (@vitrum/shared-bvh) with
 * ReSTIR-specific post-packing:
 *   - UV-pack-into-position-w (compute-cast 8-storage-buffer-per-stage limit
 *     workaround — uses the .w slot left zero by the shared stride-4 path).
 *   - bvhIndex.w RGBA8 + texType packing (4-bit transmission + 4-bit
 *     surfaceTextureId carried through the 4th u32 of every triangle index).
 *   - Emitter list construction (80-byte EmitterTri struct + power-CDF).
 */

import type { Scene } from '@vitrum/core';
import type { PrimitiveTlasBinding } from '@vitrum/shared-bvh';
import * as THREE from 'three';
import { buildSceneBVH as buildSharedBVH } from '@vitrum/shared-bvh';
import {
  buildReSTIRSceneBVHFromVitrumScene,
  rebuildReSTIRSceneBVHPrimitive,
  resolveReSTIRBvhMode,
  type ReSTIRBvhMode,
} from './sceneBvhFromCore.js';

export { rebuildReSTIRSceneBVHPrimitive };

// Packing helpers (applyBeerLambert, packUVIntoPositionW, packBVHIndexW,
// packBVHBeerColors) live in restir/packingHelpers.ts.
// Emitter list construction lives in restir/emitterList.ts.
import {
  packUVIntoPositionW,
  packBVHIndexW,
  packBVHBeerColors,
} from './packingHelpers.js';
import { buildEmitterList, buildLightTreeBuffer } from './emitterList.js';
import {
  collectRectAreaLightEmitterTris,
  enrichMeshVertexRangesWithMatrix,
} from './bvhSceneHelpers.js';

export { collectRectAreaLightEmitterTris, enrichMeshVertexRangesWithMatrix };

/** A WebGPU storage buffer handle (GPU-side ArrayBuffer wrapper). */
interface StorageBufferHandle {
  /** The Float32Array / Uint32Array data that was uploaded. */
  cpuData: ArrayBuffer;
  /** Byte length of the buffer. */
  byteLength: number;
  /** Number of elements (each element is one struct). */
  count: number;
}

/**
 * Surface-texture type ID encoding.  Packed into the low 4 bits of
 * `bvhIndex[*].w` (the byte that historically held transmission * 255).
 *
 * The shade pass reads this ID at primary glass hits, looks up the
 * matching procedural pattern function in WGSL, and modulates the
 * cell's emission accordingly — so cells with the same baseColor but
 * different surface texture (e.g. red waterglass vs red ripple) render
 * with visibly different per-pixel patterns instead of looking like
 * identical flat hexagons.
 *
 * IDs 0..7 are populated; 8..15 are reserved.
 */

export interface SceneBVHBuffers {
  /** CPU pack mode — GPU traversal stays merged until PR-3 when `tlas`. */
  bvhMode: ReSTIRBvhMode;
  /** BVHNode[] array — 32 bytes/node. */
  bvhNodes: StorageBufferHandle;
  /** vec3u[] (3×u32) per triangle — vertex indices into bvhPositions. */
  bvhIndex: StorageBufferHandle;
  /** vec3f[] — vertex positions, world-space. .w lane carries the packed
   *  UV pair via packUVIntoPositionW, so the shaders read both position
   *  and UV from this single buffer. */
  bvhPositions: StorageBufferHandle;
  /** u32[] — per-triangle material id (kept for CPU-side use). */
  triangleMaterialIds: StorageBufferHandle;
  /**
   * u32[] — per-triangle Beer-Lambert visible color (RGBA8 packed),
   * `pow(attenuationColor, thickness/attenuationDistance)`. Read by
   * shade.wgsl Lo_emit on a primary glass hit so the cell renders with
   * PT-equivalent saturation. Distinct from bvhIndex.w (which holds the
   * RAW attenuationColor used by the emitter list, ReSTIR DI sampling,
   * and bvhTraceTintedVisibility — those need un-attenuated values to
   * keep the room properly lit).
   *
   * Opaque tris pack their material color identically (no Beer-Lambert
   * applied since they're not transmissive); decoder treats this slot
   * as primary color for ALL primary hits.
   */
  bvhBeerColors: StorageBufferHandle;
  /**
   * WS1 (2026-05-29) — per-vertex world-space normals (stride-4 vec4f, `.w`
   * unused). Same `shared.normals` already surfaced as {@link emitterNormals};
   * exposed as a GPU storage handle so the scene bind group can carry a
   * `bvh_normal` buffer. The primary passes (shade/ris/risGi/risGiNrc)
   * barycentric-blend it (`normalize(w·n0+u·n1+v·n2)·side`, mirroring the DDGI
   * precedent at probeUpdateRays.wgsl.ts:443-454) for a SMOOTH shading normal
   * instead of the faceted geometric face normal. The GPU-skin compute kernel
   * writes its inverse-transpose skinned normals into this buffer's GPU copy at
   * `baseVertex+vi`.
   */
  bvhNormals: StorageBufferHandle;
  /** EmitterTri[] — 64-byte emitter struct per emissive triangle. */
  emitters: StorageBufferHandle;
  /** f32[] — CDF over emitter power (same length as emitters). */
  emitterCdf: StorageBufferHandle;
  /** Number of entries in the emitters / cdf arrays. */
  emitterCount: number;
  totalEmissivePower: number;
  /**
   * f32[] — packed light tree for ReSTIR-DI light SELECTION (Shirley 1996
   * power split + Estévez-Kulla 2018 distance-weighted descent). 12 floats per
   * node; leaf `emitterIndex` indexes the same `emitters` array. RIS uploads it
   * to a RIS-only `@group(3)` storage buffer and importance-samples lights from
   * it when `lightTreeEnabled` is true, dividing the WRS weight by the exact
   * tree selection pdf (unbiased). A single zeroed placeholder node backs the
   * buffer when disabled (`lightTreeEnabled === false`).
   */
  lightTree: StorageBufferHandle;
  /** Number of nodes in the packed light tree (0 when disabled). */
  lightTreeNodeCount: number;
  /** Whether RIS should select lights via the tree (≥ 2 emitters). */
  lightTreeEnabled: boolean;
  /** Merged geometry (CPU side, for debug / re-upload). */
  mergedGeometry: THREE.BufferGeometry;
  /**
   * Per-source-mesh vertex ranges in the merged vertex buffer, in mesh
   * traversal order. `name` is the source `Object3D.name`; for scenes
   * synthesized via `vitrumSceneToThree` the name equals the primitive id
   * (see `vitrumSceneToThree.ts:159`), so primitive id → vertex range
   * needs no extra bookkeeping.
   *
   * `matrixWorldAtBuild` is the 16-float `mesh.matrixWorld.elements`
   * snapshot at the time the BVH was built. On a transform-only patch,
   * `HybridEngine.updatePrimitive` walks the cached merged-world position
   * back through `inverse(matrixWorldAtBuild)` (to recover the local-space
   * position) and forward through the patched `matrixWorld` — yielding
   * the new world-space position without re-running the full BVH build.
   *
   * Mirrors `SceneBVHCommonResult.meshVertexRanges` from `@vitrum/shared-bvh`,
   * extended with the matrix snapshot the refit path needs.
   */
  meshVertexRanges: ReadonlyArray<{
    name: string;
    vertexStart: number;
    vertexCount: number;
    triStart: number;
    triCount: number;
    matrixWorldAtBuild: Float32Array;
  }>;
  /**
   * Stride-3 triangle index buffer (3 u32 per triangle) — the un-packed
   * indices returned by `buildSharedBVH`. ReSTIR's GPU `bvhIndex` is
   * stride-4 (packs RGBA8 material color into `.w`); the shared-bvh
   * refit helper reads indices without the `.w` lane, so we cache the
   * stride-3 form here for the refit fast path.
   */
  bvhIndicesStride3: Uint32Array;
  /** THREE materials aligned with `triangleMaterialIds` (CPU-only, for emitter rebuild). */
  buildMaterials: readonly THREE.Material[];
  /** Stride-4 world-space normals from the last BVH build (emitter list input). */
  emitterNormals: Float32Array;
  /** TLAS storage buffers — populated when `bvhMode === 'tlas'`. */
  tlas?: {
    nodes: StorageBufferHandle;
    instanceIndices: StorageBufferHandle;
    blasRoots: StorageBufferHandle;
    worldToLocal: StorageBufferHandle;
    localToWorld: StorageBufferHandle;
    nodeCount: number;
  };
  /** Stable per-primitive TLAS bindings from `packSceneFromCore`. */
  primitiveTlasBindings: readonly PrimitiveTlasBinding[];
  /** Last CPU pack from `packSceneFromCore` / `rebuildPrimitiveBlas` (TLAS mode). */
  scenePack?: import('@vitrum/shared-bvh').ScenePackResult;
  /** Non-fatal pack warnings (skipped primitives, bad transforms, …). */
  warnings?: readonly string[];
}

export type { ReSTIRBvhMode };

/** Pick merged vs TLAS CPU pack and build ReSTIR buffers. */
export function buildReSTIRSceneBVHForScene(
  scene: Scene,
  sceneRoots: THREE.Object3D[],
  options: {
    bvhMode?: ReSTIRBvhMode;
    primaryLightDir?: THREE.Vector3;
    primaryLightIntensity?: number;
    proxyMeshNames?: Set<string>;
  } = {},
): SceneBVHBuffers {
  const mode = resolveReSTIRBvhMode(scene, options.bvhMode);
  if (mode === 'tlas') {
    return buildReSTIRSceneBVHFromVitrumScene(scene, sceneRoots, options);
  }
  return buildReSTIRSceneBVH(sceneRoots, options);
}

// EmitterTri layout, EMITTER_STRIDE / EMITTER_FLOATS, and the
// WARM_GRAY_DEFAULT_* constants live in their respective modules
// (emitterList.ts / packingHelpers.ts).

/** Build the full scene BVH + emitter list from a set of Object3D roots. */
export function buildReSTIRSceneBVH(
  sceneRoots: THREE.Object3D[],
  options: {
    /** Primary directional light direction (world-space, normalized). Used
     *  to determine which back-face panel triangles face the light for
     *  the emitter list. */
    primaryLightDir?: THREE.Vector3;
    /** Primary directional light intensity (linear, unitless). Multiplied
     *  into the panel-cell self-emission Le baked into the emitter list. */
    primaryLightIntensity?: number;
    /**
     * Mesh-name allowlist for proxy substitution. Meshes whose name
     * matches one of these IDs get their dense per-vertex geometry
     * replaced with a low-poly proxy at BVH-build time. Useful for
     * detail surfaces (carpet, ceiling tiles) that are tessellated for
     * raster but where ray-trace per-triangle cost outweighs the
     * shadow-resolution gain.
     *
     * Default: empty Set — no proxy substitution unless the host passes names.
     */
    proxyMeshNames?: Set<string>;
  } = {},
): SceneBVHBuffers {
  // ── 1. Shared BVH-build core ─────────────────────────────────────────────
  // The shared module handles: world-matrix update, sky-hide, proxy
  // substitution, normalize-to-indexed, merged-geometry generation, BVH-
  // invariant per-vertex matId snapshot, single-root group-collapse build,
  // 32-byte BVH node packing, stride-4 position/normal extraction, per-tri
  // materialId LUT, and bounds.
  //
  // Filter: ReSTIR historically passed `sceneRoots` directly to
  // StaticGeometryGenerator, which internally `traverseVisible`s and picks
  // up every `isMesh` Object3D regardless of material kind.  Pass that
  // permissive filter explicitly so the shared default (MeshStandard +
  // MeshPhysical only) doesn't silently change which meshes contribute.
  const shared = buildSharedBVH(sceneRoots, {
    positionStride: 4,
    proxyMeshNames: options.proxyMeshNames ?? new Set<string>(),
    // Permissive filter — accept every visible mesh, including came/
    // solder beads. They render as opaque dark geometry and cast
    // proper shadows in the path trace, restoring the panel structure.
    filter: (obj: THREE.Object3D) => obj instanceof THREE.Mesh,
  });

  const triCount = shared.indices.length / 3;
  const vertCount = shared.positions.length / 4; // stride-4 layout

  // ── 2. UV-pack-into-position-w (compute-cast 8-buffer-per-stage limit) ──
  // The shared `positions` is stride-4 with .w left zero.  ReSTIR packs UV
  // into .w (16-bit unorm pair, bitcast-as-f32) so the GPU shade pass can
  // unpack per-vertex UV without spending another storage-buffer slot.
  const positionsWithUV = packUVIntoPositionW(
    shared.positions,
    shared.uvAttribute,
    vertCount,
  );

  // ── 3. Pack bvhIndex.w with RGBA8 raw attCol + (trans4 | texType4) ──────
  const indexBuf = packBVHIndexW(
    shared.indices,
    shared.triMaterialId,
    shared.materials,
    triCount,
  );

  // ── 3b. Pack bvh_beer with the Beer-Lambert visible color per tri. ──────
  const beerBuf = packBVHBeerColors(
    shared.triMaterialId,
    shared.materials,
    triCount,
  );

  // ── 4. Build emitter list (transmissive + emissive triangles) ──────────
  // Non-mesh scene lights (THREE.RectAreaLight from
  // `vitrumSceneToThree`'s `'rect-area'` emitter branch) are NOT in the
  // BVH and would not be discovered by the per-triangle material walk.
  // Collect them as explicit emitter triangles so ReSTIR DI can sample
  // them.
  const extraEmitters = collectRectAreaLightEmitterTris(sceneRoots);

  const { emitterFloats, cdfArray, totalEmissivePower, treeInput } = buildEmitterList(
    shared.indices,
    shared.positions, // stride-4; emitter math reads .xyz only
    shared.normals,
    shared.triMaterialId,
    shared.materials,
    { ...options, extraEmitters },
  );
  const emitterCount = cdfArray.length;
  const lightTreeBuf = buildLightTreeBuffer(treeInput);

  // triangleMaterialIds — pass through the shared per-tri matId LUT.
  const triMatIds = new Uint32Array(shared.triMaterialId);

  // ── 6. Return buffers ────────────────────────────────────────────────────
  // NOTE: bvhIndex is now vec4u[] (4 u32 per triangle):
  //   [0..2] = vertex indices, [3] = packed RGBA8 material color + transmission.
  // The triangleMaterialIds field carries the CPU-side u32[] for emitter building;
  // it is NOT uploaded to the GPU as a separate buffer.
  // materialColors was removed (M-1 cleanup) — colors are packed into bvhIndex[*].w.
  return {
    bvhMode: 'merged',
    bvhNodes: {
      cpuData: shared.bvhNodes.buffer.slice(0) as ArrayBuffer,
      byteLength: shared.bvhNodes.byteLength,
      count: shared.bvhNodes.byteLength / 32,
    },
    bvhIndex: { cpuData: indexBuf.buffer, byteLength: indexBuf.byteLength, count: triCount },
    bvhPositions: {
      cpuData: positionsWithUV.buffer,
      byteLength: positionsWithUV.byteLength,
      count: vertCount,
    },
    triangleMaterialIds: {
      cpuData: triMatIds.buffer,
      byteLength: triMatIds.byteLength,
      count: triCount,
    },
    bvhBeerColors: {
      cpuData: beerBuf.buffer,
      byteLength: beerBuf.byteLength,
      count: triCount,
    },
    // WS1 — per-vertex world-space normals (stride-4 vec4f). `shared.normals`
    // is already a Float32Array of vertCount×4; reuse it verbatim (no re-pack).
    bvhNormals: {
      cpuData: shared.normals.buffer.slice(0) as ArrayBuffer,
      byteLength: shared.normals.byteLength,
      count: vertCount,
    },
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
    mergedGeometry: shared.bvh.geometry,
    meshVertexRanges: enrichMeshVertexRangesWithMatrix(sceneRoots, shared.meshVertexRanges),
    bvhIndicesStride3: shared.indices,
    buildMaterials: shared.materials,
    emitterNormals: shared.normals,
    primitiveTlasBindings: [],
  };
}

/**
 * Rebuild only the ReSTIR emitter list + power CDF from the current scene
 * graph and the cached BVH geometry (no SAH rebuild). Used by
 * `HybridEngine.updateEmitter` fast path.
 */
export function rebuildEmitterBuffersFromSceneRoots(
  sceneRoots: THREE.Object3D[],
  bvh: Pick<
    SceneBVHBuffers,
    | 'bvhIndicesStride3'
    | 'bvhPositions'
    | 'emitterNormals'
    | 'triangleMaterialIds'
    | 'buildMaterials'
  >,
  options: {
    primaryLightDir?: THREE.Vector3;
    primaryLightIntensity?: number;
  } = {},
): Pick<
  SceneBVHBuffers,
  | 'emitters'
  | 'emitterCdf'
  | 'emitterCount'
  | 'totalEmissivePower'
  | 'lightTree'
  | 'lightTreeNodeCount'
  | 'lightTreeEnabled'
> {
  const extraEmitters = collectRectAreaLightEmitterTris(sceneRoots);
  const { emitterFloats, cdfArray, totalEmissivePower, treeInput } = buildEmitterList(
    bvh.bvhIndicesStride3,
    new Float32Array(bvh.bvhPositions.cpuData),
    bvh.emitterNormals,
    new Uint32Array(bvh.triangleMaterialIds.cpuData),
    [...bvh.buildMaterials],
    { ...options, extraEmitters },
  );
  const emitterCount = cdfArray.length;
  // Emitters changed → rebuild the selection tree from the same inputs so the
  // tree pmf and emitter array stay aligned (leaf emitterIndex must index the
  // freshly-built emitter list).
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

/** Dispose CPU-side geometry + GPU buffers (call on unmount). */
export function disposeSceneBVH(buffers: SceneBVHBuffers): void {
  buffers.mergedGeometry.dispose();
}

