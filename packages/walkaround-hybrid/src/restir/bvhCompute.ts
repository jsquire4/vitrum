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

import * as THREE from 'three';
import { buildSceneBVH as buildSharedBVH } from '@vitrum/shared-bvh';

// Packing helpers (applyBeerLambert, packUVIntoPositionW, packBVHIndexW,
// packBVHBeerColors) live in restir/packingHelpers.ts.
// Emitter list construction lives in restir/emitterList.ts.
import {
  packUVIntoPositionW,
  packBVHIndexW,
  packBVHBeerColors,
} from './packingHelpers.js';
import { buildEmitterList } from './emitterList.js';

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
  /** BVHNode[] array — 32 bytes/node. */
  bvhNodes: StorageBufferHandle;
  /** vec3u[] (3×u32) per triangle — vertex indices into bvhPositions. */
  bvhIndex: StorageBufferHandle;
  /** vec3f[] — vertex positions, world-space. */
  bvhPositions: StorageBufferHandle;
  /** vec3f[] — vertex normals, world-space. */
  bvhNormals: StorageBufferHandle;
  /** vec2f[] — UV coordinates per vertex. */
  bvhUvs: StorageBufferHandle;
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
  /** EmitterTri[] — 64-byte emitter struct per emissive triangle. */
  emitters: StorageBufferHandle;
  /** f32[] — CDF over emitter power (same length as emitters). */
  emitterCdf: StorageBufferHandle;
  /**
   * f32[] — per-emitter total radiant flux (same length as emitters).
   *
   * cellPower[i] = luminance(Le[i]) × area[i] for mesh/triangle-area
   * emitters, matching the formula used to build the power-CDF.
   * The Sprint 3 light tree (shared-samplers `buildLightTreeCDF`) reads
   * this buffer as its `powers` input for power-weighted importance
   * sampling. Not yet consumed by any WGSL shader — GPU-side consumption
   * is deferred pending walkaround dispatch integration (Sprint 9/10).
   *
   * Sentinels: the dummy emitter (inserted when no real emitters exist)
   * has cellPower = 0.5 (its synthetic power value). This keeps the
   * buffer non-empty so a zero-emitter scene doesn't break the bind group.
   */
  cellPower: StorageBufferHandle;
  /** Number of entries in the emitters / cdf / cellPower arrays. */
  emitterCount: number;
  totalEmissivePower: number;
  /** Merged geometry (CPU side, for debug / re-upload). */
  mergedGeometry: THREE.BufferGeometry;
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
  const { emitterFloats, cdfArray, cellPowerArray, totalEmissivePower } = buildEmitterList(
    shared.indices,
    shared.positions, // stride-4; emitter math reads .xyz only
    shared.normals,
    shared.triMaterialId,
    shared.materials,
    options,
  );
  const emitterCount = cdfArray.length;

  // ── 5. UV buffer (separate; CPU-side debug + future use) ────────────────
  // The GPU consumes UV via `bvh_position[*].w`; this is the contract-
  // preserving CPU-side handle.
  const uvAttr = shared.uvAttribute;
  const uvBuf = uvAttr
    ? new Float32Array(uvAttr.array)
    : new Float32Array(vertCount * 2);

  // triangleMaterialIds — pass through the shared per-tri matId LUT.
  const triMatIds = new Uint32Array(shared.triMaterialId);

  // ── 6. Return buffers ────────────────────────────────────────────────────
  // NOTE: bvhIndex is now vec4u[] (4 u32 per triangle):
  //   [0..2] = vertex indices, [3] = packed RGBA8 material color + transmission.
  // The triangleMaterialIds field carries the CPU-side u32[] for emitter building;
  // it is NOT uploaded to the GPU as a separate buffer.
  // materialColors was removed (M-1 cleanup) — colors are packed into bvhIndex[*].w.
  return {
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
    bvhNormals: {
      cpuData: shared.normals.buffer.slice(0) as ArrayBuffer,
      byteLength: shared.normals.byteLength,
      count: vertCount,
    },
    bvhUvs: { cpuData: uvBuf.buffer, byteLength: uvBuf.byteLength, count: vertCount },
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
    // Per-emitter radiant flux (f32[], same length as emitters).
    // Sprint 3 light tree (shared-samplers buildLightTreeCDF) uses this as its
    // `powers` input. Not yet consumed by any WGSL shader — GPU dispatch deferred.
    cellPower: {
      cpuData: cellPowerArray.buffer as ArrayBuffer,
      byteLength: cellPowerArray.byteLength,
      count: emitterCount,
    },
    emitterCount,
    totalEmissivePower,
    mergedGeometry: shared.bvh.geometry,
  };
}

/** Dispose CPU-side geometry + GPU buffers (call on unmount). */
export function disposeSceneBVH(buffers: SceneBVHBuffers): void {
  buffers.mergedGeometry.dispose();
}

