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
  /** EmitterTri[] — 64-byte emitter struct per emissive triangle. */
  emitters: StorageBufferHandle;
  /** f32[] — CDF over emitter power (same length as emitters). */
  emitterCdf: StorageBufferHandle;
  /** Number of entries in the emitters / cdf arrays. */
  emitterCount: number;
  totalEmissivePower: number;
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

  const { emitterFloats, cdfArray, totalEmissivePower } = buildEmitterList(
    shared.indices,
    shared.positions, // stride-4; emitter math reads .xyz only
    shared.normals,
    shared.triMaterialId,
    shared.materials,
    { ...options, extraEmitters },
  );
  const emitterCount = cdfArray.length;

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
    emitterCount,
    totalEmissivePower,
    mergedGeometry: shared.bvh.geometry,
    meshVertexRanges: enrichMeshVertexRangesWithMatrix(sceneRoots, shared.meshVertexRanges),
    bvhIndicesStride3: shared.indices,
    buildMaterials: shared.materials,
    emitterNormals: shared.normals,
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
): Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf' | 'emitterCount' | 'totalEmissivePower'> {
  const extraEmitters = collectRectAreaLightEmitterTris(sceneRoots);
  const { emitterFloats, cdfArray, totalEmissivePower } = buildEmitterList(
    bvh.bvhIndicesStride3,
    new Float32Array(bvh.bvhPositions.cpuData),
    bvh.emitterNormals,
    new Uint32Array(bvh.triangleMaterialIds.cpuData),
    [...bvh.buildMaterials],
    { ...options, extraEmitters },
  );
  const emitterCount = cdfArray.length;
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
  };
}

/**
 * Walk `sceneRoots` once to find each named mesh and snapshot its
 * `matrixWorld.elements` — required by `HybridEngine.updatePrimitive`'s
 * transform-only refit path. Names are unique (host stamps primitive ids
 * onto `mesh.name`), so the first match per name wins.
 *
 * Meshes without a match (the host renamed or removed them between
 * build and refit) get a fallback identity matrix; the refit will then
 * misposition that mesh, but the topology-rebuild path catches it on
 * the next setScene.
 */
function enrichMeshVertexRangesWithMatrix(
  sceneRoots: THREE.Object3D[],
  rawRanges: ReadonlyArray<{
    name: string;
    vertexStart: number;
    vertexCount: number;
    triStart: number;
    triCount: number;
  }>,
): ReadonlyArray<{
  name: string;
  vertexStart: number;
  vertexCount: number;
  triStart: number;
  triCount: number;
  matrixWorldAtBuild: Float32Array;
}> {
  const byName = new Map<string, THREE.Object3D>();
  for (const root of sceneRoots) {
    root.traverseVisible((obj) => {
      if (!byName.has(obj.name)) byName.set(obj.name, obj);
    });
  }
  const out = rawRanges.map((r) => {
    const obj = byName.get(r.name);
    // Snapshot the matrix-world elements at build time. Use a fresh
    // Float32Array so subsequent host edits to obj.matrixWorld don't
    // mutate our snapshot.
    const m = obj ? new Float32Array(obj.matrixWorld.elements) : new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    return {
      name: r.name,
      vertexStart: r.vertexStart,
      vertexCount: r.vertexCount,
      triStart: r.triStart,
      triCount: r.triCount,
      matrixWorldAtBuild: m,
    };
  });
  return out;
}

/** Dispose CPU-side geometry + GPU buffers (call on unmount). */
export function disposeSceneBVH(buffers: SceneBVHBuffers): void {
  buffers.mergedGeometry.dispose();
}

/**
 * Walk the scene roots for `THREE.RectAreaLight` instances and convert each
 * to a pair of emitter triangles (front-emitting along the light's local -Z
 * face, matching THREE.RectAreaLight's convention).
 *
 * Folds `light.intensity` into Le so the WGSL shade kernel (which reads only
 * `EmitterTri.Le` for radiance and ignores the legacy `intensity` slot) sees
 * the correct power.
 */
function collectRectAreaLightEmitterTris(
  sceneRoots: THREE.Object3D[],
): {
  vA: [number, number, number];
  vB: [number, number, number];
  vC: [number, number, number];
  normal: [number, number, number];
  area: number;
  Le: [number, number, number];
}[] {
  const out: {
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    Le: [number, number, number];
  }[] = [];
  const _ll = new THREE.Vector3();
  const _lr = new THREE.Vector3();
  const _ur = new THREE.Vector3();
  const _ul = new THREE.Vector3();
  const _normal = new THREE.Vector3();
  const _ab = new THREE.Vector3();
  const _ac = new THREE.Vector3();

  for (const root of sceneRoots) {
    root.updateMatrixWorld(true);
    root.traverseVisible((obj) => {
      if (!(obj instanceof THREE.RectAreaLight)) return;
      const light = obj;
      const wHalf = light.width * 0.5;
      const hHalf = light.height * 0.5;

      _ll.set(-wHalf, -hHalf, 0).applyMatrix4(light.matrixWorld);
      _lr.set( wHalf, -hHalf, 0).applyMatrix4(light.matrixWorld);
      _ur.set( wHalf,  hHalf, 0).applyMatrix4(light.matrixWorld);
      _ul.set(-wHalf,  hHalf, 0).applyMatrix4(light.matrixWorld);

      // Triangle area: half the rect parallelogram (one rect = 2 tris).
      _ab.subVectors(_lr, _ll);
      _ac.subVectors(_ur, _ll);
      _normal.crossVectors(_ab, _ac);
      const crossLen = _normal.length();
      if (crossLen < 1e-8) return;

      // Emission direction is THREE.RectAreaLight's local -Z, transformed
      // by the light's world basis. The geometric face normal from the
      // vertex cross product points along local +Z (away from emission),
      // so we cannot use it directly — sample-cosine tests in the shade
      // kernel would reject every surface in front of the light.
      _normal.setFromMatrixColumn(light.matrixWorld, 2).normalize().negate();

      const triArea = crossLen * 0.5;
      const c = light.color;
      const I = light.intensity;
      const Le: [number, number, number] = [c.r * I, c.g * I, c.b * I];
      const N: [number, number, number] = [_normal.x, _normal.y, _normal.z];

      out.push({
        vA: [_ll.x, _ll.y, _ll.z],
        vB: [_lr.x, _lr.y, _lr.z],
        vC: [_ur.x, _ur.y, _ur.z],
        normal: N,
        area: triArea,
        Le,
      });
      out.push({
        vA: [_ll.x, _ll.y, _ll.z],
        vB: [_ur.x, _ur.y, _ur.z],
        vC: [_ul.x, _ul.y, _ul.z],
        normal: N,
        area: triArea,
        Le,
      });
    });
  }
  return out;
}

