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

/**
 * Apply Beer-Lambert through-medium attenuation to a glass material's
 * authored attenuationColor. Returns the color a viewer would see after
 * sun-radiance passes through one panel thickness — i.e. PT's transmitted
 * radiance at normal incidence (modulo fresnel).
 *
 *   tinted = pow(attenuationColor, thickness / attenuationDistance)
 *
 * Returns the original color when thickness or attenuationDistance is
 * missing/non-finite — preserves legacy behavior for materials that
 * haven't yet adopted physical thickness.
 */
function applyBeerLambert(
  attCol: THREE.Color,
  thickness: number | undefined,
  attDist: number | undefined,
): THREE.Color {
  if (thickness === undefined || attDist === undefined) return attCol;
  if (!Number.isFinite(thickness) || !Number.isFinite(attDist)) return attCol;
  if (thickness <= 0 || attDist <= 0) return attCol;
  const k = thickness / attDist;
  return new THREE.Color(
    Math.pow(Math.max(1e-6, attCol.r), k),
    Math.pow(Math.max(1e-6, attCol.g), k),
    Math.pow(Math.max(1e-6, attCol.b), k),
  );
}

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
  /** Number of entries in the emitters / cdf arrays. */
  emitterCount: number;
  totalEmissivePower: number;
  /** Merged geometry (CPU side, for debug / re-upload). */
  mergedGeometry: THREE.BufferGeometry;
}

/**
 * EmitterTri struct layout (80 bytes, 16-byte aligned):
 *   0..11  : vertexA (12 bytes)
 *   12..23 : vertexB (12 bytes)
 *   24..35 : vertexC (12 bytes)
 *   36..47 : normal  (12 bytes)
 *   48..51 : area    ( 4 bytes)
 *   52..63 : color   (12 bytes)  ← r,g,b in f32
 *   64..67 : intensity (4 bytes)
 * Padded to 80 bytes (5 × vec4f) for 16-byte alignment.
 */
const EMITTER_STRIDE = 80; // bytes per emitter, 16-byte aligned
const EMITTER_FLOATS = EMITTER_STRIDE / 4; // 20 f32 per emitter

/**
 * Default warm-gray RGB fallback color for triangles with no resolvable
 * material color (≈ 0.60, 0.58, 0.55 linear). Used in two packing functions
 * (packBVHIndexW and packBVHBeerColors) — extracted to a single constant so
 * a palette change touches one place. (WARM complexity fix.)
 */
const WARM_GRAY_DEFAULT_R = 153;
const WARM_GRAY_DEFAULT_G = 148;
const WARM_GRAY_DEFAULT_B = 140;

/**
 * Default proxy-mesh allowlist for the stained-glass-app demo. Library
 * consumers will typically override via the `proxyMeshNames` opt to
 * supply their own dense-flat-surface mesh names. Empty Set = no
 * substitution (every visible mesh contributes its real geometry to
 * the BVH).
 */
const DEFAULT_PROXY_MESH_NAMES = new Set<string>([
  'surface_floor_living',
  'surface_ceiling_living',
]);

/** Build the full scene BVH + emitter list from a set of Object3D roots. */
export function buildSceneBVH(
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
     * Defaults to `DEFAULT_PROXY_MESH_NAMES`. Library consumers should
     * pass their own scene's mesh names.
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
    proxyMeshNames: options.proxyMeshNames ?? DEFAULT_PROXY_MESH_NAMES,
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
    shared.bvh.geometry.attributes['uv'] as THREE.BufferAttribute | undefined,
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
  const { emitterFloats, cdfArray, totalEmissivePower } = buildEmitterList(
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
  const uvAttr = shared.bvh.geometry.attributes['uv'] as
    | THREE.BufferAttribute
    | undefined;
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
      cpuData: emitterFloats.buffer,
      byteLength: emitterFloats.byteLength,
      count: emitterCount,
    },
    emitterCdf: {
      cpuData: cdfArray.buffer,
      byteLength: cdfArray.byteLength,
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

// ──────────────────────────────────────────────────────────────────────────────
// Per-engine sibling functions (technique-specific packing on top of shared core)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Pack UV (16-bit unorm pair) into the .w slot of every vec4f position.
 *
 * The shared `buildSceneBVH` returns positions at stride-4 with .w = 0.
 * We repack UV into the unused word so the WGSL shade pass can read both
 * position AND uv from a SINGLE storage buffer — avoiding the 8-buffer-
 * per-stage device limit on the compute cast.
 *
 *   .xyz = world-space position (12 bytes)
 *   .w   = packed UV (16-bit unorm pair: u in low 16 bits, v in high 16 bits;
 *          bitcasted to f32 for storage, decoded via `unpack2x16unorm`).
 *
 * UVs are wrapped to [0,1) via `frac` — the procedural texture functions in
 * shade.wgsl are periodic anyway, so any out-of-range UVs read a different
 * period.
 *
 * CRITICAL: WGSL `array<vec3<f32>>` has a 16-byte STRIDE per element (vec3f
 * has alignment 16, size 12 → stride = roundUp(16,12) = 16). The pre-fix
 * 12-byte stride layout garbled positions past index 1 (the visible
 * "scrambled geometry" symptom: gray walls, no recognisable floor / panel).
 */
function packUVIntoPositionW(
  positions: Float32Array,
  uvAttr: THREE.BufferAttribute | undefined,
  vertCount: number,
): Float32Array<ArrayBuffer> {
  // Defensive copy of the shared positions buffer — we are about to mutate
  // .w slots and the lib/ output may be aliased by subsequent callers.
  // Allocate a fresh ArrayBuffer-backed view so `.buffer` is concretely typed
  // ArrayBuffer (TS5.7+ propagates ArrayBufferLike from input typed arrays).
  const out = new Float32Array(positions.length);
  out.set(positions);
  const u32View = new Uint32Array(out.buffer);

  const sourceUvs = uvAttr
    ? new Float32Array(uvAttr.array)
    : new Float32Array(vertCount * 2);

  for (let i = 0; i < vertCount; i++) {
    let u = sourceUvs[i * 2 + 0]!;
    let v = sourceUvs[i * 2 + 1]!;
    // Wrap to [0, 1) — handles negative or >1 inputs cleanly.
    u = u - Math.floor(u);
    v = v - Math.floor(v);
    const u16 = Math.min(0xFFFF, Math.max(0, Math.round(u * 0xFFFF))) & 0xFFFF;
    const v16 = Math.min(0xFFFF, Math.max(0, Math.round(v * 0xFFFF))) & 0xFFFF;
    // pack2x16unorm: low 16 bits = .x (u), high 16 bits = .y (v).
    u32View[i * 4 + 3] = (v16 << 16) | u16;
  }
  return out;
}

/**
 * Pack vertex indices + RGBA8 baseColor + (trans4|texType4) into vec4u
 * per-triangle (4 u32 = 16 bytes per triangle).
 *
 * Bit layout of bvhIndex[*].w (32 bits):
 *   31..24 : R  (8-bit baseColor red channel)
 *   23..16 : G
 *   15..8  : B
 *    7..4  : transmission * 15 (4 bits — coarse but enough for the
 *             trans>0.625 / glass-vs-opaque test the shadow ray uses)
 *    3..0  : surfaceTextureId (0..15; 0..7 used today, 8..15 reserved)
 */
function packBVHIndexW(
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: THREE.Material[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  // Fresh ArrayBuffer-backed allocation (not derived from another typed
  // array) so `.buffer` is concretely ArrayBuffer for the StorageBufferHandle.
  const indexBuf = new Uint32Array(triCount * 4);

  for (let t = 0; t < triCount; t++) {
    const base4 = t * 4;
    indexBuf[base4 + 0] = indices[t * 3 + 0]!;
    indexBuf[base4 + 1] = indices[t * 3 + 1]!;
    indexBuf[base4 + 2] = indices[t * 3 + 2]!;

    const matId = triMaterialId[t]!;
    const mat = materials[matId];
    let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
    let transmission = 0;
    let texTypeId = 0;
    let isMetal = 0;
    if (mat) {
      const physMat = mat as THREE.MeshPhysicalMaterial;
      const stdMat  = mat as THREE.MeshStandardMaterial;
      transmission = (physMat.transmission ?? 0) as number;
      // ── Glass color resolution ──────────────────────────────────────────
      //
      // For textured cathedral glass `createBakedGlassMaterial` sets
      // `material.color = white(1,1,1)` when a baked `map` is present —
      // the cell's actual color is carried by the texture map AND by
      // `material.attenuationColor` (= baseColor; drives Beer-Lambert
      // absorption).  Reading `material.color` here packs WHITE into
      // bvhIndex[*].w for every glass triangle, producing an emitter list
      // whose Le for every glass cell collapsed to white regardless of cell
      // colour. For transmissive materials we therefore prefer `attenuationColor`.
      const isTransmissive = transmission > 0.01;
      const attenColor = (physMat as { attenuationColor?: THREE.Color }).attenuationColor;
      // bvhIndex.w holds the RAW attenuation color (no Beer-Lambert).
      // This is what receivers in the room sample via emitter Le and
      // bvhTraceTintedVisibility. The Beer-Lambert *visible* color
      // (what the shader uses for Lo_emit on a primary glass hit) lives
      // in a parallel bvh_beer buffer.
      const color =
        (isTransmissive && attenColor)
          ? attenColor
          : (physMat.color ?? stdMat?.color ?? new THREE.Color(0.6, 0.58, 0.55));
      r = Math.round(color.r * 255) & 0xFF;
      g = Math.round(color.g * 255) & 0xFF;
      b = Math.round(color.b * 255) & 0xFF;
      // Look up authored surfaceTexture id from the material's userData.
      // texTypeId now uses only 3 bits (0-7); bit 3 of the nybble is the
      // isMetal flag.
      const surfTex = (mat.userData as { surfaceTextureId?: number } | undefined)?.surfaceTextureId;
      texTypeId = (typeof surfTex === 'number' ? surfTex : 0) & 0x7;
      // Any non-zero metalness counts as "metal" for noisy-direct skip.
      // Threshold 1e-4 to avoid float-equality weirdness.
      const metalness = (stdMat?.metalness ?? 0) as number;
      isMetal = metalness > 1e-4 ? 1 : 0;
    }
    const trans4 = Math.min(15, Math.round(transmission * 15)) & 0xF;
    const lowByte = ((trans4 << 4) | (isMetal << 3) | (texTypeId & 0x7)) & 0xFF;
    indexBuf[base4 + 3] = (r << 24) | (g << 16) | (b << 8) | lowByte;
  }
  return indexBuf;
}

/**
 * Pack the Beer-Lambert visible color per triangle into a parallel u32 buffer.
 *
 * For transmissive materials: pow(attenuationColor, thickness/attenuationDistance)
 * mirrors PT's medium absorption at normal incidence. For opaque materials:
 * identical to the raw material color (no absorption to apply).
 *
 * Read by shade.wgsl Lo_emit on a primary glass hit. NOT used by:
 *   - emitter Le (emitter list keeps raw attCol so receivers in the
 *     room don't get 6× dimmed)
 *   - bvhTraceTintedVisibility (sun/sky paths through panels keep raw
 *     attCol for the same reason)
 *   - ReSTIR DI emitter sampling (uses emitter Le, not this buffer)
 */
function packBVHBeerColors(
  triMaterialId: Uint32Array,
  materials: THREE.Material[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const beerBuf = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    const matId = triMaterialId[t]!;
    const mat = materials[matId];
    let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
    if (mat) {
      const physMat = mat as THREE.MeshPhysicalMaterial;
      const stdMat  = mat as THREE.MeshStandardMaterial;
      const transmission = (physMat.transmission ?? 0) as number;
      const isTransmissive = transmission > 0.01;
      const attenColor = (physMat as { attenuationColor?: THREE.Color }).attenuationColor;
      const tinted = (isTransmissive && attenColor)
        ? applyBeerLambert(
            attenColor,
            (physMat as { thickness?: number }).thickness,
            (physMat as { attenuationDistance?: number }).attenuationDistance,
          )
        : null;
      const color =
        tinted
          ?? (physMat.color ?? stdMat?.color ?? new THREE.Color(0.6, 0.58, 0.55));
      r = Math.round(Math.min(1, color.r) * 255) & 0xFF;
      g = Math.round(Math.min(1, color.g) * 255) & 0xFF;
      b = Math.round(Math.min(1, color.b) * 255) & 0xFF;
    }
    beerBuf[t] = (r << 24) | (g << 16) | (b << 8);
  }
  return beerBuf;
}

// ──────────────────────────────────────────────────────────────────────────────
// Emitter list construction
// ──────────────────────────────────────────────────────────────────────────────

function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Build the EmitterTri list + power-CDF over the triangulated scene.
 *
 * Reads the shared BVH's stride-4 position + normal buffers + per-tri
 * materialId LUT directly — no merged-geometry dependency.
 *
 * Selection rules (in priority order per triangle):
 *   1. emissive (luminance > 0 AND emissiveIntensity > 0): treat as
 *      direct light source; intensity = emissiveIntensity.
 *   2. transmissive (transmission > 0.1) AND `userData.skipEmitter` not
 *      set: treat as a sun-attenuated secondary emitter when the primary
 *      light is roughly behind the glass (primaryLightDot > 0.05).
 *   3. otherwise: skip.
 *
 * Power = luminance(Le) × area; emitters with power < 1e-8 are dropped.
 * If the resulting list is empty, a synthetic dummy emitter is inserted
 * so the GPU buffer is non-empty (WGSL bind groups can't be size 0).
 */
function buildEmitterList(
  indices: Uint32Array,
  positions: Float32Array,    // stride-4: read .xyz only
  normals: Float32Array,      // stride-4: read .xyz only
  triMatIdMap: Uint32Array,
  materials: THREE.Material[],
  options: {
    primaryLightDir?: THREE.Vector3;
    primaryLightIntensity?: number;
  },
) {
  const triCount = indices.length / 3;

  // For each triangle, determine if it is emissive.
  const emitterData: {
    triIdx: number;
    vA: [number, number, number];
    vB: [number, number, number];
    vC: [number, number, number];
    normal: [number, number, number];
    area: number;
    color: [number, number, number];
    intensity: number;
    power: number;
  }[] = [];

  const _va = new THREE.Vector3();
  const _vb = new THREE.Vector3();
  const _vc = new THREE.Vector3();
  const _ab = new THREE.Vector3();
  const _ac = new THREE.Vector3();
  const _cross = new THREE.Vector3();

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3 + 0]!;
    const i1 = indices[t * 3 + 1]!;
    const i2 = indices[t * 3 + 2]!;

    // Stride-4 layout: .xyz at offsets [i*4..i*4+2]; .w is UV pack (ignored).
    _va.set(positions[i0 * 4]!, positions[i0 * 4 + 1]!, positions[i0 * 4 + 2]!);
    _vb.set(positions[i1 * 4]!, positions[i1 * 4 + 1]!, positions[i1 * 4 + 2]!);
    _vc.set(positions[i2 * 4]!, positions[i2 * 4 + 1]!, positions[i2 * 4 + 2]!);

    _ab.subVectors(_vb, _va);
    _ac.subVectors(_vc, _va);
    _cross.crossVectors(_ab, _ac);
    // Use the cross length directly + reuse for area.
    const crossLen = _cross.length();
    if (crossLen < 1e-8) continue;
    const area = crossLen * 0.5;
    const invLen = 1.0 / crossLen;
    let nx = _cross.x * invLen;
    let ny = _cross.y * invLen;
    let nz = _cross.z * invLen;
    // Stride-4 normals: read .xyz only.
    const n0x = normals[i0 * 4]!;
    const n0y = normals[i0 * 4 + 1]!;
    const n0z = normals[i0 * 4 + 2]!;
    const hasNormals = (n0x !== 0 || n0y !== 0 || n0z !== 0);
    if (hasNormals) {
      nx = (n0x + normals[i1 * 4]! + normals[i2 * 4]!) / 3;
      ny = (n0y + normals[i1 * 4 + 1]! + normals[i2 * 4 + 1]!) / 3;
      nz = (n0z + normals[i1 * 4 + 2]! + normals[i2 * 4 + 2]!) / 3;
      const nlen = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (nlen > 1e-6) { nx /= nlen; ny /= nlen; nz /= nlen; }
    }

    const matId = triMatIdMap[t]!;
    const mat = materials[matId];
    if (!mat) continue;

    let cr = 0, cg = 0, cb = 0, intensity = 1;

    // Check emissive materials.
    const meshMat = mat as THREE.MeshStandardMaterial;
    const emissiveLum = meshMat.emissive
      ? 0.2126 * meshMat.emissive.r + 0.7152 * meshMat.emissive.g + 0.0722 * meshMat.emissive.b
      : 0;
    if (emissiveLum > 0 && meshMat.emissiveIntensity && meshMat.emissiveIntensity > 0) {
      cr = meshMat.emissive.r * meshMat.emissiveIntensity;
      cg = meshMat.emissive.g * meshMat.emissiveIntensity;
      cb = meshMat.emissive.b * meshMat.emissiveIntensity;
      intensity = meshMat.emissiveIntensity;
    } else {
      // Check transmission (stained glass as emitter).
      const physMat = mat as THREE.MeshPhysicalMaterial;
      if (physMat.transmission && physMat.transmission > 0.1) {
        // ── Skip-emitter override (interior glass) ────────────────────────
        // Materials can opt out of emitter classification by stamping
        // userData.skipEmitter = true.  Used by interior glass that is
        // sealed inside the room enclosure (sun never reaches it) but
        // the buildEmitterList "transmission > 0.1 + primaryLightDot > 0.05"
        // gate would otherwise treat as a secondary emitter.
        const skipEmitter = (mat.userData as { skipEmitter?: boolean } | undefined)?.skipEmitter === true;
        if (skipEmitter) continue;

        // Face is transmissive. Treat it as a secondary emitter using
        // |cos(primaryLight, panel-normal-axis)| — bidirectional, because a
        // transmissive panel emits from BOTH faces (light entering one
        // side exits the other).
        const lightDir = options.primaryLightDir ?? new THREE.Vector3(0, 1, 0);
        const sunDot = Math.abs(lightDir.x * nx + lightDir.y * ny + lightDir.z * nz);

        if (sunDot > 0.05) {
          // Emitter Le uses RAW attenuationColor (no Beer-Lambert).
          // ReSTIR DI samples this Le when shading non-glass receivers;
          // applying Beer-Lambert here would dim every receiver pixel.
          const baseColor = physMat.color ?? new THREE.Color(1, 1, 1);
          const attenColor = physMat.attenuationColor ?? new THREE.Color(1, 1, 1);
          const sunIrradiance = options.primaryLightIntensity ?? 3.0;
          const trans = physMat.transmission;

          cr = baseColor.r * attenColor.r * trans * sunIrradiance * sunDot;
          cg = baseColor.g * attenColor.g * trans * sunIrradiance * sunDot;
          cb = baseColor.b * attenColor.b * trans * sunIrradiance * sunDot;
          intensity = sunIrradiance * trans * sunDot;
        }
      }
    }

    const power = luminance(cr, cg, cb) * area;
    if (power < 1e-8) continue;

    emitterData.push({
      triIdx: t,
      vA: [_va.x, _va.y, _va.z],
      vB: [_vb.x, _vb.y, _vb.z],
      vC: [_vc.x, _vc.y, _vc.z],
      normal: [nx, ny, nz],
      area,
      color: [cr, cg, cb],
      intensity,
      power,
    });
  }


  // If no emitters found, add a dummy one so the buffer is non-empty.
  if (emitterData.length === 0) {
    emitterData.push({
      triIdx: 0,
      vA: [0, 10, 0], vB: [1, 10, 0], vC: [0.5, 10, 1],
      normal: [0, -1, 0],
      area: 0.5,
      color: [1, 1, 1],
      intensity: 1,
      power: 0.5,
    });
  }

  // Pack emitters into Float32Array with EMITTER_STRIDE alignment.
  const emitterCount = emitterData.length;
  const emitterFloats = new Float32Array(emitterCount * EMITTER_FLOATS);
  let totalEmissivePower = 0;

  for (let i = 0; i < emitterCount; i++) {
    const e = emitterData[i]!;
    const base = i * EMITTER_FLOATS;
    emitterFloats[base + 0] = e.vA[0]; emitterFloats[base + 1] = e.vA[1]; emitterFloats[base + 2] = e.vA[2]; emitterFloats[base + 3] = 0;
    emitterFloats[base + 4] = e.vB[0]; emitterFloats[base + 5] = e.vB[1]; emitterFloats[base + 6] = e.vB[2]; emitterFloats[base + 7] = 0;
    emitterFloats[base + 8] = e.vC[0]; emitterFloats[base + 9] = e.vC[1]; emitterFloats[base + 10] = e.vC[2]; emitterFloats[base + 11] = 0;
    emitterFloats[base + 12] = e.normal[0]; emitterFloats[base + 13] = e.normal[1]; emitterFloats[base + 14] = e.normal[2]; emitterFloats[base + 15] = e.area;
    emitterFloats[base + 16] = e.color[0]; emitterFloats[base + 17] = e.color[1]; emitterFloats[base + 18] = e.color[2]; emitterFloats[base + 19] = e.intensity;
    totalEmissivePower += e.power;
  }

  // Build CDF over emitter power (for importance sampling in the RIS pass).
  const cdfArray = new Float32Array(emitterCount);
  let runningSum = 0;
  for (let i = 0; i < emitterCount; i++) {
    runningSum += emitterData[i]!.power;
    cdfArray[i] = runningSum / totalEmissivePower;
  }

  return { emitterFloats, cdfArray, totalEmissivePower };
}
