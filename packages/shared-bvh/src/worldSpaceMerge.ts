/**
 * worldSpaceMerge.ts — world-space merged tri-stream / BVH builder
 * from a `@vitrum/core` `Scene`.
 *
 * This is the canonical ingestion path for consumers that need WORLD-space
 * geometry instead of the local-space BLAS + separate TLAS matrices emitted by
 * `packSceneFromCore`:
 *
 *   1. the ReSTIR emitter list (triangle area / world face-normal / centroid /
 *      AABB — `walkaround-hybrid/restir/bvhCore.ts`),
 *   2. the DDGI merged BVH, and
 *   3. the RC merged BVH.
 *
 * `mergeWorldSpaceFromCore` iterates `scene.primitives`, transforms each
 * primitive's local `positions`/`normals` into world space by the core
 * `transform`, concatenates the streams, and builds one merged BVH via
 * `buildArrayBvh`.
 *
 * ── Parity with `buildSceneBVH` (the migration gate) ──────────────────────────
 *
 * The merged vertex stream (positions + normals, pre-BVH-reorder) is built from
 * core transforms:
 *   • positions  — `worldPos = transform · localPos` (raw 4×4).
 *   • normals    — `worldN = normalize(normalMatrix · localN)` where
 *     `normalMatrix` is the inverse-transpose of the upper-left 3×3.
 *   • winding    — when `det(transform) < 0`, the builder swaps v0↔v2 of every
 *     triangle so mirrored transforms keep front-facing winding.
 *   • order      — primitives are concatenated in `scene.primitives` order.
 *
 * To make both the "stream parity" and the "set / AABB / ray-query equivalence"
 * checkable, the result carries the pre-BVH-reorder merged stream
 * ({@link WorldSpaceMergeResult.mergedIndices} / `mergedTriMaterialId`) ALONGSIDE
 * the post-build BVH-ordered `indices` / `triMaterialId`.
 *
 */

import type { Mat4, MaterialSpec, Scene, SceneNodeId, ScenePrimitive, TextureRef } from '@vitrum/core';
import type { PlainAabb } from './aabb.js';
import { buildArrayBvh } from './buildArrayBvh.js';
import { maybeDisplaceMeshPositions } from './vertexDisplacement.js';

const IDENTITY_MAT4: readonly number[] = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

// ──────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────

/** Per-source-primitive (or per-instance) vertex range in the merged buffer.
 *  Mirrors `SceneBVHCommonResult.meshVertexRanges` (the THREE path) so the
 *  transform-only refit fast path can address one primitive's world vertices in
 *  place. `name` is the primitive id (for instanced primitives every instance
 *  shares the same id — the per-instance split is implicit in the contiguous
 *  ranges). */
export interface MergedMeshVertexRange {
  readonly name: string;
  readonly vertexStart: number;
  readonly vertexCount: number;
  readonly triStart: number;
  readonly triCount: number;
  readonly windingFlipped?: boolean;
}

export interface WorldSpaceMergeResult {
  /**
   * Packed 32-byte BVH-node buffer (single root), byte-layout-identical to
   * `buildSceneBVH`'s `bvhNodes` (8 × u32 / node, relative interior offsets,
   * leaf `0xFFFF0000 | count`). NOTE: the tree TOPOLOGY differs from
   * `buildSceneBVH` (different SAH builder — see module docstring R1); only the
   * per-node byte LAYOUT is shared.
   */
  readonly bvhNodes: Float32Array;

  /** Merged world-space positions. Stride = {@link positionStrideFloats}. */
  readonly positions: Float32Array;
  readonly positionStrideFloats: 3 | 4;

  /**
   * BVH-traversal-ordered triangle indices (3 u32 / triangle — stride 3, the
   * `array<vec3u>` form, same as `buildSceneBVH`). Permuted by the SAH build;
   * NOT the same permutation as `buildSceneBVH` (R1). For the unpermuted merge
   * order use {@link mergedIndices}.
   */
  readonly indices: Uint32Array;
  readonly bvhIndexStride: 3;

  /** Per-triangle materialId, BVH-reordered to match {@link indices}. */
  readonly triMaterialId: Uint32Array;

  /**
   * For each triangle in {@link indices} / {@link triMaterialId}, the triangle
   * index in the pre-BVH {@link mergedIndices} stream. This is required when a
   * world-space consumer needs to point back at another triangle-addressed atlas
   * whose ordering is based on merge order or local BLAS order.
   */
  readonly bvhTriToMergedTri: Uint32Array;

  /** Merged world-space normals (THREE normal-matrix + normalize). Stride =
   *  {@link positionStrideFloats}. */
  readonly normals: Float32Array;

  /** Merged world-space tangents. Stride = {@link positionStrideFloats};
   *  xyz is the transformed tangent direction, w is the bitangent handedness.
   *  Vertices without authored/generated tangents are encoded as 0,0,0,0 so
   *  consumers can fall back to a UV-gradient frame. */
  readonly tangents: Float32Array;

  /** Merged per-vertex colors, vec4f stride (rgba). Missing colors are encoded
   *  as 1,1,1,1 so downstream material paths can multiply without a presence bit. */
  readonly colors: Float32Array;

  /** Merged per-vertex texture coords (stride 2, merge/vertex order — same order as
   *  {@link positions}/{@link normals}; the BVH reorders triangles not vertices).
   *  (0,0) for vertices whose source primitive carries no UVs. */
  readonly uvs: Float32Array;

  /**
   * Pre-BVH-reorder merged triangle indices (3 u32 / triangle, MERGE order —
   * the order `StaticGeometryGenerator` concatenates, before any SAH permute).
   * This is the order-deterministic stream the golden-parity test pins against
   * `buildSceneBVH` (whose `positions`/`normals` are likewise never reordered by
   * the BVH build). Emitter-list consumers that want the merge order (not the
   * BVH order) read this.
   */
  readonly mergedIndices: Uint32Array;

  /** Per-triangle materialId in {@link mergedIndices} (merge) order. */
  readonly mergedTriMaterialId: Uint32Array;

  /**
   * Deduped source materials in first-seen primitive order — the core
   * `MaterialSpec[]` counterpart to `buildSceneBVH`'s `THREE.Material[]`.
   * `triMaterialId` / `mergedTriMaterialId` index into this list. Dedup is by
   * the same structural signature `snapshotPreBuildMaterials` uses (so React/R3F
   * material churn collapses to the structural minimum — see {@link materialSig}).
   */
  readonly materials: readonly MaterialSpec[];

  /** World-space AABB of the merged geometry (THREE-free `{min,max}` in place of
   *  `THREE.Box3`). Float-identical to `buildSceneBVH`'s `boundingBox` for an
   *  equivalent scene. */
  readonly boundingBox: PlainAabb;

  /** Per-source-primitive vertex ranges (mirrors the THREE path). */
  readonly meshVertexRanges: readonly MergedMeshVertexRange[];

  /** Non-fatal ingestion warnings, including skipped unreadable vertex displacement maps. */
  readonly warnings: readonly string[];

  /** Vertex count in the merged buffer (positions.length / stride). */
  readonly vertexCount: number;
  /** Triangle count in the merged buffer. */
  readonly triangleCount: number;
}

export interface WorldSpaceMergeOptions {
  /**
   * Position / normal buffer element stride in floats:
   *   4 → 16-byte vec3f-aligned layout (default; the WGSL `array<vec3f>` form
   *       the DDGI/RC merged-BVH shaders read, matching
   *       `buildSceneBVH({positionStride:4})`).
   *   3 → 12-byte packed layout (raster/TSL; matches `buildSceneBVH`'s default).
   */
  readonly positionStride?: 3 | 4;

  /**
   * Caller-supplied primitive filter — returns true for primitives that should
   * contribute to the merged BVH. Defaults to {@link DEFAULT_MERGE_FILTER}
   * (every mesh-like primitive: mesh / instanced-mesh / skinned-mesh; analytic
   * primitives are skipped — they have no triangle stream). Mirrors
   * `buildSceneBVH`'s mesh filter ("any mesh with a position attribute").
   */
  readonly filter?: (primitive: ScenePrimitive) => boolean;

  /**
   * Include the primitive-level `castShadow` flag in material deduplication.
   * Default false preserves the historical GI/merged-BVH behavior where shadow
   * participation is not represented in material slots. Backends with a
   * material-texture shadow bit can opt in so otherwise-identical materials
   * remain distinct when one primitive disables shadow casting.
   */
  readonly splitMaterialsByCastShadow?: boolean;

  /**
   * Fold a primitive-wide constant vertex RGB multiplier into its material
   * `baseColor` while preserving arbitrary/non-uniform vertex colors in the
   * merged color stream. Compatibility-tier renderers can use this to render
   * the common "one COLOR_0 tint for the whole primitive" case without adding a
   * per-vertex color binding.
   */
  readonly bakeConstantVertexColorIntoMaterial?: boolean;

  /** Optional warning sink. Warnings are also returned on {@link WorldSpaceMergeResult.warnings}. */
  readonly onWarning?: (warning: string) => void;
}

// ──────────────────────────────────────────────────────────────────────────
// Mesh-like predicate + default filter
// ──────────────────────────────────────────────────────────────────────────

type MeshLikePrimitive = Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }
>;

function isMeshLike(primitive: ScenePrimitive): primitive is MeshLikePrimitive {
  return (
    primitive.kind === 'mesh' ||
    primitive.kind === 'skinned-mesh' ||
    primitive.kind === 'instanced-mesh'
  );
}

/** Default merge filter — every mesh-like primitive contributes (analytic
 *  primitives have no triangle stream and are skipped). Mirrors the DDGI mesh
 *  filter ("any visible mesh with a position attribute"). */
export const DEFAULT_MERGE_FILTER = (primitive: ScenePrimitive): boolean =>
  isMeshLike(primitive);

const MAX_WORLD_MERGE_FILTER_WARNINGS = 10;

function constantVertexRgbMultiplier(primitive: MeshLikePrimitive): readonly [number, number, number] | null {
  const colors = primitive.colors;
  if (colors == null || colors.length === 0) return null;
  const vertexCount = Math.floor(primitive.positions.length / 3);
  const stride = colors.length >= vertexCount * 4
    ? 4
    : colors.length >= vertexCount * 3
      ? 3
      : 0;
  if (vertexCount === 0 || stride === 0) return null;
  const r = colors[0] ?? 1;
  const g = colors[1] ?? 1;
  const b = colors[2] ?? 1;
  const eps = 1e-6;
  for (let i = 0; i < vertexCount; i += 1) {
    const o = i * stride;
    if (
      Math.abs((colors[o] ?? 1) - r) > eps ||
      Math.abs((colors[o + 1] ?? 1) - g) > eps ||
      Math.abs((colors[o + 2] ?? 1) - b) > eps
    ) {
      return null;
    }
    if (stride === 4 && Math.abs((colors[o + 3] ?? 1) - 1) > eps) {
      return null;
    }
  }
  return [r, g, b];
}

function bakeConstantVertexColorIntoMaterial(
  material: MaterialSpec,
  primitive: MeshLikePrimitive,
): MaterialSpec {
  const multiplier = constantVertexRgbMultiplier(primitive);
  if (multiplier == null) return material;
  const base = material.baseColor;
  const baked: [number, number, number] = [
    (base[0] ?? 0) * multiplier[0],
    (base[1] ?? 0) * multiplier[1],
    (base[2] ?? 0) * multiplier[2],
  ];
  if (
    Math.abs(baked[0] - (base[0] ?? 0)) <= 1e-6 &&
    Math.abs(baked[1] - (base[1] ?? 0)) <= 1e-6 &&
    Math.abs(baked[2] - (base[2] ?? 0)) <= 1e-6
  ) {
    return material;
  }
  return { ...material, baseColor: baked };
}

// ──────────────────────────────────────────────────────────────────────────
// World transforms — bit-for-bit mirrors of THREE / StaticGeometryGenerator
// ──────────────────────────────────────────────────────────────────────────

/**
 * `worldPos = m · localPos` for a column-major 4×4 `m` (= THREE's
 * `Vector3.applyMatrix4`, including the perspective divide). Identical
 * arithmetic to `scenePack.ts:transformPoint` BUT keeps the THREE `w`-divide
 * convention (`w = 1/(…); xyz *= w`) so the f32 round-off matches SGG exactly.
 */
function applyMatrix4(m: ArrayLike<number>, x: number, y: number, z: number): [number, number, number] {
  const e0 = m[0] ?? 0, e4 = m[4] ?? 0, e8 = m[8] ?? 0, e12 = m[12] ?? 0;
  const e1 = m[1] ?? 0, e5 = m[5] ?? 0, e9 = m[9] ?? 0, e13 = m[13] ?? 0;
  const e2 = m[2] ?? 0, e6 = m[6] ?? 0, e10 = m[10] ?? 0, e14 = m[14] ?? 0;
  const e3 = m[3] ?? 0, e7 = m[7] ?? 0, e11 = m[11] ?? 0, e15 = m[15] ?? 0;
  const w = 1 / (e3 * x + e7 * y + e11 * z + e15);
  return [
    (e0 * x + e4 * y + e8 * z + e12) * w,
    (e1 * x + e5 * y + e9 * z + e13) * w,
    (e2 * x + e6 * y + e10 * z + e14) * w,
  ];
}

function finiteVec3(v: readonly [number, number, number]): boolean {
  return Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

/**
 * Normal matrix (upper-left 3×3 inverse-transpose) of a column-major 4×4, as a
 * length-9 row-major-by-THREE-convention array — bit-for-bit THREE's
 * `Matrix3.getNormalMatrix(m4) = setFromMatrix4(m4).invert().transpose()`.
 *
 * `setFromMatrix4` lays out the upper-left 3×3 of the column-major m4 into the
 * Matrix3 as: n11=m[0], n21=m[1], n31=m[2], n12=m[4], n22=m[5], n32=m[6],
 * n13=m[8], n23=m[9], n33=m[10] (so te = [m0,m1,m2, m4,m5,m6, m8,m9,m10]).
 * `invert()` then writes te' via THREE's exact cofactor expressions, and
 * `transpose()` swaps the off-diagonal pairs. We fold invert+transpose into the
 * final element assignment so there's a single, mirror-able expression set.
 */
function getNormalMatrix3(m: ArrayLike<number>): Float64Array {
  // setFromMatrix4(m4) → Matrix3 elements (te):
  const n11 = m[0] ?? 0, n21 = m[1] ?? 0, n31 = m[2] ?? 0;
  const n12 = m[4] ?? 0, n22 = m[5] ?? 0, n32 = m[6] ?? 0;
  const n13 = m[8] ?? 0, n23 = m[9] ?? 0, n33 = m[10] ?? 0;

  // invert(): THREE's exact cofactor formula (Matrix3.invert).
  const t11 = n33 * n22 - n32 * n23;
  const t12 = n32 * n13 - n33 * n12;
  const t13 = n23 * n12 - n22 * n13;
  const det = n11 * t11 + n21 * t12 + n31 * t13;

  const inv = new Float64Array(9);
  if (det === 0) {
    // THREE sets all nine elements to 0 on a singular matrix.
    return inv;
  }
  const detInv = 1 / det;
  // inv = te' (the inverted Matrix3), THREE's exact element assignments.
  inv[0] = t11 * detInv;
  inv[1] = (n31 * n23 - n33 * n21) * detInv;
  inv[2] = (n32 * n21 - n31 * n22) * detInv;
  inv[3] = t12 * detInv;
  inv[4] = (n33 * n11 - n31 * n13) * detInv;
  inv[5] = (n31 * n12 - n32 * n11) * detInv;
  inv[6] = t13 * detInv;
  inv[7] = (n21 * n13 - n23 * n11) * detInv;
  inv[8] = (n22 * n11 - n21 * n12) * detInv;

  // transpose() in place: swap (1,3),(2,6),(5,7).
  const out = new Float64Array(9);
  out[0] = inv[0]!; out[4] = inv[4]!; out[8] = inv[8]!;
  out[1] = inv[3]!; out[3] = inv[1]!;
  out[2] = inv[6]!; out[6] = inv[2]!;
  out[5] = inv[7]!; out[7] = inv[5]!;
  return out;
}

/**
 * `worldN = normalize( normalMatrix · localN )` — bit-for-bit THREE's
 * `Vector3.applyNormalMatrix(m3) = applyMatrix3(m3).normalize()`. `nm` is the
 * length-9 Matrix3 from {@link getNormalMatrix3}; `applyMatrix3` reads it as
 * `x' = nm0·x + nm3·y + nm6·z`, etc. (THREE's `Vector3.applyMatrix3`).
 */
function applyNormalMatrix(
  nm: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const nx = (nm[0] ?? 0) * x + (nm[3] ?? 0) * y + (nm[6] ?? 0) * z;
  const ny = (nm[1] ?? 0) * x + (nm[4] ?? 0) * y + (nm[7] ?? 0) * z;
  const nz = (nm[2] ?? 0) * x + (nm[5] ?? 0) * y + (nm[8] ?? 0) * z;
  // THREE's Vector3.normalize: divide by length, or by 1 when length 0.
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  return [nx / len, ny / len, nz / len];
}

/** Transform a tangent/vector direction by the upper-left 3×3 of a column-major
 * matrix and normalize it. Unlike normals, tangents are ordinary directions, so
 * they use the direct linear transform rather than the inverse-transpose. */
function applyDirectionMatrix4(
  m: ArrayLike<number>,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const tx = (m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z;
  const ty = (m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z;
  const tz = (m[2] ?? 0) * x + (m[6] ?? 0) * y + (m[10] ?? 0) * z;
  const len = Math.sqrt(tx * tx + ty * ty + tz * tz);
  if (len <= 1e-12) return [0, 0, 0];
  return [tx / len, ty / len, tz / len];
}

/** Full 4×4 determinant of a column-major matrix — = THREE's
 *  `Matrix4.determinant()` (used to detect the winding flip). */
function determinant4(m: ArrayLike<number>): number {
  const n11 = m[0] ?? 0, n12 = m[4] ?? 0, n13 = m[8] ?? 0, n14 = m[12] ?? 0;
  const n21 = m[1] ?? 0, n22 = m[5] ?? 0, n23 = m[9] ?? 0, n24 = m[13] ?? 0;
  const n31 = m[2] ?? 0, n32 = m[6] ?? 0, n33 = m[10] ?? 0, n34 = m[14] ?? 0;
  const n41 = m[3] ?? 0, n42 = m[7] ?? 0, n43 = m[11] ?? 0, n44 = m[15] ?? 0;
  return (
    n41 * (
      +n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 +
      n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34
    ) +
    n42 * (
      +n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 -
      n13 * n21 * n34 + n13 * n24 * n31 - n14 * n23 * n31
    ) +
    n43 * (
      +n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32 +
      n12 * n21 * n34 + n14 * n22 * n31 - n12 * n24 * n31
    ) +
    n44 * (
      -n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 +
      n13 * n21 * n32 - n12 * n21 * n33 + n12 * n23 * n31
    )
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Material dedup — mirrors snapshotPreBuildMaterials' value-dedup signature
// ──────────────────────────────────────────────────────────────────────────

type TextureMapField = Extract<{
  [K in keyof MaterialSpec]: MaterialSpec[K] extends TextureRef | undefined ? K : never;
}[keyof MaterialSpec], string>;

const TEXTURE_MAP_FIELDS: readonly TextureMapField[] = [
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'displacementMap',
  'lightMap',
];

function finiteSig(value: number | undefined, fallback: number): string {
  const v = Number.isFinite(value) ? (value as number) : fallback;
  return String(Math.fround(v));
}

function rawNumberSig(value: number | undefined, fallback: number): string {
  if (value === undefined) return finiteSig(undefined, fallback);
  return Number.isFinite(value) ? String(Math.fround(value)) : String(value);
}

function vecSig(
  value: readonly number[] | undefined,
  fallback: readonly number[],
  count: 2 | 3,
): string {
  const parts: string[] = [];
  for (let i = 0; i < count; i += 1) {
    parts.push(finiteSig(value?.[i], fallback[i] ?? 0));
  }
  return parts.join(',');
}

function rawVecSig(
  value: readonly number[] | undefined,
  fallback: readonly number[],
  count: 2,
): string {
  const parts: string[] = [];
  for (let i = 0; i < count; i += 1) {
    parts.push(rawNumberSig(value?.[i], fallback[i] ?? 0));
  }
  return parts.join(',');
}

function textureRefLike(value: unknown): TextureRef | undefined {
  if (value == null || typeof value !== 'object') return undefined;
  if ('handle' in value) return value as TextureRef;
  return { handle: value };
}

function textureTexCoordSig(texCoord: number | undefined): string {
  const uv = texCoord ?? 0;
  if (uv === 0 || uv === 1) return `uv${uv}`;
  return `uvUnsupported=${rawNumberSig(uv, 0)}`;
}

function textureRefSig(value: unknown): string {
  const ref = textureRefLike(value);
  if (ref?.handle == null) return '';
  const transform = ref.transform;
  return [
    handleId(ref.handle),
    textureTexCoordSig(ref.texCoord),
    `off=${rawVecSig(transform?.offset, [0, 0], 2)}`,
    `scale=${rawVecSig(transform?.scale, [1, 1], 2)}`,
    `rot=${rawNumberSig(transform?.rotation, 0)}`,
    `wrap=${ref.wrapS ?? 'repeat'},${ref.wrapT ?? 'repeat'}`,
    `filter=${ref.magFilter ?? ''},${ref.minFilter ?? ''},${ref.mipFilter ?? ''}`,
  ].join(';');
}

function textureMapSig(m: MaterialSpec): string {
  return TEXTURE_MAP_FIELDS
    .map((field) => `${field}=${textureRefSig(m[field])}`)
    .join('|');
}

function stableJsonSig(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(4) : String(value);
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJsonSig).join(',')}]`;
  if (typeof value === 'object') {
    return `{${Object.keys(value as Record<string, unknown>)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJsonSig((value as Record<string, unknown>)[key])}`)
      .join(',')}}`;
  }
  return String(value);
}

/**
 * Structural material signature — the core `MaterialSpec` counterpart to
 * `legacy/bvhCommon.ts:snapshotPreBuildMaterials`'s `matSig`. Hashes the fields
 * the merged-BVH GI/PT consumers read: base PBR/alpha/transmission scalars,
 * lobe-extension scalars, Beer-Lambert fields, all packed texture-map refs
 * including handle identity + UV transform/sampler metadata, and pt-webgl2's
 * folded mesh-emitter shadow flag. Numeric tokens use the same Float32 precision
 * as the GPU payloads, so atlas/material metadata differences that survive upload
 * cannot be rounded away by the dedup key. Map identity uses the opaque
 * `TextureRef.handle` (the core analogue of THREE's `texture.uuid`).
 *
 * Beer-Lambert fields (`attenuationColor`, `attenuationDistance`, `thickness`) are
 * included to match `materialSetHashFloats` in `sceneBvh.ts`. Infinity
 * `attenuationDistance` is normalised to the token `'Inf'` so the signature remains
 * a stable string (JSON.stringify of Infinity produces `null`).
 */
export function materialSig(m: MaterialSpec): string {
  const colS = vecSig(m.baseColor, [0, 0, 0], 3);
  const emS = vecSig(m.emissive, [0, 0, 0], 3);
  // Beer-Lambert fields — must match materialSetHashFloats in sceneBvh.ts.
  const acS = vecSig(m.attenuationColor, [1, 1, 1], 3);
  const adRaw = m.attenuationDistance;
  const adS = adRaw == null
    ? 'Inf'
    : !isFinite(adRaw)
      ? 'Inf'
      : adRaw.toFixed(4);
  const meshEmitterShadow = (m as MaterialSpec & {
    meshEmitterCastShadowDisabled?: boolean;
  }).meshEmitterCastShadowDisabled === true ? '1' : '0';
  return [
    `base=${colS}`,
    `em=${emS}`,
    `emI=${finiteSig(m.emissiveIntensity, 1)}`,
    `rough=${finiteSig(m.roughness, 0.5)}`,
    `metal=${finiteSig(m.metallic, 0)}`,
    `shade=${m.shadingModel ?? 'pbr'}`,
    `alpha=${m.alphaMode ?? 'opaque'},${finiteSig(m.alphaCutoff, 0.5)},${finiteSig(m.opacity, 1)}`,
    `trans=${finiteSig(m.transmission, 0)}`,
    `ior=${finiteSig(m.ior, 1.5)}`,
    `beer=${acS},${adS},${finiteSig(m.thickness, 0)}`,
    `mapScalar=${finiteSig(m.normalScale, 1)},${finiteSig(m.clearcoatNormalScale, 1)},${finiteSig(m.aoMapIntensity, 1)},${finiteSig(m.bumpScale, 1)},${finiteSig(m.lightMapIntensity, 1)},${finiteSig(m.envMapIntensity, 1)}`,
    `spec=${vecSig(m.specularColor, [1, 1, 1], 3)},${finiteSig(m.specularIntensity, 1)}`,
    `coatSheen=${finiteSig(m.clearcoat, 0)},${finiteSig(m.clearcoatRoughness, 0)},${finiteSig(m.sheen, 0)},${vecSig(m.sheenColor, [0, 0, 0], 3)},${finiteSig(m.sheenRoughness, 0)}`,
    `aniso=${finiteSig(m.anisotropy, 0)},${finiteSig(m.anisotropyRotation, 0)}`,
    `iridescence=${finiteSig(m.iridescence, 0)},${finiteSig(m.iridescenceIor, 1.3)},${vecSig(m.iridescenceThicknessRange, [100, 400], 2)}`,
    `reservedDisp=${textureRefSig(m.displacementMap)},${finiteSig(m.displacementScale, 1)},${finiteSig(m.displacementBias, 0)}`,
    `volume=${finiteSig(m.scatteringCoefficient, 0)},${finiteSig(m.scatteringAnisotropy, 0)},${vecSig(m.scatteringCoefficientRGB, [0, 0, 0], 3)}`,
    `spectral=${stableJsonSig(m.spectralAttenuation)},${finiteSig(m.dispersionAbbeNumber, 0)}`,
    `layers=${stableJsonSig(m.frontLayer)},${stableJsonSig(m.backLayer)},${stableJsonSig(m.thinFilmStack)}`,
    `maps=${textureMapSig(m)}`,
    `meshEmitterShadow=${meshEmitterShadow}`,
  ].join('|');
}

/**
 * Stable per-object identity registry for the material dedup signature.
 *
 * The module-level WeakMap is LOAD-BEARING: handle identity must persist
 * ACROSS merge calls so the same object (e.g. the same decoded ImageBitmap)
 * always maps to the same signature token. Wrapping it in an exported object
 * enables test-time reset without exposing the raw module globals.
 *
 * `reset()` is intentionally not called in production — it would invalidate
 * cached signatures and break dedup continuity. Call it only in test teardown
 * to prevent cross-test object-identity bleed.
 */
export const HandleIdRegistry = {
  _ids: new WeakMap<object, string>(),
  _seq: 0,
  /** Return the stable id for `handle`. Assigns a new one on first encounter. */
  get(handle: object): string {
    let id = this._ids.get(handle);
    if (id === undefined) {
      id = `h${this._seq++}`;
      this._ids.set(handle, id);
    }
    return id;
  },
  /**
   * Reset the registry — for TEST USE ONLY.
   * Clears all assigned ids and resets the sequence counter.
   * Do NOT call in production: existing handle ids become stale.
   */
  reset(): void {
    this._ids = new WeakMap();
    this._seq = 0;
  },
};

/** A stable per-handle identity string for the dedup signature. Objects use
 *  {@link HandleIdRegistry}; primitives stringify directly; absent handles
 *  contribute the empty string. */
function handleId(handle: unknown): string {
  if (handle == null) return '';
  if (typeof handle === 'object' || typeof handle === 'function') {
    return HandleIdRegistry.get(handle);
  }
  // eslint-disable-next-line @typescript-eslint/no-base-to-string -- at this point handle is a primitive (guarded: not object/function), String() is safe
  return String(handle);
}

// ──────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────

/**
 * Build a single merged WORLD-space tri stream + BVH from a `@vitrum/core`
 * `Scene` — the THREE-free analogue of `buildSceneBVH` for the merged-BVH
 * consumers (DDGI / RC / ReSTIR emitter list). See the module docstring for the
 * exact parity contract with `buildSceneBVH` (stream float-parity; BVH-topology
 * divergence by R1).
 *
 * @param scene a `@vitrum/core` `Scene`.
 * @param opts  see {@link WorldSpaceMergeOptions}.
 */
export function mergeWorldSpaceFromCore(
  scene: Scene,
  opts: WorldSpaceMergeOptions = {},
): WorldSpaceMergeResult {
  const stride = opts.positionStride ?? 4;
  const filter = opts.filter ?? DEFAULT_MERGE_FILTER;

  // Accumulators for the merged world-space stream (merge order — NOT yet
  // BVH-reordered). `mergedIndices` is stride-3 (3 u32 / triangle).
  const positions: number[] = [];
  const normals: number[] = [];
  const tangents: number[] = [];
  const colors: number[] = [];
  const uvs: number[] = []; // per-vertex texture coords (stride 2, merge order, untransformed)
  const mergedIndices: number[] = [];
  const mergedTriMaterialId: number[] = [];
  const meshVertexRanges: MergedMeshVertexRange[] = [];
  const warnings: string[] = [];
  const warn = (warning: string): void => {
    warnings.push(warning);
    opts.onWarning?.(warning);
  };

  // Material value-dedup LUT (mirrors snapshotPreBuildMaterials).
  const materials: MaterialSpec[] = [];
  const sigToSlot = new Map<string, number>();

  const resolveMaterialSlot = (primitive: MeshLikePrimitive): number => {
    const mat = opts.bakeConstantVertexColorIntoMaterial
      ? bakeConstantVertexColorIntoMaterial(primitive.material, primitive)
      : primitive.material;
    const castShadow = primitive.castShadow ?? true;
    const sig = opts.splitMaterialsByCastShadow
      ? `${materialSig(mat)}|castShadow=${castShadow ? 1 : 0}`
      : materialSig(mat);
    const existing = sigToSlot.get(sig);
    if (existing !== undefined) return existing;
    const slot = materials.length;
    sigToSlot.set(sig, slot);
    materials.push(
      opts.splitMaterialsByCastShadow
        ? ({ ...mat, castShadow } as MaterialSpec)
        : mat,
    );
    return slot;
  };

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let filteredTriangleWarnCount = 0;
  let filteredTriangleCount = 0;

  const warnFilteredTriangle = (primitiveId: SceneNodeId, tri: number, reason: string): void => {
    filteredTriangleCount += 1;
    if (filteredTriangleWarnCount < MAX_WORLD_MERGE_FILTER_WARNINGS) {
      console.warn(
        `[@vitrum/shared-bvh/mergeWorldSpaceFromCore] Primitive "${primitiveId}" triangle ${tri} ${reason}; ` +
        'filtering it from the merged world-space BVH.',
      );
      filteredTriangleWarnCount += 1;
    }
  };

  for (const primitive of scene.primitives) {
    if (!filter(primitive) || !isMeshLike(primitive)) continue;

    const matSlot = resolveMaterialSlot(primitive);

    // For an instanced-mesh, world-merge every instance (one baked copy per
    // transform — the faithful world-space expansion). mesh / skinned-mesh have
    // exactly one transform (`transform` or identity). This matches what a
    // world-baked merge of the equivalent THREE scene produces: N baked
    // InstancedMesh copies (the ReSTIR emitter path deliberately EXCLUDES
    // instanced meshes from its world merge — callers that need that exclusion
    // pass a `filter` that rejects `kind === 'instanced-mesh'`).
    const transforms: ReadonlyArray<Mat4 | undefined> =
      primitive.kind === 'instanced-mesh'
        ? primitive.instances
        : [primitive.transform];

    const basePositions = primitive.positions;
    const baseNormals = primitive.normals;
    const baseTangents = primitive.tangents;
    const baseColors = primitive.colors;
    const baseUvs = primitive.uvs; // optional; (0,0) per vertex when absent
    const baseUv1 = primitive.uv1;
    const localVertexCount = Math.floor(basePositions.length / 3);
    const hasCompleteTangents = baseTangents != null && baseTangents.length >= localVertexCount * 4;
    const colorStride = baseColors != null && baseColors.length >= localVertexCount * 4
      ? 4
      : baseColors != null && baseColors.length >= localVertexCount * 3
        ? 3
        : 0;
    if (localVertexCount < 3) continue;
    const sourcePositions = maybeDisplaceMeshPositions({
      primitiveId: primitive.id,
      material: primitive.material,
      positions: basePositions,
      normals: baseNormals,
      ...(baseUvs != null ? { uvs: baseUvs } : {}),
      ...(baseUv1 != null ? { uv1: baseUv1 } : {}),
      onWarning: warn,
    }) ?? basePositions;

    // Sequential index when the primitive carries none (triangle-list), matching
    // `packOneMeshLikePrimitive` / SGG's index synthesis.
    const baseIndices: ArrayLike<number> =
      primitive.indices ??
      (() => {
        const gen = new Uint32Array(localVertexCount);
        for (let i = 0; i < localVertexCount; i += 1) gen[i] = i;
        return gen;
      })();
    const localTriCount = Math.floor(baseIndices.length / 3);
    if (localTriCount === 0) continue;

    for (const transform of transforms) {
      const m = transform ?? IDENTITY_MAT4;
      const normalMatrix = getNormalMatrix3(m);
      const flip = determinant4(m) < 0;

      const validTriangles: Array<readonly [number, number, number]> = [];
      for (let t = 0; t < localTriCount; t += 1) {
        const a = baseIndices[t * 3] ?? 0;
        const b = baseIndices[t * 3 + 1] ?? 0;
        const c = baseIndices[t * 3 + 2] ?? 0;
        if (a >= localVertexCount || b >= localVertexCount || c >= localVertexCount) {
          warnFilteredTriangle(
            primitive.id,
            t,
            `references an out-of-range vertex index (i0=${a}, i1=${b}, i2=${c}; vertexCount=${localVertexCount})`,
          );
          continue;
        }
        const pa = applyMatrix4(m, sourcePositions[a * 3] ?? 0, sourcePositions[a * 3 + 1] ?? 0, sourcePositions[a * 3 + 2] ?? 0);
        const pb = applyMatrix4(m, sourcePositions[b * 3] ?? 0, sourcePositions[b * 3 + 1] ?? 0, sourcePositions[b * 3 + 2] ?? 0);
        const pc = applyMatrix4(m, sourcePositions[c * 3] ?? 0, sourcePositions[c * 3 + 1] ?? 0, sourcePositions[c * 3 + 2] ?? 0);
        if (!finiteVec3(pa) || !finiteVec3(pb) || !finiteVec3(pc)) {
          warnFilteredTriangle(
            primitive.id,
            t,
            'has a non-finite transformed vertex coordinate (NaN or Inf)',
          );
          continue;
        }
        validTriangles.push(flip ? [c, b, a] : [a, b, c]);
      }
      if (validTriangles.length === 0) continue;

      const vertexStart = Math.floor(positions.length / stride);
      const triStart = Math.floor(mergedTriMaterialId.length);

      // Transform + append this instance's vertices (world space).
      for (let i = 0; i < localVertexCount; i += 1) {
        const lx = sourcePositions[i * 3] ?? 0;
        const ly = sourcePositions[i * 3 + 1] ?? 0;
        const lz = sourcePositions[i * 3 + 2] ?? 0;
        const [wx, wy, wz] = applyMatrix4(m, lx, ly, lz);
        positions.push(wx, wy, wz);
        if (stride === 4) positions.push(0);

        const lnx = baseNormals[i * 3] ?? 0;
        const lny = baseNormals[i * 3 + 1] ?? 1;
        const lnz = baseNormals[i * 3 + 2] ?? 0;
        const [nx, ny, nz] = applyNormalMatrix(normalMatrix, lnx, lny, lnz);
        normals.push(nx, ny, nz);
        if (stride === 4) normals.push(0);

        if (hasCompleteTangents) {
          const ltx = baseTangents[i * 4] ?? 0;
          const lty = baseTangents[i * 4 + 1] ?? 0;
          const ltz = baseTangents[i * 4 + 2] ?? 0;
          const handedness = (baseTangents[i * 4 + 3] ?? 1) * (flip ? -1 : 1);
          const [tx, ty, tz] = applyDirectionMatrix4(m, ltx, lty, ltz);
          tangents.push(tx, ty, tz);
          if (stride === 4) tangents.push(handedness);
        } else {
          tangents.push(0, 0, 0);
          if (stride === 4) tangents.push(0);
        }

        if (colorStride > 0) {
          const src = i * colorStride;
          colors.push(
            baseColors?.[src] ?? 1,
            baseColors?.[src + 1] ?? 1,
            baseColors?.[src + 2] ?? 1,
            colorStride === 4 ? baseColors?.[src + 3] ?? 1 : 1,
          );
        } else {
          colors.push(1, 1, 1, 1);
        }

        // UVs are 2D texture coords — transform-invariant (no world-matrix applied).
        uvs.push(baseUvs?.[i * 2] ?? 0, baseUvs?.[i * 2 + 1] ?? 0);

        if (Number.isFinite(wx) && Number.isFinite(wy) && Number.isFinite(wz)) {
          if (wx < minX) minX = wx; if (wx > maxX) maxX = wx;
          if (wy < minY) minY = wy; if (wy > maxY) maxY = wy;
          if (wz < minZ) minZ = wz; if (wz > maxZ) maxZ = wz;
        }
      }

      // Append this instance's triangles (global vertex refs = local + start),
      // applying the negative-determinant winding flip (v0↔v2) when needed —
      // mirroring StaticGeometryGenerator's `invertGeometry`.
      for (const [a, b, c] of validTriangles) {
        mergedIndices.push(a + vertexStart, b + vertexStart, c + vertexStart);
        mergedTriMaterialId.push(matSlot);
      }

      meshVertexRanges.push({
        name: primitive.id,
        vertexStart,
        vertexCount: localVertexCount,
        triStart,
        triCount: validTriangles.length,
        windingFlipped: flip,
      });
    }
  }

  if (filteredTriangleCount > MAX_WORLD_MERGE_FILTER_WARNINGS) {
    console.warn(
      `[@vitrum/shared-bvh/mergeWorldSpaceFromCore] ${filteredTriangleCount} malformed triangles filtered ` +
      `(${MAX_WORLD_MERGE_FILTER_WARNINGS} individual warnings shown above).`,
    );
  }

  const vertexCount = Math.floor(positions.length / stride);
  const triangleCount = mergedTriMaterialId.length;

  // Empty scene → an empty-but-valid result (one zeroed leaf node), so DDGI/RC
  // don't have to special-case a primitive-less frame (mirrors
  // `emptyBVHResult`'s intent, THREE-free).
  if (triangleCount === 0) {
    return {
      bvhNodes: new Float32Array(8),
      positions: new Float32Array(stride === 4 ? 12 : 9),
      positionStrideFloats: stride,
      indices: new Uint32Array([0, 1, 2]),
      bvhIndexStride: 3,
      triMaterialId: new Uint32Array(1),
      bvhTriToMergedTri: new Uint32Array(0),
      normals: new Float32Array(stride === 4 ? 12 : 9),
      tangents: new Float32Array(stride === 4 ? 12 : 9),
      colors: new Float32Array(12).fill(1),
      uvs: new Float32Array(6),
      mergedIndices: new Uint32Array([0, 1, 2]),
      mergedTriMaterialId: new Uint32Array(1),
      materials,
      boundingBox: { min: [-1, -1, -1], max: [1, 1, 1] },
      meshVertexRanges,
      warnings,
      vertexCount: 3,
      triangleCount: 0,
    };
  }

  const packedPositions = new Float32Array(positions);
  const packedNormals = new Float32Array(normals);
  const packedTangents = new Float32Array(tangents);
  const packedColors = new Float32Array(colors);
  const packedUvs = new Float32Array(uvs);
  const packedMergedIndices = new Uint32Array(mergedIndices);
  const packedMergedTriMaterialId = new Uint32Array(mergedTriMaterialId);

  // Build the merged single-root BVH over the world-space stream. stride-3 index
  // in / out (the `array<vec3u>` form, matching `buildSceneBVH`'s stride-3
  // `indices`). The reorder permutes triangles but NOT vertices.
  const bvh = buildArrayBvh(packedPositions, packedMergedIndices, packedMergedTriMaterialId, {
    positionStride: stride,
    indexStride: 3,
  });

  return {
    bvhNodes: bvh.bvhNodes,
    positions: packedPositions,
    positionStrideFloats: stride,
    indices: bvh.reorderedIndices,
    bvhIndexStride: 3,
    triMaterialId: bvh.reorderedTriMaterialIds,
    bvhTriToMergedTri: bvh.reorderedToSourceTriangle,
    normals: packedNormals,
    tangents: packedTangents,
    colors: packedColors,
    uvs: packedUvs,
    mergedIndices: packedMergedIndices,
    mergedTriMaterialId: packedMergedTriMaterialId,
    materials,
    boundingBox: { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
    meshVertexRanges,
    warnings,
    vertexCount,
    triangleCount,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// UV1 merge helper — colocated with mergeWorldSpaceFromCore because it depends on
// the same range-assignment logic (MergedMeshVertexRange ordering).
// ──────────────────────────────────────────────────────────────────────────

/**
 * Build a merged UV1 array (stride 2, same vertex order as `merged.uvs`) from the
 * source scene primitives.  The `meshVertexRanges` from `mergeWorldSpaceFromCore`
 * provide the mapping: ranges[i].vertexStart/vertexCount tell us which slice of
 * the merged vertex array corresponds to the i-th source primitive instance.
 *
 * Returns `undefined` when NO primitive in the scene carries uv1 — the caller's
 * attribute packer then falls back to uv0 for every vertex.
 *
 * For instanced-mesh primitives, `mergeWorldSpaceFromCore` pushes one range per
 * instance; all instances share the same source uv1 array (instance transforms do
 * not affect UV channels).
 *
 * D10.7 — colocated with worldSpaceMerge.ts: the range ordering contract is defined
 * here, and any change to the range-push logic in `mergeWorldSpaceFromCore` must
 * also update this function to stay in sync.
 *
 * @param scene   The same `Scene` passed to `mergeWorldSpaceFromCore`.
 * @param ranges  `WorldSpaceMergeResult.meshVertexRanges` from `mergeWorldSpaceFromCore`.
 * @param totalVertexCount `WorldSpaceMergeResult.vertexCount`.
 */
export function mergeUv1FromCore(
  scene: Scene,
  ranges: readonly MergedMeshVertexRange[],
  totalVertexCount: number,
): Float32Array | undefined {
  const meshLike = scene.primitives.filter(
    (p): p is Extract<typeof p, { positions: Float32Array; uv1?: Float32Array }> =>
      p.kind === 'mesh' || p.kind === 'instanced-mesh' || p.kind === 'skinned-mesh',
  );

  const anyUv1 = meshLike.some((p) => p.uv1 != null && p.uv1.length > 0);
  if (!anyUv1) return undefined;

  const out = new Float32Array(totalVertexCount * 2);

  let rangeIdx = 0;
  for (const prim of meshLike) {
    const localVertexCount = Math.floor(prim.positions.length / 3);
    if (localVertexCount < 3) continue;
    const localTriCount = Math.floor((prim.indices?.length ?? localVertexCount) / 3);
    if (localTriCount === 0) continue;

    const instanceCount = prim.kind === 'instanced-mesh' ? prim.instances.length : 1;

    for (let inst = 0; inst < instanceCount; inst += 1) {
      const range: MergedMeshVertexRange | undefined = ranges[rangeIdx];
      if (range == null) break;
      rangeIdx += 1;

      const src = prim.uv1;
      if (src == null) continue;

      const { vertexStart, vertexCount } = range;
      for (let v = 0; v < vertexCount; v += 1) {
        out[(vertexStart + v) * 2] = src[v * 2] ?? 0;
        out[(vertexStart + v) * 2 + 1] = src[v * 2 + 1] ?? 0;
      }
    }
  }

  return out;
}
