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

import {
  getPrimitiveActiveColorSet,
  validateScene,
  type Mat4,
  type MaterialSpec,
  type PrimitiveUvSets,
  type Scene,
  type ScenePrimitive,
} from '@vitrum/core';
import type { PlainAabb } from './aabb.js';
import { buildArrayBvh, createEmptyBvhNode } from './buildArrayBvh.js';
import { resolveDisplacedGeometry } from './vertexDisplacement.js';
import {
  IDENTITY_MAT4,
  applyMatrix4,
  finiteVec3,
  getNormalMatrix3,
  applyNormalMatrix,
  applyDirectionMatrix4,
  determinant4,
} from './worldTransforms.js';
import { materialSig } from './materialSignature.js';

const EMPTY_PRIMITIVE_UV_SETS: PrimitiveUvSets = Object.freeze([]);

// D12-9: the transform kernels + the material-dedup signature were extracted to
// `worldTransforms.ts` / `materialSignature.ts` (pure move). Re-export so the
// package entrypoint and existing importers (incl. the wsl-gpu oracles that
// import `materialSig`/`HandleIdRegistry` from `@vitrum/shared-bvh`) keep working.
export {
  IDENTITY_MAT4,
  applyMatrix4,
  finiteVec3,
  getNormalMatrix3,
  applyNormalMatrix,
  applyDirectionMatrix4,
  determinant4,
} from './worldTransforms.js';
export { materialSig, HandleIdRegistry } from './materialSignature.js';

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
  /**
   * Source scene-primitive id this range was produced from (V2-4). Additive:
   * distinct from {@link name} only in intent — `name` is the display/lookup
   * label historically equal to the primitive id, while `sourcePrimitiveId`
   * is the explicit provenance a consumer keys off to attach per-primitive
   * attribute data (e.g. UV1) WITHOUT re-deriving the producer's per-instance
   * skip logic. Optional so existing consumers are unaffected.
   */
  readonly sourcePrimitiveId?: string;
  /**
   * Zero-based instance index of this range within its source primitive
   * (V2-4). For a non-instanced primitive this is always 0; for an
   * instanced-mesh it is the index of the transform that produced this range,
   * skipping any all-filtered instances. Additive/optional.
   */
  readonly sourceInstanceIndex?: number;
  /**
   * Displacement-resolved UV sources captured by the producer. Keeping these
   * with the emitted range prevents downstream UV merges from resolving
   * microdisplacement a second time (and from touching filtered primitives).
   */
  readonly sourceUvSets?: PrimitiveUvSets;
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

function constantVertexRgbMultiplier(primitive: MeshLikePrimitive): readonly [number, number, number] | null {
  const colors = getPrimitiveActiveColorSet(primitive);
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
  validateScene(scene);
  if (opts == null || typeof opts !== 'object' || Array.isArray(opts)) {
    throw new TypeError('[@vitrum/shared-bvh/mergeWorldSpaceFromCore] opts must be an object.');
  }
  const optionKeys = new Set([
    'positionStride', 'filter', 'splitMaterialsByCastShadow',
    'bakeConstantVertexColorIntoMaterial', 'onWarning',
  ]);
  for (const key of Reflect.ownKeys(opts)) {
    if (typeof key !== 'string' || !optionKeys.has(key)) {
      throw new RangeError(
        `[@vitrum/shared-bvh/mergeWorldSpaceFromCore] unknown option ${String(key)}.`,
      );
    }
  }
  const stride = opts.positionStride ?? 4;
  if (stride !== 3 && stride !== 4) {
    throw new RangeError('[@vitrum/shared-bvh/mergeWorldSpaceFromCore] positionStride must be exactly 3 or 4.');
  }
  if (opts.filter !== undefined && typeof opts.filter !== 'function') {
    throw new TypeError('[@vitrum/shared-bvh/mergeWorldSpaceFromCore] filter must be a function.');
  }
  if (opts.onWarning !== undefined && typeof opts.onWarning !== 'function') {
    throw new TypeError('[@vitrum/shared-bvh/mergeWorldSpaceFromCore] onWarning must be a function.');
  }
  for (const key of ['splitMaterialsByCastShadow', 'bakeConstantVertexColorIntoMaterial'] as const) {
    if (opts[key] !== undefined && typeof opts[key] !== 'boolean') {
      throw new TypeError(`[@vitrum/shared-bvh/mergeWorldSpaceFromCore] ${key} must be a boolean.`);
    }
  }
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
    try {
      opts.onWarning?.(warning);
    } catch {
      // Warning callbacks are advisory; keep the returned warning list authoritative.
    }
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

    const {
      basePositions,
      baseNormals,
      baseTangents,
      baseColors,
      baseUvs, // optional; (0,0) per vertex when absent
      baseUvSets,
      baseIndicesSource,
      sourcePositions,
    } = resolveDisplacedGeometry(primitive, warn);
    const localVertexCount = Math.floor(basePositions.length / 3);
    const hasCompleteTangents = baseTangents != null && baseTangents.length >= localVertexCount * 4;
    const colorStride = baseColors != null && baseColors.length >= localVertexCount * 4
      ? 4
      : baseColors != null && baseColors.length >= localVertexCount * 3
        ? 3
        : 0;
    if (localVertexCount < 3) continue;

    // Sequential index when the primitive carries none (triangle-list), matching
    // `packOneMeshLikePrimitive` / SGG's index synthesis.
    const baseIndices: ArrayLike<number> =
      baseIndicesSource ??
      (() => {
        const gen = new Uint32Array(localVertexCount);
        for (let i = 0; i < localVertexCount; i += 1) gen[i] = i;
        return gen;
      })();
    const localTriCount = Math.floor(baseIndices.length / 3);
    if (localTriCount === 0) continue;

    for (let instanceIndex = 0; instanceIndex < transforms.length; instanceIndex += 1) {
      const transform = transforms[instanceIndex];
      const m = transform ?? IDENTITY_MAT4;
      const normalMatrix = getNormalMatrix3(m);
      const flip = determinant4(m) < 0;

      const validTriangles: Array<readonly [number, number, number]> = [];
      for (let t = 0; t < localTriCount; t += 1) {
        const a = baseIndices[t * 3] ?? 0;
        const b = baseIndices[t * 3 + 1] ?? 0;
        const c = baseIndices[t * 3 + 2] ?? 0;
        if (a >= localVertexCount || b >= localVertexCount || c >= localVertexCount) {
          throw new RangeError(
            `[@vitrum/shared-bvh/mergeWorldSpaceFromCore] Primitive "${primitive.id}" triangle ${t} ` +
            `references an out-of-range vertex index (i0=${a}, i1=${b}, i2=${c}; vertexCount=${localVertexCount}).`,
          );
        }
        const pa = applyMatrix4(m, sourcePositions[a * 3] ?? 0, sourcePositions[a * 3 + 1] ?? 0, sourcePositions[a * 3 + 2] ?? 0);
        const pb = applyMatrix4(m, sourcePositions[b * 3] ?? 0, sourcePositions[b * 3 + 1] ?? 0, sourcePositions[b * 3 + 2] ?? 0);
        const pc = applyMatrix4(m, sourcePositions[c * 3] ?? 0, sourcePositions[c * 3 + 1] ?? 0, sourcePositions[c * 3 + 2] ?? 0);
        if (!finiteVec3(pa) || !finiteVec3(pb) || !finiteVec3(pc)) {
          throw new RangeError(
            `[@vitrum/shared-bvh/mergeWorldSpaceFromCore] Primitive "${primitive.id}" triangle ${t} ` +
            'has a non-finite transformed vertex coordinate (NaN or Inf).',
          );
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
        // V2-4: explicit provenance so consumers (e.g. mergeUv1FromCore) can
        // key attribute data off the range WITHOUT re-deriving the per-instance
        // skip logic above (all-filtered instances push no range).
        sourcePrimitiveId: primitive.id,
        sourceInstanceIndex: instanceIndex,
        // Always attach a cache marker. An empty array is semantically distinct
        // from `undefined`: it proves the producer already resolved this
        // UV-less primitive, so downstream UV merges must not run displacement
        // resolution again. `undefined` remains reserved for legacy/external
        // ranges that genuinely need the fallback lookup.
        sourceUvSets: baseUvSets ?? EMPTY_PRIMITIVE_UV_SETS,
      });
    }
  }

  const vertexCount = Math.floor(positions.length / stride);
  const triangleCount = mergedTriMaterialId.length;

  // Empty scene → an empty-but-valid result (one zero-count leaf node), so DDGI/RC
  // don't have to special-case a primitive-less frame (mirrors
  // `emptyBVHResult`'s intent, THREE-free).
  if (triangleCount === 0) {
    return {
      bvhNodes: createEmptyBvhNode(),
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
export function mergeUvSetFromCore(
  scene: Scene,
  ranges: readonly MergedMeshVertexRange[],
  totalVertexCount: number,
  texCoord: number,
): Float32Array | undefined {
  if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
    throw new RangeError(
      `mergeUvSetFromCore: texCoord must be a non-negative safe integer (got ${String(texCoord)}).`,
    );
  }
  // Ranges produced by mergeWorldSpaceFromCore carry the already resolved UV
  // sources. Only legacy/external ranges without that cache need a lookup.
  const sourceByPrimitiveId = new Map<string, Float32Array>();
  const missingPrimitiveIds = new Set(
    ranges
      .filter((range) => range.sourceUvSets == null)
      .map((range) => range.sourcePrimitiveId ?? range.name),
  );
  for (const prim of scene.primitives) {
    if (
      !missingPrimitiveIds.has(prim.id) ||
      (prim.kind !== 'mesh' &&
        prim.kind !== 'instanced-mesh' &&
        prim.kind !== 'skinned-mesh')
    ) {
      continue;
    }
    const resolved = resolveDisplacedGeometry(prim, () => {});
    const source = resolved.baseUvSets?.[texCoord];
    if (source != null) sourceByPrimitiveId.set(prim.id, source);
  }

  const anyUvSet = ranges.some((range) => {
    const primitiveId = range.sourcePrimitiveId ?? range.name;
    return (
      (range.sourceUvSets?.[texCoord]?.length ?? 0) > 0 ||
      (sourceByPrimitiveId.get(primitiveId)?.length ?? 0) > 0
    );
  });
  if (!anyUvSet) return undefined;

  const out = new Float32Array(totalVertexCount * 2);

  for (const range of ranges) {
    // Prefer the explicit provenance field; fall back to `name` (historically
    // equal to the primitive id) for ranges produced before the field existed.
    const primitiveId = range.sourcePrimitiveId ?? range.name;
    const source =
      range.sourceUvSets?.[texCoord] ??
      sourceByPrimitiveId.get(primitiveId);
    if (source == null) continue;

    const { vertexStart, vertexCount } = range;
    for (let v = 0; v < vertexCount; v += 1) {
      out[(vertexStart + v) * 2] = source[v * 2] ?? 0;
      out[(vertexStart + v) * 2 + 1] = source[v * 2 + 1] ?? 0;
    }
  }

  return out;
}

/** Back-compatible TEXCOORD_1 specialization of {@link mergeUvSetFromCore}. */
export function mergeUv1FromCore(
  scene: Scene,
  ranges: readonly MergedMeshVertexRange[],
  totalVertexCount: number,
): Float32Array | undefined {
  return mergeUvSetFromCore(scene, ranges, totalVertexCount, 1);
}
