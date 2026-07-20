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

import type { Mat4, MaterialSpec, Scene, SceneNodeId, ScenePrimitive } from '@vitrum/core';
import type { PlainAabb } from './aabb.js';
import { buildArrayBvh } from './buildArrayBvh.js';
import { maybeMicrodisplaceMeshGeometry, resolveDisplacedGeometry } from './vertexDisplacement.js';
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
  let filteredTriangleWarnCount = 0;
  let filteredTriangleCount = 0;

  const warnFilteredTriangle = (primitiveId: SceneNodeId, tri: number, reason: string): void => {
    filteredTriangleCount += 1;
    if (filteredTriangleWarnCount < MAX_WORLD_MERGE_FILTER_WARNINGS) {
      const message =
        `[@vitrum/shared-bvh/mergeWorldSpaceFromCore] Primitive "${primitiveId}" triangle ${tri} ${reason}; ` +
        'filtering it from the merged world-space BVH.';
      console.warn(message);
      warn(message);
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

    const {
      basePositions,
      baseNormals,
      baseTangents,
      baseColors,
      baseUvs, // optional; (0,0) per vertex when absent
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
        // V2-4: explicit provenance so consumers (e.g. mergeUv1FromCore) can
        // key attribute data off the range WITHOUT re-deriving the per-instance
        // skip logic above (all-filtered instances push no range).
        sourcePrimitiveId: primitive.id,
        sourceInstanceIndex: instanceIndex,
      });
    }
  }

  if (filteredTriangleCount > MAX_WORLD_MERGE_FILTER_WARNINGS) {
    const message =
      `[@vitrum/shared-bvh/mergeWorldSpaceFromCore] ${filteredTriangleCount} malformed triangles filtered ` +
      `(${MAX_WORLD_MERGE_FILTER_WARNINGS} individual warnings shown above).`;
    console.warn(message);
    warn(message);
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
    (p): p is Extract<ScenePrimitive, { kind: 'mesh' | 'instanced-mesh' | 'skinned-mesh' }> =>
      p.kind === 'mesh' || p.kind === 'instanced-mesh' || p.kind === 'skinned-mesh',
  );

  const anyUv1 = meshLike.some((p) => p.uv1 != null && p.uv1.length > 0);
  if (!anyUv1) return undefined;

  // V2-4: build a per-primitive UV1 source lookup ONCE (UV1 is instance- and
  // transform-invariant), then drive the output off the merged ranges directly
  // — keyed by each range's recorded source-primitive id. This eliminates the
  // replicated per-instance skip logic that used to desync `rangeIdx` when the
  // producer dropped an all-filtered instance (validTriangles.length === 0 →
  // no range pushed).
  const srcUv1ByPrimitiveId = new Map<string, Float32Array>();
  for (const prim of meshLike) {
    if (prim.uv1 == null) continue;
    const microdisplaced = maybeMicrodisplaceMeshGeometry({
      primitiveId: prim.id,
      material: prim.material,
      positions: prim.positions,
      normals: prim.normals,
      ...(prim.indices != null ? { indices: prim.indices } : {}),
      ...(prim.uvs != null ? { uvs: prim.uvs } : {}),
      ...(prim.uv1 != null ? { uv1: prim.uv1 } : {}),
      ...(prim.tangents != null ? { tangents: prim.tangents } : {}),
      ...(prim.colors != null ? { colors: prim.colors } : {}),
    });
    const srcUv1 = microdisplaced?.uv1 ?? prim.uv1;
    if (srcUv1 != null) srcUv1ByPrimitiveId.set(prim.id, srcUv1);
  }

  const out = new Float32Array(totalVertexCount * 2);

  for (const range of ranges) {
    // Prefer the explicit provenance field; fall back to `name` (historically
    // equal to the primitive id) for ranges produced before the field existed.
    const primitiveId = range.sourcePrimitiveId ?? range.name;
    const srcUv1 = srcUv1ByPrimitiveId.get(primitiveId);
    if (srcUv1 == null) continue;

    const { vertexStart, vertexCount } = range;
    for (let v = 0; v < vertexCount; v += 1) {
      out[(vertexStart + v) * 2] = srcUv1[v * 2] ?? 0;
      out[(vertexStart + v) * 2 + 1] = srcUv1[v * 2 + 1] ?? 0;
    }
  }

  return out;
}
