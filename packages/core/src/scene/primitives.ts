// Scene description — backend-agnostic.
//
// Primitives — geometry that occupies space.

import type { Mat4, SceneNodeId } from './math.js';
import type { MaterialSpec, MaterialSpecPatch } from './material.js';

/**
 * Scalable UV-set storage. Array index is `TextureRef.texCoord`; sparse entries
 * are allowed so a primitive may carry (for example) only TEXCOORD_0 and
 * TEXCOORD_3. Every present stream contains two floats per vertex.
 *
 * `uvs` / `uv1` remain source-compatible aliases for sets 0 / 1. New adapters
 * should populate both the relevant legacy aliases and `uvSets`; when both are
 * present they must contain identical values.
 */
export type PrimitiveUvSets = ReadonlyArray<Float32Array | undefined>;

/**
 * Scalable vertex-color storage. Array index is the glTF-style `COLOR_n`
 * semantic index; sparse entries are allowed. Every present stream contains
 * either RGB or RGBA floats per vertex. `colors` remains the COLOR_0 alias.
 */
export type PrimitiveColorSets = ReadonlyArray<Float32Array | undefined>;

/**
 * Morph UV deltas grouped by UV-set index. Each present lane contains one
 * `vertexCount * 2` delta stream per morph target, parallel to
 * `SkinnedMeshPrimitive.morphTargets`.
 */
export type PrimitiveMorphUvSets = ReadonlyArray<
  ReadonlyArray<Float32Array> | undefined
>;

/**
 * Morph color deltas grouped by COLOR_n semantic index. Each present lane
 * contains one RGB or RGBA delta stream per morph target, parallel to
 * `SkinnedMeshPrimitive.morphTargets` and matching its base COLOR_n width.
 */
export type PrimitiveMorphColorSets = ReadonlyArray<
  ReadonlyArray<Float32Array> | undefined
>;

/**
 * Return the canonical numeric own keys carried by a sparse array without
 * consulting its potentially enormous native `length`.
 *
 * JavaScript only treats indices below 2^32-1 as array indices, but Vitrum's
 * semantic-set contract accepts every non-negative safe integer. Keys at or
 * above 2^32-1 therefore live as ordinary own data properties and must not be
 * lost to `slice`, spread, `map`, `for...of`, or a `length`-bounded loop.
 *
 * @internal
 */
export function sparseArrayOwnIndices(
  values: ReadonlyArray<unknown>,
): number[] {
  if (!Array.isArray(values)) {
    throw new TypeError('Sparse set container must be an Array.');
  }

  const indices: number[] = [];
  for (const key of Reflect.ownKeys(values)) {
    if (key === 'length') continue;
    if (typeof key !== 'string') {
      throw new TypeError('Sparse set container keys must be canonical decimal indices.');
    }
    const index = Number(key);
    if (
      !Number.isSafeInteger(index) ||
      index < 0 ||
      String(index) !== key
    ) {
      throw new TypeError(
        `Sparse set container key "${key}" is not a canonical non-negative safe integer.`,
      );
    }
    const descriptor = Object.getOwnPropertyDescriptor(values, key);
    if (
      descriptor == null ||
      descriptor.enumerable !== true ||
      !('value' in descriptor)
    ) {
      throw new TypeError(
        `Sparse set container entry ${key} must be an enumerable own data property.`,
      );
    }
    indices.push(index);
  }
  indices.sort((a, b) => a - b);
  return indices;
}

/** Clone a validated sparse semantic-set array without walking its `length`. */
export function cloneSparseArray<T>(
  values: ReadonlyArray<T | undefined>,
): Array<T | undefined> {
  const clone: Array<T | undefined> = [];
  for (const index of sparseArrayOwnIndices(values)) {
    clone[index] = values[index];
  }
  return clone;
}

/** Test only the sparse array's present entries, independent of `length`. */
export function sparseArrayHasDefinedEntry(
  values: ReadonlyArray<unknown>,
): boolean {
  for (const index of sparseArrayOwnIndices(values)) {
    if (values[index] !== undefined) return true;
  }
  return false;
}

export interface PrimitiveUvStreams {
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: PrimitiveUvSets;
}

export interface PrimitiveColorStreams {
  readonly colors?: Float32Array;
  readonly colorSets?: PrimitiveColorSets;
  /**
   * COLOR_n lane multiplied into the material base color. Defaults to 0;
   * `null` explicitly disables vertex-color multiplication while retaining
   * authored lanes for a later mutation.
   * Backends consume the selected lane through the canonical `colors` packing
   * stream; unselected lanes remain available for a later primitive mutation.
   */
  readonly vertexColorSet?: number | null;
}

/** Resolve a material texCoord index through the scalable and legacy lanes. */
export function getPrimitiveUvSet(
  primitive: PrimitiveUvStreams,
  texCoord: number,
): Float32Array | undefined {
  if (!Number.isSafeInteger(texCoord) || texCoord < 0) return undefined;
  return primitive.uvSets?.[texCoord] ??
    (texCoord === 0 ? primitive.uvs : texCoord === 1 ? primitive.uv1 : undefined);
}

/** Resolve a vertex-color semantic through the scalable and legacy lanes. */
export function getPrimitiveColorSet(
  primitive: PrimitiveColorStreams,
  colorSet: number,
): Float32Array | undefined {
  if (!Number.isSafeInteger(colorSet) || colorSet < 0) return undefined;
  return primitive.colorSets?.[colorSet] ??
    (colorSet === 0 ? primitive.colors : undefined);
}

/** Resolve the primitive's selected vertex-color lane (COLOR_0 by default). */
export function getPrimitiveActiveColorSet(
  primitive: object,
): Float32Array | undefined {
  const streams = primitive as PrimitiveColorStreams;
  if (streams.vertexColorSet === null) return undefined;
  return getPrimitiveColorSet(streams, streams.vertexColorSet ?? 0);
}

/** Triangle mesh. Position/normal/uv arrays follow three.js convention:
 *  flat Float32Arrays where consecutive triples (or pairs for uv) describe
 *  one vertex. `indices` is optional; without it, vertices are interpreted
 *  as triangle-list. */
export interface MeshPrimitive {
  readonly kind: 'mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;            // 2nd UV channel (TextureRef.texCoord 1); uv pairs
  readonly uvSets?: PrimitiveUvSets;       // arbitrary TextureRef.texCoord lanes
  /** Tangents, xyzw per vertex (w = bitangent sign).
   *  Consumed by pt-webgl2 for normal/bump/clearcoat-normal maps when supplied;
   *  other backends may derive tangents from UV gradients. */
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;         // vertex colors; RGB(A) (components = length / vertexCount)
  readonly colorSets?: PrimitiveColorSets; // arbitrary COLOR_n streams; colors aliases set 0
  readonly vertexColorSet?: number | null; // selected COLOR_n lane; default 0; null disables
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: MaterialSpec;
  readonly transform?: Mat4;              // identity if absent
  /** Whether this mesh casts shadows on other geometry. Default true.
   *  Per-backend status (SHADOW-01, 2026-06-11 — see
   *  `BackendSupportDetails.shadows.primitiveCastShadow`):
   *    - `@vitrum/pt-webgl2` — native (material castShadow lane + the
   *      integrator's shadow-ray continuation gate).
   *    - `@vitrum/pt-webgpu` — native (any-hit/occlusion traversals skip
   *      castShadow:false triangles on both tiers; closest-hit radiance rays
   *      still hit them).
   *    - `@vitrum/walkaround-hybrid` — native: honored by DI, ReSTIR-GI,
   *      DDGI probe direct-light visibility, GRIS reuse visibility, and RC
   *      probe direct-light visibility. */
  readonly castShadow?: boolean;
}

/** Same geometry repeated at many transforms. Backend may build a single BVH
 *  once and traverse via instance transforms. */
export interface InstancedMeshPrimitive {
  readonly kind: 'instanced-mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;            // 2nd UV channel (TextureRef.texCoord 1)
  readonly uvSets?: PrimitiveUvSets;       // arbitrary TextureRef.texCoord lanes
  /** Tangents, xyzw per vertex (w = bitangent sign).
   *  Consumed by pt-webgl2 for normal/bump/clearcoat-normal maps when supplied;
   *  other backends may derive tangents from UV gradients. */
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;         // vertex colors; RGB(A) per vertex
  readonly colorSets?: PrimitiveColorSets; // arbitrary COLOR_n streams; colors aliases set 0
  readonly vertexColorSet?: number | null; // selected COLOR_n lane; default 0; null disables
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: MaterialSpec;
  readonly instances: ReadonlyArray<Mat4>;
  /** Whether these mesh instances cast shadows on other geometry. Default true.
   *  Same per-backend status as {@link MeshPrimitive.castShadow}. All shipping
   *  backends grade primitive castShadow as native; walkaround-hybrid honors it
   *  in DI, ReSTIR-GI, DDGI probe direct-light visibility, GRIS reuse visibility,
   *  and RC probe direct-light visibility. */
  readonly castShadow?: boolean;
}

/** Closed-form ray-primitive intersection. Backend-supported shapes only;
 *  unsupported shapes log a warning and degrade to skip (or to mesh
 *  tessellation if a fallback geometry is provided).
 *
 *  Future shapes (gemstones via 'ellipsoid', etc.) extend this discriminated
 *  union without breaking existing scenes. ('capsule'/'cylinder' are already
 *  in the union — see AnalyticShape below.)
 */
export interface AnalyticPrimitive {
  readonly kind: 'analytic';
  readonly id: SceneNodeId;
  readonly shape: AnalyticShape;
  readonly params: Float32Array;          // shape-specific layout, see AnalyticShape
  readonly material: MaterialSpec;
  readonly transform?: Mat4;
  /** Whether this analytic primitive casts shadows on other geometry.
   *  Defaults to true. Same per-backend status as
   *  {@link MeshPrimitive.castShadow}. */
  readonly castShadow?: boolean;
  readonly fallbackMesh?: Omit<MeshPrimitive, 'kind' | 'id' | 'material' | 'transform'>;
}

export type AnalyticShape =
  | 'sphere'           // params: [cx, cy, cz, radius]
  | 'box'              // params: [cx, cy, cz, hx, hy, hz]
  | 'capsule'          // params: [ax, ay, az, bx, by, bz, radius]
  | 'cylinder'         // params: [cx, cy, cz, radius, halfHeight]
  | 'h-channel-came';  // params: [length, railWidth, blockHeight, webThickness]

/**
 * Skinned mesh — vertex positions deformed each frame by a skeleton of
 * bone matrices. C1 (2026-05-19) initial contract for host adapters.
 * Per-frame skinning solver + BVH refit are tracked separately.
 *
 * Layout follows glTF 2.0 skinning conventions:
 * - `positions`/`normals`/`uvs`/`indices` — REST-pose geometry
 *   (positions in mesh-local space).
 * - `skinIndices` — `skinInfluencesPerVertex` bone indices per vertex.
 *   `skinInfluencesPerVertex` defaults to 4 for source compatibility but may be
 *   larger for production assets carrying JOINTS_1+ influence sets. Index `0`
 *   is the implicit "no bone" when paired with a zero weight.
 * - `skinWeights` — the parallel per-vertex weights. They must sum to 1.0 per
 *   vertex per glTF convention;
 *   the adapter does NOT renormalise.
 * - `bones` — bone-local-to-skinning-space matrices for the current pose.
 *   `boneCount` × 16 floats (column-major). Hosts update this each
 *   frame (or whenever the skeleton pose changes); engines use it +
 *   `boneInverses` to compute per-vertex deformed positions:
 *     `deformed = Σ weight[k] · bones[skinIndex[k]] · boneInverses[skinIndex[k]] · restPos`
 * - `boneInverses` — inverse bind matrices. `boneCount` × 16 floats.
 *   Captured at bind time; constant for the life of the primitive.
 * - `material` / `transform` mirror `MeshPrimitive`.
 *
 * SPACE CONVENTION: `solveSkin` must return positions in the same space as
 * mesh-local primitive geometry because consumers apply `transform` once on top.
 * Importers can satisfy that either by supplying mesh-node-local bone matrices
 * (the glTF adapter uses `inverse(meshWorld) · jointWorld`) or, for host APIs
 * that expose world-space bone chains, by supplying `bindMatrix` /
 * `bindMatrixInverse` such that the solver's final output is mesh-local.
 *
 * The CPU-side solver (`solveSkin`) lives in `@vitrum/core`. Native engine
 * mutation APIs accept authored rest/pose fields and solve once into private
 * render geometry; direct solved arrays remain useful for previews and
 * instanced-mesh deformation fallbacks that cannot retain skeleton state.
 *
 * Backends that don't implement skinning should report this in
 * `EngineCapabilities` and either skip the primitive (with a warning)
 * or render the rest pose statically.
 */
export interface SkinnedMeshPrimitive {
  readonly kind: 'skinned-mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;   // rest-pose, mesh-local
  readonly normals: Float32Array;     // rest-pose, mesh-local
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;         // 2nd UV channel (TextureRef.texCoord 1)
  readonly uvSets?: PrimitiveUvSets;    // arbitrary TextureRef.texCoord lanes
  /** Tangents, xyzw per vertex (w = bitangent sign).
   *  Rest-pose tangent data. Backends that CPU-solve skinning should skin these
   *  explicitly or derive posed tangents from the solved surface. */
  readonly tangents?: Float32Array;
  readonly colors?: Float32Array;      // vertex colors; RGB(A) per vertex
  readonly colorSets?: PrimitiveColorSets; // arbitrary COLOR_n streams; colors aliases set 0
  readonly vertexColorSet?: number | null; // selected COLOR_n lane; default 0; null disables
  readonly indices?: Uint32Array | Uint16Array;
  /** Bone indices, length `vertexCount * skinInfluencesPerVertex`. */
  readonly skinIndices: Uint32Array;
  /** Bone weights parallel to skinIndices; each vertex sums to 1. */
  readonly skinWeights: Float32Array;
  /** Number of packed influences per vertex. Defaults to 4. */
  readonly skinInfluencesPerVertex?: number;
  /** Per-frame bone matrices in skinning space: `boneCount * 16` column-major f32s. */
  readonly bones: Float32Array;
  /** Inverse bind matrices: `boneCount * 16` column-major f32s. */
  readonly boneInverses: Float32Array;
  /**
   * Mesh bind matrix (the mesh's world transform at bind time). 16
   * column-major f32s. Optional; identity if omitted. Three.js's
   * `SkinnedMesh.bindMatrix` ≠ identity when the host called
   * `mesh.bind(skeleton, customBindMatrix)` OR when `mesh.bind(skeleton)`
   * was called after positioning the mesh away from the origin.
   *
   * Solver formula with non-identity bindMatrix (see three.js's skinning
   * vertex shader for the canonical reference):
   *   skinVertex   = bindMatrix       · restPos
   *   skinnedWorld = Σ w[k] · ( bones[idx[k]] · boneInverses[idx[k]] ) · skinVertex
   *   skinnedLocal = bindMatrixInverse · skinnedWorld
   *
   * For glTF (typical: mesh at origin, bind at origin) bindMatrix is
   * identity and the formula collapses to the simpler form.
   */
  readonly bindMatrix?: Float32Array;
  readonly bindMatrixInverse?: Float32Array;
  /**
   * Blend-shape (morph-target) deltas. Each entry is a position delta of
   * length `vertexCount * 3` (Δx,Δy,Δz per vertex). glTF convention —
   * displacement from the rest pose, applied before skinning:
   *
   *   morphedPos = restPos + Σ_t morphWeights[t] · morphTargets[t]
   *
   * If a host has absolute-position morphs (the older three.js
   * `morphTargetsRelative === false` mode), the adapter converts to
   * deltas at extract time. Optional `morphTargetNormals` carries
   * matching normal deltas; optional `morphTargetTangents` carries glTF
   * TANGENT direction deltas (xyz only; tangent handedness stays on the base
   * `tangents` stream); optional `morphTargetUvSets` carries arbitrary glTF
   * TEXCOORD_N deltas; optional `morphTargetColorSets` carries additive
   * COLOR_N deltas. `morphTargetUvs` / `morphTargetUv1s` remain compatibility
   * aliases for UV sets 0 / 1, while `morphTargetColors` aliases COLOR_0; each
   * alias must match its scalable lane when both are present.
   * `solveSkin()` applies these deltas when the corresponding rest stream
   * exists. Omit for position-only morphs.
   */
  readonly morphTargets?: ReadonlyArray<Float32Array>;
  readonly morphTargetNormals?: ReadonlyArray<Float32Array>;
  readonly morphTargetTangents?: ReadonlyArray<Float32Array>;
  readonly morphTargetUvs?: ReadonlyArray<Float32Array>;
  readonly morphTargetUv1s?: ReadonlyArray<Float32Array>;
  readonly morphTargetUvSets?: PrimitiveMorphUvSets;
  readonly morphTargetColors?: ReadonlyArray<Float32Array>;
  readonly morphTargetColorSets?: PrimitiveMorphColorSets;
  /** Per-target influence weights, length = morphTargets.length. Omission means all-zero weights. */
  readonly morphWeights?: Float32Array;
  readonly material: MaterialSpec;
  readonly transform?: Mat4;
  /** Whether this mesh casts shadows on other geometry. Default true.
   *  Same per-backend status as {@link MeshPrimitive.castShadow}: all shipping
   *  backends grade primitive castShadow as native; walkaround-hybrid honors it
   *  in DI, ReSTIR-GI, DDGI probe direct-light visibility, GRIS reuse visibility,
   *  and RC probe direct-light visibility. */
  readonly castShadow?: boolean;
}

export type ScenePrimitive =
  | MeshPrimitive
  | InstancedMeshPrimitive
  | AnalyticPrimitive
  | SkinnedMeshPrimitive;

/**
 * Keys whose properties are optional on `T`.
 *
 * This local helper lets primitive patches express an explicit clear only for
 * fields that are optional on the complete primitive contract.
 */
type OptionalPrimitivePropertyKeys<T extends object> = {
  [K in keyof T]-?: Pick<T, K> extends Required<Pick<T, K>> ? never : K;
}[keyof T];

type ExactPrimitivePatch<T extends object> = {
  readonly [K in keyof T]?: K extends OptionalPrimitivePropertyKeys<T>
    ? T[K] | undefined
    : T[K];
};

type PrimitivePatchMember<T extends ScenePrimitive> = ExactPrimitivePatch<
  Omit<T, 'id' | 'kind' | 'material'>
> & {
  /** Identity is selected by the `updatePrimitive(id, ...)` argument. */
  readonly id?: never;
  /** Changing a discriminated-union member requires a new scene snapshot. */
  readonly kind?: never;
  readonly material?: MaterialSpecPatch;
};

type PrimitivePatchMembers =
  | PrimitivePatchMember<MeshPrimitive>
  | PrimitivePatchMember<InstancedMeshPrimitive>
  | PrimitivePatchMember<AnalyticPrimitive>
  | PrimitivePatchMember<SkinnedMeshPrimitive>;

type UnionKeys<T> = T extends T ? keyof T : never;
type StrictUnionMember<T, TAll> = T extends T
  ? T & Partial<Record<Exclude<UnionKeys<TAll>, keyof T>, never>>
  : never;

/**
 * Strict incremental primitive patch.
 *
 * Each union member exposes only fields belonging to that primitive kind, so
 * incompatible combinations such as `{ instances, transform }` or
 * `{ positions, shape }` are rejected. `id` and `kind` are intentionally
 * unassignable: identity/kind changes require `setScene`.
 *
 * Optional primitive fields may be cleared with an own `undefined`; omission
 * preserves the current value. Material updates use {@link MaterialSpecPatch}.
 */
export type ScenePrimitivePatch = StrictUnionMember<
  PrimitivePatchMembers,
  PrimitivePatchMembers
>;
