// Scene description — backend-agnostic.
//
// Primitives — geometry that occupies space.

import type { Mat4, SceneNodeId } from './math.js';
import type { MaterialSpec } from './material.js';

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
  readonly tangents?: Float32Array;       // xyzw per vertex; w = bitangent sign
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: MaterialSpec;
  readonly transform?: Mat4;              // identity if absent
  readonly castShadow?: boolean;          // default true
  readonly receiveShadow?: boolean;       // default true
}

/** Same geometry repeated at many transforms. Backend may build a single BVH
 *  once and traverse via instance transforms. */
export interface InstancedMeshPrimitive {
  readonly kind: 'instanced-mesh';
  readonly id: SceneNodeId;
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs?: Float32Array;
  readonly tangents?: Float32Array;
  readonly indices?: Uint32Array | Uint16Array;
  readonly material: MaterialSpec;
  readonly instances: ReadonlyArray<Mat4>;
}

/** Closed-form ray-primitive intersection. Backend-supported shapes only;
 *  unsupported shapes log a warning and degrade to skip (or to mesh
 *  tessellation if a fallback geometry is provided).
 *
 *  Phase 6 sprint 5 introduces 'h-channel-came' for our analytic came/solder
 *  geometry. Future shapes (gemstones via 'ellipsoid', pillars via 'capsule',
 *  etc.) extend this discriminated union without breaking existing scenes.
 */
export interface AnalyticPrimitive {
  readonly kind: 'analytic';
  readonly id: SceneNodeId;
  readonly shape: AnalyticShape;
  readonly params: Float32Array;          // shape-specific layout, see AnalyticShape
  readonly material: MaterialSpec;
  readonly transform?: Mat4;
  readonly fallbackMesh?: Omit<MeshPrimitive, 'kind' | 'id' | 'material' | 'transform'>;
}

export type AnalyticShape =
  | 'sphere'           // params: [cx, cy, cz, radius]
  | 'box'              // params: [cx, cy, cz, hx, hy, hz]
  | 'capsule'          // params: [ax, ay, az, bx, by, bz, radius]
  | 'cylinder'         // params: [cx, cy, cz, radius, halfHeight]
  | 'h-channel-came';  // params: [length, railWidth, blockHeight, webThickness] — H-channel rail primitive, Phase 6 sprint 5

/**
 * Skinned mesh — vertex positions deformed each frame by a skeleton of
 * bone matrices. C1 (2026-05-19) initial contract for the foundation
 * adapter (`sceneFromThreeJS` acceptance). Per-frame skinning solver
 * + BVH refit are tracked separately.
 *
 * Layout follows glTF 2.0 / THREE.SkinnedMesh:
 * - `positions`/`normals`/`uvs`/`indices` — REST-pose geometry
 *   (positions in mesh-local space).
 * - `skinIndices` — 4 bone indices per vertex (Uint32Array, length
 *   `vertexCount * 4`). Index `0` is the implicit "no bone" with weight
 *   forcing it to a no-op when paired with `skinWeights[i*4+k] === 0`.
 * - `skinWeights` — 4 weights per vertex (Float32Array, length
 *   `vertexCount * 4`). Must sum to 1.0 per vertex per glTF convention;
 *   the adapter does NOT renormalise.
 * - `bones` — bone-local-to-world matrices for the current pose.
 *   `boneCount` × 16 floats (column-major). Hosts update this each
 *   frame (or whenever the skeleton pose changes); engines use it +
 *   `boneInverses` to compute per-vertex deformed positions:
 *     `deformed = Σ weight[k] · bones[skinIndex[k]] · boneInverses[skinIndex[k]] · restPos`
 * - `boneInverses` — inverse bind matrices. `boneCount` × 16 floats.
 *   Captured at bind time; constant for the life of the primitive.
 * - `material` / `transform` mirror `MeshPrimitive`.
 *
 * The CPU-side solver lives in `@vitrum/three-bindings` (C1 follow-up);
 * the engine ingests the deformed positions through the existing
 * `HybridEngine.updatePrimitive` positions-refit fast path (A3).
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
  readonly tangents?: Float32Array;
  readonly indices?: Uint32Array | Uint16Array;
  /** 4 bone indices per vertex (length `vertexCount * 4`). */
  readonly skinIndices: Uint32Array;
  /** 4 bone weights per vertex (length `vertexCount * 4`); sum to 1. */
  readonly skinWeights: Float32Array;
  /** Per-frame bone world matrices: `boneCount * 16` column-major f32s. */
  readonly bones: Float32Array;
  /** Inverse bind matrices: `boneCount * 16` column-major f32s. */
  readonly boneInverses: Float32Array;
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
   * matching normal deltas; omit for position-only morphs.
   */
  readonly morphTargets?: ReadonlyArray<Float32Array>;
  readonly morphTargetNormals?: ReadonlyArray<Float32Array>;
  /** Per-target influence weights, length = morphTargets.length. */
  readonly morphWeights?: Float32Array;
  readonly material: MaterialSpec;
  readonly transform?: Mat4;
  readonly castShadow?: boolean;
  readonly receiveShadow?: boolean;
}

export type ScenePrimitive =
  | MeshPrimitive
  | InstancedMeshPrimitive
  | AnalyticPrimitive
  | SkinnedMeshPrimitive;
