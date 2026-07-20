/**
 * scenePack.ts — THREE-free scene geometry packer (BLAS concat + optional TLAS).
 *
 * Canonical CPU pack used by walkaround-hybrid and pt-webgpu. Extracted from
 * pt-webgpu `uploadSceneBuffers.ts` (PR-2.1).
 */

import { asMat4, type Mat4, type Scene, type ScenePrimitive, type Vec3 } from '@vitrum/core';
import { buildArrayBvh, isLeafSplit } from './buildArrayBvh.js';
import { BVH_NODE_FLOATS, VERTEX_STRIDE_F32, MAT4_STRIDE_F32 } from './strides.js';
import { buildTlas, refitTlas } from './tlas.js';
import { invertMat4 as _invertMat4 } from './mathUtils.js';
import { rebaseLeafTriOffset as _rebaseLeafTriOffset, copyVec4Strided as _copyVec4Strided, rebaseIndexWords as _rebaseIndexWords } from './splicePack.js';
import { resolveDisplacedGeometry } from './vertexDisplacement.js';

// ── Back-compat re-export from extracted module ───────────────────────────────
// invertMat4 was previously defined in this file; now canonical in mathUtils.
// The re-export keeps existing `from './scenePack.js'` imports working (D11.1).
export { invertMat4 } from './mathUtils.js';

const IDENTITY_MAT4 = asMat4([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

/** Per-primitive bookkeeping for transform-only TLAS refit (stable across frames). */
export interface PrimitiveTlasBinding {
  readonly primitiveId: string;
  readonly primitiveKind: 'mesh' | 'instanced-mesh' | 'skinned-mesh';
  readonly blasRoot: number;
  readonly instanceCount: number;
  readonly vertexStart: number;
  readonly vertexCount: number;
  readonly triStart: number;
  readonly triCount: number;
  readonly localAabbMin: readonly [number, number, number];
  readonly localAabbMax: readonly [number, number, number];
}

export interface ScenePackOptions {
  /** Build per-primitive local BLAS + TLAS. If false, TLAS buffers are empty. */
  readonly tlas?: boolean;
  /** Material id resolver: (primitiveId) → u32 mat slot */
  readonly resolveMaterialId: (primitiveId: string) => number;
}

export interface ScenePackResult {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  /** Per-vertex UVs, vec4f-strided (.xy = uv0, .zw = uv1); same vertex order /
   *  count as {@link positions}. Always present (the packer emits it even for
   *  UV-less geometry, all-zero). Consumers that don't texture may ignore it. */
  readonly uvs: Float32Array;
  /** Per-vertex authored/generated tangents, vec4f-strided (xyz = tangent,
   *  w = bitangent handedness). Same vertex order/count as positions. Missing
   *  tangents are encoded as 0,0,0,0 so consumers can fall back to deriving a
   *  frame from positions + UVs. */
  readonly tangents: Float32Array;
  /** Per-vertex colors, vec4f-strided (rgba). Missing colors are encoded as
   *  1,1,1,1 so consumers can multiply them into material base color/alpha
   *  without adding an authored-color presence bit. */
  readonly colors: Float32Array;
  readonly indices: Uint32Array;
  readonly triMaterialIds: Uint32Array;
  readonly bvhNodes: Float32Array;
  readonly triangleCount: number;
  readonly tlasNodes: Uint32Array;
  readonly tlasInstanceIndices: Uint32Array;
  readonly tlasBlasRoots: Uint32Array;
  readonly tlasInstanceWorldToLocal: Float32Array;
  readonly tlasInstanceLocalToWorld: Float32Array;
  readonly tlasNodeCount: number;
  readonly primitiveTlasBindings: readonly PrimitiveTlasBinding[];
  readonly warnings: readonly string[];
}

export interface TlasGpuSnapshot {
  readonly tlasNodes: Uint32Array;
  readonly tlasInstanceIndices: Uint32Array;
  readonly tlasBlasRoots: Uint32Array;
  readonly tlasInstanceWorldToLocal: Float32Array;
}

export type RefitTlasResult =
  | {
      readonly ok: true;
      readonly tlasNodes: Uint32Array;
      readonly tlasInstanceIndices: Uint32Array;
      readonly tlasBlasRoots: Uint32Array;
      readonly tlasInstanceWorldToLocal: Float32Array;
      readonly tlasInstanceLocalToWorld: Float32Array;
      readonly warnings: readonly string[];
    }
  | {
      readonly ok: false;
      readonly reason: string;
    };

interface PendingTlasInstance {
  readonly aabbMin: readonly [number, number, number];
  readonly aabbMax: readonly [number, number, number];
  readonly worldToLocal: Float32Array;
  readonly localToWorld: Float32Array;
  readonly blasRoot: number;
}

function transformPoint(m: Mat4, p: Vec3): [number, number, number] {
  const at = (index: number): number => m[index] ?? 0;
  const x = p[0], y = p[1], z = p[2];
  const tx = at(0) * x + at(4) * y + at(8) * z + at(12);
  const ty = at(1) * x + at(5) * y + at(9) * z + at(13);
  const tz = at(2) * x + at(6) * y + at(10) * z + at(14);
  const tw = at(3) * x + at(7) * y + at(11) * z + at(15);
  if (Math.abs(tw) > 1e-8) return [tx / tw, ty / tw, tz / tw];
  return [tx, ty, tz];
}

export function computeLocalAabb(positions: Float32Array, stride = 3): {
  min: readonly [number, number, number];
  max: readonly [number, number, number];
} | null {
  if (positions.length < 3 || stride < 3) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i + 2 < positions.length; i += stride) {
    const x = positions[i] ?? 0;
    const y = positions[i + 1] ?? 0;
    const z = positions[i + 2] ?? 0;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

function transformAabb(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
  localToWorld: Float32Array,
): { min: readonly [number, number, number]; max: readonly [number, number, number] } {
  let outMinX = Number.POSITIVE_INFINITY;
  let outMinY = Number.POSITIVE_INFINITY;
  let outMinZ = Number.POSITIVE_INFINITY;
  let outMaxX = Number.NEGATIVE_INFINITY;
  let outMaxY = Number.NEGATIVE_INFINITY;
  let outMaxZ = Number.NEGATIVE_INFINITY;
  for (let c = 0; c < 8; c += 1) {
    const corner: Vec3 = [
      (c & 1) === 0 ? min[0] : max[0],
      (c & 2) === 0 ? min[1] : max[1],
      (c & 4) === 0 ? min[2] : max[2],
    ];
    const p = transformPoint(asMat4(localToWorld), corner);
    outMinX = Math.min(outMinX, p[0]);
    outMinY = Math.min(outMinY, p[1]);
    outMinZ = Math.min(outMinZ, p[2]);
    outMaxX = Math.max(outMaxX, p[0]);
    outMaxY = Math.max(outMaxY, p[1]);
    outMaxZ = Math.max(outMaxZ, p[2]);
  }
  return { min: [outMinX, outMinY, outMinZ], max: [outMaxX, outMaxY, outMaxZ] };
}

function buildTlasFromInstances(instances: readonly PendingTlasInstance[]): {
  tlasNodes: Uint32Array;
  tlasInstanceIndices: Uint32Array;
  tlasBlasRoots: Uint32Array;
  tlasInstanceWorldToLocal: Float32Array;
  tlasInstanceLocalToWorld: Float32Array;
  tlasNodeCount: number;
} {
  if (instances.length === 0) {
    return {
      tlasNodes: new Uint32Array(0),
      tlasInstanceIndices: new Uint32Array(0),
      tlasBlasRoots: new Uint32Array(0),
      tlasInstanceWorldToLocal: new Float32Array(0),
      tlasInstanceLocalToWorld: new Float32Array(0),
      tlasNodeCount: 0,
    };
  }
  const tlas = buildTlas(
    instances.map((instance) => ({
      blasId: instance.blasRoot,
      aabbMin: instance.aabbMin,
      aabbMax: instance.aabbMax,
      worldToLocal: instance.worldToLocal,
    })),
  );
  const l2w = new Float32Array(instances.length * MAT4_STRIDE_F32);
  for (let i = 0; i < instances.length; i += 1) {
    l2w.set(instances[i]!.localToWorld, i * MAT4_STRIDE_F32);
  }
  return {
    tlasNodes: tlas.nodes,
    tlasInstanceIndices: tlas.instanceIndices,
    tlasBlasRoots: tlas.blasRoots,
    tlasInstanceWorldToLocal: tlas.instanceTransforms,
    tlasInstanceLocalToWorld: l2w,
    tlasNodeCount: tlas.nodeCount,
  };
}

function isMeshLike(primitive: ScenePrimitive): primitive is Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }
> {
  return primitive.kind === 'mesh'
    || primitive.kind === 'skinned-mesh'
    || primitive.kind === 'instanced-mesh';
}

interface PackedPrimitiveSlice {
  readonly localPositions: Float32Array;
  readonly localNormals: Float32Array;
  /** Per-vertex UVs, vec4f-strided: .xy = uv0, .zw = uv1 (0 when a channel is
   *  absent). Same 1:1 vertex order as localPositions/localNormals (the vertex
   *  expansion below does not weld or reorder; buildArrayBvh permutes only the
   *  triangle index list), so uvs[i] follows vertex i exactly. */
  readonly localUvs: Float32Array;
  readonly localTangents: Float32Array;
  readonly localColors: Float32Array;
  readonly indexWords: readonly number[];
  readonly triMaterialIds: readonly number[];
  readonly bvhNodeWords: readonly number[];
  readonly vertexCount: number;
  readonly triCount: number;
  readonly bvhNodeCount: number;
  readonly localAabbMin: readonly [number, number, number];
  readonly localAabbMax: readonly [number, number, number];
  readonly warnings: readonly string[];
}

/** Return type for {@link packOneMeshLikePrimitive}: warnings are always available. */
interface PackOneMeshLikeResult {
  readonly slice: PackedPrimitiveSlice | null;
  readonly warnings: readonly string[];
}

function packOneMeshLikePrimitive(
  primitive: Extract<ScenePrimitive, { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }>,
  matId: number,
): PackOneMeshLikeResult {
  const warnings: string[] = [];
  const {
    basePositions,
    baseNormals,
    baseIndicesSource,
    baseUvs,
    baseUv1,
    baseTangents,
    baseColors,
    sourcePositions,
  } = resolveDisplacedGeometry(primitive, (warning) => warnings.push(warning));
  const vertexCount = Math.floor(basePositions.length / 3);
  if (vertexCount < 3) {
    warnings.push(`Primitive "${primitive.id}" has fewer than 3 vertices; skipping.`);
    return { slice: null, warnings };
  }
  const baseIndices =
    baseIndicesSource ??
    (() => {
      const generated = new Uint32Array(vertexCount);
      for (let i = 0; i < generated.length; i += 1) generated[i] = i;
      return generated;
    })();
  const triCount = Math.floor(baseIndices.length / 3);
  if (triCount === 0) {
    warnings.push(`Primitive "${primitive.id}" has no triangles; skipping.`);
    return { slice: null, warnings };
  }

  const localPositions = new Float32Array(vertexCount * VERTEX_STRIDE_F32);
  const localNormals = new Float32Array(vertexCount * VERTEX_STRIDE_F32);
  // Per-vertex UVs, vec4f-strided: .xy = uv0 (base channel), .zw = uv1 (second
  // channel; TextureRef.texCoord 1). Absent channels stay 0. Expanded 1:1 with
  // the vertex loop below — the renderer interpolates these by barycentrics at
  // the hit to drive baseColor/normal/etc. texture sampling.
  const localUvs = new Float32Array(vertexCount * VERTEX_STRIDE_F32);
  const localTangents = new Float32Array(vertexCount * VERTEX_STRIDE_F32);
  const hasCompleteTangents = baseTangents != null && baseTangents.length >= vertexCount * 4;
  if (baseTangents != null && !hasCompleteTangents) {
    warnings.push(
      `Primitive "${primitive.id}" provides ${baseTangents.length} tangent floats; ` +
      `expected ${vertexCount * 4}. Ignoring authored tangents and falling back to derived frames.`,
    );
  }
  const localColors = new Float32Array(vertexCount * VERTEX_STRIDE_F32);
  const hasRgbaColors = baseColors != null && baseColors.length >= vertexCount * 4;
  const hasRgbColors = baseColors != null && !hasRgbaColors && baseColors.length >= vertexCount * 3;
  if (baseColors != null && !hasRgbaColors && !hasRgbColors) {
    warnings.push(
      `Primitive "${primitive.id}" provides ${baseColors.length} color floats; ` +
      `expected at least ${vertexCount * 3}. Ignoring authored vertex colors.`,
    );
  }
  for (let i = 0; i < vertexCount; i += 1) {
    localPositions[i * VERTEX_STRIDE_F32] = sourcePositions[i * 3] ?? 0;
    localPositions[i * VERTEX_STRIDE_F32 + 1] = sourcePositions[i * 3 + 1] ?? 0;
    localPositions[i * VERTEX_STRIDE_F32 + 2] = sourcePositions[i * 3 + 2] ?? 0;
    localPositions[i * VERTEX_STRIDE_F32 + 3] = 0;
    localNormals[i * VERTEX_STRIDE_F32] = baseNormals[i * 3] ?? 0;
    localNormals[i * VERTEX_STRIDE_F32 + 1] = baseNormals[i * 3 + 1] ?? 1;
    localNormals[i * VERTEX_STRIDE_F32 + 2] = baseNormals[i * 3 + 2] ?? 0;
    localNormals[i * VERTEX_STRIDE_F32 + 3] = 0;
    if (baseUvs != null) {
      localUvs[i * VERTEX_STRIDE_F32] = baseUvs[i * 2] ?? 0;
      localUvs[i * VERTEX_STRIDE_F32 + 1] = baseUvs[i * 2 + 1] ?? 0;
    }
    if (baseUv1 != null) {
      localUvs[i * VERTEX_STRIDE_F32 + 2] = baseUv1[i * 2] ?? 0;
      localUvs[i * VERTEX_STRIDE_F32 + 3] = baseUv1[i * 2 + 1] ?? 0;
    }
    if (hasCompleteTangents) {
      localTangents[i * VERTEX_STRIDE_F32] = baseTangents[i * 4] ?? 0;
      localTangents[i * VERTEX_STRIDE_F32 + 1] = baseTangents[i * 4 + 1] ?? 0;
      localTangents[i * VERTEX_STRIDE_F32 + 2] = baseTangents[i * 4 + 2] ?? 0;
      localTangents[i * VERTEX_STRIDE_F32 + 3] = baseTangents[i * 4 + 3] ?? 0;
    }
    if (hasRgbaColors) {
      localColors[i * VERTEX_STRIDE_F32] = baseColors[i * 4] ?? 1;
      localColors[i * VERTEX_STRIDE_F32 + 1] = baseColors[i * 4 + 1] ?? 1;
      localColors[i * VERTEX_STRIDE_F32 + 2] = baseColors[i * 4 + 2] ?? 1;
      localColors[i * VERTEX_STRIDE_F32 + 3] = baseColors[i * 4 + 3] ?? 1;
    } else if (hasRgbColors) {
      localColors[i * VERTEX_STRIDE_F32] = baseColors[i * 3] ?? 1;
      localColors[i * VERTEX_STRIDE_F32 + 1] = baseColors[i * 3 + 1] ?? 1;
      localColors[i * VERTEX_STRIDE_F32 + 2] = baseColors[i * 3 + 2] ?? 1;
      localColors[i * VERTEX_STRIDE_F32 + 3] = 1;
    } else {
      localColors[i * VERTEX_STRIDE_F32] = 1;
      localColors[i * VERTEX_STRIDE_F32 + 1] = 1;
      localColors[i * VERTEX_STRIDE_F32 + 2] = 1;
      localColors[i * VERTEX_STRIDE_F32 + 3] = 1;
    }
  }

  const localIndices = new Uint32Array(triCount * 4);
  for (let t = 0; t < triCount; t += 1) {
    localIndices[t * 4] = baseIndices[t * 3] ?? 0;
    localIndices[t * 4 + 1] = baseIndices[t * 3 + 1] ?? 0;
    localIndices[t * 4 + 2] = baseIndices[t * 3 + 2] ?? 0;
    localIndices[t * 4 + 3] = 0;
  }
  const localTriMaterialIds = new Uint32Array(triCount);
  localTriMaterialIds.fill(matId);
  const localBvh = buildArrayBvh(localPositions, localIndices, localTriMaterialIds);

  const indexWords: number[] = [];
  for (let i = 0; i + 3 < localBvh.reorderedIndices.length; i += 4) {
    indexWords.push(
      localBvh.reorderedIndices[i] ?? 0,
      localBvh.reorderedIndices[i + 1] ?? 0,
      localBvh.reorderedIndices[i + 2] ?? 0,
      localBvh.reorderedIndices[i + 3] ?? 0,
    );
  }
  const triMaterialIds: number[] = [];
  for (let i = 0; i < localBvh.reorderedTriMaterialIds.length; i += 1) {
    triMaterialIds.push(localBvh.reorderedTriMaterialIds[i] ?? matId);
  }

  const localNodeWords = new Uint32Array(
    localBvh.bvhNodes.buffer,
    localBvh.bvhNodes.byteOffset,
    localBvh.bvhNodes.length,
  );
  const bvhNodeWords: number[] = [];
  for (let n = 0; n + 7 < localNodeWords.length; n += 8) {
    bvhNodeWords.push(
      localNodeWords[n] ?? 0,
      localNodeWords[n + 1] ?? 0,
      localNodeWords[n + 2] ?? 0,
      localNodeWords[n + 3] ?? 0,
      localNodeWords[n + 4] ?? 0,
      localNodeWords[n + 5] ?? 0,
      localNodeWords[n + 6] ?? 0,
      localNodeWords[n + 7] ?? 0,
    );
  }

  const localAabb = computeLocalAabb(localPositions, VERTEX_STRIDE_F32);
  if (localAabb == null) return { slice: null, warnings };

  return {
    slice: {
      localPositions,
      localNormals,
      localUvs,
      localTangents,
      localColors,
      indexWords,
      triMaterialIds,
      bvhNodeWords,
      vertexCount,
      triCount,
      bvhNodeCount: Math.floor(bvhNodeWords.length / BVH_NODE_FLOATS),
      localAabbMin: localAabb.min,
      localAabbMax: localAabb.max,
      warnings,
    },
    warnings,
  };
}

/** Index a scene's primitives by id for O(1) binding resolution. */
function mapPrimitivesById(scene: Scene): Map<string, ScenePrimitive> {
  const primitiveById = new Map<string, ScenePrimitive>();
  for (const primitive of scene.primitives) {
    primitiveById.set(primitive.id, primitive);
  }
  return primitiveById;
}

/** A single resolved instance plus whether its transform was non-invertible
 *  (so the caller can decide whether/how to warn). */
interface ResolvedInstance {
  readonly instance: PendingTlasInstance;
  readonly nonInvertible: boolean;
}

/**
 * Resolve a single raw `transform` (Mat4 or undefined) into a
 * {@link PendingTlasInstance}: invert the local→world matrix, fall back to
 * identity when non-invertible, transform the local AABB into world space.
 * This is the shared inner kernel used by both {@link resolveInstanceTransforms}
 * (binding-aware batch) and the {@link packSceneFromCore} TLAS loop.
 */
function resolveOneTransform(
  transform: Mat4 | undefined,
  localAabbMin: readonly [number, number, number],
  localAabbMax: readonly [number, number, number],
  blasRoot: number,
): ResolvedInstance {
  const candidateLocalToWorld = asMat4(transform ?? IDENTITY_MAT4);
  const maybeWorldToLocal = _invertMat4(candidateLocalToWorld);
  const localToWorld = maybeWorldToLocal == null ? IDENTITY_MAT4 : candidateLocalToWorld;
  const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
  const worldAabb = transformAabb(localAabbMin, localAabbMax, localToWorld);
  return {
    nonInvertible: maybeWorldToLocal == null,
    instance: {
      blasRoot,
      worldToLocal,
      localToWorld,
      aabbMin: worldAabb.min,
      aabbMax: worldAabb.max,
    },
  };
}

/**
 * Resolve a primitive's instance transforms into {@link PendingTlasInstance}s
 * relative to a binding (invert each local→world, transform the local AABB into
 * world space, fall back to identity when non-invertible). Shared by all three
 * TLAS-instance collectors — they differ only in their instance-count policy and
 * whether they emit the non-invertible warning.
 */
function resolveInstanceTransforms(
  binding: PrimitiveTlasBinding,
  primitive: Extract<ScenePrimitive, { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }>,
): { readonly transformCount: number; readonly resolved: readonly ResolvedInstance[] } {
  const transforms =
    primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];
  const resolved: ResolvedInstance[] = [];
  for (const transform of transforms) {
    resolved.push(resolveOneTransform(transform, binding.localAabbMin, binding.localAabbMax, binding.blasRoot));
  }
  return { transformCount: transforms.length, resolved };
}

function collectTlasInstancesFromBindings(
  scene: Scene,
  bindings: readonly PrimitiveTlasBinding[],
): { readonly ok: true; readonly instances: readonly PendingTlasInstance[] } | { readonly ok: false; readonly reason: string } {
  const primitiveById = mapPrimitivesById(scene);
  const instances: PendingTlasInstance[] = [];
  for (const binding of bindings) {
    const primitive = primitiveById.get(binding.primitiveId);
    if (primitive == null) {
      return { ok: false, reason: `primitive "${binding.primitiveId}" no longer exists` };
    }
    if (!isMeshLike(primitive) || primitive.kind !== binding.primitiveKind) {
      return {
        ok: false,
        reason: `primitive "${binding.primitiveId}" kind mismatch or not mesh-like`,
      };
    }
    const { resolved } = resolveInstanceTransforms(binding, primitive);
    // V2-3: mirror the initial-pack skip — non-invertible instances are NOT
    // inserted (no identity-at-origin fallback), and the membership comparison
    // uses the INSERTED count so it matches `binding.instanceCount` (which is now
    // itself the inserted-instance count). A membership change (e.g. a formerly
    // singular instance became invertible) rejects → caller falls back to a full
    // repack.
    const insertedInstances: PendingTlasInstance[] = [];
    for (const { instance, nonInvertible } of resolved) {
      if (nonInvertible) continue;
      insertedInstances.push(instance);
    }
    if (insertedInstances.length !== binding.instanceCount) {
      return {
        ok: false,
        reason: `primitive "${binding.primitiveId}" instance count changed from ${binding.instanceCount} to ${insertedInstances.length}`,
      };
    }
    for (const instance of insertedInstances) {
      instances.push(instance);
    }
  }
  return { ok: true, instances };
}

/**
 * Size-changing BLAS splice (slice-2). The changed primitive at `bindingIndex`
 * has a NEW vertex/tri/BVH-node count, so its concat slices grow or shrink and
 * every DOWNSTREAM primitive must shift forward/back by the deltas. This rebuilds
 * ONLY the changed primitive's BLAS (the `slice` was already built by the caller)
 * and copies every other primitive's BLAS bytes verbatim — re-rebasing the global
 * vertex refs in their index words and the global triangle offsets in their leaf
 * nodes by the new deltas. The TLAS is rebuilt from the updated bindings (BLAS
 * roots moved, and the changed primitive's local AABB changed).
 *
 * Layout recap (see packSceneFromCore):
 *   positions/normals — stride 4 (vec4f/vertex); primitive p owns
 *     [vertexStart*4, (vertexStart+vertexCount)*4).
 *   indices — stride 4 (vec4u/triangle); .x.y.z are GLOBAL vertex indices
 *     (local + vertexStart), .w = 0. Primitive p owns [triStart*4, …).
 *   triMaterialIds — 1 u32/triangle at triStart.
 *   bvhNodes — 8 words/node at blasRoot*8. Leaf word[6] = GLOBAL tri offset
 *     (local + triStart); interior word[6] = RELATIVE child offset (unchanged).
 */
/** The mutated buffer set + counts shared by both splice paths' return value. */
interface SplicedPackBuffers {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs: Float32Array;
  readonly tangents: Float32Array;
  readonly colors: Float32Array;
  readonly indices: Uint32Array;
  readonly triMaterialIds: Uint32Array;
  readonly bvhNodes: Float32Array;
  readonly triangleCount: number;
}

/**
 * Rebuild the TLAS over `primitiveTlasBindings` and assemble the spliced pack
 * result (D12-4). Shared by the same-size and resize splice paths — both, after
 * mutating the BLAS buffers, collect TLAS instances (falling back to a full
 * `packSceneFromCore` when a transform became non-invertible), build the TLAS,
 * and return `{ ok, strategy: 'splice', pack }`. Behavior is bit-for-bit
 * identical to the former inline tails.
 */
function finalizeSplicedPack(
  prev: ScenePackResult,
  scene: Scene,
  opts: ScenePackOptions,
  buffers: SplicedPackBuffers,
  primitiveTlasBindings: readonly PrimitiveTlasBinding[],
  sliceWarnings: readonly string[],
): RebuildPrimitiveBlasResult {
  const collected = collectTlasInstancesFromBindings(scene, primitiveTlasBindings);
  if (!collected.ok) {
    return { ok: true, pack: packSceneFromCore(scene, opts), strategy: 'full' };
  }
  const tlasBuild = buildTlasFromInstances(collected.instances);
  return {
    ok: true,
    strategy: 'splice',
    pack: {
      positions: buffers.positions,
      normals: buffers.normals,
      uvs: buffers.uvs,
      tangents: buffers.tangents,
      colors: buffers.colors,
      indices: buffers.indices,
      triMaterialIds: buffers.triMaterialIds,
      bvhNodes: buffers.bvhNodes,
      triangleCount: buffers.triangleCount,
      tlasNodes: tlasBuild.tlasNodes,
      tlasInstanceIndices: tlasBuild.tlasInstanceIndices,
      tlasBlasRoots: tlasBuild.tlasBlasRoots,
      tlasInstanceWorldToLocal: tlasBuild.tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld: tlasBuild.tlasInstanceLocalToWorld,
      tlasNodeCount: tlasBuild.tlasNodeCount,
      primitiveTlasBindings,
      warnings: [...prev.warnings, ...sliceWarnings],
    },
  };
}

function spliceResizedPrimitiveBlasIntoPack(
  prev: ScenePackResult,
  bindingIndex: number,
  slice: PackedPrimitiveSlice,
  oldNodeCount: number,
  scene: Scene,
  opts: ScenePackOptions,
): RebuildPrimitiveBlasResult {
  const binding = prev.primitiveTlasBindings[bindingIndex]!;

  // Pre-edit offsets of the changed primitive stay fixed (they are determined by
  // the verbatim prefix). The deltas shift everything after it.
  const deltaVert = slice.vertexCount - binding.vertexCount;
  const deltaTri = slice.triCount - binding.triCount;
  const deltaNode = slice.bvhNodeCount - oldNodeCount;

  const prevTotalVerts = Math.floor(prev.positions.length / VERTEX_STRIDE_F32);
  const prevTotalTris = prev.triangleCount;
  const prevTotalNodes = Math.floor(prev.bvhNodes.length / BVH_NODE_FLOATS);

  // The changed primitive's verbatim node span in the OLD pack.
  const oldNodeStart = binding.blasRoot;
  const oldNodeEnd = oldNodeStart + oldNodeCount;
  // Its tri span and vert span in the OLD pack.
  const oldTriStart = binding.triStart;
  const oldTriEnd = oldTriStart + binding.triCount;
  const oldVertStart = binding.vertexStart;
  const oldVertEnd = oldVertStart + binding.vertexCount;

  const newTotalVerts = prevTotalVerts + deltaVert;
  const newTotalTris = prevTotalTris + deltaTri;
  const newTotalNodes = prevTotalNodes + deltaNode;

  const positions = new Float32Array(newTotalVerts * VERTEX_STRIDE_F32);
  const normals = new Float32Array(newTotalVerts * VERTEX_STRIDE_F32);
  const uvs = new Float32Array(newTotalVerts * VERTEX_STRIDE_F32);
  const tangents = new Float32Array(newTotalVerts * VERTEX_STRIDE_F32);
  const colors = new Float32Array(newTotalVerts * VERTEX_STRIDE_F32);
  const indices = new Uint32Array(newTotalTris * 4);
  const triMaterialIds = new Uint32Array(newTotalTris);
  const newNodeView = new Uint32Array(newTotalNodes * BVH_NODE_FLOATS);
  const prevNodeView = new Uint32Array(
    prev.bvhNodes.buffer,
    prev.bvhNodes.byteOffset,
    prev.bvhNodes.length,
  );

  // ── Positions / normals (vec4f-strided) ──────────────────────────────────
  // Prefix [0, oldVertStart) verbatim.
  positions.set(prev.positions.subarray(0, oldVertStart * VERTEX_STRIDE_F32), 0);
  normals.set(prev.normals.subarray(0, oldVertStart * VERTEX_STRIDE_F32), 0);
  uvs.set(prev.uvs.subarray(0, oldVertStart * VERTEX_STRIDE_F32), 0);
  tangents.set(prev.tangents.subarray(0, oldVertStart * VERTEX_STRIDE_F32), 0);
  colors.set(prev.colors.subarray(0, oldVertStart * VERTEX_STRIDE_F32), 0);
  // Changed primitive's new local slice at the SAME vertexStart.
  positions.set(slice.localPositions, oldVertStart * VERTEX_STRIDE_F32);
  normals.set(slice.localNormals, oldVertStart * VERTEX_STRIDE_F32);
  uvs.set(slice.localUvs, oldVertStart * VERTEX_STRIDE_F32);
  tangents.set(slice.localTangents, oldVertStart * VERTEX_STRIDE_F32);
  colors.set(slice.localColors, oldVertStart * VERTEX_STRIDE_F32);
  // Suffix (downstream primitives) shifted by deltaVert*VERTEX_STRIDE_F32 floats.
  if (oldVertEnd < prevTotalVerts) {
    positions.set(prev.positions.subarray(oldVertEnd * VERTEX_STRIDE_F32), (oldVertEnd + deltaVert) * VERTEX_STRIDE_F32);
    normals.set(prev.normals.subarray(oldVertEnd * VERTEX_STRIDE_F32), (oldVertEnd + deltaVert) * VERTEX_STRIDE_F32);
    uvs.set(prev.uvs.subarray(oldVertEnd * VERTEX_STRIDE_F32), (oldVertEnd + deltaVert) * VERTEX_STRIDE_F32);
    tangents.set(prev.tangents.subarray(oldVertEnd * VERTEX_STRIDE_F32), (oldVertEnd + deltaVert) * VERTEX_STRIDE_F32);
    colors.set(prev.colors.subarray(oldVertEnd * VERTEX_STRIDE_F32), (oldVertEnd + deltaVert) * VERTEX_STRIDE_F32);
  }

  // ── Indices (vec4u-strided; .x.y.z global vertex refs, .w = 0) ────────────
  const prevIndices = prev.indices;
  // Prefix triangles [0, oldTriStart) verbatim (their vertex refs are unaffected
  // — they reference vertices before oldVertStart).
  indices.set(prevIndices.subarray(0, oldTriStart * 4), 0);
  triMaterialIds.set(prev.triMaterialIds.subarray(0, oldTriStart), 0);
  // Changed primitive's new index words rebased to its (unchanged) vertexStart.
  const newTriStart = oldTriStart; // unchanged for the spliced primitive
  _rebaseIndexWords(indices, newTriStart * 4, slice.indexWords, binding.vertexStart);
  for (let t = 0; t < slice.triMaterialIds.length; t += 1) {
    triMaterialIds[newTriStart + t] = slice.triMaterialIds[t] ?? 0;
  }
  // Downstream triangles: copy with vertex refs shifted by deltaVert.
  for (let t = oldTriEnd; t < prevTotalTris; t += 1) {
    _copyVec4Strided(indices, triMaterialIds, prevIndices, prev.triMaterialIds, t, t + deltaTri, deltaVert);
  }

  // ── BVH nodes (BVH_NODE_FLOATS words/node) ───────────────────────────────
  // Prefix nodes [0, oldNodeStart) verbatim. Leaf word[6] is a GLOBAL tri offset
  // into a region BEFORE the changed primitive, so it is unaffected; interior
  // word[6] is relative and self-contained within the prefix subtrees.
  newNodeView.set(prevNodeView.subarray(0, oldNodeStart * BVH_NODE_FLOATS), 0);
  // Changed primitive's new nodes at the SAME blasRoot, leaf offsets rebased to
  // its (unchanged) triStart.
  const newBlasRoot = oldNodeStart; // unchanged for the spliced primitive
  for (let n = 0; n + 7 < slice.bvhNodeWords.length; n += BVH_NODE_FLOATS) {
    _rebaseLeafTriOffset(newNodeView, newBlasRoot * BVH_NODE_FLOATS + n, slice.bvhNodeWords, n, binding.triStart);
  }
  // Downstream nodes shifted by deltaNode. Leaf global tri offsets shift by
  // deltaTri; interior relative child offsets are unchanged (the subtree shape
  // moves rigidly).
  for (let n = oldNodeEnd; n < prevTotalNodes; n += 1) {
    _rebaseLeafTriOffset(newNodeView, (n + deltaNode) * BVH_NODE_FLOATS, prevNodeView, n * BVH_NODE_FLOATS, deltaTri);
  }

  const bvhNodes = new Float32Array(newNodeView.buffer);

  // ── Rebase bindings: changed primitive gets the new counts + AABB; downstream
  // bindings shift by the deltas. ──────────────────────────────────────────────
  const primitiveTlasBindings = prev.primitiveTlasBindings.map((b, i) => {
    if (i < bindingIndex) return b;
    if (i === bindingIndex) {
      return {
        ...b,
        vertexCount: slice.vertexCount,
        triCount: slice.triCount,
        localAabbMin: slice.localAabbMin,
        localAabbMax: slice.localAabbMax,
      };
    }
    return {
      ...b,
      blasRoot: b.blasRoot + deltaNode,
      vertexStart: b.vertexStart + deltaVert,
      triStart: b.triStart + deltaTri,
    };
  });

  // BLAS roots moved (and the changed primitive's local AABB changed), so the
  // TLAS must be fully rebuilt from the updated bindings.
  return finalizeSplicedPack(
    prev,
    scene,
    opts,
    { positions, normals, uvs, tangents, colors, indices, triMaterialIds, bvhNodes, triangleCount: newTotalTris },
    primitiveTlasBindings,
    slice.warnings,
  );
}

function splicePrimitiveBlasIntoPack(
  prev: ScenePackResult,
  bindingIndex: number,
  slice: PackedPrimitiveSlice,
  scene: Scene,
  opts: ScenePackOptions,
): RebuildPrimitiveBlasResult {
  const binding = prev.primitiveTlasBindings[bindingIndex]!;
  const nodeStart = binding.blasRoot;
  const nextBinding = prev.primitiveTlasBindings[bindingIndex + 1];
  const nodeEnd = nextBinding != null ? nextBinding.blasRoot : Math.floor(prev.bvhNodes.length / BVH_NODE_FLOATS);
  const oldNodeCount = nodeEnd - nodeStart;

  if (
    slice.vertexCount !== binding.vertexCount
    || slice.triCount !== binding.triCount
    || slice.bvhNodeCount !== oldNodeCount
  ) {
    // PR-4.3 deepening (slice-2): the changed primitive's BLAS genuinely changed
    // SIZE. Grow/shrink the concat buffers around it and rebase every downstream
    // primitive's offsets + the TLAS BLAS roots — no full per-primitive rebuild.
    return spliceResizedPrimitiveBlasIntoPack(
      prev,
      bindingIndex,
      slice,
      oldNodeCount,
      scene,
      opts,
    );
  }

  const positions = new Float32Array(prev.positions);
  const normals = new Float32Array(prev.normals);
  const uvs = new Float32Array(prev.uvs);
  const tangents = new Float32Array(prev.tangents);
  const colors = new Float32Array(prev.colors);
  const indices = new Uint32Array(prev.indices);
  const triMaterialIds = new Uint32Array(prev.triMaterialIds);
  const bvhNodes = new Float32Array(prev.bvhNodes);

  const vertOff = binding.vertexStart * VERTEX_STRIDE_F32;
  positions.set(slice.localPositions, vertOff);
  normals.set(slice.localNormals, vertOff);
  uvs.set(slice.localUvs, vertOff);
  tangents.set(slice.localTangents, vertOff);
  colors.set(slice.localColors, vertOff);

  const indexOff = binding.triStart * 4;
  _rebaseIndexWords(indices, indexOff, slice.indexWords, binding.vertexStart);
  for (let t = 0; t < slice.triMaterialIds.length; t += 1) {
    triMaterialIds[binding.triStart + t] = slice.triMaterialIds[t] ?? 0;
  }

  const nodeWordStart = nodeStart * BVH_NODE_FLOATS;
  const nodeView = new Uint32Array(bvhNodes.buffer);
  for (let n = 0; n + 7 < slice.bvhNodeWords.length; n += BVH_NODE_FLOATS) {
    _rebaseLeafTriOffset(nodeView, nodeWordStart + n, slice.bvhNodeWords, n, binding.triStart);
  }

  const primitiveTlasBindings = prev.primitiveTlasBindings.map((b, i) =>
    i === bindingIndex
      ? {
          ...b,
          localAabbMin: slice.localAabbMin,
          localAabbMax: slice.localAabbMax,
        }
      : b,
  );

  return finalizeSplicedPack(
    prev,
    scene,
    opts,
    { positions, normals, uvs, tangents, colors, indices, triMaterialIds, bvhNodes, triangleCount: prev.triangleCount },
    primitiveTlasBindings,
    slice.warnings,
  );
}

/**
 * Pack mesh-like primitives from a `@vitrum/core` `Scene` into concatenated
 * local BLAS buffers and an optional TLAS over instances.
 *
 * **`tlas:false` + multiple primitives:** when the caller explicitly disables TLAS
 * but the scene contains more than one mesh-like primitive, the resulting
 * per-BLAS-only concat is an untraversable forest (the WGSL traversal expects a
 * single root node, not multiple disjoint subtrees). The packer automatically
 * upgrades to `tlas:true` in this case and emits a warning. If you intentionally
 * want a single-primitive pack without a TLAS, ensure the scene contains exactly
 * one mesh-like primitive.
 */
export function packSceneFromCore(scene: Scene, opts: ScenePackOptions): ScenePackResult {
  // H34-d: guard the `tlas:false` + multiple-primitive footgun. Count mesh-like
  // primitives first (excluding analytics that are already skipped below). Auto-
  // upgrade to tlas mode if >1 mesh-like primitive is present.
  const meshLikeCount = scene.primitives.filter(
    (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
  ).length;
  const requestedTlas = opts.tlas !== false;
  const buildTlasTree = requestedTlas || meshLikeCount > 1;
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const tangents: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const triMaterialIds: number[] = [];
  const bvhNodeWords: number[] = [];
  const pendingTlasInstances: PendingTlasInstance[] = [];
  const primitiveTlasBindings: PrimitiveTlasBinding[] = [];
  const warnings: string[] = [];

  if (!requestedTlas && meshLikeCount > 1) {
    warnings.push(
      `packSceneFromCore: tlas:false was requested but the scene has ${meshLikeCount} mesh-like ` +
      `primitives. A per-BLAS-only concat of multiple primitives is an untraversable forest; ` +
      `automatically upgrading to tlas:true.`,
    );
  }

  for (const primitive of scene.primitives) {
    if (!isMeshLike(primitive)) {
      warnings.push(
        `Primitive "${primitive.id}" (${primitive.kind}) skipped; scenePack supports mesh, skinned-mesh, and instanced-mesh only.`,
      );
      continue;
    }

    const matId = opts.resolveMaterialId(primitive.id);
    // Theme-F dedup: build this primitive's LOCAL slice via the shared
    // `packOneMeshLikePrimitive` (same vec4 expansion + buildArrayBvh + node
    // extraction); the loop here only concatenates + rebases + does TLAS
    // bookkeeping.
    const { slice, warnings: sliceWarnings } = packOneMeshLikePrimitive(primitive, matId);
    warnings.push(...sliceWarnings);
    if (slice == null) {
      continue;
    }

    // H34-c: Skip zero-instance instanced meshes entirely — before concatenating
    // any geometry — so no orphan BLAS nodes are emitted that would break the
    // bindings-tile-the-node-array invariant.
    if (primitive.kind === 'instanced-mesh' && primitive.instances.length === 0) {
      warnings.push(
        `Primitive "${primitive.id}" is an instanced-mesh with zero instances; ` +
        `skipping (no geometry or BLAS nodes contributed).`,
      );
      continue;
    }

    const vertexCount = slice.vertexCount;
    const triCount = slice.triCount;
    const vertexBase = Math.floor(positions.length / VERTEX_STRIDE_F32);
    const triBase = triMaterialIds.length;
    const nodeBase = Math.floor(bvhNodeWords.length / BVH_NODE_FLOATS);

    for (let i = 0; i < slice.localPositions.length; i += 1) positions.push(slice.localPositions[i] ?? 0);
    for (let i = 0; i < slice.localNormals.length; i += 1) normals.push(slice.localNormals[i] ?? 0);
    for (let i = 0; i < slice.localUvs.length; i += 1) uvs.push(slice.localUvs[i] ?? 0);
    for (let i = 0; i < slice.localTangents.length; i += 1) tangents.push(slice.localTangents[i] ?? 0);
    for (let i = 0; i < slice.localColors.length; i += 1) colors.push(slice.localColors[i] ?? 1);
    for (let i = 0; i + 3 < slice.indexWords.length; i += 4) {
      indices.push(
        (slice.indexWords[i] ?? 0) + vertexBase,
        (slice.indexWords[i + 1] ?? 0) + vertexBase,
        (slice.indexWords[i + 2] ?? 0) + vertexBase,
        slice.indexWords[i + 3] ?? 0,
      );
    }
    for (let i = 0; i < slice.triMaterialIds.length; i += 1) {
      triMaterialIds.push(slice.triMaterialIds[i] ?? matId);
    }

    for (let n = 0; n + 7 < slice.bvhNodeWords.length; n += BVH_NODE_FLOATS) {
      const splitOrCount = slice.bvhNodeWords[n + 7] ?? 0;
      const isLeaf = isLeafSplit(splitOrCount);
      bvhNodeWords.push(
        slice.bvhNodeWords[n] ?? 0,
        slice.bvhNodeWords[n + 1] ?? 0,
        slice.bvhNodeWords[n + 2] ?? 0,
        slice.bvhNodeWords[n + 3] ?? 0,
        slice.bvhNodeWords[n + 4] ?? 0,
        slice.bvhNodeWords[n + 5] ?? 0,
        isLeaf ? (slice.bvhNodeWords[n + 6] ?? 0) + triBase : (slice.bvhNodeWords[n + 6] ?? 0),
        splitOrCount,
      );
    }

    const localAabb = { min: slice.localAabbMin, max: slice.localAabbMax };

    if (buildTlasTree) {
      const transforms =
        primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];
      // Zero-instance case is handled above (before geometry concatenation); this
      // branch should never be reached with an empty transforms array now.
      // `insertedInstanceCount` counts only the instances that actually land in
      // the TLAS — non-invertible (singular) transforms are skipped, so the
      // per-primitive `instanceCount` reflects TLAS MEMBERSHIP, not the raw
      // transform count (V2-3: count-vs-membership reconcile).
      let insertedInstanceCount = 0;
      for (const transform of transforms) {
        const { instance, nonInvertible } = resolveOneTransform(
          transform,
          localAabb.min,
          localAabb.max,
          nodeBase,
        );
        if (nonInvertible) {
          // H34-e: singular transform → skip this TLAS instance with a warning
          // rather than silently placing geometry at the origin.
          warnings.push(
            `Primitive "${primitive.id}" has non-invertible instance transform; ` +
            `skipping this TLAS instance (geometry would be placed at the origin otherwise).`,
          );
          continue;
        }
        pendingTlasInstances.push(instance);
        insertedInstanceCount += 1;
      }
      primitiveTlasBindings.push({
        primitiveId: primitive.id,
        primitiveKind: primitive.kind,
        blasRoot: nodeBase,
        instanceCount: insertedInstanceCount,
        vertexStart: vertexBase,
        vertexCount,
        triStart: triBase,
        triCount,
        localAabbMin: localAabb.min,
        localAabbMax: localAabb.max,
      });
    }
  }

  const packedPositions = new Float32Array(positions);
  const packedNormals = new Float32Array(normals);
  const packedUvs = new Float32Array(uvs);
  const packedTangents = new Float32Array(tangents);
  const packedColors = new Float32Array(colors);
  const packedIndices = new Uint32Array(indices);
  const packedTriMaterialIds = new Uint32Array(triMaterialIds);
  const packedBvhNodes = new Float32Array(new Uint32Array(bvhNodeWords).buffer);
  const tlasBuild = buildTlasTree
    ? buildTlasFromInstances(pendingTlasInstances)
    : {
        tlasNodes: new Uint32Array(0),
        tlasInstanceIndices: new Uint32Array(0),
        tlasBlasRoots: new Uint32Array(0),
        tlasInstanceWorldToLocal: new Float32Array(0),
        tlasInstanceLocalToWorld: new Float32Array(0),
        tlasNodeCount: 0,
      };

  return {
    positions: packedPositions,
    normals: packedNormals,
    uvs: packedUvs,
    tangents: packedTangents,
    colors: packedColors,
    indices: packedIndices,
    triMaterialIds: packedTriMaterialIds,
    bvhNodes: packedBvhNodes,
    triangleCount: packedTriMaterialIds.length,
    tlasNodes: tlasBuild.tlasNodes,
    tlasInstanceIndices: tlasBuild.tlasInstanceIndices,
    tlasBlasRoots: tlasBuild.tlasBlasRoots,
    tlasInstanceWorldToLocal: tlasBuild.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: tlasBuild.tlasInstanceLocalToWorld,
    tlasNodeCount: tlasBuild.tlasNodeCount,
    primitiveTlasBindings,
    warnings,
  };
}

/** Refit TLAS instance transforms / world AABBs without topology change. */
export function refitTlasTransforms(
  scene: Scene,
  primitiveTlasBindings: readonly PrimitiveTlasBinding[],
  prevTlas?: TlasGpuSnapshot,
): RefitTlasResult {
  const primitiveById = mapPrimitivesById(scene);
  const warnings: string[] = [];
  const pendingTlasInstances: PendingTlasInstance[] = [];
  const refitAabbs: Array<{ min: readonly [number, number, number]; max: readonly [number, number, number] }> = [];

  for (const binding of primitiveTlasBindings) {
    const primitive = primitiveById.get(binding.primitiveId);
    if (primitive == null) {
      return {
        ok: false,
        reason: `refitTlasTransforms: primitive "${binding.primitiveId}" no longer exists.`,
      };
    }
    if (primitive.kind !== binding.primitiveKind) {
      return {
        ok: false,
        reason: `refitTlasTransforms: primitive "${binding.primitiveId}" changed kind ` +
          `from "${binding.primitiveKind}" to "${primitive.kind}".`,
      };
    }
    if (!isMeshLike(primitive)) {
      return {
        ok: false,
        reason: `refitTlasTransforms: primitive "${binding.primitiveId}" is not mesh-like.`,
      };
    }
    const { transformCount, resolved } = resolveInstanceTransforms(binding, primitive);
    if (transformCount !== binding.instanceCount) {
      return {
        ok: false,
        reason: `refitTlasTransforms: primitive "${binding.primitiveId}" instance count changed ` +
          `from ${binding.instanceCount} to ${transformCount}.`,
      };
    }
    for (const { instance, nonInvertible } of resolved) {
      if (nonInvertible) {
        return {
          ok: false,
          reason:
            `refitTlasTransforms: primitive "${primitive.id}" has a non-invertible ` +
            `instance transform. Rebuild the pack so the singular instance can be skipped ` +
            `instead of refit as identity-at-origin.`,
        };
      }
      refitAabbs.push({ min: instance.aabbMin, max: instance.aabbMax });
      pendingTlasInstances.push(instance);
    }
  }

  if (
    prevTlas != null &&
    prevTlas.tlasNodes.length > 0 &&
    prevTlas.tlasInstanceIndices.length === refitAabbs.length &&
    prevTlas.tlasBlasRoots.length === refitAabbs.length &&
    prevTlas.tlasInstanceWorldToLocal.length === refitAabbs.length * MAT4_STRIDE_F32
  ) {
    const refitNodes = new Uint32Array(prevTlas.tlasNodes);
    refitTlas(
      {
        nodes: refitNodes,
        nodeCount: Math.floor(refitNodes.length / BVH_NODE_FLOATS),
        instanceIndices: prevTlas.tlasInstanceIndices,
        blasRoots: prevTlas.tlasBlasRoots,
        instanceTransforms: prevTlas.tlasInstanceWorldToLocal,
      },
      refitAabbs,
    );
    const l2w = new Float32Array(pendingTlasInstances.length * MAT4_STRIDE_F32);
    const w2l = new Float32Array(pendingTlasInstances.length * MAT4_STRIDE_F32);
    for (let i = 0; i < pendingTlasInstances.length; i += 1) {
      l2w.set(pendingTlasInstances[i]!.localToWorld, i * MAT4_STRIDE_F32);
      w2l.set(pendingTlasInstances[i]!.worldToLocal, i * MAT4_STRIDE_F32);
    }
    return {
      ok: true,
      tlasNodes: refitNodes,
      tlasInstanceIndices: prevTlas.tlasInstanceIndices,
      tlasBlasRoots: prevTlas.tlasBlasRoots,
      tlasInstanceWorldToLocal: w2l,
      tlasInstanceLocalToWorld: l2w,
      warnings,
    };
  }

  const tlas = buildTlasFromInstances(pendingTlasInstances);
  return {
    ok: true,
    tlasNodes: tlas.tlasNodes,
    tlasInstanceIndices: tlas.tlasInstanceIndices,
    tlasBlasRoots: tlas.tlasBlasRoots,
    tlasInstanceWorldToLocal: tlas.tlasInstanceWorldToLocal,
    tlasInstanceLocalToWorld: tlas.tlasInstanceLocalToWorld,
    warnings,
  };
}

/** Union world-space AABB over all TLAS instances (for RC cascade bounds). */
export function computeWorldAabbForBindings(
  scene: Scene,
  bindings: readonly PrimitiveTlasBinding[],
): {
  min: readonly [number, number, number];
  max: readonly [number, number, number];
} | null {
  const byId = mapPrimitivesById(scene);
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  let any = false;
  for (const binding of bindings) {
    const primitive = byId.get(binding.primitiveId);
    if (primitive == null || !isMeshLike(primitive)) continue;
    const transforms =
      primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? IDENTITY_MAT4];
    for (const transform of transforms) {
      const l2w = asMat4(transform ?? IDENTITY_MAT4);
      const world = transformAabb(binding.localAabbMin, binding.localAabbMax, l2w);
      minX = Math.min(minX, world.min[0]);
      minY = Math.min(minY, world.min[1]);
      minZ = Math.min(minZ, world.min[2]);
      maxX = Math.max(maxX, world.max[0]);
      maxY = Math.max(maxY, world.max[1]);
      maxZ = Math.max(maxZ, world.max[2]);
      any = true;
    }
  }
  if (!any) return null;
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/**
 * Collect TLAS instances from the CURRENT scene, taking each primitive's live
 * instance list (whose length may differ from the binding's stored
 * `instanceCount`). Unlike {@link collectTlasInstancesFromBindings}, this does
 * NOT bail on an instance-count change — it is the basis for a TLAS-only rebuild
 * that reuses verbatim BLAS buffers when only the instance count changed. BLAS
 * geometry (positions/normals/indices/nodes) is shared across instances, so an
 * instance-count change leaves every BLAS byte identical; only the TLAS and the
 * per-primitive `instanceCount` move.
 */
function collectLiveTlasInstancesFromBindings(
  scene: Scene,
  bindings: readonly PrimitiveTlasBinding[],
):
  | { readonly ok: true; readonly instances: readonly PendingTlasInstance[]; readonly liveCounts: readonly number[]; readonly warnings: readonly string[] }
  | { readonly ok: false; readonly reason: string } {
  const primitiveById = mapPrimitivesById(scene);
  const instances: PendingTlasInstance[] = [];
  const liveCounts: number[] = [];
  const warnings: string[] = [];
  for (const binding of bindings) {
    const primitive = primitiveById.get(binding.primitiveId);
    if (primitive == null) {
      return { ok: false, reason: `primitive "${binding.primitiveId}" no longer exists` };
    }
    if (!isMeshLike(primitive) || primitive.kind !== binding.primitiveKind) {
      return {
        ok: false,
        reason: `primitive "${binding.primitiveId}" kind mismatch or not mesh-like`,
      };
    }
    const { transformCount, resolved } = resolveInstanceTransforms(binding, primitive);
    if (transformCount === 0) {
      return {
        ok: false,
        reason: `primitive "${binding.primitiveId}" has zero instances (TLAS-only rebuild needs at least one)`,
      };
    }
    // V2-3: mirror the initial-pack skip behavior — a non-invertible (singular)
    // instance is SKIPPED (not inserted at identity), and `liveCounts` counts
    // only the instances that actually land in the TLAS so the count matches
    // membership on both build paths.
    let insertedCount = 0;
    for (const { instance, nonInvertible } of resolved) {
      if (nonInvertible) {
        warnings.push(
          `Primitive "${binding.primitiveId}" has non-invertible instance transform; ` +
          `skipping this TLAS instance (geometry would be placed at the origin otherwise).`,
        );
        continue;
      }
      instances.push(instance);
      insertedCount += 1;
    }
    liveCounts.push(insertedCount);
  }
  return { ok: true, instances, liveCounts, warnings };
}

export type RebuildTlasReuseBlasResult =
  | { readonly ok: true; readonly pack: ScenePackResult }
  | { readonly ok: false; readonly reason: string };

/**
 * Rebuild ONLY the TLAS after an instanced-mesh's instance COUNT changed,
 * reusing the previous pack's BLAS node/index/position/normal arrays VERBATIM
 * (no per-triangle `buildArrayBvh` SAH rebuild). This is the clean slice-1 win
 * for incremental instanced-mesh add/remove: shared BLAS geometry is byte-
 * identical across instances, so only the TLAS tree + per-primitive
 * `instanceCount` need to change.
 *
 * Rejects (returns `ok:false`) when the patch is NOT a pure instance-count
 * change — e.g. a primitive's vertex/tri/BVH-node geometry changed, a primitive
 * disappeared, or no instance count actually moved — so the caller can fall back
 * to a full {@link packSceneFromCore} rebuild. The caller is responsible for
 * having already verified that the changed primitive is an `instanced-mesh`
 * whose only mutated facet is `instances`.
 */
export function rebuildTlasReuseBlas(
  scene: Scene,
  prev: ScenePackResult,
): RebuildTlasReuseBlasResult {
  if (prev.primitiveTlasBindings.length === 0) {
    return { ok: false, reason: 'previous pack has no TLAS bindings; full rebuild required' };
  }
  const collected = collectLiveTlasInstancesFromBindings(scene, prev.primitiveTlasBindings);
  if (!collected.ok) {
    return { ok: false, reason: collected.reason };
  }

  // Verify the ONLY thing that changed is one-or-more instanced-mesh instance
  // counts. If every live count equals its binding's stored count, there is no
  // count change and the caller should have taken the transform-only refit path
  // instead — reject so we don't silently shadow that path. (A non-instanced
  // binding can never legitimately change count here; `liveCounts` for a mesh /
  // skinned-mesh binding is always 1, matching its stored `instanceCount` of 1.)
  let anyCountChanged = false;
  for (let i = 0; i < prev.primitiveTlasBindings.length; i += 1) {
    const binding = prev.primitiveTlasBindings[i]!;
    const liveCount = collected.liveCounts[i]!;
    if (liveCount !== binding.instanceCount) {
      if (binding.primitiveKind !== 'instanced-mesh') {
        return {
          ok: false,
          reason: `primitive "${binding.primitiveId}" (${binding.primitiveKind}) changed instance count; only instanced-mesh count changes are TLAS-only`,
        };
      }
      anyCountChanged = true;
    }
  }
  if (!anyCountChanged) {
    return { ok: false, reason: 'no instance count changed; use the transform-only refit path' };
  }

  const tlasBuild = buildTlasFromInstances(collected.instances);
  const primitiveTlasBindings = prev.primitiveTlasBindings.map((binding, i) => {
    const liveCount = collected.liveCounts[i]!;
    return liveCount === binding.instanceCount ? binding : { ...binding, instanceCount: liveCount };
  });

  return {
    ok: true,
    pack: {
      // BLAS buffers reused verbatim — no per-triangle rebuild.
      positions: prev.positions,
      normals: prev.normals,
      uvs: prev.uvs,
      tangents: prev.tangents,
      colors: prev.colors,
      indices: prev.indices,
      triMaterialIds: prev.triMaterialIds,
      bvhNodes: prev.bvhNodes,
      triangleCount: prev.triangleCount,
      // TLAS rebuilt from the new instance list.
      tlasNodes: tlasBuild.tlasNodes,
      tlasInstanceIndices: tlasBuild.tlasInstanceIndices,
      tlasBlasRoots: tlasBuild.tlasBlasRoots,
      tlasInstanceWorldToLocal: tlasBuild.tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld: tlasBuild.tlasInstanceLocalToWorld,
      tlasNodeCount: tlasBuild.tlasNodeCount,
      primitiveTlasBindings,
      warnings: [...prev.warnings, ...collected.warnings],
    },
  };
}

export type RebuildPrimitiveBlasResult =
  | { readonly ok: true; readonly pack: ScenePackResult; readonly strategy: 'splice' | 'full' }
  | { readonly ok: false; readonly reason: string };

/**
 * Repack after one primitive's topology changed (PR-4.3).
 * In-place BLAS splice when vertex/tri/BVH node counts are unchanged; else full repack.
 */
export function rebuildPrimitiveBlas(
  scene: Scene,
  primitiveId: string,
  prev: ScenePackResult,
  opts: ScenePackOptions,
): RebuildPrimitiveBlasResult {
  const bindingIndex = prev.primitiveTlasBindings.findIndex((b) => b.primitiveId === primitiveId);
  if (bindingIndex < 0) {
    const prim = scene.primitives.find((p) => p.id === primitiveId);
    if (prim == null) {
      return { ok: false, reason: `primitive "${primitiveId}" not found in scene or previous pack` };
    }
    return {
      ok: false,
      reason: `primitive "${primitiveId}" was not in the previous TLAS pack (topology-only rebuild requires full scene pack)`,
    };
  }

  const primitive = scene.primitives.find((p) => p.id === primitiveId);
  if (primitive == null || !isMeshLike(primitive)) {
    return { ok: false, reason: `primitive "${primitiveId}" is missing or not mesh-like` };
  }

  const { slice } = packOneMeshLikePrimitive(primitive, opts.resolveMaterialId(primitiveId));
  if (slice == null) {
    return { ok: true, pack: packSceneFromCore(scene, opts), strategy: 'full' };
  }

  return splicePrimitiveBlasIntoPack(prev, bindingIndex, slice, scene, opts);
}
