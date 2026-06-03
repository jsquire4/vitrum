/**
 * scenePack.ts — THREE-free scene geometry packer (BLAS concat + optional TLAS).
 *
 * Canonical CPU pack used by walkaround-hybrid and pt-webgpu. Extracted from
 * pt-webgpu `uploadSceneBuffers.ts` (PR-2.1).
 */

import { asMat4, type Mat4, type Scene, type ScenePrimitive, type Vec3 } from '@vitrum/core';
import { buildArrayBvh, isLeafSplit } from './buildArrayBvh.js';
import { buildTlas, refitTlas } from './tlas.js';

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

export function invertMat4(m: Mat4): Float32Array | null {
  const at = (index: number): number => m[index] ?? 0;
  const out = new Float32Array(16);
  const a00 = at(0), a01 = at(1), a02 = at(2), a03 = at(3);
  const a10 = at(4), a11 = at(5), a12 = at(6), a13 = at(7);
  const a20 = at(8), a21 = at(9), a22 = at(10), a23 = at(11);
  const a30 = at(12), a31 = at(13), a32 = at(14), a33 = at(15);
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-10) return null;
  const invDet = 1.0 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * invDet;
  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * invDet;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return out;
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

export function computeLocalAabb(positions: Float32Array): {
  min: readonly [number, number, number];
  max: readonly [number, number, number];
} | null {
  if (positions.length < 3) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i + 2 < positions.length; i += 3) {
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
  const l2w = new Float32Array(instances.length * 16);
  for (let i = 0; i < instances.length; i += 1) {
    l2w.set(instances[i]!.localToWorld, i * 16);
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
  const basePositions = primitive.positions;
  const vertexCount = Math.floor(basePositions.length / 3);
  if (vertexCount < 3) {
    warnings.push(`Primitive "${primitive.id}" has fewer than 3 vertices; skipping.`);
    return { slice: null, warnings };
  }
  const baseIndices =
    primitive.indices ??
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

  const localPositions = new Float32Array(vertexCount * 4);
  const localNormals = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i += 1) {
    localPositions[i * 4] = basePositions[i * 3] ?? 0;
    localPositions[i * 4 + 1] = basePositions[i * 3 + 1] ?? 0;
    localPositions[i * 4 + 2] = basePositions[i * 3 + 2] ?? 0;
    localPositions[i * 4 + 3] = 0;
    localNormals[i * 4] = primitive.normals[i * 3] ?? 0;
    localNormals[i * 4 + 1] = primitive.normals[i * 3 + 1] ?? 1;
    localNormals[i * 4 + 2] = primitive.normals[i * 3 + 2] ?? 0;
    localNormals[i * 4 + 3] = 0;
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

  const localAabb = computeLocalAabb(basePositions);
  if (localAabb == null) return { slice: null, warnings };

  return {
    slice: {
      localPositions,
      localNormals,
      indexWords,
      triMaterialIds,
      bvhNodeWords,
      vertexCount,
      triCount,
      bvhNodeCount: Math.floor(bvhNodeWords.length / 8),
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
  const maybeWorldToLocal = invertMat4(candidateLocalToWorld);
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
    const { transformCount, resolved } = resolveInstanceTransforms(binding, primitive);
    if (transformCount !== binding.instanceCount) {
      return {
        ok: false,
        reason: `primitive "${binding.primitiveId}" instance count changed from ${binding.instanceCount} to ${transformCount}`,
      };
    }
    for (const { instance } of resolved) {
      instances.push(instance);
    }
  }
  return { ok: true, instances };
}

/**
 * Copy one 8-word BVH node from `src` (at `srcWordBase`) into `dst` (at
 * `dstWordBase`), adding `leafTriDelta` to word[6] iff the node is a LEAF.
 *
 * Word[6] of a leaf is a GLOBAL triangle offset; of an interior node it is a
 * RELATIVE child offset (which must NOT shift when the subtree moves rigidly).
 * Shared by both BLAS-splice paths — an off-by-one here silently corrupts BVH
 * traversal, so it lives in exactly one place.
 */
function rebaseLeafTriOffset(
  dst: Uint32Array,
  dstWordBase: number,
  src: ArrayLike<number>,
  srcWordBase: number,
  leafTriDelta: number,
): void {
  const splitOrCount = src[srcWordBase + 7] ?? 0;
  const isLeaf = isLeafSplit(splitOrCount);
  dst[dstWordBase] = src[srcWordBase] ?? 0;
  dst[dstWordBase + 1] = src[srcWordBase + 1] ?? 0;
  dst[dstWordBase + 2] = src[srcWordBase + 2] ?? 0;
  dst[dstWordBase + 3] = src[srcWordBase + 3] ?? 0;
  dst[dstWordBase + 4] = src[srcWordBase + 4] ?? 0;
  dst[dstWordBase + 5] = src[srcWordBase + 5] ?? 0;
  dst[dstWordBase + 6] = isLeaf ? (src[srcWordBase + 6] ?? 0) + leafTriDelta : (src[srcWordBase + 6] ?? 0);
  dst[dstWordBase + 7] = splitOrCount;
}

/**
 * Copy `triCount` stride-4 (vec4u) index triangles from `src` (starting at
 * triangle `srcTri`) into `dst` (starting at triangle `dstTri`), shifting each
 * of the three GLOBAL vertex refs (.x.y.z) by `vertexDelta` and zeroing the .w
 * padding lane. Also copies the parallel per-triangle material id.
 *
 * This is the downstream-rebase inner loop of the resize splice — the one place
 * a wrong stride or delta corrupts which vertices a triangle references.
 */
function copyVec4Strided(
  dstIndices: Uint32Array,
  dstTriMaterialIds: Uint32Array,
  srcIndices: Uint32Array,
  srcTriMaterialIds: Uint32Array,
  srcTri: number,
  dstTri: number,
  vertexDelta: number,
): void {
  for (let k = 0; k < 3; k += 1) {
    dstIndices[dstTri * 4 + k] = (srcIndices[srcTri * 4 + k] ?? 0) + vertexDelta;
  }
  dstIndices[dstTri * 4 + 3] = 0;
  dstTriMaterialIds[dstTri] = srcTriMaterialIds[srcTri] ?? 0;
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

  const prevTotalVerts = Math.floor(prev.positions.length / 4);
  const prevTotalTris = prev.triangleCount;
  const prevTotalNodes = Math.floor(prev.bvhNodes.length / 8);

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

  const positions = new Float32Array(newTotalVerts * 4);
  const normals = new Float32Array(newTotalVerts * 4);
  const indices = new Uint32Array(newTotalTris * 4);
  const triMaterialIds = new Uint32Array(newTotalTris);
  const newNodeView = new Uint32Array(newTotalNodes * 8);
  const prevNodeView = new Uint32Array(
    prev.bvhNodes.buffer,
    prev.bvhNodes.byteOffset,
    prev.bvhNodes.length,
  );

  // ── Positions / normals (vec4f-strided) ──────────────────────────────────
  // Prefix [0, oldVertStart) verbatim.
  positions.set(prev.positions.subarray(0, oldVertStart * 4), 0);
  normals.set(prev.normals.subarray(0, oldVertStart * 4), 0);
  // Changed primitive's new local slice at the SAME vertexStart.
  positions.set(slice.localPositions, oldVertStart * 4);
  normals.set(slice.localNormals, oldVertStart * 4);
  // Suffix (downstream primitives) shifted by deltaVert*4 floats.
  if (oldVertEnd < prevTotalVerts) {
    positions.set(prev.positions.subarray(oldVertEnd * 4), (oldVertEnd + deltaVert) * 4);
    normals.set(prev.normals.subarray(oldVertEnd * 4), (oldVertEnd + deltaVert) * 4);
  }

  // ── Indices (vec4u-strided; .x.y.z global vertex refs, .w = 0) ────────────
  const prevIndices = prev.indices;
  // Prefix triangles [0, oldTriStart) verbatim (their vertex refs are unaffected
  // — they reference vertices before oldVertStart).
  indices.set(prevIndices.subarray(0, oldTriStart * 4), 0);
  triMaterialIds.set(prev.triMaterialIds.subarray(0, oldTriStart), 0);
  // Changed primitive's new index words rebased to its (unchanged) vertexStart.
  const newTriStart = oldTriStart; // unchanged for the spliced primitive
  for (let i = 0; i < slice.indexWords.length; i += 1) {
    const localIdx = slice.indexWords[i] ?? 0;
    indices[newTriStart * 4 + i] = i % 4 === 3 ? localIdx : localIdx + binding.vertexStart;
  }
  for (let t = 0; t < slice.triMaterialIds.length; t += 1) {
    triMaterialIds[newTriStart + t] = slice.triMaterialIds[t] ?? 0;
  }
  // Downstream triangles: copy with vertex refs shifted by deltaVert.
  for (let t = oldTriEnd; t < prevTotalTris; t += 1) {
    copyVec4Strided(indices, triMaterialIds, prevIndices, prev.triMaterialIds, t, t + deltaTri, deltaVert);
  }

  // ── BVH nodes (8 words/node) ──────────────────────────────────────────────
  // Prefix nodes [0, oldNodeStart) verbatim. Leaf word[6] is a GLOBAL tri offset
  // into a region BEFORE the changed primitive, so it is unaffected; interior
  // word[6] is relative and self-contained within the prefix subtrees.
  newNodeView.set(prevNodeView.subarray(0, oldNodeStart * 8), 0);
  // Changed primitive's new nodes at the SAME blasRoot, leaf offsets rebased to
  // its (unchanged) triStart.
  const newBlasRoot = oldNodeStart; // unchanged for the spliced primitive
  for (let n = 0; n + 7 < slice.bvhNodeWords.length; n += 8) {
    rebaseLeafTriOffset(newNodeView, newBlasRoot * 8 + n, slice.bvhNodeWords, n, binding.triStart);
  }
  // Downstream nodes shifted by deltaNode. Leaf global tri offsets shift by
  // deltaTri; interior relative child offsets are unchanged (the subtree shape
  // moves rigidly).
  for (let n = oldNodeEnd; n < prevTotalNodes; n += 1) {
    rebaseLeafTriOffset(newNodeView, (n + deltaNode) * 8, prevNodeView, n * 8, deltaTri);
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
  const collected = collectTlasInstancesFromBindings(scene, primitiveTlasBindings);
  if (!collected.ok) {
    return { ok: true, pack: packSceneFromCore(scene, opts), strategy: 'full' };
  }
  const tlasBuild = buildTlasFromInstances(collected.instances);

  return {
    ok: true,
    strategy: 'splice',
    pack: {
      positions,
      normals,
      indices,
      triMaterialIds,
      bvhNodes,
      triangleCount: newTotalTris,
      tlasNodes: tlasBuild.tlasNodes,
      tlasInstanceIndices: tlasBuild.tlasInstanceIndices,
      tlasBlasRoots: tlasBuild.tlasBlasRoots,
      tlasInstanceWorldToLocal: tlasBuild.tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld: tlasBuild.tlasInstanceLocalToWorld,
      tlasNodeCount: tlasBuild.tlasNodeCount,
      primitiveTlasBindings,
      warnings: [...prev.warnings, ...slice.warnings],
    },
  };
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
  const nodeEnd = nextBinding != null ? nextBinding.blasRoot : Math.floor(prev.bvhNodes.length / 8);
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
  const indices = new Uint32Array(prev.indices);
  const triMaterialIds = new Uint32Array(prev.triMaterialIds);
  const bvhNodes = new Float32Array(prev.bvhNodes);

  const vertOff = binding.vertexStart * 4;
  positions.set(slice.localPositions, vertOff);
  normals.set(slice.localNormals, vertOff);

  const indexOff = binding.triStart * 4;
  for (let i = 0; i < slice.indexWords.length; i += 1) {
    const localIdx = slice.indexWords[i] ?? 0;
    indices[indexOff + i] =
      i % 4 === 3 ? localIdx : localIdx + binding.vertexStart;
  }
  for (let t = 0; t < slice.triMaterialIds.length; t += 1) {
    triMaterialIds[binding.triStart + t] = slice.triMaterialIds[t] ?? 0;
  }

  const nodeWordStart = nodeStart * 8;
  const nodeView = new Uint32Array(bvhNodes.buffer);
  for (let n = 0; n + 7 < slice.bvhNodeWords.length; n += 8) {
    rebaseLeafTriOffset(nodeView, nodeWordStart + n, slice.bvhNodeWords, n, binding.triStart);
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

  const collected = collectTlasInstancesFromBindings(scene, primitiveTlasBindings);
  if (!collected.ok) {
    return { ok: true, pack: packSceneFromCore(scene, opts), strategy: 'full' };
  }
  const tlasBuild = buildTlasFromInstances(collected.instances);

  return {
    ok: true,
    strategy: 'splice',
    pack: {
      positions,
      normals,
      indices,
      triMaterialIds,
      bvhNodes,
      triangleCount: prev.triangleCount,
      tlasNodes: tlasBuild.tlasNodes,
      tlasInstanceIndices: tlasBuild.tlasInstanceIndices,
      tlasBlasRoots: tlasBuild.tlasBlasRoots,
      tlasInstanceWorldToLocal: tlasBuild.tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld: tlasBuild.tlasInstanceLocalToWorld,
      tlasNodeCount: tlasBuild.tlasNodeCount,
      primitiveTlasBindings,
      warnings: [...prev.warnings, ...slice.warnings],
    },
  };
}

/**
 * Pack mesh-like primitives from a `@vitrum/core` `Scene` into concatenated
 * local BLAS buffers and an optional TLAS over instances.
 */
export function packSceneFromCore(scene: Scene, opts: ScenePackOptions): ScenePackResult {
  const buildTlasTree = opts.tlas !== false;
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const triMaterialIds: number[] = [];
  const bvhNodeWords: number[] = [];
  const pendingTlasInstances: PendingTlasInstance[] = [];
  const primitiveTlasBindings: PrimitiveTlasBinding[] = [];
  const warnings: string[] = [];

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

    const vertexCount = slice.vertexCount;
    const triCount = slice.triCount;
    const vertexBase = Math.floor(positions.length / 4);
    const triBase = triMaterialIds.length;
    const nodeBase = Math.floor(bvhNodeWords.length / 8);

    for (let i = 0; i < slice.localPositions.length; i += 1) positions.push(slice.localPositions[i] ?? 0);
    for (let i = 0; i < slice.localNormals.length; i += 1) normals.push(slice.localNormals[i] ?? 0);
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

    for (let n = 0; n + 7 < slice.bvhNodeWords.length; n += 8) {
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
      if (transforms.length === 0) {
        warnings.push(`Instanced primitive "${primitive.id}" has no instances; skipping TLAS instance upload.`);
        continue;
      }
      for (const transform of transforms) {
        const { instance, nonInvertible } = resolveOneTransform(
          transform,
          localAabb.min,
          localAabb.max,
          nodeBase,
        );
        if (nonInvertible) {
          warnings.push(
            `Primitive "${primitive.id}" has non-invertible instance transform; using identity fallback for TLAS transform.`,
          );
        }
        pendingTlasInstances.push(instance);
      }
      primitiveTlasBindings.push({
        primitiveId: primitive.id,
        primitiveKind: primitive.kind,
        blasRoot: nodeBase,
        instanceCount: transforms.length,
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
        warnings.push(
          `Primitive "${primitive.id}" has non-invertible instance transform; using identity fallback for TLAS transform.`,
        );
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
    prevTlas.tlasInstanceWorldToLocal.length === refitAabbs.length * 16
  ) {
    const refitNodes = new Uint32Array(prevTlas.tlasNodes);
    refitTlas(
      {
        nodes: refitNodes,
        nodeCount: Math.floor(refitNodes.length / 8),
        instanceIndices: prevTlas.tlasInstanceIndices,
        blasRoots: prevTlas.tlasBlasRoots,
        instanceTransforms: prevTlas.tlasInstanceWorldToLocal,
      },
      refitAabbs,
    );
    const l2w = new Float32Array(pendingTlasInstances.length * 16);
    const w2l = new Float32Array(pendingTlasInstances.length * 16);
    for (let i = 0; i < pendingTlasInstances.length; i += 1) {
      l2w.set(pendingTlasInstances[i]!.localToWorld, i * 16);
      w2l.set(pendingTlasInstances[i]!.worldToLocal, i * 16);
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
  const byId = new Map<string, ScenePrimitive>();
  for (const primitive of scene.primitives) {
    byId.set(primitive.id, primitive);
  }
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
    liveCounts.push(transformCount);
    for (const { instance, nonInvertible } of resolved) {
      if (nonInvertible) {
        warnings.push(
          `Primitive "${binding.primitiveId}" has non-invertible instance transform; using identity fallback for TLAS transform.`,
        );
      }
      instances.push(instance);
    }
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
