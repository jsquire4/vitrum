/**
 * scenePack.ts — THREE-free scene geometry packer (BLAS concat + optional TLAS).
 *
 * Canonical CPU pack used by walkaround-hybrid and pt-webgpu. Extracted from
 * pt-webgpu `uploadSceneBuffers.ts` (PR-2.1).
 */

import { asMat4, type Mat4, type Scene, type ScenePrimitive, type Vec3 } from '@vitrum/core';
import { buildArrayBvh } from './buildArrayBvh.js';
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

function invertMat4(m: Mat4): Float32Array | null {
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

function packOneMeshLikePrimitive(
  primitive: Extract<ScenePrimitive, { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }>,
  matId: number,
): PackedPrimitiveSlice | null {
  const warnings: string[] = [];
  const basePositions = primitive.positions;
  const vertexCount = Math.floor(basePositions.length / 3);
  if (vertexCount < 3) {
    warnings.push(`Primitive "${primitive.id}" has fewer than 3 vertices; skipping.`);
    return null;
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
    return null;
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
  if (localAabb == null) return null;

  return {
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
  };
}

function collectTlasInstancesFromBindings(
  scene: Scene,
  bindings: readonly PrimitiveTlasBinding[],
): { readonly ok: true; readonly instances: readonly PendingTlasInstance[] } | { readonly ok: false; readonly reason: string } {
  const primitiveById = new Map<string, ScenePrimitive>();
  for (const primitive of scene.primitives) {
    primitiveById.set(primitive.id, primitive);
  }
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
    const transforms =
      primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];
    if (transforms.length !== binding.instanceCount) {
      return {
        ok: false,
        reason: `primitive "${binding.primitiveId}" instance count changed from ${binding.instanceCount} to ${transforms.length}`,
      };
    }
    for (const transform of transforms) {
      const candidateLocalToWorld = asMat4(transform ?? IDENTITY_MAT4);
      const maybeWorldToLocal = invertMat4(candidateLocalToWorld);
      const localToWorld = maybeWorldToLocal == null ? IDENTITY_MAT4 : candidateLocalToWorld;
      const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
      const worldAabb = transformAabb(binding.localAabbMin, binding.localAabbMax, localToWorld);
      instances.push({
        blasRoot: binding.blasRoot,
        worldToLocal,
        localToWorld,
        aabbMin: worldAabb.min,
        aabbMax: worldAabb.max,
      });
    }
  }
  return { ok: true, instances };
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
    return { ok: true, pack: packSceneFromCore(scene, opts), strategy: 'full' };
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
    const splitOrCount = slice.bvhNodeWords[n + 7] ?? 0;
    const isLeaf = (splitOrCount & 0xffff0000) === 0xffff0000;
    const w = nodeWordStart + n;
    nodeView[w] = slice.bvhNodeWords[n] ?? 0;
    nodeView[w + 1] = slice.bvhNodeWords[n + 1] ?? 0;
    nodeView[w + 2] = slice.bvhNodeWords[n + 2] ?? 0;
    nodeView[w + 3] = slice.bvhNodeWords[n + 3] ?? 0;
    nodeView[w + 4] = slice.bvhNodeWords[n + 4] ?? 0;
    nodeView[w + 5] = slice.bvhNodeWords[n + 5] ?? 0;
    nodeView[w + 6] = isLeaf
      ? (slice.bvhNodeWords[n + 6] ?? 0) + binding.triStart
      : (slice.bvhNodeWords[n + 6] ?? 0);
    nodeView[w + 7] = splitOrCount;
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
    const basePositions = primitive.positions;
    const vertexCount = Math.floor(basePositions.length / 3);
    if (vertexCount < 3) {
      warnings.push(`Primitive "${primitive.id}" has fewer than 3 vertices; skipping.`);
      continue;
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
      continue;
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

    const vertexBase = Math.floor(positions.length / 4);
    const triBase = triMaterialIds.length;
    const nodeBase = Math.floor(bvhNodeWords.length / 8);
    const localNodeWords = new Uint32Array(
      localBvh.bvhNodes.buffer,
      localBvh.bvhNodes.byteOffset,
      localBvh.bvhNodes.length,
    );

    for (let i = 0; i < localPositions.length; i += 1) positions.push(localPositions[i] ?? 0);
    for (let i = 0; i < localNormals.length; i += 1) normals.push(localNormals[i] ?? 0);
    for (let i = 0; i + 3 < localBvh.reorderedIndices.length; i += 4) {
      indices.push(
        (localBvh.reorderedIndices[i] ?? 0) + vertexBase,
        (localBvh.reorderedIndices[i + 1] ?? 0) + vertexBase,
        (localBvh.reorderedIndices[i + 2] ?? 0) + vertexBase,
        localBvh.reorderedIndices[i + 3] ?? 0,
      );
    }
    for (let i = 0; i < localBvh.reorderedTriMaterialIds.length; i += 1) {
      triMaterialIds.push(localBvh.reorderedTriMaterialIds[i] ?? matId);
    }

    for (let n = 0; n + 7 < localNodeWords.length; n += 8) {
      const splitOrCount = localNodeWords[n + 7] ?? 0;
      const isLeaf = (splitOrCount & 0xffff0000) === 0xffff0000;
      bvhNodeWords.push(
        localNodeWords[n] ?? 0,
        localNodeWords[n + 1] ?? 0,
        localNodeWords[n + 2] ?? 0,
        localNodeWords[n + 3] ?? 0,
        localNodeWords[n + 4] ?? 0,
        localNodeWords[n + 5] ?? 0,
        isLeaf ? (localNodeWords[n + 6] ?? 0) + triBase : (localNodeWords[n + 6] ?? 0),
        splitOrCount,
      );
    }

    const localAabb = computeLocalAabb(basePositions);
    if (localAabb == null) continue;

    if (buildTlasTree) {
      const transforms =
        primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];
      if (transforms.length === 0) {
        warnings.push(`Instanced primitive "${primitive.id}" has no instances; skipping TLAS instance upload.`);
        continue;
      }
      for (const transform of transforms) {
        const candidateLocalToWorld = asMat4(transform ?? IDENTITY_MAT4);
        const maybeWorldToLocal = invertMat4(candidateLocalToWorld);
        if (maybeWorldToLocal == null) {
          warnings.push(
            `Primitive "${primitive.id}" has non-invertible instance transform; using identity fallback for TLAS transform.`,
          );
        }
        const localToWorld = maybeWorldToLocal == null ? IDENTITY_MAT4 : candidateLocalToWorld;
        const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
        const worldAabb = transformAabb(localAabb.min, localAabb.max, localToWorld);
        pendingTlasInstances.push({
          blasRoot: nodeBase,
          worldToLocal,
          localToWorld,
          aabbMin: worldAabb.min,
          aabbMax: worldAabb.max,
        });
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
  const primitiveById = new Map<string, ScenePrimitive>();
  for (const primitive of scene.primitives) {
    primitiveById.set(primitive.id, primitive);
  }
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
    const transforms =
      primitive.kind === 'instanced-mesh' ? primitive.instances : [primitive.transform ?? undefined];
    if (transforms.length !== binding.instanceCount) {
      return {
        ok: false,
        reason: `refitTlasTransforms: primitive "${binding.primitiveId}" instance count changed ` +
          `from ${binding.instanceCount} to ${transforms.length}.`,
      };
    }
    for (const transform of transforms) {
      const candidateLocalToWorld = asMat4(transform ?? IDENTITY_MAT4);
      const maybeWorldToLocal = invertMat4(candidateLocalToWorld);
      if (maybeWorldToLocal == null) {
        warnings.push(
          `Primitive "${primitive.id}" has non-invertible instance transform; using identity fallback for TLAS transform.`,
        );
      }
      const localToWorld = maybeWorldToLocal == null ? IDENTITY_MAT4 : candidateLocalToWorld;
      const worldToLocal = asMat4(maybeWorldToLocal ?? IDENTITY_MAT4);
      const worldAabb = transformAabb(binding.localAabbMin, binding.localAabbMax, localToWorld);
      refitAabbs.push(worldAabb);
      pendingTlasInstances.push({
        blasRoot: binding.blasRoot,
        worldToLocal,
        localToWorld,
        aabbMin: worldAabb.min,
        aabbMax: worldAabb.max,
      });
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

  const slice = packOneMeshLikePrimitive(primitive, opts.resolveMaterialId(primitiveId));
  if (slice == null) {
    return { ok: true, pack: packSceneFromCore(scene, opts), strategy: 'full' };
  }

  return splicePrimitiveBlasIntoPack(prev, bindingIndex, slice, scene, opts);
}
