export * from './aabb.js';
export { BVH_NODE_FLOATS, VERTEX_STRIDE_F32, MAT4_STRIDE_F32 } from './strides.js';
export { pickPrimitiveCpu, type PickCamera } from './pickPrimitiveCpu.js';
export { refitBvhBounds } from './refitBvhBounds.js';
export * from './buildArrayBvh.js';
export { validateBvhEncoding } from './validateBvhEncoding.js';
export * from './sceneBvh.js';
export * from './materialEntry.js';
export {
  mergeWorldSpaceFromCore,
  materialSig,
  DEFAULT_MERGE_FILTER,
  type WorldSpaceMergeResult,
  type WorldSpaceMergeOptions,
  type MergedMeshVertexRange,
} from './worldSpaceMerge.js';
export * from './tlas.js';
export {
  packSceneFromCore,
  refitTlasTransforms,
  rebuildPrimitiveBlas,
  rebuildTlasReuseBlas,
  computeLocalAabb,
  computeWorldAabbForBindings,
  invertMat4,
  type PrimitiveTlasBinding,
  type ScenePackOptions,
  type ScenePackResult,
  type TlasGpuSnapshot,
  type RefitTlasResult,
  type RebuildPrimitiveBlasResult,
  type RebuildTlasReuseBlasResult,
} from './scenePack.js';
export * from './wgsl/octahedral.wgsl.js';
export * from './wgsl/materialEntry.wgsl.js';
export * from './wgsl/bvhIntersect.wgsl.js';
export * from './wgsl/tlasTraversal.wgsl.js';
export * from './wgsl/tlasSceneHitTraversal.wgsl.js';
export {
  fingerprintBuffer,
  fingerprintBuffers,
  fingerprintTlasBuffers,
  isTlasOnlyVersionBump,
} from './bufferFingerprint.js';
export {
  deriveSceneAABBFromBvhPositions,
  type SceneAabb,
} from './sceneAabbFromBvh.js';

/**
 * Index-buffer stride used by the BVH index array.
 *
 *   3 — `array<vec3u>` form: three u32 per triangle; `.w` not present.
 *       Used by RC and DDGI traversal shaders.
 *
 *   4 — `array<vec4u>` form: three u32 per triangle + one u32 padding/payload.
 *       Used by pt-webgpu (`.w = 0`, zero-fill contract) and by ReSTIR
 *       (`.w` packs packed-RGBA material color + texType).
 *
 * Callers that need stride 4 can expand a stride-3 source with
 * `expandIndicesToStride4`. Callers that have stride-4 data and need stride-3
 * can collapse with `collapseIndicesToStride3`.
 *
 * Upload-time assertion (recommended for all callers):
 * ```ts
 * if (indexData.byteLength % (stride * 4) !== 0)
 *   throw new Error(`BVH index buffer not aligned to stride ${stride}`);
 * ```
 */
export type BvhIndexStride = 3 | 4;

/**
 * Expand a stride-3 index buffer (`array<vec3u>` form: three u32 per triangle)
 * into a stride-4 buffer (`array<vec4u>` form: three u32 + one payload u32 per
 * triangle).
 *
 * The `.w` lane defaults to `0` (the pt-webgpu zero-fill contract). A caller
 * that packs its own payload into `.w` (e.g. ReSTIR packing RGBA material color
 * + texType) passes `payloadFn(triIndex)` to supply it.
 *
 * @param indices   stride-3 index buffer (length must be a multiple of 3).
 * @param payloadFn optional `.w` value per triangle; defaults to `0`.
 */
export function expandIndicesToStride4(
  indices: Uint32Array,
  payloadFn?: (triIndex: number) => number,
): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  const out = new Uint32Array(triCount * 4);
  for (let t = 0; t < triCount; t += 1) {
    out[t * 4] = indices[t * 3] ?? 0;
    out[t * 4 + 1] = indices[t * 3 + 1] ?? 0;
    out[t * 4 + 2] = indices[t * 3 + 2] ?? 0;
    out[t * 4 + 3] = payloadFn != null ? payloadFn(t) : 0;
  }
  return out;
}

/**
 * Collapse a stride-4 index buffer (`array<vec4u>` form: three u32 per triangle
 * + one payload u32) into a stride-3 buffer (`array<vec3u>` form). The `.w`
 * lane is discarded.
 *
 * This is the inverse of {@link expandIndicesToStride4}. Used by traversal
 * paths that require stride-3 (RC, DDGI, `refitBvhBounds`) when geometry was
 * originally packed as stride-4 by `packSceneFromCore`.
 *
 * @param indices  stride-4 index buffer (length must be a multiple of 4).
 */
export function collapseIndicesToStride3(indices: Uint32Array): Uint32Array {
  const triCount = Math.floor(indices.length / 4);
  const out = new Uint32Array(triCount * 3);
  for (let t = 0; t < triCount; t += 1) {
    out[t * 3] = indices[t * 4] ?? 0;
    out[t * 3 + 1] = indices[t * 4 + 1] ?? 0;
    out[t * 3 + 2] = indices[t * 4 + 2] ?? 0;
  }
  return out;
}
