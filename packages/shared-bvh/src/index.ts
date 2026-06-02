// Re-apply of W7-G5 (lost in May-17 merge race per items_to_fix.md E7):
// list each bvhCommon export explicitly so internal helpers like
// `validateBvhEncoding` (test-only) don't leak onto the public surface.
export { buildSceneBVH, refitBvhBounds } from './bvhCommon.js';
export type { SceneBVHCommonResult, SceneBVHCommonOpts } from './bvhCommon.js';
export * from './buildArrayBvh.js';
export * from './sceneBvh.js';
export * from './materialEntry.js';
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
 * `buildSceneBVH` always returns stride 3 (`indices` is a raw `Uint32Array`
 * with 3 u32 per triangle).  Callers that need stride 4 must post-process
 * the output (expand each triple to a four-element group, zeroing `.w`, or
 * pack caller-specific payload into `.w`).
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
 * triangle), the post-processing step every stride-4 consumer must apply to
 * `buildSceneBVH`'s stride-3 output (see {@link BvhIndexStride}).
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
