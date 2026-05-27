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
  computeLocalAabb,
  computeWorldAabbForBindings,
  type PrimitiveTlasBinding,
  type ScenePackOptions,
  type ScenePackResult,
  type TlasGpuSnapshot,
  type RefitTlasResult,
  type RebuildPrimitiveBlasResult,
} from './scenePack.js';
export * from './wgsl/octahedral.wgsl.js';
export * from './wgsl/materialEntry.wgsl.js';
export * from './wgsl/bvhIntersect.wgsl.js';
export * from './wgsl/tlasTraversal.wgsl.js';
export { fingerprintBuffer, fingerprintBuffers } from './bufferFingerprint.js';

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
