/**
 * Re-export the shared CPU pick helper from `@vitrum/shared-bvh`.
 *
 * `pickPrimitiveCpu` and `PickCamera` now live in `@vitrum/shared-bvh` so that
 * all three backends (walkaround-hybrid, pt-webgpu, pt-webgl2) can share the
 * same implementation without a circular dependency.
 *
 * HybridEngineDebug.ts imports from here (unchanged import path); `PickCamera`
 * is re-exported so callers that already import from this path keep compiling.
 */
export { pickPrimitiveCpu, type PickCamera } from '@vitrum/shared-bvh';
