// @vitrum/core — public façade.
//
// This package contains ONLY types + lifecycle contracts. No GPU code, no
// scene-binding code, no backend-specific code. Backends and bindings depend
// on @vitrum/core; @vitrum/core depends on nothing.

export * from './scene.js';
export * from './frame.js';
export * from './engine.js';
// Explicit named exports — _resetCacheUnsafe is intentionally excluded
// (test-only; accessible only via gpuDetection.test-utils.ts).
export type { GpuDetection } from './gpuDetection.js';
export { detectGpu } from './gpuDetection.js';
