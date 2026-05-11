// @vitrum/core — public façade.
//
// Primary surface: scene/frame/engine contracts. Small Tier-2 browser probes
// (`probeWebGPU`, `detectGpu`) live here for shared chroma / mount gates.

export * from './scene.js';
export * from './frame.js';
export * from './engine.js';
// Explicit named exports — _resetCacheUnsafe is intentionally excluded
// (test-only; accessible only via gpuDetection.test-utils.ts).
export type { GpuDetection, DetectGpuOptions } from './gpuDetection.js';
export type { WgpuAdapterKind, WgpuProbeResult } from './wgpuSupport.js';
export { detectGpu } from './gpuDetection.js';
export { probeWebGPU, isSwiftShaderAdapter } from './wgpuSupport.js';
