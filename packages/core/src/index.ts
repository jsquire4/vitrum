// @vitrum/core — public façade.
//
// Primary surface: scene/frame/engine contracts. Small Tier-2 browser probes
// (`probeWebGPU`, `detectGpu`) live here for shared chroma / mount gates.

export * from './scene/index.js';
export * from './frame.js';
export * from './engine/index.js';
export type { GpuDetection, DetectGpuOptions } from './gpuDetection.js';
export type { WgpuAdapterKind, WgpuProbeResult } from './wgpuSupport.js';
export { detectGpu, _resetCacheUnsafe } from './gpuDetection.js';
export { probeWebGPU, isSwiftShaderAdapter } from './wgpuSupport.js';
