// @vitrum/core — public façade.
//
// Primary surface: scene/frame/engine contracts. Small Tier-2 browser probes
// (`probeWebGPU`, `detectGpu`) live here for shared chroma / mount gates.

export * from './scene/index.js';
export * from './frame.js';
export * from './inverse.js';
export * from './engine/index.js';
export { solveSkin, combineSkinMatrices, mat3InverseTranspose } from './skinSolver.js';
export type { GpuDetection, DetectGpuOptions } from './gpuDetection.js';
export type { WgpuAdapterKind, WgpuProbeResult } from './wgpuSupport.js';
export type {
  AdapterProfile,
  RealtimeTier,
  HeroBackendRec,
  PtWebgpuTierRec,
} from './adapterProfile.js';
export { detectGpu, probeWebGPU, extractGpuLimits, resetGpuDetectionCache } from './gpuDetection.js';
export { isSwiftShaderAdapter } from './wgpuSupport.js';
