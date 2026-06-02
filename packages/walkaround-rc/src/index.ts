// @vitrum/walkaround-rc — Radiance Cascades subsystem.
//
// Hoisted from @vitrum/walkaround-hybrid/src/rc/ on 2026-05-18 (W8 follow-up).
// Composition with DDGI / ReSTIR-GI happens in `@vitrum/walkaround-hybrid`
// via `HybridEngineRC`; this package owns the algorithm itself: cascade
// pyramid layout, BVH compute, dispatch state machine, buffer manager,
// receiver material wrapper, raw WGSL shader strings.
//
// Reference: Sannikov 2023, "Radiance Cascades: A Novel Approach to
// Calculating Global Illumination."

// Cascade pyramid storage layout.
export {
  CASCADE_DIMS,
  CASCADE_COUNT,
  allocateCascades,
  disposeCascades,
  fillCascadeDebug,
} from './cascadePyramid.js';
export type { CascadeAABB, CascadeDim, CascadeBuffers } from './cascadePyramid.js';

// Cascade dispatch — raw WebGPU compute. The THREE-tied `RCDispatchOpts`
// and `dispatchFrame` path were dropped 2026-05-18 once the raw-GPU
// `dispatchFrameRaw` / `RCDispatchOptsRaw` path absorbed the only consumer
// (`@vitrum/walkaround-hybrid`'s `RCSubsystem`).
export { RCDispatcher } from './cascadeDispatch.js';
export type { RCDispatchOptsRaw } from './cascadeDispatch.js';

// Cascade buffer manager.
export { CascadeBufferManager } from './cascadeBuffers.js';

// GI receiver material wrapper (TSL-preserved; requires three/webgpu + three/tsl).
export { GIReceiver } from './giReceiver.js';
export type { GIReceiverExclusionPredicate, GIReceiverOptions } from './giReceiver.js';

// Walkaround diffuse lighting node (TSL-preserved; requires three/tsl).
export { buildWalkaroundLightingNode } from './walkaroundDiffuseLighting.js';
export type { WalkaroundLightingNodes } from './walkaroundDiffuseLighting.js';

// Raw WGSL shader strings (for host inspection or headless testing).
export { PROBE_RAY_CAST_WGSL } from './wgsl/probeRayCast.wgsl.js';
export { CASCADE_MERGE_WGSL } from './wgsl/cascadeMerge.wgsl.js';

// Octahedral solid-angle helper (pure CPU; consumed by cascade-merge math).
export { computeOctahedralSolidAngles } from './octahedralSolidAngles.js';
