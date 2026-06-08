// @vitrum/walkaround-rc — Radiance Cascades subsystem.
//
// Hoisted from @vitrum/walkaround-hybrid/src/rc/ on 2026-05-18 (W8 follow-up).
// Composition with DDGI / ReSTIR-GI happens in `@vitrum/walkaround-hybrid`
// via `HybridEngineRC`; this package owns the algorithm itself: cascade
// pyramid layout, BVH compute, dispatch state machine, and raw WGSL shader
// strings. THREE/TSL receiver helpers live behind the explicit `/three` subpath.
//
// Reference: Sannikov 2023, "Radiance Cascades: A Novel Approach to
// Calculating Global Illumination."

// Cascade pyramid geometry and raw storage layout.
export { CASCADE_DIMS, CASCADE_COUNT } from './cascadePyramid.js';
export type { CascadeAABB, CascadeDim } from './cascadePyramid.js';

// Cascade dispatch — raw WebGPU compute. The THREE-tied `RCDispatchOpts`
// and `dispatchFrame` path were dropped 2026-05-18 once the raw-GPU
// `dispatchFrameRaw` / `RCDispatchOptsRaw` path absorbed the only consumer
// (`@vitrum/walkaround-hybrid`'s `RCSubsystem`).
export { RCDispatcher } from './cascadeDispatch.js';
export type { RCDispatchOptsRaw } from './cascadeDispatch.js';

// Optional THREE/TSL bridge exports live at `@vitrum/walkaround-rc/three`.

// Raw WGSL shader strings (for host inspection or headless testing).
export { PROBE_RAY_CAST_WGSL } from './wgsl/probeRayCast.wgsl.js';
export { CASCADE_MERGE_WGSL } from './wgsl/cascadeMerge.wgsl.js';

// Octahedral solid-angle helper (pure CPU; consumed by cascade-merge math).
export { computeOctahedralSolidAngles } from './octahedralSolidAngles.js';
