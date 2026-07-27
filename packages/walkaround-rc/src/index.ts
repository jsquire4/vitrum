// @vitrum/walkaround-rc — Radiance Cascades subsystem.
//
// Hoisted from @vitrum/walkaround-hybrid/src/rc/ on 2026-05-18 (W8 follow-up).
// Composition with DDGI / ReSTIR-GI happens in `@vitrum/walkaround-hybrid`
// via `HybridEngineRC`; this package owns the algorithm itself: cascade
// pyramid layout, BVH compute, dispatch state machine, and raw WGSL shader
// strings.
//
// Reference: Sannikov 2023, "Radiance Cascades: A Novel Approach to
// Calculating Global Illumination."

// Cascade pyramid geometry and raw storage layout.
export { CASCADE_DIMS, CASCADE_COUNT, validateCascadeDims } from './cascadePyramid.js';
export type { CascadeAABB, CascadeDim } from './cascadePyramid.js';
export { allocateCascades, disposeCascades } from './cascadeBuffers.js';
export type { CascadeBuffers } from './cascadeBuffers.js';

// Cascade dispatch — raw WebGPU compute.
export {
  RCDispatcher,
  RC_DEFAULT_TRANSMITTED_INTERFACE_BUDGET,
  RC_MAX_TRANSMITTED_INTERFACE_BUDGET,
  RC_MIN_TRANSMITTED_INTERFACE_BUDGET,
  RC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
} from './cascadeDispatch.js';
export type { RCDispatchOptsRaw } from './cascadeDispatch.js';

// Raw WGSL shader strings (for host inspection or headless testing).
export { PROBE_RAY_CAST_WGSL } from './wgsl/probeRayCast.wgsl.js';
export { CASCADE_MERGE_WGSL } from './wgsl/cascadeMerge.wgsl.js';
export {
  RC_OCTAHEDRAL_SOLID_ANGLE_WGSL,
  RC_OCTAHEDRAL_STRATIFIED_SAMPLING_WGSL,
} from './wgsl/octahedralSampling.wgsl.js';

// Octahedral solid-angle helper (pure CPU; consumed by cascade-merge math).
export {
  computeOctahedralSolidAngles,
  MAX_OCTAHEDRAL_SOLID_ANGLE_GRID_SIZE,
} from './octahedralSolidAngles.js';
