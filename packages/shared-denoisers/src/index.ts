// @vitrum/shared-denoisers — denoiser building blocks (à-trous, SVGF, BMFR, OIDN bridge).
//
// Phase 5 deliverable: atrous + temporalAccum WGSL fragments.
// Sprint 6 (Phase 6): 37-tap hexagonal-kernel edge-stopping spatial filter.
// Future Phase 6 sprints add: SVGF (Sprint 10a), BMFR candidate (Sprint 10a),
// OIDN ONNX final-pass bridge (Sprint 10b).

export * from './wgsl/atrous.wgsl.js';
export * from './wgsl/temporalAccum.wgsl.js';
export { SPATIAL_FILTER_WGSL } from './wgsl/spatialFilter.wgsl.js';
export type { SpatialFilterBindGroupLayout } from './wgsl/spatialFilter.wgsl.js';
