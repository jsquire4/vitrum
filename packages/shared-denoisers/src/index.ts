// @vitrum/shared-denoisers — denoiser building blocks (à-trous, SVGF, BMFR, OIDN bridge).
//
// Phase 5 deliverable: atrous + temporalAccum WGSL fragments.
// Sprint 6 (Phase 6): 37-tap hexagonal-kernel edge-stopping spatial filter.
// Sprint 10a (Phase 6): SVGF spatiotemporal variance-guided denoiser.
// Sprint 10b (Phase 6): OIDN ONNX final-pass bridge.

export * from './wgsl/atrous.wgsl.js';
export * from './wgsl/temporalAccum.wgsl.js';
export { SPATIAL_FILTER_WGSL } from './wgsl/spatialFilter.wgsl.js';
export type { SpatialFilterBindGroupLayout } from './wgsl/spatialFilter.wgsl.js';

// Sprint 10a — SVGF
export { SVGF_WGSL } from './wgsl/svgf.wgsl.js';
export {
  SVGF_UNIFORMS_SIZE_BYTES,
  SVGF_VARIANCE_UNIFORMS_SIZE_BYTES,
  SVGF_DEFAULT_UNIFORMS,
  packSVGFUniforms,
  packSVGFVarianceUniforms,
} from './svgfBindings.js';
export type {
  SVGFUniforms,
  SVGFVarianceUniforms,
  SVGFVarianceBindGroupLayout,
  SVGFAtrousBindGroupLayout,
} from './svgfBindings.js';

// Sprint 10b — OIDN
export {
  denoiseFinal,
  preloadOIDNModel,
  clearOIDNCache,
} from './oidnBridge.js';
export type {
  OIDNDenoiseInputs,
  OIDNDenoiseOptions,
} from './oidnBridge.js';
