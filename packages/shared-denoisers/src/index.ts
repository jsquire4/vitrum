// @vitrum/shared-denoisers — denoiser building blocks (à-trous, SVGF, OIDN bridge).
// BMFR remains a roadmap candidate; no BMFR module is exported from this package.
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
export {
  SVGF_WGSL,
  SVGF_COMPUTE_WORKGROUP_SIZE,
} from './wgsl/svgf.wgsl.js';
export {
  SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT,
  SVGF_DEFAULT_ATROUS_ITERATIONS,
  SVGF_MAX_ATROUS_ITERATIONS,
  SVGF_FRAME_COUNT_INPUT_GUARD_MAX,
} from './svgfConstants.js';
export {
  WEBGPU_COPY_BYTES_PER_ROW_ALIGNMENT,
  alignedTextureCopyBytesPerRow,
} from './webGpuTextureCopy.js';
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

// HDR bilateral (WebGPU compute, luminance edge-stop — no G-buffer)
export {
  HDR_LUMINANCE_BILATERAL_WGSL,
  HDR_LUMINANCE_BILATERAL_ENTRY,
  HDR_LUMINANCE_BILATERAL_WORKGROUP_SIZE,
} from './wgsl/hdrLuminanceBilateral.wgsl.js';
export {
  runHdrLuminanceBilateralWebGPU,
  HDR_LUMINANCE_BILATERAL_DEFAULT_SIGMA_LUMINANCE,
} from './hdrLuminanceBilateralWebGPU.js';
export type { HdrLuminanceBilateralWebGPUOptions } from './hdrLuminanceBilateralWebGPU.js';

// Half-precision helpers + SVGF WebGPU host path (synthetic G-buffer).
export { float16BitsToFloat32, float32ToFloat16Bits } from './halfFloat.js';
export {
  runSvgfWebGPU,
  assertSvgfWebGPUBufferShapes,
  SVGF_SYNTHETIC_GBUFFER_DEFAULTS,
} from './svgfWebGPU.js';
export type { SvgfWebGPUOptions, SvgfSyntheticGbufferFallback } from './svgfWebGPU.js';
export { getSharedWebGPUDevice, disposeSharedWebGPUDevice } from './sharedWebGpuDevice.js';
