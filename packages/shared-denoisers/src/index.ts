// @vitrum/shared-denoisers — denoiser building blocks (à-trous, atrous-variance, OIDN bridge).
// BMFR remains a roadmap candidate; no BMFR module is exported from this package.
//
// Phase 5 deliverable: atrous + temporalAccum WGSL fragments.
// Sprint 6 (Phase 6): 37-tap hexagonal-kernel edge-stopping spatial filter.
// Sprint 10a (Phase 6): à-trous + variance-guided denoiser (renamed from SVGF by sweep-2026-05-11 D3).
//   Real Schied 2017 SVGF is tracked in plan/sprint-svgf-real-future.md.
// Sprint 10b (Phase 6): OIDN ONNX final-pass bridge.

export * from './wgsl/atrous.wgsl.js';
export * from './wgsl/temporalAccum.wgsl.js';

// Canonical WelfordVariance — single source for cross-package variance state.
export {
  WELFORD_VARIANCE_WGSL,
  WELFORD_VARIANCE_VERSION,
} from './wgsl/welfordVariance.wgsl.js';

// Sprint 10a — à-trous + variance-guided denoiser (formerly SVGF)
export {
  ATROUS_VARIANCE_WGSL,
  ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE,
} from './wgsl/atrousVariance.wgsl.js';
export {
  ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT,
  ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_MAX_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_FRAME_COUNT_INPUT_GUARD_MAX,
} from './atrousVarianceConstants.js';
export {
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  packAtrousVarianceAtrousUniforms,
  packAtrousVarianceVarianceUniforms,
} from './atrousVarianceBindings.js';
export type {
  AtrousVarianceAtrousUniforms,
  AtrousVarianceVarianceUniforms,
  AtrousVarianceVarianceBindGroupLayout,
  AtrousVarianceAtrousBindGroupLayout,
} from './atrousVarianceBindings.js';

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

// À-trous + variance WebGPU host path (synthetic G-buffer).
export {
  runAtrousVarianceWebGPU,
  assertAtrousVarianceWebGPUBufferShapes,
  ATROUS_VARIANCE_SYNTHETIC_GBUFFER_DEFAULTS,
} from './atrousVarianceWebGPU.js';
export type { AtrousVarianceWebGPUOptions, AtrousVarianceSyntheticGbufferFallback } from './atrousVarianceWebGPU.js';

// ── T2.H1 — Real Schied 2017 SVGF ('svgf-real' mode) ─────────────────────────
// Implements bilinear reprojection + disocclusion test + per-pixel history +
// variance-from-moments (Eq. 1–5) + 7×7 spatial fallback (§4.3).
// WGSL fragment exports (for WalkaroundGPUPipeline persistent-texture path):
export {
  SVGF_REPROJECTION_WGSL,
  SVGF_REAL_REPROJECTION_WORKGROUP_SIZE,
} from './wgsl/svgfReprojection.wgsl.js';
export {
  SVGF_VARIANCE_FROM_MOMENTS_WGSL,
  SVGF_HISTORY_MIN_FOR_MOMENTS,
  SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE,
} from './wgsl/svgfVarianceFromMoments.wgsl.js';
export {
  SVGF_7X7_SPATIAL_FALLBACK_WGSL,
  SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD,
  SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE,
} from './wgsl/svgf7x7SpatialFallback.wgsl.js';
// Constants:
export {
  SVGF_REAL_DEFAULT_ALPHA_MIN,
  SVGF_REAL_DEFAULT_SIGMA_DEPTH,
  SVGF_REAL_DEFAULT_SIGMA_NORMAL,
  SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
  SVGF_REAL_MAX_ATROUS_ITERATIONS,
} from './svgfRealConstants.js';
// Bindings + packer:
export {
  SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
  SVGF_REPROJ_DEFAULT_UNIFORMS,
  packSVGFReprojUniforms,
} from './svgfRealBindings.js';
export type { SVGFReprojUniforms } from './svgfRealBindings.js';
// One-shot WebGPU host pipeline (CPU-backed, allocates transient textures):
export { runSVGFRealWebGPU } from './svgfRealWebGPU.js';
export type { SVGFRealWebGPUOptions } from './svgfRealWebGPU.js';
// CPU emulation oracles (test helpers; live in svgfRealCpu.ts but also re-exported
// from svgfRealWebGPU.ts for backward compatibility with existing test imports):
export {
  svgfReprojCPU,
  svgfVarianceFromMomentsCPU,
  svgf7x7FallbackCPU,
} from './svgfRealCpu.js';
export {
  svgfRealDemodulateAlbedo,
  svgfRealRemodulateAlbedo,
} from './svgfRealWebGPU.js';
export type {
  SVGFReprojCPUInput,
  SVGFReprojCPUOutput,
} from './svgfRealCpu.js';
