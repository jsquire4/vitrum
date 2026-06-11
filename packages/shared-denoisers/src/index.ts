// @vitrum/shared-denoisers — denoiser building blocks (à-trous, atrous-variance,
// SVGF-real, OIDN bridge, BMFR).
//
// Phase 5 deliverable: atrous + temporalAccum WGSL fragments.
// Sprint 10a (Phase 6): à-trous + variance-guided denoiser (renamed from SVGF by sweep-2026-05-11 D3).
//   Real Schied 2017 SVGF ('svgf-real' mode) is implemented in svgfRealWebGPU.ts.
// Sprint 10b (Phase 6): OIDN ONNX final-pass bridge.

export * from './wgsl/atrous.wgsl.js';
export * from './wgsl/temporalAccum.wgsl.js';

// Temporal accumulation pass UBO helper (D12.11 — defineUbo pattern).
export {
  TEMPORAL_ACCUM_UBO_SIZE_BYTES,
  packTemporalAccumUniforms,
} from './temporalAccumBindings.js';
export type { TemporalAccumUniforms } from './temporalAccumBindings.js';

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
  releaseOIDNCacheEntry,
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
// One-shot WebGPU host pipeline (CPU-backed, allocates transient textures).
// STANDALONE BUILDING BLOCK — intentionally has no in-engine consumer. The
// walkaround-hybrid 'svgf-real' denoiser mode (see HybridEngine `_svgfReal`
// pass-graph, assembled in pipeline/wgslModules.ts) does NOT call this wrapper;
// it composes the SVGF_* WGSL fragments exported above into its own
// persistent-texture pass graph, which is the right shape for a realtime engine
// (no per-frame transient texture alloc/free). This one-shot entry point exists
// for host/offline denoising and GPU-execution coverage
// (__tests__/webgpuDenoiserExecution.gpu.test.ts). So "zero engine consumers" is by
// design, not a gap: the algorithm is wired, the convenience wrapper is standalone.
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
export {
  demodulateAlbedo,
  remodulateAlbedo,
} from './albedoModulation.js';
export type {
  SVGFReprojCPUInput,
  SVGFReprojCPUOutput,
} from './svgfRealCpu.js';

// ── BMFR — Koskela et al. 2019 blockwise multi-order feature regression ──────
// Per-32×32-block least-squares fit of noisy color against [1, p, n, p²] via
// Householder QR, + temporal EMA. WGSL kernel + one-shot host pipeline + the
// CPU-unit-testable regression core.
export { BMFR_WGSL, BMFR_WORKGROUP_SIZE, BMFR_WGSL_FEATURE_COUNT } from './wgsl/bmfr.wgsl.js';
export {
  BMFR_FEATURE_COUNT,
  BMFR_BLOCK_SIZE,
  BMFR_QR_REGULARISATION,
  bmfrFeatureRow,
  bmfrSolveChannel,
  householderSolve,
  bmfrFitBlock,
} from './bmfrRegression.js';
export {
  BMFR_DEFAULT_TEMPORAL_ALPHA,
  BMFR_DEFAULT_POSITION_SCALE,
} from './bmfrConstants.js';
export {
  BMFR_UNIFORMS_SIZE_BYTES,
  BMFR_DEFAULT_UNIFORMS,
  packBmfrUniforms,
} from './bmfrBindings.js';
export type { BmfrUniforms } from './bmfrBindings.js';
// One-shot WebGPU host pipeline — STANDALONE BUILDING BLOCK, same contract as
// runSVGFRealWebGPU above. The walkaround-hybrid 'bmfr' denoiser mode consumes
// BMFR_WGSL via its own pipeline (pipeline/denoisers/bmfr.ts + wgslModules.ts), not
// this transient-texture wrapper. This entry point is for host/offline use +
// GPU-execution tests; no in-engine consumer by design.
export { runBmfrWebGPU } from './bmfrWebGPU.js';
export type { BmfrWebGPUOptions } from './bmfrWebGPU.js';

// ── OIDNDispatcherCore — shared cohort state machine for converged backends ──
export {
  OIDNDispatcherCore,
  _defaultLoader as oidnDefaultLoader,
} from './oidnDispatcherCore.js';
export type {
  OIDNFinalDispatcherOptions,
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
  ReadbackResult,
  ReadbackFn,
  OIDNDispatcherCoreOptions,
} from './oidnDispatcherCore.js';

// ── Cross-package primitives (consumed by walkaround-hybrid OIDN denoiser) ───
// These helpers existed inside the package but were not re-exported from the
// index. The walkaround-hybrid OIDNFinalDenoiser previously inlined its own
// copies; route through these canonicals now.
export {
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from './halfFloat.js';
export { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';

// ── RGBA16F ↔ Float32-RGB conversion helpers ─────────────────────────────────
// Shared by walkaround-hybrid OIDNFinalDenoiser and any future backend that
// needs the same rgba16float readback / upload round-trip (e.g. pt-webgpu).
export {
  rgba16fBufferToRgbF32,
  rgbF32ToRgba16fRowAligned,
} from './rgba16fConversions.js';
