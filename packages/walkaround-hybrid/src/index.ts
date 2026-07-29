// @vitrum/walkaround-hybrid — public engine-facing surface.
// Internal shader/pipeline/PPG/neural symbols are intentionally NOT exported
// from this package root; consume them from explicit internal paths when needed.

import type { HybridEngineOptions } from './HybridEngineOptions.js';
import type { HybridEngine as HybridEnginePublic } from './HybridEnginePublic.js';

export type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';
export {
  validateHybridEngineAdvancedOptions,
  resolveHybridNrcConfig,
  type HybridEngineAdvancedOptions,
} from './HybridEngineConfig.js';
export type {
  HybridEngine,
  HybridEngineGISurface,
  HybridRenderLayer,
} from './HybridEnginePublic.js';
export {
  WALKAROUND_WEBGPU_TEXTURE_SOURCE_KIND,
  WALKAROUND_CPU_MIRROR_SNAPSHOT_BUDGET_BYTES,
  createWalkaroundWebGpuTextureSource,
  isWalkaroundWebGpuTextureSource,
  type WalkaroundTextureColorSpace,
  type WalkaroundTextureCpuMirror,
  type WalkaroundTextureCpuMirrorDataType,
  type WalkaroundTextureCpuMirrorInput,
  type WalkaroundWebGpuTextureSource,
  type WalkaroundWebGpuTextureSourceOptions,
} from './materialTextureSource.js';

export async function createWalkaroundEngine_Hybrid(
  opts: HybridEngineOptions,
): Promise<HybridEnginePublic> {
  const concrete = await import('./HybridEngine.js');
  return concrete.createWalkaroundEngine_Hybrid(opts);
}
export {
  FrameBudgetController,
  DEFAULT_FRAME_BUDGET_CONFIG,
  type FrameBudgetControllerConfig,
  type FrameBudgetDecision,
  type FrameBudgetAction,
} from './FrameBudgetController.js';
// NRC memory planning: exact GPU-buffer residency estimates plus the resolved
// default configuration. WebGPU has no adapter-wide aggregate-memory limit, so
// hosts may use this estimate to set
// HybridEngineOptions.nrcConfig.maxNrcResidentBytes.
export {
  DEFAULT_NRC_CONFIG,
  resolveNrcConfig,
} from './neural/nrc/nrcSubsystem.js';
export type { NrcConfig } from './neural/nrc/nrcSubsystem.js';
export type { NrcDiagnostics } from './neural/nrc/nrcDiagnostics.js';
export {
  computeNrcResourceFootprint as estimateNrcResourceFootprint,
} from './neural/nrc/nrcPreflight.js';
export type {
  NrcBufferAllocation,
  NrcResourceFootprint,
} from './neural/nrc/nrcPreflight.js';

export { serializeGIState, deserializeGIState, type GIStateSnapshot } from './giStateSnapshot.js';

export {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
  HYBRID_WEBGPU_REQUIRED_FEATURES,
  NRC_WEBGPU_REQUIRED_LIMITS,
  NRC_REQUIRED_MAX_BIND_GROUPS,
  NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
  NRC_REQUIRED_WORKGROUP_STORAGE_BYTES,
  nrcWebGpuRequiredLimitsForConfig,
  nrcWebGpuRequiredFeaturesForConfig,
  assertHybridDeviceCapable,
  assertNrcDeviceCapable,
} from './pipeline/WalkaroundGPUPipeline.js';

// THREE/TSL bridge subpath (`@vitrum/walkaround-hybrid/three`) was removed;
// no runtime three dependency remains in this package.

// Neural-denoiser host wiring surface (kept public for example hosts).
export {
  assessNeuralCheckpointProductionReadiness,
  isNeuralCheckpointF16Compatible,
  isNeuralCheckpointProductionReady,
  loadWeightsFromArrayBuffer,
  VITRUM_MODEL_LEGACY_VERSION,
  serializeWeightsToArrayBuffer,
  VITRUM_MODEL_MAGIC,
  VITRUM_MODEL_VERSION,
  NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS,
} from './neural/weights.js';
export type {
  ModelWeights,
  LayerWeights,
  NeuralCheckpointMetadata,
  NeuralCheckpointProductionAssessment,
  NeuralCheckpointQualityReport,
} from './neural/weights.js';
export {
  NEURAL_PREPROCESSING_CONTRACT,
  preprocessNeuralRadiance,
  postprocessNeuralRadiance,
  sanitizeNeuralAlbedo,
  sanitizeNeuralNormal,
  type NeuralPreprocessingContract,
} from './neural/preprocessing.js';
export {
  executeNeuralInferenceCpu,
  type NeuralCpuInputs,
  type NeuralCpuInferenceResult,
} from './neural/cpuInference.js';

export { WALKAROUND_DENOISER_UNET_SPEC } from './neural/unetArchitecture.js';
export {
  WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
  walkaroundNeuralDenoiserShapeError,
  assertWalkaroundNeuralDenoiserShape,
} from './neural/shapeContract.js';

// Quality preset public surface — hosts can enumerate tiers, resolve preset
// knob values, and build quality-picker UIs without importing internal paths.
export {
  QUALITY_PRESETS,
  resolveQualityPreset,
  CHECKERBOARD_MEASURED_PERF_PROOF,
  CHECKERBOARD_PENDING_PERF_PROOF,
  CHECKERBOARD_SUPPORT_DETAILS,
  type QualityTier,
  type QualityPreset,
  type CheckerboardPerfProof,
  type CheckerboardPerPassSpeedups,
  type CheckerboardQualitySummary,
} from './HybridEngineQualityPreset.js';
