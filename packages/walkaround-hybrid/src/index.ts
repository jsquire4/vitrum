// @vitrum/walkaround-hybrid — public engine-facing surface.
// Internal shader/pipeline/PPG/neural symbols are intentionally NOT exported
// from this package root; consume them from explicit internal paths when needed.

import type { Engine } from '@vitrum/core';
import type { HybridEngineOptions } from './HybridEngineOptions.js';
import type { HybridEngineGISurface } from './HybridEnginePublic.js';

export type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';
export type { HybridEngine, HybridEngineGISurface } from './HybridEnginePublic.js';

export async function createWalkaroundEngine_Hybrid(
  opts: HybridEngineOptions,
): Promise<Engine & HybridEngineGISurface> {
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
export { serializeGIState, deserializeGIState, type GIStateSnapshot } from './giStateSnapshot.js';

export {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
  HYBRID_WEBGPU_REQUIRED_FEATURES,
} from './pipeline/WalkaroundGPUPipeline.js';

// THREE/TSL bridge subpath (`@vitrum/walkaround-hybrid/three`) was removed;
// no runtime three dependency remains in this package.

// Neural-denoiser host wiring surface (kept public for example hosts).
export {
  buildRandomWeightsForSpec,
  loadWeightsFromArrayBuffer,
  serializeWeightsToArrayBuffer,
  VITRUM_MODEL_MAGIC,
  VITRUM_MODEL_VERSION,
} from './neural/weights.js';
export type { ModelWeights, LayerWeights } from './neural/weights.js';
export { WALKAROUND_DENOISER_UNET_SPEC } from './neural/unetArchitecture.js';

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
