// @vitrum/walkaround-hybrid — public engine-facing surface.
// Internal shader/pipeline/PPG/neural symbols are intentionally NOT exported
// from this package root; consume them from explicit internal paths when needed.

export type {
  WalkaroundBVHSceneRoot,
  WalkaroundDDGIScene,
  WalkaroundThreeHostScene,
} from './hostScene/types.js';

export { HybridEngine, createWalkaroundEngine_Hybrid } from './HybridEngine.js';
export type { HybridEngineOptions, LightingOptions } from './HybridEngine.js';
export {
  serializeGIState,
  deserializeGIState,
  type GIStateSnapshot,
} from './giStateSnapshot.js';

export {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
  HYBRID_WEBGPU_REQUIRED_FEATURES,
} from './pipeline/WalkaroundGPUPipeline.js';

// DDGI shading injection (TSL path).
export { applyDDGIShading, disposeApplyDDGIShadingCache } from './ddgi/applyDDGIShading.js';

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
