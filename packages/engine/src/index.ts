// @vitrum/engine — drop-in entry point.
//
// Re-exports the public Engine surface from @vitrum/core so hosts can take
// a single dependency for the common case.

export {
  createEngine,
  pickBackend,
  deriveScaleDefaults,
  mergeWalkaroundTlasExtension,
  type CreateEngineOptions,
  type EnginePreference,
  type ScaleDefaults,
} from './createEngine.js';
export { auditSceneNeedsTlas, type SceneTlasAudit } from '@vitrum/core';
export { computeSceneAABB, type SceneAABB } from './sceneAABB.js';

// Phase-0 productization — graceful-degradation adapter probe. The function
// lives here (it imports the real backend limit thresholds); the data shape
// (`AdapterProfile`) is re-exported from @vitrum/core for one-import ergonomics.
export { probeAdapterProfile } from './adapterProfile.js';
export type {
  AdapterProfile,
  RealtimeTier,
  HeroBackendRec,
  PtWebgpuTierRec,
} from '@vitrum/core';

// Lifecycle helpers — vanilla attachVitrum() is also available as a
// dedicated entrypoint via `@vitrum/engine/lifecycle`. React's
// <VitrumCanvas> ships under `@vitrum/engine/react`.
export {
  attachVitrum,
  type AttachVitrumOptions,
  type AttachVitrumHandle,
} from './lifecycle/vanilla.js';

// Convenience re-exports — hosts can `import { createEngine, type Engine } from '@vitrum/engine'`.
export type {
  Engine,
  EngineState,
  EngineCapabilities,
  Scene,
  ScenePrimitive,
  SceneEmitter,
  MaterialSpec,
  FrameInput,
  FrameOutput,
  Viewport,
} from '@vitrum/core';

// Backend-specific lighting vocabulary re-export. `Engine.updateLighting` is
// contractually opaque in @vitrum/core (Readonly<Record<string, unknown>>),
// so hosts that drive HybridEngine's per-frame time-of-day scrub can import
// this concrete type for compile-time key checking without @vitrum/core baking
// a backend-specific shape into the universal contract.
export type { LightingOptions } from '@vitrum/walkaround-hybrid';
