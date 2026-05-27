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
