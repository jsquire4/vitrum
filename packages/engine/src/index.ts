// @vitrum/engine — drop-in entry point.
//
// Re-exports the public Engine surface from @vitrum/core so hosts can take
// a single dependency for the common case.

export {
  createEngine,
  pickBackend,
  deriveScaleDefaults,
  type CreateEngineOptions,
  type EnginePreference,
  type ScaleDefaults,
} from './createEngine.js';
export { computeSceneAABB, type SceneAABB } from './sceneAABB.js';

// Convenience re-exports — hosts can `import { createEngine, type Engine } from '@vitrum/engine'`.
export type {
  Engine,
  EngineState,
  EngineCapabilities,
  Scene,
  ScenePrimitive,
  SceneEmitter,
  Material,
  FrameInput,
  FrameOutput,
  Viewport,
} from '@vitrum/core';
