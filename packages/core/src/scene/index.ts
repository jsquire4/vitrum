// Scene description — backend-agnostic.
//
// Design principle: every scene the engine renders is composed of three things —
// PRIMITIVES (what occupies space), EMITTERS (what gives off light), and an
// ENVIRONMENT (the world's hemispheric light source). The camera lives in
// FrameInput because it changes per-frame; the scene itself is camera-free.
//
// The general case is the union of every shape we might support. The current
// concrete needs are triangle meshes (panels, walls, floors) and analytic
// primitives (architectural-pattern shapes such as H-channel came rails;
// Phase 6 sprint 5 lands them). Future kinds extend the discriminated union
// without breaking older backends — backends pattern-match on `kind` and
// ignore unknown kinds with a warning, not a crash.

export * from './math.js';
export * from './material.js';
export * from './analyticParams.js';
export * from './primitives.js';
export * from './emitters.js';
export * from './environment.js';

import type { ScenePrimitive } from './primitives.js';
import type { SceneEmitter } from './emitters.js';
import type { SceneEnvironment } from './environment.js';

// ────────────────────────────────────────────────────────────────────────────
// The Scene
// ────────────────────────────────────────────────────────────────────────────

/** A complete, immutable scene description. Hosts call `engine.setScene(scene)`
 *  with a new Scene whenever the geometry, materials, or lighting topology
 *  changes. For frequent property edits (color sliders, intensity scrubs),
 *  prefer `engine.updatePrimitive` / `engine.updateEmitter` if the backend
 *  reports `capabilities.supportsIncrementalScene = true`. */
export interface Scene {
  readonly primitives: ReadonlyArray<ScenePrimitive>;
  readonly emitters: ReadonlyArray<SceneEmitter>;
  readonly environment: SceneEnvironment;
}
