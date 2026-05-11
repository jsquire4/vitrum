import type { Scene } from '@vitrum/core';

/**
 * Babylon scene graph → `@vitrum/core` `Scene`.
 *
 * See `plan/binding-babylon-sketch.md`. Intentionally no `@babylonjs/core`
 * dependency until implementation work is scheduled; calling this throws — it
 * only reserves the import path until a real Babylon→vitrum walker lands.
 *
 * Prefer the explicit **`experimental`** name in new hosts so callers know they
 * are wiring a stub. `sceneFromBabylonScene` remains an alias for compatibility.
 *
 * Migration: convert in the host (e.g. via glTF ↔ vitrum adapters) until this API
 * is implemented end-to-end.
 */
export function experimentalSceneFromBabylonScene(_babylonScene: unknown): Scene {
  void _babylonScene;
  throw new Error(
    '[@vitrum/babylon-bindings] Babylon→vitrum conversion is not implemented — see plan/binding-babylon-sketch.md',
  );
}

/** @deprecated Alias of {@link experimentalSceneFromBabylonScene}; both throw until implemented. */
export function sceneFromBabylonScene(babylonScene: unknown): Scene {
  return experimentalSceneFromBabylonScene(babylonScene);
}
