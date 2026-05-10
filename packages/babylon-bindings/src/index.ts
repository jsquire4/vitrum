import type { Scene } from '@vitrum/core';

/**
 * Stub adapter — Babylon scene graph → `@vitrum/core` `Scene`.
 *
 * See `plan/binding-babylon-sketch.md`. Intentionally no `@babylonjs/core`
 * dependency until implementation work is scheduled; the package exists to
 * reserve the import path and pressure-test naming.
 */
export function sceneFromBabylonScene(_babylonScene: unknown): Scene {
  void _babylonScene;
  throw new Error(
    '[@vitrum/babylon-bindings] sceneFromBabylonScene is not implemented — see plan/binding-babylon-sketch.md',
  );
}
