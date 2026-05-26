/**
 * Pure helpers to patch `@vitrum/core` `Scene` snapshots in sync with
 * THREE mesh edits during incremental engine updates.
 */

import type { Scene, SceneEmitter, ScenePrimitive } from '@vitrum/core';

export function applyPrimitivePatchToScene(
  scene: Scene,
  id: string,
  patch: Partial<ScenePrimitive>,
): Scene {
  const idx = scene.primitives.findIndex((p) => String(p.id) === id);
  if (idx < 0) {
    throw new Error(`applyPrimitivePatchToScene: primitive "${id}" not found.`);
  }
  const current = scene.primitives[idx]!;
  if (patch.kind !== undefined && patch.kind !== current.kind) {
    throw new Error(
      `applyPrimitivePatchToScene: primitive "${id}" kind cannot change from ` +
      `"${current.kind}" to "${patch.kind}"; call setScene().`,
    );
  }
  if (patch.id !== undefined && String(patch.id) !== String(current.id)) {
    throw new Error(`applyPrimitivePatchToScene: primitive "${id}" id cannot be changed.`);
  }
  const nextPrimitives = scene.primitives.slice();
  nextPrimitives[idx] = { ...current, ...patch, id: current.id } as ScenePrimitive;
  return { ...scene, primitives: nextPrimitives };
}

export function applyEmitterPatchToScene(
  scene: Scene,
  id: string,
  patch: Partial<SceneEmitter>,
): Scene {
  const idx = scene.emitters.findIndex((e) => String(e.id) === id);
  if (idx < 0) {
    throw new Error(`applyEmitterPatchToScene: emitter "${id}" not found.`);
  }
  const current = scene.emitters[idx]!;
  if (patch.kind !== undefined && patch.kind !== current.kind) {
    throw new Error(
      `applyEmitterPatchToScene: emitter "${id}" kind cannot change from ` +
      `"${current.kind}" to "${patch.kind}"; call setScene().`,
    );
  }
  if (patch.id !== undefined && String(patch.id) !== String(current.id)) {
    throw new Error(`applyEmitterPatchToScene: emitter "${id}" id cannot be changed.`);
  }
  const nextEmitters = scene.emitters.slice();
  nextEmitters[idx] = { ...current, ...patch, id: current.id } as SceneEmitter;
  return { ...scene, emitters: nextEmitters };
}
