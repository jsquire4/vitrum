import type { Scene, SceneEmitter, ScenePrimitive } from '@vitrum/core';

export function patchPrimitiveInScene(scene: Scene, id: string, patch: Partial<ScenePrimitive>): Scene {
  let matched = false;
  const primitives = scene.primitives.map((primitive) => {
    if (primitive.id !== id) return primitive;
    matched = true;
    return { ...primitive, ...patch } as ScenePrimitive;
  });
  if (!matched) {
    throw new Error(`updatePrimitive: primitive "${id}" not found in current scene`);
  }
  return {
    ...scene,
    primitives,
  };
}

export function patchEmitterInScene(scene: Scene, id: string, patch: Partial<SceneEmitter>): Scene {
  let matched = false;
  const emitters = scene.emitters.map((emitter) => {
    if (emitter.id !== id) return emitter;
    matched = true;
    return { ...emitter, ...patch } as SceneEmitter;
  });
  if (!matched) {
    throw new Error(`updateEmitter: emitter "${id}" not found in current scene`);
  }
  return {
    ...scene,
    emitters,
  };
}
