import type { Scene, SceneEmitter, ScenePrimitive } from '@vitrum/core';

export function patchPrimitiveInScene(scene: Scene, id: string, patch: Partial<ScenePrimitive>): Scene {
  let matched = false;
  const primitives = scene.primitives.map((primitive) => {
    if (primitive.id !== id) return primitive;
    matched = true;
    if ('kind' in patch && patch.kind != null && patch.kind !== primitive.kind) {
      throw new Error(`updatePrimitive: primitive "${id}" kind cannot change from "${primitive.kind}" to "${patch.kind}"`);
    }
    if ('id' in patch && patch.id != null && patch.id !== primitive.id) {
      throw new Error(`updatePrimitive: primitive "${id}" id cannot be changed`);
    }
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
    if ('kind' in patch && patch.kind != null && patch.kind !== emitter.kind) {
      throw new Error(`updateEmitter: emitter "${id}" kind cannot change from "${emitter.kind}" to "${patch.kind}"`);
    }
    if ('id' in patch && patch.id != null && patch.id !== emitter.id) {
      throw new Error(`updateEmitter: emitter "${id}" id cannot be changed`);
    }
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
