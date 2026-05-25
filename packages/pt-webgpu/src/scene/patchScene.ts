import type { Scene, SceneEmitter, ScenePrimitive } from '@vitrum/core';
import { validateAnalyticParams } from '@vitrum/core';

const MESH_LIKE_KINDS = new Set(['mesh', 'skinned-mesh', 'instanced-mesh']);

function assertPrimitivePatch(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
): void {
  if ('kind' in patch && patch.kind != null && patch.kind !== primitive.kind) {
    throw new Error(
      `updatePrimitive: primitive "${primitive.id}" kind cannot change from "${primitive.kind}" to "${patch.kind}"`,
    );
  }
  if ('id' in patch && patch.id != null && patch.id !== primitive.id) {
    throw new Error(`updatePrimitive: primitive "${primitive.id}" id cannot be changed`);
  }

  const nextKind = (patch.kind ?? primitive.kind) as ScenePrimitive['kind'];
  if (nextKind === 'analytic') {
    const shape = ('shape' in patch && patch.shape != null)
      ? patch.shape
      : primitive.kind === 'analytic'
        ? primitive.shape
        : undefined;
    const params = ('params' in patch && patch.params != null)
      ? patch.params
      : primitive.kind === 'analytic'
        ? primitive.params
        : undefined;
    if (shape == null || params == null) {
      throw new Error(
        `updatePrimitive: analytic primitive "${primitive.id}" requires shape and params`,
      );
    }
    validateAnalyticParams(shape, params);
  }

  if (MESH_LIKE_KINDS.has(nextKind) && 'params' in patch) {
    throw new Error(
      `updatePrimitive: primitive "${primitive.id}" (${nextKind}) cannot accept analytic "params"`,
    );
  }
}

export function patchPrimitiveInScene(scene: Scene, id: string, patch: Partial<ScenePrimitive>): Scene {
  let matched = false;
  const primitives = scene.primitives.map((primitive) => {
    if (primitive.id !== id) return primitive;
    matched = true;
    assertPrimitivePatch(primitive, patch);
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
