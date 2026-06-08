// Scene description — backend-agnostic.
//
// Canonical, pure scene-snapshot patch helpers. Backends call these to apply a
// `engine.updatePrimitive` / `engine.updateEmitter` diff to an immutable Scene
// while enforcing the discriminated-union invariants that hold across every
// backend:
//
//   - a node's `id` can never change (identity is the patch key);
//   - a node's `kind` can never change (kind selects the union member, and a
//     cross-kind morph is a topology change that requires a full `setScene`);
//   - an `analytic` primitive's `shape`/`params` must stay length-consistent
//     (validated via {@link validateAnalyticParams});
//   - mesh-like primitives cannot smuggle in analytic `params`.
//
// These were previously reimplemented (and had drifted) in pt-webgpu, pt-webgl,
// and walkaround-hybrid; the strict pt-webgpu variant is the superset lifted
// here so all backends share one invariant layer. Backend-specific fast-path
// detection (material-only / transform-only / geometry-refit) stays in the
// backends — only the snapshot-patch + invariant enforcement is canonical.

import { validateAnalyticParams } from './analyticParams.js';
import type { ScenePrimitive } from './primitives.js';
import type { SceneEmitter } from './emitters.js';
import type { Scene } from './index.js';

/** Primitive kinds that carry triangle geometry rather than analytic params. */
const MESH_LIKE_KINDS: ReadonlySet<ScenePrimitive['kind']> = new Set([
  'mesh',
  'skinned-mesh',
  'instanced-mesh',
]);

/**
 * Enforce the cross-backend invariants for a single primitive patch. Throws on
 * any violation; otherwise returns silently (the caller performs the spread).
 */
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

  if (MESH_LIKE_KINDS.has(nextKind) && 'shape' in patch) {
    throw new Error(
      `updatePrimitive: primitive "${primitive.id}" (${nextKind}) cannot accept analytic "shape"`,
    );
  }
  if (MESH_LIKE_KINDS.has(nextKind) && 'params' in patch) {
    throw new Error(
      `updatePrimitive: primitive "${primitive.id}" (${nextKind}) cannot accept analytic "params"`,
    );
  }
  if (MESH_LIKE_KINDS.has(nextKind) && 'fallbackMesh' in patch) {
    throw new Error(
      `updatePrimitive: primitive "${primitive.id}" (${nextKind}) cannot accept analytic "fallbackMesh"`,
    );
  }
}

/**
 * Apply a primitive patch to a Scene snapshot, returning a NEW Scene (the input
 * is never mutated). Throws if `id` is not present, or if the patch violates an
 * id/kind immutability or analytic-param invariant.
 */
export function patchPrimitiveInScene(
  scene: Scene,
  id: string,
  patch: Partial<ScenePrimitive>,
): Scene {
  let matched = false;
  const primitives = scene.primitives.map((primitive) => {
    if (String(primitive.id) !== id) return primitive;
    matched = true;
    assertPrimitivePatch(primitive, patch);
    const merged = { ...primitive, ...patch, id: primitive.id } as ScenePrimitive;
    // Deep-merge the `material` sub-object so a PARTIAL material patch (e.g.
    // `{ material: { emissive } }`) PRESERVES the primitive's other material
    // fields instead of replacing the whole material. A shallow spread silently
    // resets unspecified fields to packer defaults — and drops `baseColor`, which
    // crashes the material packer. Non-material patch fields keep replace semantics.
    const primMat = (primitive as unknown as { material?: Record<string, unknown> }).material;
    const patchMat = (patch as unknown as { material?: Record<string, unknown> }).material;
    if (patchMat != null && primMat != null) {
      (merged as unknown as { material: Record<string, unknown> }).material = { ...primMat, ...patchMat };
    }
    return merged;
  });
  if (!matched) {
    throw new Error(`updatePrimitive: primitive "${id}" not found in current scene`);
  }
  return {
    ...scene,
    primitives,
  };
}

/**
 * Apply an emitter patch to a Scene snapshot, returning a NEW Scene (the input
 * is never mutated). Throws if `id` is not present, or if the patch attempts to
 * change the emitter's `id` or `kind`.
 */
export function patchEmitterInScene(
  scene: Scene,
  id: string,
  patch: Partial<SceneEmitter>,
): Scene {
  let matched = false;
  const emitters = scene.emitters.map((emitter) => {
    if (String(emitter.id) !== id) return emitter;
    matched = true;
    if ('kind' in patch && patch.kind != null && patch.kind !== emitter.kind) {
      throw new Error(
        `updateEmitter: emitter "${id}" kind cannot change from "${emitter.kind}" to "${patch.kind}"`,
      );
    }
    if ('id' in patch && patch.id != null && patch.id !== emitter.id) {
      throw new Error(`updateEmitter: emitter "${id}" id cannot be changed`);
    }
    return { ...emitter, ...patch, id: emitter.id } as SceneEmitter;
  });
  if (!matched) {
    throw new Error(`updateEmitter: emitter "${id}" not found in current scene`);
  }
  return {
    ...scene,
    emitters,
  };
}
