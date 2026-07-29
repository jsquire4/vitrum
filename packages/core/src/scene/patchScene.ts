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
// These were previously reimplemented (and had drifted) in pt-webgpu, pt-webgl2,
// and walkaround-hybrid; the strict pt-webgpu variant is the superset lifted
// here so all backends share one invariant layer. Backend-specific fast-path
// detection (material-only / transform-only / geometry-refit) stays in the
// backends — only the snapshot-patch + invariant enforcement is canonical.

import { validateAnalyticParams } from './analyticParams.js';
import {
  validateEmitterForScenePatch,
  validatePrimitiveFastFieldsForScenePatch,
  validatePrimitiveForScenePatch,
} from './validation.js';
import type { ScenePrimitive } from './primitives.js';
import type { SceneEmitter } from './emitters.js';
import type { Scene } from './index.js';

/** Primitive kinds that carry triangle geometry rather than analytic params. */
const MESH_LIKE_KINDS: ReadonlySet<ScenePrimitive['kind']> = new Set([
  'mesh',
  'skinned-mesh',
  'instanced-mesh',
]);

const LAYERED_MATERIAL_KEYS = ['frontLayer', 'backLayer'] as const;
const FAST_PRIMITIVE_PATCH_FIELDS = new Set([
  'material',
  'transform',
  'castShadow',
] as const);
type FastPrimitivePatchField =
  typeof FAST_PRIMITIVE_PATCH_FIELDS extends ReadonlySet<infer T> ? T : never;

function isMergeableRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value) && !ArrayBuffer.isView(value);
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function mergeMaterialPatch(
  base: Record<string, unknown>,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  const merged = { ...base, ...patch };
  for (const key of LAYERED_MATERIAL_KEYS) {
    if (!hasOwn(patch, key)) continue;
    const baseLayer = base[key];
    const patchLayer = patch[key];
    merged[key] = isMergeableRecord(baseLayer) && isMergeableRecord(patchLayer)
      ? { ...baseLayer, ...patchLayer }
      : patchLayer;
  }
  return merged;
}

function classifyFastPrimitivePatch(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
): readonly FastPrimitivePatchField[] | undefined {
  const fields: FastPrimitivePatchField[] = [];
  for (const key of Reflect.ownKeys(patch)) {
    const descriptor = Object.getOwnPropertyDescriptor(patch, key);
    if (
      typeof key !== 'string' ||
      descriptor?.enumerable !== true ||
      !FAST_PRIMITIVE_PATCH_FIELDS.has(key as FastPrimitivePatchField) ||
      (key === 'transform' && primitive.kind === 'instanced-mesh')
    ) {
      return undefined;
    }
    fields.push(key as FastPrimitivePatchField);
  }
  return fields;
}

/**
 * Descriptor-preserving shallow merge that does not invoke getters for
 * unchanged primitive fields. This matters for hot material edits: cloning via
 * object spread would read every geometry property before validation had a
 * chance to take the field-aware path.
 */
function mergePrimitivePatch(
  primitive: ScenePrimitive,
  patch: Partial<ScenePrimitive>,
  materialOverride: Record<string, unknown> | undefined,
): ScenePrimitive {
  const descriptors = new Map<PropertyKey, PropertyDescriptor>();
  for (const source of [primitive, patch] as const) {
    for (const key of Reflect.ownKeys(source)) {
      const descriptor = Object.getOwnPropertyDescriptor(source, key);
      if (descriptor !== undefined) descriptors.set(key, descriptor);
    }
  }
  descriptors.set('id', {
    configurable: true,
    enumerable: true,
    value: primitive.id,
    writable: true,
  });
  if (materialOverride !== undefined) {
    descriptors.set('material', {
      configurable: true,
      enumerable: true,
      value: materialOverride,
      writable: true,
    });
  }

  const merged = {};
  for (const [key, descriptor] of descriptors) {
    Object.defineProperty(merged, key, descriptor);
  }
  return merged as ScenePrimitive;
}

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

  const nextKind = (patch.kind ?? primitive.kind);
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
 *
 * Precondition: `scene` is the immutable snapshot previously accepted through
 * `validateScene`. This incremental helper validates the newly merged primitive
 * only; it intentionally does not re-traverse unchanged geometry every frame.
 */
export function patchPrimitiveInScene(
  scene: Scene,
  id: string,
  patch: Partial<ScenePrimitive>,
): Scene {
  const primitiveIndex = scene.primitives.findIndex(
    (primitive) => String(primitive.id) === id,
  );
  if (primitiveIndex < 0) {
    throw new Error(`updatePrimitive: primitive "${id}" not found in current scene`);
  }
  const primitive = scene.primitives[primitiveIndex]!;
  assertPrimitivePatch(primitive, patch);
  // Deep-merge the `material` sub-object so a PARTIAL material patch (e.g.
  // `{ material: { emissive } }`) PRESERVES the primitive's other material
  // fields instead of replacing the whole material. Layered material objects
  // are merged one level deeper so `{ material: { frontLayer: { roughness } } }`
  // does not drop an existing `frontLayer.normalMap`/`normalScale`.
  // Non-material patch fields keep replace semantics.
  const primMat = (primitive as unknown as { material?: Record<string, unknown> }).material;
  const patchMat = (patch as unknown as { material?: Record<string, unknown> }).material;
  let materialOverride: Record<string, unknown> | undefined;
  if (patchMat != null && primMat != null) {
    materialOverride = mergeMaterialPatch(primMat, patchMat);
  }
  const merged = mergePrimitivePatch(primitive, patch, materialOverride);
  const fastFields = classifyFastPrimitivePatch(primitive, patch);
  if (fastFields === undefined) {
    validatePrimitiveForScenePatch(merged, `scene.primitives[${primitiveIndex}]`);
  } else {
    validatePrimitiveFastFieldsForScenePatch(
      merged,
      fastFields,
      `scene.primitives[${primitiveIndex}]`,
    );
  }
  const primitives = scene.primitives.slice();
  primitives[primitiveIndex] = merged;
  return { ...scene, primitives };
}

/**
 * Apply an emitter patch to a Scene snapshot, returning a NEW Scene (the input
 * is never mutated). Throws if `id` is not present, or if the patch attempts to
 * change the emitter's `id` or `kind`.
 *
 * Precondition: `scene` is the immutable snapshot previously accepted through
 * `validateScene`. The merged emitter and any mesh-area reference/ownership
 * change are validated; unchanged primitive payloads are not re-traversed.
 */
export function patchEmitterInScene(
  scene: Scene,
  id: string,
  patch: Partial<SceneEmitter>,
): Scene {
  const emitterIndex = scene.emitters.findIndex(
    (emitter) => String(emitter.id) === id,
  );
  if (emitterIndex < 0) {
    throw new Error(`updateEmitter: emitter "${id}" not found in current scene`);
  }
  const emitter = scene.emitters[emitterIndex]!;
  if ('kind' in patch && patch.kind != null && patch.kind !== emitter.kind) {
    throw new Error(
      `updateEmitter: emitter "${id}" kind cannot change from "${emitter.kind}" to "${patch.kind}"`,
    );
  }
  if ('id' in patch && patch.id != null && patch.id !== emitter.id) {
    throw new Error(`updateEmitter: emitter "${id}" id cannot be changed`);
  }
  const merged = { ...emitter, ...patch, id: emitter.id } as SceneEmitter;
  validateEmitterForScenePatch(merged, `scene.emitters[${emitterIndex}]`);
  if (merged.kind === 'mesh-area') {
    const target = scene.primitives.find((primitive) => primitive.id === merged.meshId);
    if (target == null) {
      throw new RangeError(
        `validateScene: scene.emitters[${emitterIndex}].meshId references missing primitive "${merged.meshId}"`,
      );
    }
    if (target.kind === 'analytic') {
      throw new RangeError(
        `validateScene: scene.emitters[${emitterIndex}].meshId must reference a mesh-like primitive (got analytic "${merged.meshId}")`,
      );
    }
    const duplicateIndex = scene.emitters.findIndex(
      (candidate, index) =>
        index !== emitterIndex &&
        candidate.kind === 'mesh-area' &&
        candidate.meshId === merged.meshId,
    );
    if (duplicateIndex >= 0) {
      throw new RangeError(
        `validateScene: scene.emitters[${emitterIndex}].meshId duplicates mesh-area ownership of primitive "${merged.meshId}" already claimed by scene.emitters[${duplicateIndex}].meshId`,
      );
    }
  }
  const emitters = scene.emitters.slice();
  emitters[emitterIndex] = merged;
  return { ...scene, emitters };
}
