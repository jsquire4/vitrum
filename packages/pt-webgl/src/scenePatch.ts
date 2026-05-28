// Stateless scene-patch helpers for PTEngineWebGL2 — no engine-instance
// dependency. Extracted from ptEngineWebGL2.ts (theme T14 god-class
// decomposition).
//
// Two concerns live here:
//   1. Fast-path CLASSIFIERS — predicates that decide whether an incremental
//      `updatePrimitive` / `updateEmitter` patch can take a cheap fork-side
//      update (materials / lights / geometry refit) or must fall back to a
//      full `setScene` (BVH rebuild). They are the backend-specific companion
//      to the shared snapshot-patch layer in `@vitrum/core`
//      (`patchPrimitiveInScene` / `patchEmitterInScene`), which is a separate
//      concern (immutable record update + invariant checks).
//   2. The THREE-side geometry mutators (`applyPositionsPatchToMesh`) and the
//      fork geometry-refresh (`refreshPathTracerSceneGeometry`), which mutate
//      a THREE scene root + the fork's merged BVH but hold no engine state.

import { BufferAttribute } from 'three';
import type { Mesh as TMesh, Scene as ThreeScene } from 'three';
import type { WebGLPathTracer } from 'three-gpu-pathtracer';
import type { ScenePrimitive, SceneEmitter, MeshPrimitive } from '@vitrum/core';
import { ForkAccess } from './forkAccess.js';

/** Distributive `keyof` over a discriminated union. `T extends T` with `T` as
 *  a naked generic parameter forces distribution, so this yields the UNION of
 *  every variant's keys (`kind | id | color | … | meshId`), not just the common
 *  keys a bare `keyof` would give. */
type KeysOfUnion<T> = T extends T ? keyof T : never;

/** Every key that appears on any `SceneEmitter` variant. Referencing a phantom
 *  field name against this type (e.g. the old `meshPrimitiveId` typo this guard
 *  previously tested) fails typecheck. */
type SceneEmitterKey = KeysOfUnion<SceneEmitter>;

/**
 * True when an emitter patch touches ONLY light parameters (color, intensity,
 * direction, position, …) and can therefore take the cheap fork-side
 * `updateLights()` fast path with no BVH rebuild.
 *
 * Returns false when the patch changes the discriminant `kind` or — for a
 * `mesh-area` emitter — the `meshId` it samples. A `meshId` repoint changes
 * which mesh geometry feeds the area emitter, so it is a geometry/topology
 * change that REQUIRES a full scene rebuild, not the light-only path.
 *
 * `blocksFastPath` is typed `SceneEmitterKey[]`, so a phantom field name would
 * fail typecheck rather than silently never matching at runtime (the A4 bug:
 * the guard tested the non-existent `meshPrimitiveId`, letting `meshId`
 * repoints incorrectly take the cheap light-only path).
 */
export function isEmitterOnlyPatch(patch: Partial<SceneEmitter>): boolean {
  const blocksFastPath: ReadonlyArray<SceneEmitterKey> = ['kind', 'meshId'];
  const rec = patch as Partial<Record<SceneEmitterKey, unknown>>;
  for (const blocked of blocksFastPath) {
    if (blocked in patch && rec[blocked] !== undefined) {
      return false;
    }
  }
  return Object.keys(patch).some((k) => k !== 'id');
}

export function isMaterialOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  if (patch.material === undefined) return false;
  for (const key of Object.keys(patch) as (keyof ScenePrimitive)[]) {
    if (key === 'id' || key === 'material') continue;
    if ((patch as Record<string, unknown>)[key] !== undefined) return false;
  }
  return true;
}

export function isTransformOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  const rec = patch as Record<string, unknown>;
  if (rec['transform'] === undefined) return false;
  for (const key of Object.keys(rec)) {
    if (key === 'id' || key === 'transform') continue;
    if (rec[key] !== undefined) return false;
  }
  return true;
}

export function isPositionsOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  const rec = patch as Record<string, unknown>;
  if (rec['positions'] === undefined) return false;
  for (const key of Object.keys(rec)) {
    if (key === 'id' || key === 'positions' || key === 'normals') continue;
    if (rec[key] !== undefined) return false;
  }
  return true;
}

/**
 * Apply a positions(+normals) patch to a THREE mesh in place. Returns false
 * when the vertex count changed (topology change — caller must full-rebuild).
 */
export function applyPositionsPatchToMesh(mesh: TMesh, patch: Partial<MeshPrimitive>): boolean {
  const positions = patch.positions;
  if (positions == null) return false;
  const posAttr = mesh.geometry.getAttribute('position');
  const vertCount = positions.length / 3;
  if (posAttr != null && posAttr.count !== vertCount) {
    return false;
  }
  mesh.geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  if (patch.normals != null) {
    mesh.geometry.setAttribute('normal', new BufferAttribute(new Float32Array(patch.normals), 3));
  }
  return true;
}

/**
 * Regenerate merged geometry + BVH via the fork generator — no full
 * `setScene`. Thin wrapper over {@link ForkAccess.regenerateSceneGeometry}
 * that supplies `pathTracer.reset()` as the accumulator-clear callback.
 * Returns false when the fork generator is not yet initialized (caller falls
 * back to a full `setScene`).
 */
export function refreshPathTracerSceneGeometry(
  pathTracer: WebGLPathTracer,
  threeRoot: ThreeScene,
): boolean {
  return ForkAccess.regenerateSceneGeometry(pathTracer, threeRoot, () => {
    pathTracer.reset();
  });
}
