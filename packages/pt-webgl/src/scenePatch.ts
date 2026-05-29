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
import type { Mesh as TMesh, Scene as ThreeScene, BufferGeometry } from 'three';
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

/** Mesh-geometry fields a topology patch may touch (per `MeshPrimitive`). A
 *  patch limited to these can rebuild ONE mesh's THREE BufferGeometry and take
 *  the fork's targeted geometry/BVH regen instead of a full `setScene`. */
const GEOMETRY_PATCH_FIELDS: ReadonlySet<string> = new Set([
  'positions',
  'normals',
  'uvs',
  'tangents',
  'indices',
]);

/**
 * True when a primitive patch touches ONLY mesh-geometry buffers
 * (positions / normals / uvs / tangents / indices) and at least one of them —
 * crucially WITHOUT touching `material`, `transform`, `kind`, or any other
 * field. Such a patch (including an arbitrary vertex-/index-COUNT change) can
 * rebuild that one mesh's geometry and take the fork's targeted geometry+BVH
 * regen (`refreshPathTracerSceneGeometry`) rather than a full `setScene`
 * teardown.
 *
 * Why `material` blocks this path: the fork's geometry-only regen skips
 * `updateMaterials()`/`updateLights()`/`updateEnvironment()` (it only rebuilds
 * geometry/BVH/attributes + the material-INDEX attribute). A patch that also
 * changes the material must therefore fall back to a full `setScene` so the
 * MaterialsTexture is re-packed — otherwise the new geometry would render with
 * the stale material. The same-material constraint is what keeps this safe.
 *
 * The simpler {@link isPositionsOnlyPrimitivePatch} path stays its own branch
 * (same-vertex-count refit via an in-place attribute swap); this predicate is
 * the superset that also admits count changes and index/uv/tangent surgery via
 * a full per-mesh geometry rebuild.
 */
export function isGeometryOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  const rec = patch as Record<string, unknown>;
  let touchesGeometry = false;
  for (const key of Object.keys(rec)) {
    if (key === 'id') continue;
    if (rec[key] === undefined) continue;
    if (!GEOMETRY_PATCH_FIELDS.has(key)) return false;
    touchesGeometry = true;
  }
  return touchesGeometry;
}

/**
 * True when a primitive patch touches ONLY the `instances` array (the
 * per-instance transform list of an `instanced-mesh`) and nothing else. Such a
 * patch — including one that GROWS or SHRINKS the instance COUNT — can be
 * serviced by re-expanding just that one instanced-mesh's baked children in the
 * live THREE scene root + the fork's targeted geometry+BVH regen, rather than a
 * full `setScene`.
 *
 * Like {@link isGeometryOnlyPrimitivePatch}, this deliberately blocks on a
 * co-present `material` (which would need the MaterialsTexture re-packed via a
 * full rebuild) or any other field: an instances-only patch keeps the same
 * shared geometry + material, so the geometry-only regen that skips
 * `updateMaterials()` is safe. The caller still checks the primitive's `kind`
 * is `instanced-mesh` before re-expanding — a stray `instances` field on a
 * non-instanced primitive falls through to the full-rebuild path.
 */
export function isInstanceCountOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  const rec = patch as Record<string, unknown>;
  if (rec['instances'] === undefined) return false;
  for (const key of Object.keys(rec)) {
    if (key === 'id' || key === 'instances') continue;
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
 * Apply an arbitrary geometry patch (any subset of positions / normals / uvs /
 * tangents / indices, INCLUDING a vertex- or index-COUNT change) to a THREE
 * mesh in place by replacing each present attribute / index buffer on the
 * mesh's existing BufferGeometry. Unlike {@link applyPositionsPatchToMesh},
 * this never bails on a count change — the fork's `StaticGeometryGenerator`
 * detects the changed attribute lengths (its `BakedGeometry.isCompatible`
 * returns false on a count mismatch), discards the stale baked geometry, and
 * force-rebuilds the merged geometry + BVH on the next `regenerateSceneGeometry`
 * call. Buffers are copied (fresh typed arrays) so the engine's THREE scene
 * owns its own geometry storage independent of the caller's patch buffers.
 *
 * Returns false when the patch carried no recognised geometry buffer (caller
 * must full-rebuild — should not happen behind `isGeometryOnlyPrimitivePatch`).
 */
export function applyGeometryPatchToMesh(mesh: TMesh, patch: Partial<MeshPrimitive>): boolean {
  const geom = mesh.geometry as BufferGeometry;
  let applied = false;
  if (patch.positions != null) {
    geom.setAttribute('position', new BufferAttribute(new Float32Array(patch.positions), 3));
    applied = true;
  }
  if (patch.normals != null) {
    geom.setAttribute('normal', new BufferAttribute(new Float32Array(patch.normals), 3));
    applied = true;
  }
  if (patch.uvs != null) {
    geom.setAttribute('uv', new BufferAttribute(new Float32Array(patch.uvs), 2));
    applied = true;
  }
  if (patch.tangents != null) {
    geom.setAttribute('tangent', new BufferAttribute(new Float32Array(patch.tangents), 4));
    applied = true;
  }
  if (patch.indices != null) {
    // Preserve the patch's index width (Uint16Array vs Uint32Array). A
    // BakedGeometry index-array constructor mismatch also trips
    // `validateAttributes`, so width changes are handled correctly downstream.
    const indexCopy =
      patch.indices instanceof Uint16Array
        ? new Uint16Array(patch.indices)
        : new Uint32Array(patch.indices);
    geom.setIndex(new BufferAttribute(indexCopy, 1));
    applied = true;
  }
  return applied;
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
