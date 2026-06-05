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

import { BufferAttribute, Matrix4 } from 'three';
import type { Mesh as TMesh, Scene as ThreeScene, BufferGeometry } from 'three';
import type { WebGLPathTracer } from 'three-gpu-pathtracer';
import type {
  ScenePrimitive,
  SceneEmitter,
  MeshPrimitive,
  InstancedMeshPrimitive,
  Scene,
} from '@vitrum/core';
import {
  applyVitrumMaterialToMesh,
  findMeshByPrimitiveId,
} from '@vitrum/three-bindings';
import {
  findAllMeshesByPrimitiveId,
  reexpandInstancedMeshInScene,
} from './expandInstancedMeshes.js';
import { ForkAccess, type WebGLPathTracerCompat } from './forkAccess.js';

function matrix4FromArrayLike(values: ArrayLike<number>): Matrix4 {
  const m = new Matrix4();
  const e = m.elements;
  for (let i = 0; i < 16; i += 1) {
    e[i] = values[i] ?? 0;
  }
  return m;
}

function replaceOrUpdateFloatAttribute(
  geometry: BufferGeometry,
  name: string,
  values: ArrayLike<number>,
  itemSize: number,
): boolean {
  const count = values.length / itemSize;
  if (!Number.isInteger(count)) return false;

  const attr = geometry.getAttribute(name);
  if (
    attr instanceof BufferAttribute &&
    attr.array instanceof Float32Array &&
    attr.count === count &&
    attr.array.length === values.length
  ) {
    (attr.array as Float32Array).set(values);
    attr.needsUpdate = true;
    return true;
  }

  geometry.setAttribute(name, new BufferAttribute(new Float32Array(values), itemSize));
  return true;
}

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

function isTransformOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
  const rec = patch as Record<string, unknown>;
  if (rec['transform'] === undefined) return false;
  for (const key of Object.keys(rec)) {
    if (key === 'id' || key === 'transform') continue;
    if (rec[key] !== undefined) return false;
  }
  return true;
}

function isPositionsOnlyPrimitivePatch(patch: Partial<ScenePrimitive>): boolean {
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
function applyPositionsPatchToMesh(mesh: TMesh, patch: Partial<MeshPrimitive>): boolean {
  const positions = patch.positions;
  if (positions == null) return false;
  const posAttr = mesh.geometry.getAttribute('position');
  const vertCount = positions.length / 3;
  if (!Number.isInteger(vertCount) || (posAttr != null && posAttr.count !== vertCount)) {
    return false;
  }
  if (patch.normals != null && patch.normals.length !== positions.length) {
    return false;
  }
  if (!replaceOrUpdateFloatAttribute(mesh.geometry, 'position', positions, 3)) return false;
  if (patch.normals != null) {
    if (!replaceOrUpdateFloatAttribute(mesh.geometry, 'normal', patch.normals, 3)) return false;
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
function refreshPathTracerSceneGeometry(
  pathTracer: WebGLPathTracer,
  threeRoot: ThreeScene,
): boolean {
  return ForkAccess.regenerateSceneGeometry(pathTracer, threeRoot, () => {
    pathTracer.reset();
  });
}

/* ──────────────────────────────────────────────────────────────────────────
 * routePrimitivePatch — the `updatePrimitive` dispatch cascade (Task 4.4)
 *
 * Lifted verbatim from `PTEngineWebGL2.updatePrimitive` + its inline
 * `#primitivePatchHandlers` table. The engine still owns its state — it passes
 * a {@link PrimitivePatchContext} (the live `pathTracer` + THREE scene root +
 * vitrum scene) and runs the shared commit/fallback EPILOGUE itself (oidn
 * invalidate + `patchPrimitiveInScene` + `setScene`), which mutates
 * engine-owned scene records. This function only performs the THREE-side
 * mutations + (for geometry handlers) the fork's targeted geometry+BVH regen,
 * and reports which epilogue the engine should run.
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Read-only engine state the patch handlers operate on. The engine constructs
 * this per `updatePrimitive` call from its own (non-null-asserted) fields.
 */
export interface PrimitivePatchContext {
  readonly pathTracer: WebGLPathTracer;
  readonly threeSceneRoot: ThreeScene;
  /** The live vitrum scene — read ONLY for the instance-count handler's
   *  current-primitive kind guard. */
  readonly vitrumScene: Scene;
}

/**
 * Outcome the engine acts on after {@link routePrimitivePatch} returns:
 *   • `'commit'`   — an incremental fast path landed; the engine runs the
 *                    shared commit epilogue (oidn invalidate +
 *                    `patchPrimitiveInScene`);
 *   • `'fallback'` — no fast path landed (no classifier matched, every match
 *                    passed through, or a mutator/fork-regen declined); the
 *                    engine takes the full-`setScene` rebuild from the patched
 *                    scene.
 */
export type RoutePrimitivePatchOutcome = 'commit' | 'fallback';

/**
 * Per-handler outcome inside the dispatch cascade. Identical semantics to the
 * pre-extraction `PatchApplyOutcome`:
 *   • `'commit'`      — fast path landed; engine runs the commit epilogue;
 *   • `'fallback'`    — incremental path declined; full-`setScene` rebuild;
 *   • `'passthrough'` — classifier matched but the handler's own guard rejected
 *                       it; continue to the next handler / tail.
 */
type PatchApplyOutcome = 'commit' | 'fallback' | 'passthrough';

interface PrimitivePatchHandler {
  classify(patch: Partial<ScenePrimitive>): boolean;
  apply(id: string, patch: Partial<ScenePrimitive>): PatchApplyOutcome;
}

/**
 * Build the ordered, first-match-wins dispatch table for one routing call. The
 * order MUST stay material → transform → positions → geometry → instance-count
 * to preserve the original cascade's branch precedence; each `apply` reproduces
 * exactly one of the original per-branch bodies (sans the shared commit
 * epilogue, which the engine runs once). `apply` may THROW on a missing-id mesh
 * lookup — the throw propagates BEFORE any scene commit, exactly as the original
 * per-branch `throw` did.
 */
function buildPrimitivePatchHandlers(ctx: PrimitivePatchContext): readonly PrimitivePatchHandler[] {
  const { pathTracer, threeSceneRoot, vitrumScene } = ctx;
  return [
    {
      // Material-only. NOTE: no geometry regen + no `setScene` fallback — this
      // handler always 'commit's (or throws on a missing mesh).
      classify: (patch) => isMaterialOnlyPrimitivePatch(patch),
      apply: (id, patch) => {
        // An expanded instanced-mesh has N THREE.Mesh children sharing the
        // primitive id (and, at setScene time, one shared material). Re-point
        // the material on ALL of them — `applyVitrumMaterialToMesh` REPLACES
        // `mesh.material`, so applying to only the first child (what
        // findMeshByPrimitiveId returns) would leave the other N-1 instances
        // on the stale material. For a plain mesh this is the single match.
        const meshes = findAllMeshesByPrimitiveId(threeSceneRoot, id);
        if (meshes.length === 0) {
          throw new Error(`updatePrimitive: primitive "${id}" not found in internal THREE scene`);
        }
        for (const mesh of meshes) {
          applyVitrumMaterialToMesh(mesh, (patch as Partial<MeshPrimitive>).material!);
        }
        const tracerCompat = pathTracer as unknown as WebGLPathTracerCompat;
        if (typeof tracerCompat.updateMaterials === 'function') {
          tracerCompat.updateMaterials();
        } else {
          tracerCompat.reset();
        }
        return 'commit';
      },
    },
    {
      // Transform-only.
      classify: (patch) => isTransformOnlyPrimitivePatch(patch),
      apply: (id, patch) => {
        const mesh = findMeshByPrimitiveId(threeSceneRoot, id);
        if (mesh == null) {
          throw new Error(`updatePrimitive: primitive "${id}" not found in internal THREE scene`);
        }
        const transform = (patch as Partial<MeshPrimitive>).transform;
        if (transform != null && transform.length >= 16) {
          const m = matrix4FromArrayLike(transform);
          mesh.matrix.copy(m);
          mesh.matrixWorld.copy(m);
          mesh.matrixAutoUpdate = false;
        }
        if (!refreshPathTracerSceneGeometry(pathTracer, threeSceneRoot)) {
          return 'fallback';
        }
        return 'commit';
      },
    },
    {
      // Positions-only (same vertex count).
      classify: (patch) => isPositionsOnlyPrimitivePatch(patch),
      apply: (id, patch) => {
        const mesh = findMeshByPrimitiveId(threeSceneRoot, id);
        if (mesh == null) {
          throw new Error(`updatePrimitive: primitive "${id}" not found in internal THREE scene`);
        }
        const meshPatch = patch as Partial<MeshPrimitive>;
        if (!applyPositionsPatchToMesh(mesh, meshPatch)) {
          return 'fallback';
        }
        if (!refreshPathTracerSceneGeometry(pathTracer, threeSceneRoot)) {
          return 'fallback';
        }
        return 'commit';
      },
    },
    {
      // Geometry-only — arbitrary same-material geometry surgery, INCLUDING a
      // vertex- or index-COUNT change, on ONE existing mesh. Rebuild that
      // mesh's THREE BufferGeometry (positions/normals/uvs/tangents/indices)
      // in place, then run the fork's targeted geometry+BVH regen. Because the
      // patch carries no `material`, the regen's skipped `updateMaterials()` is
      // harmless (the material slot is unchanged) — a material change would
      // route through the material-only handler or the full-rebuild
      // fallthrough instead. The fork's StaticGeometryGenerator detects the
      // changed attribute lengths and force-rebuilds the merged geometry + BVH
      // (GEOMETRY_REBUILT), so no full `setScene` teardown / material+light
      // re-pack is needed.
      classify: (patch) => isGeometryOnlyPrimitivePatch(patch),
      apply: (id, patch) => {
        const mesh = findMeshByPrimitiveId(threeSceneRoot, id);
        if (mesh == null) {
          throw new Error(`updatePrimitive: primitive "${id}" not found in internal THREE scene`);
        }
        const meshPatch = patch as Partial<MeshPrimitive>;
        if (!applyGeometryPatchToMesh(mesh, meshPatch)) {
          return 'fallback';
        }
        if (!refreshPathTracerSceneGeometry(pathTracer, threeSceneRoot)) {
          return 'fallback';
        }
        return 'commit';
      },
    },
    {
      // Instance-count-only — an `instances`-only patch on an `instanced-mesh`:
      // a per-instance transform list change, INCLUDING an instance-COUNT
      // grow/shrink. At setScene the single THREE.InstancedMesh was expanded
      // into N baked THREE.Mesh children (the fork bakes only
      // `mesh.matrixWorld`, ignoring `instanceMatrix`), so an instances change
      // is a topology change on the baked children, not a field the fork reads
      // off a live InstancedMesh. Re-expand ONLY this primitive's children
      // (swap the live root's N children for N' fresh ones, reusing the shared
      // geometry + material), then take the fork's targeted geometry+BVH regen
      // — the StaticGeometryGenerator detects the changed child set (added
      // uuids built fresh, removed uuids evicted; a count delta forces
      // GEOMETRY_REBUILT), exactly like the vertex-count path. Because
      // `instances` carries no material, the regen's skipped `updateMaterials()`
      // is harmless; a co-present material would have been blocked by the
      // classifier and routed to the full-rebuild fallthrough.
      //
      // Guard on the CURRENT primitive's kind: a stray `instances` field on a
      // non-instanced primitive is not a valid instanced re-expansion, so it
      // 'passthrough's to the tail full-rebuild path.
      classify: (patch) => isInstanceCountOnlyPrimitivePatch(patch),
      apply: (id, patch) => {
        const current = vitrumScene.primitives.find((p) => String(p.id) === id);
        if (current == null || current.kind !== 'instanced-mesh') {
          return 'passthrough';
        }
        const instances = (patch as Partial<InstancedMeshPrimitive>).instances;
        if (instances == null) {
          return 'passthrough';
        }
        if (!reexpandInstancedMeshInScene(threeSceneRoot, id, instances)) {
          // No existing expanded children to swap (generator state we can't
          // incrementally patch) — full rebuild.
          return 'fallback';
        }
        if (!refreshPathTracerSceneGeometry(pathTracer, threeSceneRoot)) {
          return 'fallback';
        }
        return 'commit';
      },
    },
  ];
}

/**
 * Run the `updatePrimitive` incremental-fast-path dispatch cascade against the
 * engine's live state. Performs the THREE-side mutations (and, for geometry
 * handlers, the fork's targeted geometry+BVH regen) and returns which epilogue
 * the engine should run:
 *   • `'commit'`   — an incremental fast path landed;
 *   • `'fallback'` — no fast path landed (no classifier matched, every match
 *                    passed through, or a mutator/fork-regen declined).
 *
 * ORDER is load-bearing (first match wins). A missing-id mesh lookup THROWS
 * here BEFORE returning, exactly as the original per-branch `throw` did —
 * before any scene commit.
 */
export function routePrimitivePatch(
  ctx: PrimitivePatchContext,
  id: string,
  patch: Partial<ScenePrimitive>,
): RoutePrimitivePatchOutcome {
  for (const handler of buildPrimitivePatchHandlers(ctx)) {
    if (!handler.classify(patch)) continue;
    const outcome = handler.apply(id, patch);
    if (outcome === 'passthrough') continue;
    // 'commit' | 'fallback' — both map 1:1 to the engine epilogue.
    return outcome;
  }
  // No fast path matched (or every matching handler passed through) — full
  // rebuild.
  return 'fallback';
}
