// solveSkinPrimitives.ts — pre-pass that replaces skinned-mesh rest-pose
// positions/normals with CPU-solved posed positions before scene ingestion.
//
// WHY here, not in shared-bvh:
//   `mergeWorldSpaceFromCore` reads `primitive.positions` and `primitive.normals`
//   directly without awareness of the skinning solver.  The cleanest seam is a
//   pre-pass that produces a new Scene whose skinned-mesh primitives carry SOLVED
//   positions so the rest of the pipeline (mergeWorldSpaceFromCore, BVH packers,
//   attribute packers) is position-agnostic.
//
// CONTRACT:
//   • Primitives that are NOT skinned-mesh are returned as-is (referential
//     identity preserved — no re-allocation, no copy).
//   • A skinned-mesh whose `bones` array is zero-length is left as rest pose
//     with a console.warn (degenerate primitive — solveSkin would throw).
//   • If the scene contains no skinned-mesh primitives the input scene object
//     is returned unchanged (fast-path, no allocations).
//
// updatePrimitive path:
//   pt-webgl2's `updatePrimitive` calls `patchPrimitiveInScene` then `setScene`
//   which rebuilds the entire scene texture bundle via `buildSceneTextures`.
//   This pre-pass runs inside `buildSceneTextures` (after partitionSceneBySupport,
//   before mergeWorldSpaceFromCore) so a host that updates bone matrices via
//   `updatePrimitive(id, { bones: newBones })` will automatically get a fresh
//   solve — no special incremental path needed.
//
// morphTargets:
//   `solveSkin` fully handles morph-target + normal blending when the primitive
//   carries `morphTargets` / `morphWeights` / `morphTargetNormals`.  No gap here
//   — morph application is part of the solver's pre-LBS step.

import type { Scene, ScenePrimitive } from '@vitrum/core';
import { solveSkin } from '@vitrum/core';

/**
 * Replace each `skinned-mesh` primitive's rest-pose `positions` and `normals`
 * with the CPU-solved posed values from {@link solveSkin}.  All other
 * primitives and all scene-level fields are returned unchanged.
 *
 * The returned primitives still carry `kind: 'skinned-mesh'` so downstream
 * code that filters by kind continues to work; only the geometry arrays are
 * substituted.
 */
export function solveSkinPrimitives(scene: Scene): Scene {
  let anyResolved = false;

  const primitives = scene.primitives.map((prim): ScenePrimitive => {
    if (prim.kind !== 'skinned-mesh') return prim;

    // Guard: degenerate primitive with no bones — pass through rest pose with a
    // warning so the scene at least renders (as a static rest-pose mesh) rather
    // than crashing the solver.
    if (prim.bones.length === 0) {
      console.warn(
        `[vitrum/pt-webgl2] solveSkinPrimitives: primitive "${String(prim.id)}" is skinned-mesh ` +
          'but has an empty bones array — rendering rest pose.',
      );
      return prim;
    }

    const { positions, normals } = solveSkin(prim);
    anyResolved = true;
    // Return a new object that shares every field except positions/normals, which
    // are now the solved (posed) geometry.  The rest-pose arrays on the original
    // prim are still referenced by the host — we do NOT mutate them.
    return { ...prim, positions, normals };
  });

  if (!anyResolved) return scene;
  return { ...scene, primitives };
}
