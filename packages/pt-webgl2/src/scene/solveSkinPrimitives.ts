// solveSkinPrimitives.ts — pre-pass that replaces skinned-mesh rest-pose
// positions/normals/tangents/uvs with CPU-solved posed geometry before scene ingestion.
//
// WHY here, not in shared-bvh:
//   `mergeWorldSpaceFromCore` reads `primitive.positions` and `primitive.normals`
//   directly without awareness of the skinning solver.  The cleanest seam is a
//   pre-pass that produces a new Scene whose skinned-mesh primitives carry SOLVED
//   positions/normals/tangents/uvs so the rest of the pipeline
//   (mergeWorldSpaceFromCore, BVH packers, attribute packers) is
//   skinning-agnostic.
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
//   `solveSkin` fully handles morph-target + normal/UV blending when the primitive
//   carries `morphTargets` / `morphWeights` / `morphTargetNormals` /
//   `morphTargetUvs`.  No gap here
//   — morph application is part of the solver's pre-LBS step.

import type { EngineWarning, Scene, ScenePrimitive } from '@vitrum/core';
import { solveSkin } from '@vitrum/core';

export interface SolveSkinPrimitivesWarningOptions {
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: EngineWarning['phase'];
  readonly warningMethod?: string;
}

/**
 * Replace each `skinned-mesh` primitive's rest-pose `positions`, `normals`,
 * optional `tangents`, and morph-animated `uvs`/`uv1` with the CPU-solved posed
 * values from {@link solveSkin}.
 * All other primitives and all scene-level fields are returned unchanged.
 *
 * The returned primitives still carry `kind: 'skinned-mesh'` so downstream
 * code that filters by kind continues to work; only the geometry arrays are
 * substituted.
 */
export function solveSkinPrimitives(scene: Scene, warningOptions: SolveSkinPrimitivesWarningOptions = {}): Scene {
  let anyResolved = false;

  const primitives = scene.primitives.map((prim): ScenePrimitive => {
    if (prim.kind !== 'skinned-mesh') return prim;

    // Guard: degenerate primitive with no bones — pass through rest pose with a
    // warning so the scene at least renders (as a static rest-pose mesh) rather
    // than crashing the solver.
    if (prim.bones.length === 0) {
      emitEmptyBonesWarning(prim.id, warningOptions);
      return prim;
    }

    const { positions, normals, tangents, uvs, uv1 } = solveSkin(prim);
    anyResolved = true;
    // Return a new object that shares every field except solved geometry arrays.
    // Core solveSkin() now handles tangent morphing + skinning, so preserve its
    // posed tangent frame when present instead of falling back to derived frames.
    return {
      ...prim,
      positions,
      normals,
      ...(tangents != null ? { tangents } : {}),
      ...(uvs != null ? { uvs } : {}),
      ...(uv1 != null ? { uv1 } : {}),
    };
  });

  if (!anyResolved) return scene;
  return { ...scene, primitives };
}

function emitEmptyBonesWarning(
  primitiveId: ScenePrimitive['id'],
  warningOptions: SolveSkinPrimitivesWarningOptions,
): void {
  const message =
    `[vitrum/pt-webgl2] solveSkinPrimitives: primitive "${String(primitiveId)}" is skinned-mesh ` +
    'but has an empty bones array — rendering rest pose.';
  const warning: EngineWarning = {
    code: 'pt-webgl2.skinned-mesh-empty-bones',
    message,
    backend: 'pt-webgl2',
    phase: warningOptions.warningPhase ?? 'setScene',
    method: warningOptions.warningMethod ?? 'solveSkinPrimitives',
    details: {
      primitiveId: String(primitiveId),
      primitiveKind: 'skinned-mesh',
      fallback: 'rest-pose',
    },
  };

  if (warningOptions.onWarning) {
    warningOptions.onWarning(warning);
    return;
  }
  console.warn(message);
}
