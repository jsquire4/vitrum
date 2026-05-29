/**
 * HybridEngineGiPropagation — single owner of the "new BVH buffers ⇒ re-sync
 * the DDGI + RC GI subsystems" cascade (Wave 3.1, sub-task 3).
 *
 * Before this module the same cascade was hand-rolled (with drifted branches)
 * in three places:
 *   - HybridEngine._applyPrimitiveUpdateSubsystems (post primitive update —
 *     the variant that can carry an `rcRefitBounds` cheap cascade refit);
 *   - HybridEngine's init-host `publishBvh` (post async BVH publish — 2-way
 *     tlas-sync vs `setScene(root)` rebuild);
 *   - HybridEngineFrameOrchestrator.runDdgiAndRc (per-frame — tlas-sync only,
 *     NEVER the merged-mode `setScene(root)` rebuild, which would re-build the
 *     RC scene BVH every frame).
 *
 * The cascade, encoded once:
 *   1. DDGI — when `syncDdgi` is requested and BVH buffers exist, point the
 *      DDGI probe rays at the same BLAS/TLAS as ReSTIR.
 *   2. RC — when an RC subsystem is active:
 *        a. if `rcRefitBounds` is supplied:
 *             - tlas mode → cheap cascade-bounds refit, then sync the shared
 *               ReSTIR BVH buffers;
 *             - merged mode → in-place merged refit (`refitMergedInstance`) from
 *               the moved instance's re-derived world positions (the stride-4
 *               `bvhPositions` the post-update refit already updated), falling
 *               back to a full `setScene` rebuild (when `allowRcSceneRebuild`)
 *               only if the fast path declines (no retained CPU mirrors / a
 *               vertex-count change). A plain cascade-bounds refit is kept as
 *               the floor so the probe grid still tracks the new AABB even when
 *               neither the in-place refit nor the rebuild can run.
 *        b. else if BVH is in `tlas` mode → sync the shared ReSTIR BVH buffers;
 *        c. else (merged mode) and `allowRcSceneRebuild` → rebuild the RC scene
 *           BVH from the THREE root.
 *
 * The two flags capture the only axes the three sites differed on:
 *   - `rcRefitBounds`        — present only on the post-update path.
 *   - `allowRcSceneRebuild`  — true post-update / post-publish; false per-frame
 *                              (so the per-frame path stays tlas-sync-only).
 *   - `syncDdgi`             — true post-update / post-publish; per-frame it
 *                              mirrors the `ddgi`-layer gate the orchestrator
 *                              already computed.
 */

import type { Scene } from '@vitrum/core';
import type * as THREE from 'three';
import type { DDGI } from './ddgi/DDGI.js';
import type { RCSubsystem } from './HybridEngineRC.js';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';

export interface GiPropagationDeps {
  ddgi: DDGI;
  rc: RCSubsystem | null;
  bvhBuffers: SceneBVHBuffers | null;
  /** Scene threaded into DDGI's snapshot for material/emissive correlation. */
  lastScene: Scene | null;
  /** When true, re-point DDGI probe rays at the live BVH buffers. */
  syncDdgi: boolean;
  /** When true, the merged-mode RC fallback may rebuild the RC scene BVH from
   *  the THREE root. False on the per-frame path (would rebuild every frame). */
  allowRcSceneRebuild: boolean;
  /** Lazy THREE-root accessor for the merged-mode RC `setScene` fallback. */
  ensureThreeSceneRoot: () => THREE.Scene | null;
  /** Cheap cascade-bounds refit bounds (post primitive-update path only). */
  rcRefitBounds?: {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
  } | null | undefined;
}

/**
 * Run the BVH ⇒ GI-subsystem propagation cascade. Behaviour-preserving union
 * of the three previously-duplicated sites — see module header for the flag
 * semantics that recover each call site's exact behaviour.
 */
export function propagateBvhToGiSubsystems(deps: GiPropagationDeps): void {
  if (deps.syncDdgi && deps.bvhBuffers != null) {
    deps.ddgi.syncRestirBvhBuffers(deps.bvhBuffers, deps.lastScene ?? undefined);
  }

  const rc = deps.rc;
  if (rc == null) return;

  const isTlas = deps.bvhBuffers?.bvhMode === 'tlas';

  if (deps.rcRefitBounds != null) {
    if (isTlas && deps.bvhBuffers != null) {
      // TLAS mode: cheap cascade-bounds refit + share the live ReSTIR buffers
      // (the instance transforms live in the TLAS payload, not baked vertices).
      rc.refitCascadeBounds(deps.rcRefitBounds.min, deps.rcRefitBounds.max);
      rc.syncRestirBvhBuffers(deps.bvhBuffers);
      return;
    }

    // Merged mode: the moved instance's world transform is baked into the merged
    // vertex positions, so the RC merged BVH geometry is now stale. The
    // post-update refit (transformRefit / positionsRefit / topologyRebuild) has
    // already written the FULL re-derived stride-4 world positions into
    // `bvhBuffers.bvhPositions.cpuData` (old-world → local via
    // inverse(matrixWorldAtBuild) → new-world). Hand that buffer to the in-place
    // merged refit so RC re-uploads positions + node AABBs WITHOUT a teardown.
    const mergedPositions =
      deps.bvhBuffers != null
        ? new Float32Array(deps.bvhBuffers.bvhPositions.cpuData)
        : null;
    const refit =
      mergedPositions != null
        ? rc.refitMergedInstance(
            mergedPositions,
            deps.rcRefitBounds.min,
            deps.rcRefitBounds.max,
          )
        : false;
    if (refit) return;

    // Fast path declined (no retained merged CPU mirrors / vertex-count change).
    // Prefer a full RC scene rebuild from the THREE root when allowed; that path
    // refreshes the cascade bounds itself. Otherwise keep at least the cascade
    // probe grid in sync with the new AABB.
    if (deps.allowRcSceneRebuild) {
      const rcRoot = deps.ensureThreeSceneRoot();
      if (rcRoot != null) {
        rc.setScene(rcRoot);
        return;
      }
    }
    rc.refitCascadeBounds(deps.rcRefitBounds.min, deps.rcRefitBounds.max);
    return;
  }

  if (isTlas && deps.bvhBuffers != null) {
    rc.syncRestirBvhBuffers(deps.bvhBuffers);
    return;
  }

  if (deps.allowRcSceneRebuild) {
    const rcRoot = deps.ensureThreeSceneRoot();
    if (rcRoot != null) rc.setScene(rcRoot);
  }
}
