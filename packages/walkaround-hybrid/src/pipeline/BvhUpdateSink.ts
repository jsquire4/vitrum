/**
 * BvhUpdateSink — narrow interface covering the 7 BVH / accumulator mutation
 * methods that `HybridEnginePrimitiveUpdates` calls on the pipeline.
 *
 * Extracted from `WalkaroundGPUPipeline` (complexity sweep 2026-06-02) so that
 * the primitive-update helpers are decoupled from the full pipeline class and
 * can be tested with a lightweight stub instead of a real GPU pipeline.
 *
 * `WalkaroundGPUPipeline implements BvhUpdateSink` — all 7 methods exist on
 * the pipeline already; the interface just formalises the contract.
 */

import type { SceneBVHBuffers } from '../restir/bvhCompute.js';

/**
 * Sink for BVH-mutation + accumulator-reset calls issued by
 * {@link transformRefit}, {@link positionsRefit},
 * {@link refitSkinnedMeshAfterGpuWrite}, {@link topologyRebuild}, and
 * {@link materialPatch} in `HybridEnginePrimitiveUpdates`.
 *
 * Callers obtain an instance via `PrimitiveUpdateContext.pipeline`; the
 * concrete implementation is `WalkaroundGPUPipeline`.
 */
export interface BvhUpdateSink {
  /**
   * BVH-refit fast path — overwrite the bvhNodes + bvhPositions GPU
   * buffers in place via `device.queue.writeBuffer`. Caller passes the
   * already-refit BVH node bytes and the affected position slice
   * (byte-offset relative to the start of `bvhPositions`).
   */
  refreshBvhRefit(
    bvhNodesBytes: ArrayBuffer,
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void;

  /** PR-7 — upload refit BVH nodes only (positions already on GPU). */
  refreshBvhNodesOnly(bvhNodesBytes: ArrayBuffer): void;

  /** PR-4 — upload refit TLAS nodes + instance transforms (topology unchanged). */
  refreshTlasRefit(
    tlasNodes: ArrayBuffer,
    worldToLocal: ArrayBuffer,
    localToWorld: ArrayBuffer,
  ): void;

  /**
   * Full BVH-buffer reupload — destroy + recreate the four BVH GPU buffers
   * from a freshly-built `SceneBVHBuffers`. Emitter buffers are NOT touched —
   * call `updateEmitters` separately if the emitter list also changed.
   */
  refreshBvhFullRebuild(
    bvhBuffers: Pick<
      SceneBVHBuffers,
      | 'bvhNodes'
      | 'bvhIndex'
      | 'bvhBeerColors'
      | 'bvhEmissiveLe'
      | 'bvhNormals'
      | 'bvhPositions'
      | 'bvhMode'
      | 'tlas'
    >,
  ): void;

  /** Re-upload emitter GPU buffers after an emitter list change. */
  updateEmitters(
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'emitters' | 'emitterCdf' | 'lightTree' | 'lightTreeNodeCount' | 'lightTreeEnabled'
    >,
  ): void;

  /**
   * Material-only fast path — partial upload of the packed `bvhIndex` slice
   * after CPU re-pack (PR-1). `bvhBeer` is a texture, so the whole beer
   * texture is re-uploaded from the full beer data + triCount.
   */
  refreshBvhMaterialSlice(
    indexSlice: { byteOffset: number; data: ArrayBuffer },
    beerFull: { data: ArrayBuffer; triCount: number },
    emissiveFull: { data: ArrayBuffer; triCount: number },
  ): void;

  /**
   * Reset the temporal accumulator to frame 0 (α=1 next frame).
   * Called after any geometric or material change that invalidates
   * temporal history.
   */
  requestAccumReset(): void;
}
