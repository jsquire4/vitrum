/**
 * BvhUpdateSink — narrow interface covering the BVH / accumulator mutation
 * methods that `HybridEnginePrimitiveUpdates` calls on the pipeline.
 *
 * Extracted from `WalkaroundGPUPipeline` (complexity sweep 2026-06-02) so that
 * the primitive-update helpers are decoupled from the full pipeline class and
 * can be tested with a lightweight stub instead of a real GPU pipeline.
 *
 * `WalkaroundGPUPipeline implements BvhUpdateSink` — these methods exist on
 * the pipeline already; the interface just formalises the contract.
 */

import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import type { MaterialTextureAtlasPayload } from './materialTextureAtlas.js';

export interface BvhBufferSlice {
  readonly byteOffset: number;
  readonly data: ArrayBuffer;
}

export interface TlasRefitMutation {
  readonly nodes: readonly BvhBufferSlice[];
  readonly worldToLocal: readonly BvhBufferSlice[];
  readonly localToWorld: readonly BvhBufferSlice[];
}

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
    bvhNodesByteOffset?: number,
  ): void;

  /**
   * H19 — upload a per-vertex normals slice into the GPU `bvhNormals` buffer.
   * Used by transform/positions fast paths to keep world-space smooth-shading
   * normals consistent after a refit. The skin path is exempt (the GPU-skin
   * kernel writes normals directly via `skinnedNormals`).
   *
   * @param normalsSlice - Byte offset + data for the affected vertex range.
   *   Each vertex is 16 bytes (vec4f, .w unused). The offset must be aligned
   *   to 4 bytes (GPU requirement for writeBuffer).
   */
  refreshBvhNormalsSlice(
    normalsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void;

  /**
   * CPU position bytes changed although the GPU position buffer was written by
   * an earlier command in the same pending transaction (GPU skinning).
   */
  recordLearningBvhPositionsSlice?(
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void;

  /** PR-7 — upload refit BVH nodes only (positions already on GPU). */
  refreshBvhNodesOnly(
    bvhNodesBytes: ArrayBuffer,
    bvhNodesByteOffset?: number,
  ): void;

  /** Upload exact refit TLAS node + changed-instance matrix slices. */
  refreshTlasRefit(mutation: TlasRefitMutation): void;

  /** Transactionally replace geometry, BLAS/TLAS, and emitter-selection GPU
   * resources. Implementations must leave all live bindings untouched when any
   * candidate allocation/upload fails. */
  replaceBvhAndEmitters(bvhBuffers: SceneBVHBuffers): void;

  /** Re-upload emitter GPU buffers after an emitter list change. */
  updateEmitters(
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'emitters' | 'emitterCdf' | 'emitterAlias' | 'lightTree' | 'lightTreeNodeCount' | 'lightTreeEnabled'
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
    /** B1 — FULL per-tri roughness+metalness re-upload (optional). */
    roughMetalFull?: { data: ArrayBuffer; triCount: number },
  ): void;

  /** Refresh only the material texture atlas + metadata textures after an
   *  atlas-affecting material patch. BVH/TLAS/emitter buffers are handled by the
   *  caller's separate material/emitter refresh paths. */
  refreshMaterialTextureAtlas(materialTextureAtlas: MaterialTextureAtlasPayload): void;

  /**
   * Reset the temporal accumulator to frame 0 (α=1 next frame).
   * Called after any geometric or material change that invalidates
   * temporal history.
   */
  requestAccumReset(): void;
}
