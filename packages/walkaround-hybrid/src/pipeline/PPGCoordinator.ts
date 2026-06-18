/**
 * PPGCoordinator — owns the Practical Path Guiding (Müller 2017) bootstrap
 * state that lives alongside the WebGPU pipeline.
 *
 * Extracted from {@link WalkaroundGPUPipeline} in the 2026-05-18 refactor
 * sweep: the pipeline used to hold `_ppgEnabled`, `_ppgSTree`, `_ppgSceneAABB`
 * plus tree/update UBO writers (`_uploadPPGTree`, `_writePPGUpdateUBO`) as
 * private members. Concentrating them here keeps
 * the orchestrator focused on pass scheduling and gives PPG a single owner
 * for its CPU-side sTree, scene bounds, and serialise / upload lifecycle.
 *
 * Lifecycle: the pipeline constructs one `PPGCoordinator` and forwards
 * `initialize`, `onResize`, and `dispose` calls. When PPG
 * is disabled (host opt-out, or one of the compute pipelines failed to
 * compile) every method is a cheap no-op — `enabled` stays `false`.
 */

import { deriveSceneAABBFromBvhPositions } from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import { buildSTree, splitOverflowLeaves } from '../ppg/sTree.js';
import { refineDTree } from '../ppg/dTree.js';
import { serialiseSTree, deserialiseSTree, type SerialisedSTree } from '../ppg/serialise.js';
import {
  PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL,
  PPG_DEFAULT_SPATIAL_CELLS,
  PPG_MIS_ALPHA,
  PPG_FLUX_DECAY,
} from '../ppg/ppgConstants.js';
import type { AABB, STree } from '../ppg/types.js';
import { allocatePPGResources, type FrameResources, type PPGFrameResources } from './resourceManager.js';
import type { PipelineSubsystem } from './PipelineSubsystem.js';
import type { EngineError, EngineWarning } from '@vitrum/core';

/**
 * W9 — derive a world-space AABB for the PPG sTree from the uploaded BVH data.
 *
 * Delegates to the canonical {@link deriveSceneAABBFromBvhPositions} (shared with
 * the NRC hash-grid in `WalkaroundGPUPipeline` and the ReGIR grid in
 * `ReGIRCoordinator`): scan the BVH position buffer (which the host always
 * uploads, per `restir/bvhCore.ts`), pad 1%, fall back to ±10 when empty.
 *
 * This AABB is used for the sTree root cell extents so adaptive splits
 * subdivide the actual scene volume.
 */
function derivePPGSceneAABB(bvh: { bvhPositions: { cpuData: ArrayBuffer; count: number } }): AABB {
  return deriveSceneAABBFromBvhPositions(bvh);
}

/**
 * Type guard — returns `true` when `ppg` is a fully-allocated
 * `PPGFrameResources` (i.e. `allocatePPGResources` was called). Distinguishes
 * the PPG-enabled state from the empty-record default.
 */
function isPPGAllocated(ppg: FrameResources['ppg']): ppg is PPGFrameResources {
  return 'sTreeBuf' in ppg;
}

/**
 * Owns the PPG bootstrap state (enable flag, sTree, scene AABB) and the
 * serialise/upload writers. All methods are safe no-ops when PPG was not
 * enabled (or never initialized).
 */
export class PPGCoordinator implements PipelineSubsystem {
  private readonly _device: GPUDevice;
  private readonly _onWarning: ((warning: EngineWarning) => void) | null;
  private readonly _onError: ((error: EngineError) => void) | null;
  private static readonly _FLUX_SCALE = 65536.0;
  private static readonly _DEFAULT_READBACK_INTERVAL_FRAMES = 64;
  private _enabled = false;
  /** Retained from initialize() so onResize() forwards the same cap to
   *  allocatePPGResources — without this, an sTree that has grown beyond
   *  the default 1024 cells would overflow the under-allocated buffer. */
  private _maxSpatialCells: number | undefined = undefined;
  /** Retained from initialize() so onResize(), export, and import enforce the
   *  same per-cell dTree stride as the compiled update shader. */
  private _maxDTreeNodesPerCell: number | undefined = undefined;
  private _mixAlpha = PPG_MIS_ALPHA;
  /** CPU-side PPG model (sTree + per-cell dTrees). Allocated at
   *  initialize() when ppgEnabled is true; serialised to GPU buffers per
   *  frame (Phase 1: static empty tree uploaded once). */
  private _sTree: STree | null = null;
  /** Scene-bounds AABB used to initialise the spatial tree root. Set from the
   *  BVH bounds at initialize() time. */
  private _sceneAABB: AABB = { min: [-10, -10, -10], max: [10, 10, 10] };
  private _fluxReadbackBuffer: GPUBuffer | null = null;
  /** A2 — staging buffer for the per-cell sample counts read back alongside flux. */
  private _cellCountReadbackBuffer: GPUBuffer | null = null;
  private _fluxReadbackInFlight = false;
  private _lastFluxReadbackFrame = -1;
  /** Incremented on each onResize() call. The async readback chain captures
   *  this at launch and checks on completion — a mismatch means a resize
   *  happened mid-flight, so the frameResources arg is stale and any write
   *  through it would target destroyed GPU buffers. */
  private _frameResourcesGeneration = 0;
  private _lastTrainingReadbackErrorMessage: string | null = null;
  /**
   * Reusable zero-fill scratch for clearing the GPU flux accumulators after a
   * refine cycle. Grown on demand to the active-prefix byte count we actually
   * clear (see {@link _mergeFluxAndRefine}); a fresh `Uint32Array` per cycle
   * would churn the GC with a multi-MB allocation every readback window.
   */
  private _fluxZeroScratch: Uint32Array | null = null;
  constructor(
    device: GPUDevice,
    diagnostics: {
      onWarning?: (warning: EngineWarning) => void;
      onError?: (error: EngineError) => void;
    } = {},
  ) {
    this._device = device;
    this._onWarning = diagnostics.onWarning ?? null;
    this._onError = diagnostics.onError ?? null;
  }

  /** Whether PPG dispatch is live. Mirrors the gate the pipeline forwards
   *  into {@link PassGateOptions.ppgEnabled}. */
  get enabled(): boolean {
    return this._enabled;
  }

  /** MIS mixing weight alpha (Muller section 3.4) the gi-ris RIS source pdf uses
   *  for the guided/cosine mixture `p_src = alpha*p_guide + (1-alpha)*p_cos`. */
  get mixAlpha(): number {
    return this._mixAlpha;
  }

  /**
   * W9 — initialize PPG resources at engine boot.
   *
   * Derives scene bounds from the uploaded BVH, builds a single-cell sTree
   * at those bounds, allocates the PPG GPU buffers via
   * {@link allocatePPGResources}, then uploads the serialised tree and packs
   * the update UBO. No-op when `ppgEnabled` is false — leaves `enabled` at false.
   *
   * The kernels descend the serialised buffers each frame; the CPU refines +
   * re-uploads on rebuild cycles (Phase 2 follow-up).
   */
  initialize(
    bvhBuffers: SceneBVHBuffers,
    frameResources: FrameResources,
    width: number,
    height: number,
    ppgEnabled: boolean,
    _frameCount: number,
    /** H47 — forwarded to allocatePPGResources `maxSpatialCells`; undefined = default 1 024. */
    maxSpatialCells?: number,
    /** H29 — forwarded to allocatePPGResources `maxDTreeNodesPerCell`; undefined = default 341. */
    maxDTreeNodesPerCell?: number,
    /** PPG guide/cosine MIS mixture alpha; undefined = paper default. */
    mixAlpha?: number,
  ): void {
    if (!ppgEnabled) {
      this._enabled = false;
      return;
    }
    this._enabled = true;
    // Retain for onResize() so it forwards the same cap on resize.
    this._maxSpatialCells = maxSpatialCells;
    this._maxDTreeNodesPerCell = maxDTreeNodesPerCell;
    this._mixAlpha = resolvePpgMixAlpha(mixAlpha);
    // Derive scene bounds from the uploaded BVH if available; the initial
    // single-cell sTree root must cover the rendered scene before adaptive
    // splits begin.
    this._sceneAABB = derivePPGSceneAABB(bvhBuffers);
    this._sTree = buildSTree(this._sceneAABB);
    frameResources.ppg = allocatePPGResources(
      this._device,
      width,
      height,
      this._allocationOptions(),
    );
    this._uploadTree(frameResources);
    this._writeUpdateUBO(frameResources, width, height);
  }

  /**
   * W9 — re-allocate PPG resolution-dependent buffers + re-upload the
   * (unchanged) sTree topology so the new bind groups have valid GPU
   * buffers to bind. The CPU sTree itself isn't size-dependent and
   * survives the resize unchanged.
   *
   * No-op when PPG is disabled.
   */
  onResize(
    frameResources: FrameResources,
    width: number,
    height: number,
    _frameCount: number,
  ): void {
    if (!this._enabled) return;
    // Forward the same maxSpatialCells cap used at initialize() time so a
    // tree that has grown past the default 1024-cell cap doesn't overflow
    // the re-allocated buffer on resize.
    frameResources.ppg = allocatePPGResources(
      this._device,
      width,
      height,
      this._allocationOptions(),
    );
    // Bump the generation so any in-flight readback chain that captured the
    // old frameResources knows its resource references are now stale.
    this._frameResourcesGeneration++;
    this._uploadTree(frameResources);
    this._writeUpdateUBO(frameResources, width, height);
  }

  /**
   * Run one PPG training/refine cycle when enough frames have elapsed:
   *
   * 1) Copy `fluxAtomicsBuf` into a MAP_READ staging buffer.
   * 2) Merge decoded flux into CPU-side dTrees.
   * 3) Run `refineDTree` + `splitOverflowLeaves`.
   * 4) Re-serialise and upload the updated sTree.
   * 5) Reset CPU and GPU accumulators for the next training window.
   *
   * Fire-and-forget async; renderFrame remains synchronous.
   */
  maybeRunTrainingRefine(
    frameResources: FrameResources,
    frameCount: number,
    intervalFrames: number = PPGCoordinator._DEFAULT_READBACK_INTERVAL_FRAMES,
  ): void {
    if (!this._enabled || this._sTree == null) return;
    if (!isPPGAllocated(frameResources.ppg)) return;
    const { fluxAtomicsBuf, dTreeOffsetsBuf: offsetsBuf, cellSampleCountsBuf: cellCountsBuf } = frameResources.ppg;
    if (this._fluxReadbackInFlight) return;
    if (this._lastFluxReadbackFrame >= 0
      && frameCount - this._lastFluxReadbackFrame < intervalFrames) {
      return;
    }

    // Active-prefix bound (perf): the update kernel only ever writes to slots
    // `dTreeIndex * MAX_DTREE_NODES_PER_CELL + nodeIdx` for dTreeIndex in
    // [0, activeCells). Every slot past `activeCells * maxDTreeNodesPerCell`
    // is guaranteed zero (no sTree leaf maps to it). Copy / map / zero only
    // that prefix instead of the whole (up to ~22 MB) buffer — bit-identical
    // to reading the full buffer, since the tail is always zero.
    const fluxByteSize = fluxAtomicsBuf.size;
    const maxSpatialCells = Math.max(1, Math.floor(offsetsBuf.size / 4));
    const maxDTreeNodesPerCell = Math.max(1, Math.floor((fluxByteSize / 4) / maxSpatialCells));
    const activeCells = Math.min(this._sTree.dTrees.length, maxSpatialCells);
    // Round up to a 4-byte (u32) multiple — copyBufferToBuffer requires a
    // multiple-of-4 size, which `activeCells * maxDTreeNodesPerCell * 4`
    // already is.
    const activeBytes = Math.min(
      fluxByteSize,
      Math.max(4, activeCells * maxDTreeNodesPerCell * 4),
    );

    this._lastFluxReadbackFrame = frameCount;
    this._fluxReadbackInFlight = true;
    // Capture the current frameResources generation. If onResize() fires
    // before the async chain completes, the generation will have changed
    // and the _mergeFluxAndRefine writes would target destroyed GPU buffers.
    const capturedGeneration = this._frameResourcesGeneration;

    // A2 — the per-cell sample counter holds one u32 per spatial cell. Read back
    // the same active prefix (activeCells cells).
    const cellCountBytes = Math.min(
      cellCountsBuf.size,
      Math.max(4, activeCells * 4),
    );

    // The readback staging buffer only needs to hold the active prefix. Grow
    // it on demand (it never shrinks within a session, which keeps it stable
    // as the sTree subdivides across training windows).
    if (this._fluxReadbackBuffer == null || this._fluxReadbackBuffer.size < activeBytes) {
      this._fluxReadbackBuffer?.destroy();
      this._fluxReadbackBuffer = this._device.createBuffer({
        label: 'ppg-flux-readback',
        size: activeBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    if (this._cellCountReadbackBuffer == null || this._cellCountReadbackBuffer.size < cellCountBytes) {
      this._cellCountReadbackBuffer?.destroy();
      this._cellCountReadbackBuffer = this._device.createBuffer({
        label: 'ppg-cellcount-readback',
        size: cellCountBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    const enc = this._device.createCommandEncoder({ label: 'ppg-flux-readback-copy' });
    enc.copyBufferToBuffer(fluxAtomicsBuf, 0, this._fluxReadbackBuffer, 0, activeBytes);
    enc.copyBufferToBuffer(cellCountsBuf, 0, this._cellCountReadbackBuffer, 0, cellCountBytes);
    this._device.queue.submit([enc.finish()]);

    void this._device.queue.onSubmittedWorkDone()
      .then(async () => {
        if (this._fluxReadbackBuffer == null || this._cellCountReadbackBuffer == null
          || this._sTree == null) return;
        // If a resize happened after we launched the copy, frameResources is
        // stale (its GPU buffers have been destroyed). Bail out to avoid
        // writing into destroyed buffers.
        if (this._frameResourcesGeneration !== capturedGeneration) return;
        await this._fluxReadbackBuffer.mapAsync(GPUMapMode.READ, 0, activeBytes);
        const mapped = this._fluxReadbackBuffer.getMappedRange(0, activeBytes);
        const raw = new Uint32Array(mapped.slice(0));
        this._fluxReadbackBuffer.unmap();
        await this._cellCountReadbackBuffer.mapAsync(GPUMapMode.READ, 0, cellCountBytes);
        const cellMapped = this._cellCountReadbackBuffer.getMappedRange(0, cellCountBytes);
        const cellCounts = new Uint32Array(cellMapped.slice(0));
        this._cellCountReadbackBuffer.unmap();
        this._mergeFluxAndRefine(
          raw, cellCounts, frameResources, maxSpatialCells, maxDTreeNodesPerCell,
        );
        this._lastTrainingReadbackErrorMessage = null;
      })
      .catch((err) => {
        this._reportTrainingReadbackFailure(err);
      })
      .finally(() => {
        this._fluxReadbackInFlight = false;
      });
  }

  /**
   * Export the current CPU sTree + per-cell dTree guiding distribution as flat
   * serialised buffers (the same wire format `_uploadTree` sends to the GPU).
   *
   * Returns `null` when PPG is disabled or not yet initialised (the caller
   * should treat a null return as "no PPG section in snapshot").
   *
   * The returned `SerialisedSTree` plus `maxSpatialCells`,
   * `maxDTreeNodesPerCell`, and `sceneBounds` are the pieces
   * `GIStateSnapshot.ppg` stores.
   */
  exportSTree(): (SerialisedSTree & {
    maxSpatialCells: number;
    maxDTreeNodesPerCell: number;
    sceneBoundsMin: readonly [number, number, number];
    sceneBoundsMax: readonly [number, number, number];
  }) | null {
    if (!this._enabled || !this._sTree) return null;
    const maxDTreeNodesPerCell = this._maxDTreeNodesPerCell ?? PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
    const serialised = serialiseSTree(this._sTree, maxDTreeNodesPerCell);
    const maxSpatialCells = this._maxSpatialCells ?? PPG_DEFAULT_SPATIAL_CELLS;
    return {
      ...serialised,
      maxSpatialCells,
      maxDTreeNodesPerCell,
      sceneBoundsMin: [
        this._sceneAABB.min[0], this._sceneAABB.min[1], this._sceneAABB.min[2],
      ] as const,
      sceneBoundsMax: [
        this._sceneAABB.max[0], this._sceneAABB.max[1], this._sceneAABB.max[2],
      ] as const,
    };
  }

  /**
   * Restore a PPG snapshot into the live coordinator, replacing the current
   * sTree with the deserialised snapshot tree and immediately re-uploading it
   * to the GPU buffers so guided sampling picks up the restored distribution
   * on the very next frame.
   *
   * Compatibility checks:
   *   1. `maxSpatialCells` and `maxDTreeNodesPerCell` must match the live
   *      coordinator caps (defaults 1 024 / 341 when unset). A mismatch means
   *      the sTree indices or dTree stride may be out-of-bounds for the live GPU
   *      buffer allocation — reject loudly.
   *   2. `sceneBoundsMin/Max` must match `_sceneAABB` within ε=1e-3 so a
   *      snapshot trained on a different scene's geometry is rejected before
   *      its guiding distribution poisons the live training.
   *
   * Returns `false` and emits a structured warning for any mismatch, `true` on success.
   *
   * No-op (returns false) when PPG is disabled or not yet initialised —
   * the importGIState caller treats false as "PPG restore skipped" and
   * continues with the atlas-only success.
   */
  importSTree(
    snapshot: {
      maxSpatialCells: number;
      maxDTreeNodesPerCell?: number;
      sTreeBuf: Float32Array;
      dTreeBuf: Float32Array;
      dTreeOffsets: Uint32Array;
      sceneBoundsMin: readonly [number, number, number];
      sceneBoundsMax: readonly [number, number, number];
    },
    frameResources: FrameResources,
  ): boolean {
    if (!this._enabled) return false;

    // ── Compatibility: maxSpatialCells ───────────────────────────────────────
    const liveCap = this._maxSpatialCells ?? PPG_DEFAULT_SPATIAL_CELLS;
    if (snapshot.maxSpatialCells !== liveCap) {
      this._warn({
        code: 'walkaround-hybrid.ppg-import-max-spatial-cells-mismatch',
        backend: 'walkaround-hybrid',
        phase: 'lifecycle',
        method: 'importGIState',
        message:
          `[PPGCoordinator] importSTree: maxSpatialCells mismatch — ` +
          `snapshot=${snapshot.maxSpatialCells}, live=${liveCap}. ` +
          `PPG restore rejected; guided sampling will restart cold.`,
        details: {
          snapshotMaxSpatialCells: snapshot.maxSpatialCells,
          liveMaxSpatialCells: liveCap,
          fallback: 'cold PPG restart',
        },
      });
      return false;
    }

    // ── Compatibility: maxDTreeNodesPerCell ─────────────────────────────────
    const liveDTreeCap = this._maxDTreeNodesPerCell ?? PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
    const snapshotDTreeCap = snapshot.maxDTreeNodesPerCell ?? PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
    if (snapshotDTreeCap !== liveDTreeCap) {
      this._warn({
        code: 'walkaround-hybrid.ppg-import-max-dtree-nodes-mismatch',
        backend: 'walkaround-hybrid',
        phase: 'lifecycle',
        method: 'importGIState',
        message:
          `[PPGCoordinator] importSTree: maxDTreeNodesPerCell mismatch — ` +
          `snapshot=${snapshotDTreeCap}, live=${liveDTreeCap}. ` +
          `PPG restore rejected; guided sampling will restart cold.`,
        details: {
          snapshotMaxDTreeNodesPerCell: snapshotDTreeCap,
          liveMaxDTreeNodesPerCell: liveDTreeCap,
          fallback: 'cold PPG restart',
        },
      });
      return false;
    }

    // ── Compatibility: scene bounds ──────────────────────────────────────────
    const eps = 1e-3;
    const sb = this._sceneAABB;
    const boundsOk =
      Math.abs(snapshot.sceneBoundsMin[0] - sb.min[0]) <= eps &&
      Math.abs(snapshot.sceneBoundsMin[1] - sb.min[1]) <= eps &&
      Math.abs(snapshot.sceneBoundsMin[2] - sb.min[2]) <= eps &&
      Math.abs(snapshot.sceneBoundsMax[0] - sb.max[0]) <= eps &&
      Math.abs(snapshot.sceneBoundsMax[1] - sb.max[1]) <= eps &&
      Math.abs(snapshot.sceneBoundsMax[2] - sb.max[2]) <= eps;
    if (!boundsOk) {
      this._warn({
        code: 'walkaround-hybrid.ppg-import-scene-bounds-mismatch',
        backend: 'walkaround-hybrid',
        phase: 'lifecycle',
        method: 'importGIState',
        message:
          `[PPGCoordinator] importSTree: scene-bounds mismatch — snapshot covers a different ` +
          `scene geometry. PPG restore rejected; guided sampling will restart cold.`,
        details: {
          snapshotSceneBounds: {
            min: snapshot.sceneBoundsMin,
            max: snapshot.sceneBoundsMax,
          },
          liveSceneBounds: {
            min: sb.min,
            max: sb.max,
          },
          epsilon: eps,
          fallback: 'cold PPG restart',
        },
      });
      return false;
    }

    // ── Deserialise and install ──────────────────────────────────────────────
    const restored = deserialiseSTree(snapshot, {
      min: snapshot.sceneBoundsMin,
      max: snapshot.sceneBoundsMax,
    });
    this._sTree = restored;

    // Upload the restored tree to the GPU so the very next frame samples from
    // the recovered distribution. `_uploadTree` already guards on PPG buffers
    // being allocated and on `_enabled`.
    this._uploadTree(frameResources);
    return true;
  }

  dispose(): void {
    this._enabled = false;
    this._sTree = null;
    this._fluxReadbackInFlight = false;
    this._lastFluxReadbackFrame = -1;
    this._fluxReadbackBuffer?.destroy();
    this._fluxReadbackBuffer = null;
    this._cellCountReadbackBuffer?.destroy();
    this._cellCountReadbackBuffer = null;
    this._fluxZeroScratch = null;
    this._lastTrainingReadbackErrorMessage = null;
  }

  private _warn(warning: EngineWarning): void {
    if (this._onWarning) {
      try {
        this._onWarning(warning);
      } catch {
        // Host warning callbacks must not break PPG training or state restore.
      }
      return;
    }
    console.warn(warning.message);
  }

  private _reportTrainingReadbackFailure(raw: unknown): void {
    if (!this._enabled) return;
    const detail = raw instanceof Error ? raw.message : String(raw);
    if (this._lastTrainingReadbackErrorMessage === detail) return;
    this._lastTrainingReadbackErrorMessage = detail;
    const message =
      `[PPGCoordinator] training refine readback failed; retaining previous PPG guide. ${detail}`;
    if (this._onError) {
      try {
        this._onError({
          kind: 'render',
          message,
          fatal: false,
          raw,
        });
      } catch {
        // Host error callbacks must not break the render loop.
      }
      return;
    }
    console.warn(message);
  }

  /**
   * W9 — Serialise the CPU sTree + per-cell dTrees and upload to the GPU
   * storage buffers. Called once at init; Phase 2 will call this after each
   * refinement cycle. No-op when PPG is disabled.
   */
  private _uploadTree(frameResources: FrameResources): void {
    if (!this._enabled || !this._sTree) return;
    if (!isPPGAllocated(frameResources.ppg)) return;
    const ppg = frameResources.ppg;
    // Item A — overflow guard. `serialiseSTree` with NO clamp sums every CPU
    // dTree's full node count; a host that allocated with
    // `maxDTreeNodesPerCell < 341` (while `refineDTree` still grows dTrees up to
    // the 341-node depth-4 cap) would produce a `dTreeBuf` LARGER than the GPU
    // allocation, so the `writeBuffer` below would throw / truncate. Derive the
    // per-cell node cap from the live GPU buffers (one flux slot per dTree node,
    // exactly as `_mergeFluxAndRefine` does) and clamp the serialised tree to
    // it. The DEFAULT 341-per-cell config is unaffected (cap ≥ tree size ⇒
    // no-op clamp), so this only changes behaviour for sub-341 hosts — for which
    // it keeps the upload valid instead of crashing.
    const cap = this._deriveMaxDTreeNodesPerCell(frameResources);
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(this._sTree, cap);
    this._device.queue.writeBuffer(ppg.sTreeBuf, 0, sTreeBuf.buffer, sTreeBuf.byteOffset, sTreeBuf.byteLength);
    this._device.queue.writeBuffer(ppg.dTreeBuf, 0, dTreeBuf.buffer, dTreeBuf.byteOffset, dTreeBuf.byteLength);
    this._device.queue.writeBuffer(ppg.dTreeOffsetsBuf, 0, dTreeOffsets.buffer, dTreeOffsets.byteOffset, dTreeOffsets.byteLength);
  }

  private _allocationOptions(): {
    maxSpatialCells?: number;
    maxDTreeNodesPerCell?: number;
  } | undefined {
    if (this._maxSpatialCells === undefined && this._maxDTreeNodesPerCell === undefined) {
      return undefined;
    }
    return {
      ...(this._maxSpatialCells !== undefined ? { maxSpatialCells: this._maxSpatialCells } : {}),
      ...(this._maxDTreeNodesPerCell !== undefined ? { maxDTreeNodesPerCell: this._maxDTreeNodesPerCell } : {}),
    };
  }

  /**
   * Derive the per-cell dTree node cap (`maxDTreeNodesPerCell`) baked into the
   * allocated GPU buffers, identically to {@link maybeRunTrainingRefine}'s
   * active-prefix bound: `fluxAtomicsBuf` holds one u32 slot per dTree node, so
   * `slots / maxSpatialCells` is the per-cell stride, where
   * `maxSpatialCells = dTreeOffsetsBuf.size / 4`. Returns `undefined` (= "no
   * clamp") when the buffers are missing — the caller then serialises the full
   * tree, matching the historical path.
   */
  private _deriveMaxDTreeNodesPerCell(frameResources: FrameResources): number | undefined {
    if (!isPPGAllocated(frameResources.ppg)) return undefined;
    const { fluxAtomicsBuf, dTreeOffsetsBuf: offsetsBuf } = frameResources.ppg;
    const maxSpatialCells = Math.max(1, Math.floor(offsetsBuf.size / 4));
    return Math.max(1, Math.floor((fluxAtomicsBuf.size / 4) / maxSpatialCells));
  }

  /**
   * W9 — Pack and upload the update-kernel UBO. Layout (16 bytes):
   *   [0] sampleCount       (u32) — half-res ReSTIR-GI reservoir entries
   *   [1] fluxBudget        (u32) — total flux atomic slots
   *   [2] sampleCountBudget (u32) — A2: cell-sample-counter slots (= maxSpatialCells)
   *   [3] padding
   */
  private _writeUpdateUBO(
    frameResources: FrameResources,
    width: number,
    height: number,
  ): void {
    if (!this._enabled) return;
    if (!isPPGAllocated(frameResources.ppg)) return;
    const { updateUboBuffer: buf, fluxAtomicsBuf: fluxAtomics, cellSampleCountsBuf: cellCounts } = frameResources.ppg;
    const fluxBudget = Math.floor(fluxAtomics.size / 4);
    const sampleCountBudget = Math.floor(cellCounts.size / 4);
    const halfW = Math.max(1, Math.floor(width / 2));
    const halfH = Math.max(1, Math.floor(height / 2));
    const data = new ArrayBuffer(16);
    const u32 = new Uint32Array(data);
    u32[0] = halfW * halfH;
    u32[1] = fluxBudget;
    u32[2] = sampleCountBudget;
    u32[3] = 0;
    this._device.queue.writeBuffer(buf, 0, data);
  }

  /**
   * Merge the read-back flux atomics into the CPU dTrees, refine, re-upload,
   * and zero the GPU accumulators for the next window.
   *
   * @param rawFlux              The active-prefix slice copied back from the
   *                             GPU (length = activeCells × maxDTreeNodesPerCell,
   *                             possibly shorter than the full GPU buffer).
   * @param cellCounts           A2 — per-spatial-cell sample counts (one u32 per
   *                             cell, indexed by dTreeIndex) read back this window;
   *                             drives `splitOverflowLeaves`.
   * @param maxSpatialCells      Cell capacity of the GPU buffers (= offsets /4).
   * @param maxDTreeNodesPerCell Per-cell slot stride baked into the buffers and
   *                             the update kernel (MAX_DTREE_NODES_PER_CELL).
   */
  private _mergeFluxAndRefine(
    rawFlux: Uint32Array,
    cellCounts: Uint32Array,
    frameResources: FrameResources,
    maxSpatialCells: number,
    maxDTreeNodesPerCell: number,
  ): void {
    const sTree = this._sTree;
    if (!sTree) return;
    if (!isPPGAllocated(frameResources.ppg)) return;
    const { fluxAtomicsBuf, cellSampleCountsBuf: cellCountsBuf } = frameResources.ppg;

    const activeCells = Math.min(sTree.dTrees.length, maxSpatialCells);
    // RUNAWAY FIX — Müller §5 per-window decay of the persistent flux
    // accumulator. The GPU buffer is zeroed each window (below), so `rawFlux`
    // is THIS window's fresh deposits. We combine it with the decayed carry-over
    // already stored on each CPU dTree node: `flux ← decay·prevFlux + fresh`.
    // Under steady input this converges to the bounded geometric steady state
    // F/(1−decay) instead of growing without bound (the filed runaway is the
    // decay=1 / no-reset regime). decay=0 reproduces the historical full reset.
    const decay = PPG_FLUX_DECAY;

    for (let dTreeIdx = 0; dTreeIdx < activeCells; dTreeIdx++) {
      const dTree = sTree.dTrees[dTreeIdx]!;

      // ── Step 1: combine fresh GPU readback with decayed carry-over ─────────
      // Interior nodes in the GPU atomic buffer are never written (the update
      // kernel descends to a leaf, then atomicAdd to that leaf's slot). So we
      // only set flux for leaf nodes here; interior node flux will be computed
      // by the bottom-up propagation pass below.
      let totalFlux = 0;
      const nodeLimit = Math.min(dTree.nodes.length, maxDTreeNodesPerCell);
      for (let nodeIdx = 0; nodeIdx < nodeLimit; nodeIdx++) {
        const slot = dTreeIdx * maxDTreeNodesPerCell + nodeIdx;
        const node = dTree.nodes[nodeIdx]!;
        // `rawFlux` only spans the active prefix; slots within it are dense.
        const fresh = (rawFlux[slot] ?? 0) / PPGCoordinator._FLUX_SCALE;
        if (node.isLeaf) {
          // Decay the retained leaf flux (temporal prior), add this window's
          // fresh deposit. (A freshly-split child carries its parent's already
          // -merged flux as the prior, so the inherited distribution is kept.)
          node.flux = decay * node.flux + fresh;
          totalFlux += node.flux;
        } else {
          // Interior node: zero out first; the propagation pass fills it.
          node.flux = 0;
        }
      }
      // Nodes beyond nodeLimit are untouched (already zero or from a prior cycle;
      // they are orphaned after refinement and dropped by compactDTree).
      dTree.totalFlux = totalFlux;

      // ── Step 2: bottom-up interior-flux propagation (H25) ────────────────
      // The GPU sampler (ppgDTreeSampleLeafBase in ppgPdf.wgsl) reads child
      // flux at `cBase + 4u` to do proportional CDF descent at EVERY interior
      // node. If an interior node's children are themselves interior (depth > 1
      // tree), those interior children need subtree sums for the descent to
      // work. The BFS layout produced by buildSubtree / compactDTree guarantees:
      //   parent at index p → children at firstChild, firstChild+1, ..+3
      //   and children always appear AFTER their parent.
      // So a single REVERSE pass over nodes[] propagates subtree sums bottom-up.
      //
      // We also apply the same propagation BEFORE dTreeSample/dTreePdf calls in
      // the CPU oracle (dTree.ts) so the oracle agrees with the GPU path — the
      // oracle was previously green-while-wrong because it shared the same bug
      // (interior node flux = 0 → uniform fall-back → leaf-selection biased to
      // the last child). After this pass both paths see the correct subtree sums.
      for (let nodeIdx = dTree.nodes.length - 1; nodeIdx >= 0; nodeIdx--) {
        const node = dTree.nodes[nodeIdx]!;
        if (node.isLeaf || node.firstChild < 0) continue;
        // Interior: accumulate children's flux (which may itself be a subtree sum
        // if the children are interior, since we scan in reverse = bottom-up).
        let childrenFlux = 0;
        for (let ci = 0; ci < 4; ci++) {
          const childIdx = node.firstChild + ci;
          if (childIdx < dTree.nodes.length) {
            childrenFlux += dTree.nodes[childIdx]!.flux;
          }
        }
        node.flux = childrenFlux;
      }

      refineDTree(dTree);
    }

    // Spatial refinement after directional refinement.
    // Deliberately run once per readback window, not per frame.
    //
    // Bound tree growth to the GPU buffer capacity (`maxSpatialCells`), NOT
    // the library default of 16 384. The flux/sTree/dTree buffers are sized
    // for `maxSpatialCells` cells (see allocatePPGResources); letting the CPU
    // tree grow past that would make serialiseSTree emit a buffer larger than
    // the allocation — `_uploadTree`'s writeBuffer would throw or silently
    // truncate the live tree. Passing the real cap keeps the CPU model and
    // the GPU buffers in lockstep.
    //
    // A2 — the split decision now reads the GPU per-cell sample counts (the
    // CPU-side node.sampleCount is never written on this path). On a split, the
    // child cells inherit a CLONE of the parent's (decayed-and-merged) dTree as
    // their directional prior (Müller §3.1 — handled inside splitOverflowLeaves).
    splitOverflowLeaves(sTree, undefined, maxSpatialCells, cellCounts);

    this._uploadTree(frameResources);
    // Clear ONLY the sTree leaf sampleCounts (the CPU split path is unused; the
    // GPU counter is the source). The dTree flux is the retained temporal prior
    // and is DELIBERATELY NOT zeroed here — decay was already folded into Step 1.
    for (const node of sTree.nodes) {
      if (node.splitAxis === -1) node.sampleCount = 0;
    }

    // The cell set may have GROWN during splitOverflowLeaves; clear the GPU
    // accumulators for the POST-split active prefix so the new child cells'
    // slots start clean for the next window.
    const postSplitActiveCells = Math.min(sTree.dTrees.length, maxSpatialCells);

    // Reset GPU flux accumulators for the next training window. Only the
    // active prefix was ever written (every other slot is still zero), so we
    // only need to clear that prefix — and we reuse a growable scratch buffer
    // instead of allocating a fresh multi-MB zero array each window.
    const clearU32 = Math.min(
      Math.floor(fluxAtomicsBuf.size / 4),
      Math.max(1, postSplitActiveCells * maxDTreeNodesPerCell),
    );
    // A2 — also clear the per-cell sample-count buffer (one u32 per cell) so the
    // next window's split decision is based only on the next window's traffic.
    const clearCellU32 = Math.min(
      Math.floor(cellCountsBuf.size / 4),
      Math.max(1, postSplitActiveCells),
    );
    const scratchNeeded = Math.max(clearU32, clearCellU32);
    if (this._fluxZeroScratch == null || this._fluxZeroScratch.length < scratchNeeded) {
      this._fluxZeroScratch = new Uint32Array(scratchNeeded);
    } else {
      // Grown-but-reused scratch may carry stale zeros only (we never write
      // non-zero into it), so no fill is needed; it is allocated zeroed and
      // we never mutate its contents.
    }
    this._device.queue.writeBuffer(
      fluxAtomicsBuf,
      0,
      this._fluxZeroScratch.buffer,
      this._fluxZeroScratch.byteOffset,
      clearU32 * 4,
    );
    this._device.queue.writeBuffer(
      cellCountsBuf,
      0,
      this._fluxZeroScratch.buffer,
      this._fluxZeroScratch.byteOffset,
      clearCellU32 * 4,
    );
  }
}

function resolvePpgMixAlpha(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) return PPG_MIS_ALPHA;
  return Math.min(1, Math.max(0, value));
}
