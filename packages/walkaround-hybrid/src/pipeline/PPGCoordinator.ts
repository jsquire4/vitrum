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
import { recomputeDTreeInteriorFlux, refineDTree } from '../ppg/dTree.js';
import {
  serialiseSTree,
  deserialiseSTree,
  type OwnedSerialisedSTree,
  type SerialisedSTree,
} from '../ppg/serialise.js';
import { validateSerialisedSTree } from '../ppg/validateSerialisedSTree.js';
import {
  PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL,
  PPG_DEFAULT_SPATIAL_CELLS,
  PPG_MIS_ALPHA,
  PPG_FLUX_DECAY,
  PPG_DTREE_INITIAL_DEPTH,
} from '../ppg/ppgConstants.js';
import type { AABB, STree } from '../ppg/types.js';
import { allocatePPGResources, type FrameResources, type PPGFrameResources } from './resourceManager.js';
import type { PipelineSubsystem } from './PipelineSubsystem.js';
import type { EngineError, EngineWarning } from '@vitrum/core';
import {
  rethrowWithSceneMutationCleanup,
  runSceneMutationCleanups,
  type PreparedSceneMutation,
} from '../SceneMutationTransaction.js';
import {
  assertPpgQueryArenaPayloadFits,
  buildPpgQueryArenaHeader,
  nextPpgQueryArenaEpoch,
} from '../ppg/ppgQueryArena.js';
import { f32SnapshotMetadataMatches } from '../giStateSnapshot.js';

export type PPGTrainingEpochStatus =
  | 'collecting'
  | 'readback'
  | 'retry-pending'
  | 'failed'
  | 'disposed';

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
function derivePPGSceneAABB(bvh: SceneBvhPositionData): AABB {
  return deriveSceneAABBFromBvhPositions(bvh);
}

interface SceneBvhPositionData {
  readonly bvhPositions: {
    readonly cpuData: ArrayBuffer;
  };
}

export interface PPGSTreeSnapshot extends SerialisedSTree {
  readonly maxSpatialCells: number;
  readonly maxDTreeNodesPerCell?: number;
  readonly sceneBoundsMin: readonly [number, number, number];
  readonly sceneBoundsMax: readonly [number, number, number];
}

export type PreparedSTreeImport = PreparedSceneMutation;

/**
 * Type guard — returns `true` when `ppg` is a fully-allocated
 * `PPGFrameResources` (i.e. `allocatePPGResources` was called). Distinguishes
 * the PPG-enabled state from the empty-record default.
 */
function isPPGAllocated(ppg: FrameResources['ppg']): ppg is PPGFrameResources {
  return 'queryArenaBuf' in ppg;
}

function destroyPPGResources(ppg: PPGFrameResources): void {
  for (const buffer of new Set(ppgResourceBuffers(ppg))) {
    try { buffer.destroy(); } catch { /* release every independently-owned buffer */ }
  }
}

function ppgResourceBuffers(ppg: PPGFrameResources): readonly GPUBuffer[] {
  return [
    ppg.queryArenaBuf,
    ppg.fluxAtomicsBuf,
    ppg.cellSampleCountsBuf,
    ppg.updateUboBuffer,
  ];
}

function assertCandidatePPGResources(
  candidate: PPGFrameResources,
  live: PPGFrameResources,
): void {
  const candidateBuffers = ppgResourceBuffers(candidate);
  const liveBuffers = new Set(ppgResourceBuffers(live));
  if (
    new Set(candidateBuffers).size !== candidateBuffers.length ||
    candidateBuffers.some((buffer) => liveBuffers.has(buffer))
  ) {
    throw new Error(
      'PPG import candidate aliases a live or sibling resource buffer.',
    );
  }
}

function destroyCandidatePPGResources(
  candidate: PPGFrameResources,
  live: PPGFrameResources,
): void {
  const liveBuffers = new Set(ppgResourceBuffers(live));
  for (const buffer of new Set(ppgResourceBuffers(candidate))) {
    if (liveBuffers.has(buffer)) continue;
    try { buffer.destroy(); } catch { /* release every isolated candidate buffer */ }
  }
}

function isPPGSTreeSnapshot(value: unknown): value is PPGSTreeSnapshot {
  if (value == null || typeof value !== 'object') return false;
  const candidate = value as Partial<PPGSTreeSnapshot>;
  const boundsAreTriples = (
    bounds: unknown,
  ): bounds is readonly [number, number, number] =>
    Array.isArray(bounds) &&
    bounds.length === 3 &&
    bounds.every((entry) => typeof entry === 'number');
  return (
    typeof candidate.maxSpatialCells === 'number' &&
    (
      candidate.maxDTreeNodesPerCell === undefined ||
      typeof candidate.maxDTreeNodesPerCell === 'number'
    ) &&
    candidate.sTreeBuf instanceof Float32Array &&
    candidate.dTreeBuf instanceof Float32Array &&
    candidate.dTreeOffsets instanceof Uint32Array &&
    boundsAreTriples(candidate.sceneBoundsMin) &&
    boundsAreTriples(candidate.sceneBoundsMax)
  );
}

function cloneSTreeForRefine(source: STree): STree {
  return {
    sceneBounds: {
      min: [...source.sceneBounds.min] as [number, number, number],
      max: [...source.sceneBounds.max] as [number, number, number],
    },
    nodes: source.nodes.map((node) => ({
      ...node,
      aabb: {
        min: [...node.aabb.min] as [number, number, number],
        max: [...node.aabb.max] as [number, number, number],
      },
    })),
    dTrees: source.dTrees.map((tree) => ({
      totalFlux: tree.totalFlux,
      nodes: tree.nodes.map((node) => ({ ...node })),
    })),
  };
}

function nextCoordinatorGeneration(current: number): number {
  return current >= Number.MAX_SAFE_INTEGER ? 0 : current + 1;
}
function initialDTreeDepthForNodeCap(maxNodes: number | undefined): number {
  const cap = Math.max(
    1,
    Math.floor(maxNodes ?? PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL),
  );
  let depth = 0;
  let nodeCount = 1;
  let nextLevelNodes = 4;
  while (depth < PPG_DTREE_INITIAL_DEPTH && nodeCount + nextLevelNodes <= cap) {
    nodeCount += nextLevelNodes;
    nextLevelNodes *= 4;
    depth++;
  }
  return depth;
}

const NOOP_MUTATION: PreparedSceneMutation = {
  commit: () => undefined,
  rollback: () => undefined,
  finalize: () => undefined,
};


/**
 * Owns the PPG bootstrap state (enable flag, sTree, scene AABB) and the
 * serialise/upload writers. All methods are safe no-ops when PPG was not
 * enabled (or never initialized).
 */
export class PPGCoordinator implements PipelineSubsystem {
  private readonly _device: GPUDevice;
  private readonly _onWarning: ((warning: EngineWarning) => void) | null;
  private readonly _onError: ((error: EngineError) => void) | null;
  private static readonly _MAX_FINITE_F32 = 3.402823466e38;
  private static readonly _DEFAULT_READBACK_INTERVAL_FRAMES = 64;
  private static readonly _MAX_READBACK_FAILURES = 3;
  private _enabled = false;
  /** Retained from initialize() so onResize() forwards the same cap to
   *  allocatePPGResources — without this, an sTree that has grown beyond
   *  the default 1024 cells would overflow the under-allocated buffer. */
  private _maxSpatialCells: number | undefined = undefined;
  /** Retained from initialize() so onResize(), export, and import enforce the
   *  same per-cell dTree stride as the compiled update shader. */
  private _maxDTreeNodesPerCell: number | undefined = undefined;
  /** Last successfully published render dimensions, used to size import candidates. */
  private _width = 1;
  private _height = 1;
  private _mixAlpha = PPG_MIS_ALPHA;
  /** CPU-side PPG model (sTree + per-cell dTrees). Allocated at
   *  initialize() when ppgEnabled is true; refined from GPU training epochs
   *  and re-serialised transactionally after each successful readback. */
  private _sTree: STree | null = null;
  /** Scene-bounds AABB used to initialise the spatial tree root. Set from the
   *  BVH bounds at initialize() time. */
  private _sceneAABB: AABB = { min: [-10, -10, -10], max: [10, 10, 10] };
  private _fluxReadbackBuffer: GPUBuffer | null = null;
  /** A2 — staging buffer for the per-cell sample counts read back alongside flux. */
  private _cellCountReadbackBuffer: GPUBuffer | null = null;
  private _fluxReadbackInFlight = false;
  private _trainingEpochState: PPGTrainingEpochStatus = 'collecting';
  private _trainingReadbackFailures = 0;
  /** Number of update-pass dispatches accumulated in the current topology epoch. */
  private _trainingDispatchesSinceRefine = 0;
  /** Incremented on each onResize() call. The async readback chain captures
   *  this at launch and checks on completion — a mismatch means a resize
   *  happened mid-flight, so the frameResources arg is stale and any write
   *  through it would target destroyed GPU buffers. */
  private _frameResourcesGeneration = 0;
  private _lastTrainingReadbackErrorMessage: string | null = null;
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

  /** False while a sealed readback/refine epoch owns the live topology. */
  get trainingDispatchAllowed(): boolean {
    return this._enabled && this._trainingEpochState === 'collecting';
  }

  /** MIS mixing weight alpha (Muller section 3.4) the gi-ris RIS source pdf uses
   *  for the guided/cosine mixture `p_src = alpha*p_guide + (1-alpha)*p_cos`. */
  /** Durable lifecycle state for diagnostics and explicit recovery. */
  get trainingStatus(): PPGTrainingEpochStatus {
    return this._trainingEpochState;
  }

  /** Retry a sealed epoch after three consecutive readback failures. */
  requestTrainingRecovery(): boolean {
    if (!this._enabled || this._trainingEpochState !== 'failed') return false;
    this._trainingReadbackFailures = 0;
    this._lastTrainingReadbackErrorMessage = null;
    this._trainingEpochState = 'retry-pending';
    return true;
  }

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
   * The kernels descend the serialised buffers each frame; the CPU refines and
   * re-uploads the trees after each completed training epoch.
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
    // Validate caller-controlled proposal support before mutating lifecycle state.
    const resolvedMixAlpha = resolvePpgMixAlpha(mixAlpha);
    // Derive scene bounds from the uploaded BVH if available; the initial
    // single-cell sTree root must cover the rendered scene before adaptive
    // splits begin.
    const nextSceneAABB = derivePPGSceneAABB(bvhBuffers);
    const nextSTree = buildSTree(
      nextSceneAABB,
      initialDTreeDepthForNodeCap(maxDTreeNodesPerCell),
    );
    const allocationOptions = maxSpatialCells === undefined
      && maxDTreeNodesPerCell === undefined
      ? undefined
      : {
          ...(maxSpatialCells !== undefined ? { maxSpatialCells } : {}),
          ...(maxDTreeNodesPerCell !== undefined ? { maxDTreeNodesPerCell } : {}),
        };
    const candidate = allocatePPGResources(
      this._device,
      width,
      height,
      allocationOptions,
    );
    try {
      this._uploadTreeModel(candidate, nextSTree);
      this._writeUpdateUBOForResources(candidate, width, height);
    } catch (error) {
      destroyPPGResources(candidate);
      throw error;
    }

    const previous = frameResources.ppg;
    frameResources.ppg = candidate;
    // Publish coordinator state only after the complete candidate resource set
    // has accepted its tree and UBO uploads.
    this._enabled = true;
    this._fluxReadbackInFlight = false;
    this._trainingEpochState = 'collecting';
    this._trainingReadbackFailures = 0;
    this._trainingDispatchesSinceRefine = 0;
    this._maxSpatialCells = maxSpatialCells;
    this._maxDTreeNodesPerCell = maxDTreeNodesPerCell;
    this._mixAlpha = resolvedMixAlpha;
    this._sceneAABB = nextSceneAABB;
    this._sTree = nextSTree;
    this._width = width;
    this._height = height;
    if (isPPGAllocated(previous)) destroyPPGResources(previous);
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
    const candidate = allocatePPGResources(
      this._device,
      width,
      height,
      this._allocationOptions(),
    );
    try {
      this._uploadTreeModel(candidate, this._sTree!);
      this._writeUpdateUBOForResources(candidate, width, height);
    } catch (error) {
      destroyPPGResources(candidate);
      throw error;
    }
    const previous = frameResources.ppg;
    frameResources.ppg = candidate;
    // Bump the generation so any in-flight readback chain that captured the
    // old frameResources knows its resource references are now stale.
    this._frameResourcesGeneration = nextCoordinatorGeneration(
      this._frameResourcesGeneration,
    );
    this._discardReadbackBuffers();
    this._fluxReadbackInFlight = false;
    this._trainingEpochState = 'collecting';
    this._trainingReadbackFailures = 0;
    this._trainingDispatchesSinceRefine = 0;
    this._width = width;
    this._height = height;
    if (isPPGAllocated(previous)) destroyPPGResources(previous);
  }

  /**
   * Cold-restart the guide after scene-geometry mutation.
   *
   * A BVH/TLAS refit can move or resize the scene volume. Reusing the previous
   * sTree would train/sample against stale bounds and stale per-cell flux, so
   * mutation paths rebuild the single-cell root from the current BVH and clear
   * in-flight training state while preserving the already-allocated GPU buffers.
   */
  resetForSceneBvh(
    bvhBuffers: SceneBvhPositionData,
    frameResources: FrameResources,
    width: number,
    height: number,
  ): void {
    if (!this._enabled) return;
    const encoder = this._device.createCommandEncoder({ label: 'ppg-scene-reset' });
    const mutation = this.prepareResetForSceneBvh(
      bvhBuffers,
      frameResources,
      width,
      height,
      encoder,
    );
    try {
      mutation.commit();
      this._device.queue.submit([encoder.finish()]);
    } catch (error) {
      rethrowWithSceneMutationCleanup(
        error,
        [() => mutation.rollback()],
        'PPG scene reset failed and rollback also failed',
      );
    }
    mutation.finalize();
  }

  prepareResetForSceneBvh(
    bvhBuffers: SceneBvhPositionData,
    frameResources: FrameResources,
    width: number,
    height: number,
    encoder: GPUCommandEncoder,
  ): PreparedSceneMutation {
    if (!this._enabled) return NOOP_MUTATION;
    if (!isPPGAllocated(frameResources.ppg)) {
      throw new Error('PPG scene mutation requires allocated live resources.');
    }

    const ppg = frameResources.ppg;
    const nextAabb = derivePPGSceneAABB(bvhBuffers);
    const nextTree = buildSTree(
      nextAabb,
      initialDTreeDepthForNodeCap(this._maxDTreeNodesPerCell),
    );
    const cap = this._deriveMaxDTreeNodesPerCell(frameResources);
    const serialised = serialiseSTree(nextTree, cap);
    assertPpgQueryArenaPayloadFits(ppg.queryArenaLayout, serialised);
    const nextArenaEpoch = nextPpgQueryArenaEpoch(ppg.queryArenaEpoch);
    const arenaHeader = buildPpgQueryArenaHeader(
      ppg.queryArenaLayout,
      serialised,
      nextArenaEpoch,
    );
    const ubo = new Uint32Array(4);
    ubo[0] = Math.max(1, Math.floor(width / 2)) * Math.max(1, Math.floor(height / 2));
    ubo[1] = Math.floor(ppg.fluxAtomicsBuf.size / 4);
    ubo[2] = Math.floor(ppg.cellSampleCountsBuf.size / 4);

    const staging: GPUBuffer[] = [];
    const stageCopy = (
      destination: GPUBuffer,
      destinationOffset: number,
      data: ArrayBufferView,
    ): void => {
      if ((data.byteLength & 3) !== 0) {
        throw new RangeError('PPG staged copies require four-byte aligned payloads.');
      }
      const upload = this._device.createBuffer({
        label: 'ppg-scene-reset-staging',
        size: Math.max(4, data.byteLength),
        usage: 0x4,
        mappedAtCreation: true,
      });
      staging.push(upload);
      new Uint8Array(upload.getMappedRange()).set(
        new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      );
      upload.unmap();
      encoder.copyBufferToBuffer(upload, 0, destination, destinationOffset, data.byteLength);
    };
    try {
      stageCopy(ppg.queryArenaBuf, ppg.queryArenaLayout.sTreeByteOffset, serialised.sTreeBuf);
      stageCopy(ppg.queryArenaBuf, ppg.queryArenaLayout.dTreeByteOffset, serialised.dTreeBuf);
      stageCopy(ppg.queryArenaBuf, ppg.queryArenaLayout.dTreeOffsetsByteOffset, serialised.dTreeOffsets);
      // Header/epoch is copied last: it is the publication record for the three
      // query segments in this command buffer.
      stageCopy(ppg.queryArenaBuf, 0, arenaHeader);
      stageCopy(ppg.updateUboBuffer, 0, ubo);
      encoder.clearBuffer(ppg.fluxAtomicsBuf);
      encoder.clearBuffer(ppg.cellSampleCountsBuf);
    } catch (error) {
      rethrowWithSceneMutationCleanup(
        error,
        staging.map((buffer) => () => buffer.destroy()),
        'PPG scene-reset preparation failed and cleanup also failed',
      );
    }

    const oldAabb = this._sceneAABB;
    const oldTree = this._sTree;
    const oldTrainingDispatches = this._trainingDispatchesSinceRefine;
    const oldEpochState = this._trainingEpochState;
    const oldReadbackFailures = this._trainingReadbackFailures;
    const oldReadbackInFlight = this._fluxReadbackInFlight;
    const oldReadbackError = this._lastTrainingReadbackErrorMessage;
    const oldGeneration = this._frameResourcesGeneration;
    const oldArenaEpoch = ppg.queryArenaEpoch;
    const oldWidth = this._width;
    const oldHeight = this._height;
    let committed = false;
    let closed = false;
    const releaseStaging = (): void => {
      runSceneMutationCleanups(
        staging.map((buffer) => () => buffer.destroy()),
        'PPG scene-reset staging retirement failed',
      );
    };
    return {
      commit: () => {
        if (closed || committed) return;
        this._sceneAABB = nextAabb;
        this._sTree = nextTree;
        ppg.queryArenaEpoch = nextArenaEpoch;
        this._trainingDispatchesSinceRefine = 0;
        this._lastTrainingReadbackErrorMessage = null;
        this._trainingEpochState = 'collecting';
        this._trainingReadbackFailures = 0;
        this._fluxReadbackInFlight = false;
        this._frameResourcesGeneration = nextCoordinatorGeneration(oldGeneration);
        this._width = width;
        this._height = height;
        committed = true;
      },
      rollback: () => {
        if (closed) return;
        closed = true;
        runSceneMutationCleanups(
          [
            () => {
              if (!committed) return;
              this._sceneAABB = oldAabb;
              this._sTree = oldTree;
              ppg.queryArenaEpoch = oldArenaEpoch;
              this._trainingDispatchesSinceRefine = oldTrainingDispatches;
              this._trainingEpochState = oldEpochState;
              this._trainingReadbackFailures = oldReadbackFailures;
              this._fluxReadbackInFlight = oldReadbackInFlight;
              this._lastTrainingReadbackErrorMessage = oldReadbackError;
              this._frameResourcesGeneration = oldGeneration;
              this._width = oldWidth;
              this._height = oldHeight;
            },
            releaseStaging,
          ],
          'PPG scene-reset rollback failed',
        );
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        runSceneMutationCleanups(
          [
            () => {
              if (committed) this._discardReadbackBuffers();
            },
            releaseStaging,
          ],
          'PPG scene-reset retirement failed',
        );
      },
    };
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
    trainingDispatchOccurred: boolean,
    intervalDispatches: number = PPGCoordinator._DEFAULT_READBACK_INTERVAL_FRAMES,
  ): void {
    if (!this._enabled || this._sTree == null) return;
    if (!isPPGAllocated(frameResources.ppg)) return;
    if (!Number.isSafeInteger(intervalDispatches)
        || intervalDispatches <= 0
    ) {
      throw new RangeError(
        `PPG refine interval must be a positive safe integer; got ${intervalDispatches}`,
      );
    }
    const requiredDispatches = intervalDispatches;
    if (
      this._trainingEpochState === 'readback' ||
      this._trainingEpochState === 'failed' ||
      this._trainingEpochState === 'disposed'
    ) return;
    if (
      this._trainingEpochState === 'collecting' &&
      trainingDispatchOccurred
    ) {
      this._trainingDispatchesSinceRefine = Math.min(
        requiredDispatches,
        this._trainingDispatchesSinceRefine + 1,
      );
    }
    if (
      this._trainingEpochState === 'collecting' &&
      this._trainingDispatchesSinceRefine < requiredDispatches
    ) return;

    // Seal the topology and training buffers before copying. A retry-pending
    // epoch re-reads the same sealed buffers without accepting new deposits.
    const {
      fluxAtomicsBuf,
      queryArenaLayout,
      cellSampleCountsBuf: cellCountsBuf,
    } = frameResources.ppg;
    this._trainingDispatchesSinceRefine = 0;
    this._fluxReadbackInFlight = true;
    this._trainingEpochState = 'readback';

    // Active-prefix bound
    // `dTreeIndex * MAX_DTREE_NODES_PER_CELL + nodeIdx` for dTreeIndex in
    // [0, activeCells). Every slot past `activeCells * maxDTreeNodesPerCell`
    // is guaranteed zero (no sTree leaf maps to it). Copy / map / zero only
    // that prefix instead of the whole (up to ~22 MB) buffer — bit-identical
    // to reading the full buffer, since the tail is always zero.
    const fluxByteSize = fluxAtomicsBuf.size;
    const maxSpatialCells = queryArenaLayout.maxSpatialCells;
    const maxDTreeNodesPerCell = Math.max(1, Math.floor((fluxByteSize / 4) / maxSpatialCells));
    const activeCells = Math.min(this._sTree.dTrees.length, maxSpatialCells);
    // Round up to a 4-byte (u32) multiple — copyBufferToBuffer requires a
    // multiple-of-4 size, which `activeCells * maxDTreeNodesPerCell * 4`
    // already is.
    const activeBytes = Math.min(
      fluxByteSize,
      Math.max(4, activeCells * maxDTreeNodesPerCell * 4),
    );

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
    let fluxReadback: GPUBuffer;
    let cellCountReadback: GPUBuffer;
    try {
      if (this._fluxReadbackBuffer == null || this._fluxReadbackBuffer.size < activeBytes) {
        try { this._fluxReadbackBuffer?.destroy(); } catch { /* replace independently */ }
        this._fluxReadbackBuffer = this._device.createBuffer({
          label: 'ppg-flux-readback',
          size: activeBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
      }
      if (this._cellCountReadbackBuffer == null || this._cellCountReadbackBuffer.size < cellCountBytes) {
        try { this._cellCountReadbackBuffer?.destroy(); } catch { /* replace independently */ }
        this._cellCountReadbackBuffer = this._device.createBuffer({
          label: 'ppg-cellcount-readback',
          size: cellCountBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
      }
      fluxReadback = this._fluxReadbackBuffer;
      cellCountReadback = this._cellCountReadbackBuffer;
      const enc = this._device.createCommandEncoder({ label: 'ppg-flux-readback-copy' });
      enc.copyBufferToBuffer(fluxAtomicsBuf, 0, fluxReadback, 0, activeBytes);
      enc.copyBufferToBuffer(cellCountsBuf, 0, cellCountReadback, 0, cellCountBytes);
      this._device.queue.submit([enc.finish()]);
    } catch (raw) {
      this._fluxReadbackInFlight = false;
      this._trainingReadbackFailures++;
      this._trainingEpochState = this._trainingReadbackFailures >= PPGCoordinator._MAX_READBACK_FAILURES
        ? 'failed'
        : 'retry-pending';
      this._reportTrainingReadbackFailure(raw);
      return;
    }

    let refineCompleted = false;
    let readbackFailure: unknown = null;
    let fluxMapped = false;
    let cellCountsMapped = false;
    void this._device.queue.onSubmittedWorkDone()
      .then(async () => {
        try {
          if (this._sTree == null) return;
          if (this._frameResourcesGeneration !== capturedGeneration) return;
          await fluxReadback.mapAsync(GPUMapMode.READ, 0, activeBytes);
          fluxMapped = true;
          const mapped = fluxReadback.getMappedRange(0, activeBytes);
          const raw = new Float32Array(mapped.slice(0));
          fluxReadback.unmap();
          fluxMapped = false;
          if (this._frameResourcesGeneration !== capturedGeneration) return;
          await cellCountReadback.mapAsync(GPUMapMode.READ, 0, cellCountBytes);
          cellCountsMapped = true;
          const cellMapped = cellCountReadback.getMappedRange(0, cellCountBytes);
          const cellCounts = new Uint32Array(cellMapped.slice(0));
          cellCountReadback.unmap();
          cellCountsMapped = false;
          if (this._frameResourcesGeneration !== capturedGeneration) return;
          this._mergeFluxAndRefine(
            raw, cellCounts, frameResources, maxSpatialCells, maxDTreeNodesPerCell,
          );
          refineCompleted = true;
          this._lastTrainingReadbackErrorMessage = null;
        } finally {
          if (fluxMapped) fluxReadback.unmap();
          if (cellCountsMapped) cellCountReadback.unmap();
        }
      })
      .catch((err) => {
        readbackFailure = err;
        if (this._frameResourcesGeneration === capturedGeneration) {
          this._reportTrainingReadbackFailure(err);
        }
      })
      .finally(() => {
        if (this._frameResourcesGeneration !== capturedGeneration) return;
        this._fluxReadbackInFlight = false;
        if (refineCompleted) {
          this._trainingReadbackFailures = 0;
          this._trainingEpochState = this._enabled ? 'collecting' : 'disposed';
          return;
        }
        if (!this._enabled) {
          this._trainingEpochState = 'disposed';
          return;
        }
        this._trainingReadbackFailures++;
        if (this._trainingReadbackFailures >= PPGCoordinator._MAX_READBACK_FAILURES) {
          this._trainingEpochState = 'failed';
          this._reportTrainingReadbackFailure(new Error(
            `PPG training entered a durable failed state after ${this._trainingReadbackFailures} readback attempts; call HybridEngine.requestPpgTrainingRecovery() to retry the sealed epoch. Last failure: ${String(readbackFailure)}`,
          ));
        } else {
          this._trainingEpochState = 'retry-pending';
        }
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
   * Non-mutating compatibility probe used by the outer GI restore transaction.
   * It deliberately emits no diagnostics: the caller can inspect several
   * optional snapshot sections before choosing which transaction to prepare.
   */
  canImportSTree(
    snapshot: unknown,
    frameResources: FrameResources,
  ): snapshot is PPGSTreeSnapshot {
    return this._validateSTreeImport(snapshot, frameResources, false) !== null;
  }

  /**
   * Allocate and populate a complete replacement PPG generation without
   * touching the live guide. The returned transaction publishes handles and
   * CPU state only in commit, can restore them in rollback, and retires the old
   * GPU/readback generation only in finalize.
   *
   * The restored directional distribution starts a deliberately cold training
   * epoch: snapshot persistence covers the guide, not volatile per-frame flux
   * atomics, sample counters, readback attempts, or an in-flight readback.
   */
  prepareSTreeImport(
    snapshot: PPGSTreeSnapshot,
    frameResources: FrameResources,
  ): PreparedSTreeImport | null {
    const validated = this._validateSTreeImport(
      snapshot,
      frameResources,
      true,
    );
    if (validated === null) return null;

    const { restored, packed, liveResources } = validated;
    let candidate: PPGFrameResources | null = null;
    try {
      candidate = allocatePPGResources(
        this._device,
        this._width,
        this._height,
        this._allocationOptions(),
      );
      assertCandidatePPGResources(candidate, liveResources);
      this._uploadSerialisedTreeModel(candidate, packed);
      this._writeUpdateUBOForResources(candidate, this._width, this._height);

      // These accumulators are intentionally not part of the persisted guide.
      // Clear the isolated candidate cohort before it can become live.
      const encoder = this._device.createCommandEncoder({
        label: 'ppg-snapshot-import-candidate-clear',
      });
      encoder.clearBuffer(candidate.fluxAtomicsBuf);
      encoder.clearBuffer(candidate.cellSampleCountsBuf);
      this._device.queue.submit([encoder.finish()]);
    } catch (error) {
      if (candidate !== null) {
        destroyCandidatePPGResources(candidate, liveResources);
      }
      throw error;
    }

    const replacement = candidate;
    const oldTree = this._sTree;
    const oldGeneration = this._frameResourcesGeneration;
    const oldDispatches = this._trainingDispatchesSinceRefine;
    const oldReadbackFailures = this._trainingReadbackFailures;
    const oldReadbackError = this._lastTrainingReadbackErrorMessage;
    const oldReadbackInFlight = this._fluxReadbackInFlight;
    const oldEpochState = this._trainingEpochState;
    let state: 'prepared' | 'committed' | 'closed' = 'prepared';

    return {
      commit: () => {
        if (state !== 'prepared') return;
        if (
          frameResources.ppg !== liveResources ||
          this._sTree !== oldTree ||
          this._frameResourcesGeneration !== oldGeneration
        ) {
          throw new Error(
            'PPG import commit rejected because the live generation changed after preparation.',
          );
        }
        frameResources.ppg = replacement;
        this._sTree = restored;
        this._frameResourcesGeneration =
          nextCoordinatorGeneration(oldGeneration);
        this._trainingDispatchesSinceRefine = 0;
        this._trainingReadbackFailures = 0;
        this._lastTrainingReadbackErrorMessage = null;
        this._fluxReadbackInFlight = false;
        this._trainingEpochState = 'collecting';
        state = 'committed';
      },
      rollback: () => {
        if (state === 'closed') return;
        if (state === 'committed') {
          frameResources.ppg = liveResources;
          this._sTree = oldTree;
          this._frameResourcesGeneration = oldGeneration;
          this._trainingDispatchesSinceRefine = oldDispatches;
          this._trainingReadbackFailures = oldReadbackFailures;
          this._lastTrainingReadbackErrorMessage = oldReadbackError;
          this._fluxReadbackInFlight = oldReadbackInFlight;
          this._trainingEpochState = oldEpochState;
        }
        state = 'closed';
        destroyPPGResources(replacement);
      },
      finalize: () => {
        if (state === 'closed') return;
        if (state === 'prepared') {
          state = 'closed';
          destroyPPGResources(replacement);
          return;
        }
        state = 'closed';
        this._discardReadbackBuffers();
        destroyPPGResources(liveResources);
      },
    };
  }

  /** Prepare, publish, and retire a snapshot as one convenience operation. */
  importSTree(
    snapshot: PPGSTreeSnapshot,
    frameResources: FrameResources,
  ): boolean {
    let transaction: PreparedSTreeImport | null = null;
    try {
      transaction = this.prepareSTreeImport(snapshot, frameResources);
      if (transaction === null) return false;
      transaction.commit();
      transaction.finalize();
      return true;
    } catch (raw) {
      try { transaction?.rollback(); } catch { /* preserve the primary failure */ }
      this._warn({
        code: 'walkaround-hybrid.ppg-import-upload-failed',
        backend: 'walkaround-hybrid',
        phase: 'lifecycle',
        method: 'importGIState',
        message: `[PPGCoordinator] importSTree: candidate GPU preparation failed; current guide retained. ${raw instanceof Error ? raw.message : String(raw)}`,
        details: { fallback: 'retain current PPG guide' },
      });
      return false;
    }
  }

  private _validateSTreeImport(
    rawSnapshot: unknown,
    frameResources: FrameResources,
    emitWarning: boolean,
  ): {
    restored: STree;
    packed: OwnedSerialisedSTree;
    liveResources: PPGFrameResources;
  } | null {
    if (!this._enabled) return null;
    if (!isPPGSTreeSnapshot(rawSnapshot)) {
      if (emitWarning) {
        this._warn({
          code: 'walkaround-hybrid.ppg-import-malformed-snapshot',
          backend: 'walkaround-hybrid',
          phase: 'lifecycle',
          method: 'importGIState',
          message: '[PPGCoordinator] importSTree: malformed PPG snapshot rejected before live state mutation.',
          details: { fallback: 'retain current PPG guide' },
        });
      }
      return null;
    }
    const snapshot = rawSnapshot;
    const maxSpatialCells =
      this._maxSpatialCells ?? PPG_DEFAULT_SPATIAL_CELLS;
    if (snapshot.maxSpatialCells !== maxSpatialCells) {
      if (emitWarning) {
        this._warn({
          code: 'walkaround-hybrid.ppg-import-max-spatial-cells-mismatch',
          backend: 'walkaround-hybrid',
          phase: 'lifecycle',
          method: 'importGIState',
          message:
            `[PPGCoordinator] importSTree: maxSpatialCells mismatch — ` +
            `snapshot=${snapshot.maxSpatialCells}, live=${maxSpatialCells}. ` +
            `PPG restore rejected; the current guide is retained.`,
          details: {
            snapshotMaxSpatialCells: snapshot.maxSpatialCells,
            liveMaxSpatialCells: maxSpatialCells,
            fallback: 'retain current PPG guide',
          },
        });
      }
      return null;
    }

    const maxDTreeNodesPerCell =
      this._maxDTreeNodesPerCell ??
      PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
    const snapshotDTreeCap =
      snapshot.maxDTreeNodesPerCell ??
      PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
    if (snapshotDTreeCap !== maxDTreeNodesPerCell) {
      if (emitWarning) {
        this._warn({
          code: 'walkaround-hybrid.ppg-import-max-dtree-nodes-mismatch',
          backend: 'walkaround-hybrid',
          phase: 'lifecycle',
          method: 'importGIState',
          message:
            `[PPGCoordinator] importSTree: maxDTreeNodesPerCell mismatch — ` +
            `snapshot=${snapshotDTreeCap}, live=${maxDTreeNodesPerCell}. ` +
            `PPG restore rejected; the current guide is retained.`,
          details: {
            snapshotMaxDTreeNodesPerCell: snapshotDTreeCap,
            liveMaxDTreeNodesPerCell: maxDTreeNodesPerCell,
            fallback: 'retain current PPG guide',
          },
        });
      }
      return null;
    }

    const sceneBounds = this._sceneAABB;
    const boundsMatch =
      f32SnapshotMetadataMatches(
        snapshot.sceneBoundsMin[0],
        sceneBounds.min[0],
      ) &&
      f32SnapshotMetadataMatches(
        snapshot.sceneBoundsMin[1],
        sceneBounds.min[1],
      ) &&
      f32SnapshotMetadataMatches(
        snapshot.sceneBoundsMin[2],
        sceneBounds.min[2],
      ) &&
      f32SnapshotMetadataMatches(
        snapshot.sceneBoundsMax[0],
        sceneBounds.max[0],
      ) &&
      f32SnapshotMetadataMatches(
        snapshot.sceneBoundsMax[1],
        sceneBounds.max[1],
      ) &&
      f32SnapshotMetadataMatches(
        snapshot.sceneBoundsMax[2],
        sceneBounds.max[2],
      );
    if (!boundsMatch) {
      if (emitWarning) {
        this._warn({
          code: 'walkaround-hybrid.ppg-import-scene-bounds-mismatch',
          backend: 'walkaround-hybrid',
          phase: 'lifecycle',
          method: 'importGIState',
          message:
            '[PPGCoordinator] importSTree: scene-bounds mismatch — snapshot covers a different ' +
            'scene geometry. PPG restore rejected; the current guide is retained.',
          details: {
            snapshotSceneBounds: {
              min: snapshot.sceneBoundsMin,
              max: snapshot.sceneBoundsMax,
            },
            liveSceneBounds: {
              min: sceneBounds.min,
              max: sceneBounds.max,
            },
            tolerance: 'same-f32-representation',
            fallback: 'retain current PPG guide',
          },
        });
      }
      return null;
    }

    let restored: STree;
    let packed: OwnedSerialisedSTree;
    try {
      validateSerialisedSTree(snapshot, {
        maxSpatialCells,
        maxDTreeNodesPerCell,
        sceneBounds: {
          min: snapshot.sceneBoundsMin,
          max: snapshot.sceneBoundsMax,
        },
        epsilon: 1e-5,
      });
      restored = deserialiseSTree(snapshot, {
        min: snapshot.sceneBoundsMin,
        max: snapshot.sceneBoundsMax,
      });
      packed = serialiseSTree(restored, maxDTreeNodesPerCell);
    } catch (raw) {
      if (emitWarning) {
        this._warn({
          code: 'walkaround-hybrid.ppg-import-malformed-snapshot',
          backend: 'walkaround-hybrid',
          phase: 'lifecycle',
          method: 'importGIState',
          message: `[PPGCoordinator] importSTree: malformed PPG snapshot rejected before live state mutation. ${raw instanceof Error ? raw.message : String(raw)}`,
          details: { fallback: 'retain current PPG guide' },
        });
      }
      return null;
    }

    if (!isPPGAllocated(frameResources.ppg)) {
      if (emitWarning) {
        this._warn({
          code: 'walkaround-hybrid.ppg-import-resources-unavailable',
          backend: 'walkaround-hybrid',
          phase: 'lifecycle',
          method: 'importGIState',
          message: '[PPGCoordinator] importSTree: live PPG resources are unavailable; current guide retained.',
          details: { fallback: 'retain current PPG guide' },
        });
      }
      return null;
    }
    return {
      restored,
      packed,
      liveResources: frameResources.ppg,
    };
  }

  dispose(): void {
    this._enabled = false;
    this._sTree = null;
    this._fluxReadbackInFlight = false;
    this._frameResourcesGeneration = nextCoordinatorGeneration(
      this._frameResourcesGeneration,
    );
    this._trainingEpochState = 'disposed';
    this._trainingReadbackFailures = 0;
    this._trainingDispatchesSinceRefine = 0;
    this._discardReadbackBuffers();
    this._lastTrainingReadbackErrorMessage = null;
  }

  private _discardReadbackBuffers(): void {
    try { this._fluxReadbackBuffer?.destroy(); } catch { /* stale async owner */ }
    try { this._cellCountReadbackBuffer?.destroy(); } catch { /* stale async owner */ }
    this._fluxReadbackBuffer = null;
    this._cellCountReadbackBuffer = null;
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
  private _uploadTreeModel(ppg: PPGFrameResources, tree: STree): void {
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
    const cap = this._deriveMaxDTreeNodesPerCellFromResources(ppg);
    const serialised = serialiseSTree(tree, cap);
    this._uploadSerialisedTreeModel(ppg, serialised);
  }

  /** Upload a previously validated canonical tree into one isolated cohort. */
  private _uploadSerialisedTreeModel(
    ppg: PPGFrameResources,
    serialised: OwnedSerialisedSTree,
  ): void {
    assertPpgQueryArenaPayloadFits(ppg.queryArenaLayout, serialised);
    const nextEpoch = nextPpgQueryArenaEpoch(ppg.queryArenaEpoch);
    const header = buildPpgQueryArenaHeader(ppg.queryArenaLayout, serialised, nextEpoch);
    const write = (
      offset: number,
      data: ArrayBufferView<ArrayBuffer>,
    ): void => {
      this._device.queue.writeBuffer(
        ppg.queryArenaBuf,
        offset,
        data.buffer,
        data.byteOffset,
        data.byteLength,
      );
    };
    write(ppg.queryArenaLayout.sTreeByteOffset, serialised.sTreeBuf);
    write(ppg.queryArenaLayout.dTreeByteOffset, serialised.dTreeBuf);
    write(ppg.queryArenaLayout.dTreeOffsetsByteOffset, serialised.dTreeOffsets);
    // Publish only after every segment write is queued.
    write(0, header);
    ppg.queryArenaEpoch = nextEpoch;
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
    return this._deriveMaxDTreeNodesPerCellFromResources(frameResources.ppg);
  }

  private _deriveMaxDTreeNodesPerCellFromResources(
    ppg: PPGFrameResources,
  ): number {
    const { fluxAtomicsBuf, queryArenaLayout } = ppg;
    const maxSpatialCells = queryArenaLayout.maxSpatialCells;
    return Math.max(1, Math.floor((fluxAtomicsBuf.size / 4) / maxSpatialCells));
  }

  /**
   * W9 — Pack and upload the update-kernel UBO. Layout (16 bytes):
   *   [0] sampleCount       (u32) — half-res ReSTIR-GI reservoir entries
   *   [1] fluxBudget        (u32) — total flux atomic slots
   *   [2] sampleCountBudget (u32) — A2: cell-sample-counter slots (= maxSpatialCells)
   *   [3] padding
   */
  private _writeUpdateUBOForResources(
    ppg: PPGFrameResources,
    width: number,
    height: number,
  ): void {
    const {
      updateUboBuffer: buf,
      fluxAtomicsBuf: fluxAtomics,
      cellSampleCountsBuf: cellCounts,
    } = ppg;
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
   * Submit one indivisible guide publication: staged tree segments, header last,
   * and accumulator clears share a single command buffer. The live CPU tree is
   * swapped by the caller only after submit succeeds.
   */
  private _submitRefinedTree(
    ppg: PPGFrameResources,
    tree: STree,
    clearFluxBytes: number,
    clearCellCountBytes: number,
  ): void {
    const cap = this._deriveMaxDTreeNodesPerCellFromResources(ppg);
    const serialised = serialiseSTree(tree, cap);
    assertPpgQueryArenaPayloadFits(ppg.queryArenaLayout, serialised);
    const nextEpoch = nextPpgQueryArenaEpoch(ppg.queryArenaEpoch);
    const header = buildPpgQueryArenaHeader(
      ppg.queryArenaLayout,
      serialised,
      nextEpoch,
    );
    const encoder = this._device.createCommandEncoder({
      label: 'ppg-refine-publication',
    });
    const staging: GPUBuffer[] = [];
    const stageCopy = (
      destinationOffset: number,
      data: ArrayBufferView<ArrayBuffer>,
      label: string,
    ): void => {
      const upload = this._device.createBuffer({
        label,
        size: Math.max(4, data.byteLength),
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      staging.push(upload);
      new Uint8Array(upload.getMappedRange()).set(
        new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      );
      upload.unmap();
      encoder.copyBufferToBuffer(
        upload,
        0,
        ppg.queryArenaBuf,
        destinationOffset,
        data.byteLength,
      );
    };
    const releaseStaging = (): void => {
      for (const buffer of staging) {
        try { buffer.destroy(); } catch { /* retire every staging buffer */ }
      }
    };

    try {
      stageCopy(
        ppg.queryArenaLayout.sTreeByteOffset,
        serialised.sTreeBuf,
        'ppg-refine-stree',
      );
      stageCopy(
        ppg.queryArenaLayout.dTreeByteOffset,
        serialised.dTreeBuf,
        'ppg-refine-dtree',
      );
      stageCopy(
        ppg.queryArenaLayout.dTreeOffsetsByteOffset,
        serialised.dTreeOffsets,
        'ppg-refine-offsets',
      );
      // Epoch/header is the final publication record for the staged payload.
      stageCopy(0, header, 'ppg-refine-header');
      encoder.clearBuffer(ppg.fluxAtomicsBuf, 0, clearFluxBytes);
      encoder.clearBuffer(ppg.cellSampleCountsBuf, 0, clearCellCountBytes);
      this._device.queue.submit([encoder.finish()]);
    } catch (error) {
      releaseStaging();
      throw error;
    }

    ppg.queryArenaEpoch = nextEpoch;
    try {
      void this._device.queue.onSubmittedWorkDone().then(
        releaseStaging,
        releaseStaging,
      );
    } catch {
      releaseStaging();
    }
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
    rawFlux: Float32Array,
    cellCounts: Uint32Array,
    frameResources: FrameResources,
    maxSpatialCells: number,
    maxDTreeNodesPerCell: number,
  ): void {
    const liveSTree = this._sTree;
    if (!liveSTree) return;
    if (!isPPGAllocated(frameResources.ppg)) return;
    const ppg = frameResources.ppg;
    const { fluxAtomicsBuf, cellSampleCountsBuf: cellCountsBuf } = ppg;
    // Refine a private candidate. Any malformed readback, topology failure, or
    // upload/submit exception leaves the currently published CPU guide intact.
    const sTree = cloneSTreeForRefine(liveSTree);

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
      // Bound the CPU topology to the compiled per-cell stride before mutation.
      if (dTree.nodes.length > maxDTreeNodesPerCell) {
        throw new RangeError('PPG dTree topology exceeds its live GPU stride');
      }
      let totalFlux = 0;
      const nodeLimit = dTree.nodes.length;
      for (let nodeIdx = 0; nodeIdx < nodeLimit; nodeIdx++) {
        const slot = dTreeIdx * maxDTreeNodesPerCell + nodeIdx;
        const node = dTree.nodes[nodeIdx]!;
        // `rawFlux` only spans the active prefix; slots within it are dense.
        const rawFresh = rawFlux[slot] ?? 0;
        const fresh = Number.isFinite(rawFresh) && rawFresh > 0 ? rawFresh : 0;
        if (node.isLeaf) {
          // Decay the retained leaf flux (temporal prior), add this window's
          // fresh deposit. (A freshly-split child carries its parent's already
          // -merged flux as the prior, so the inherited distribution is kept.)
          const previous = Number.isFinite(node.flux) && node.flux > 0 ? node.flux : 0;
          node.flux = Math.min(
            PPGCoordinator._MAX_FINITE_F32,
            decay * previous + fresh,
          );
          totalFlux += node.flux;
        } else {
          // Interior node: zero out first; the propagation pass fills it.
          node.flux = 0;
        }
      }
      // Multiple saturated leaf atomics can sum beyond f32 even though every
      // individual slot is finite. Renormalise all leaves together so the
      // learned distribution is preserved and headers/interior masses remain
      // representable when uploaded as Float32Array.
      if (totalFlux > PPGCoordinator._MAX_FINITE_F32) {
        const scale = (PPGCoordinator._MAX_FINITE_F32 * 0.999999) / totalFlux;
        for (const node of dTree.nodes) {
          if (node.isLeaf) node.flux *= scale;
        }
      }
      recomputeDTreeInteriorFlux(dTree);

      refineDTree(dTree, undefined, undefined, undefined, maxDTreeNodesPerCell);
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
    this._submitRefinedTree(
      ppg,
      sTree,
      clearU32 * Uint32Array.BYTES_PER_ELEMENT,
      clearCellU32 * Uint32Array.BYTES_PER_ELEMENT,
    );
    this._sTree = sTree;
  }
}

function resolvePpgMixAlpha(value: number | undefined): number {
  if (value === undefined) return PPG_MIS_ALPHA;
  if (!Number.isFinite(value) || value <= 0 || value >= 1) {
    throw new RangeError('PPG mix alpha must be finite and strictly between 0 and 1.');
  }
  return value;
}
