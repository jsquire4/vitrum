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
import { buildSTree, resetAccumulators, splitOverflowLeaves } from '../ppg/sTree.js';
import { refineDTree } from '../ppg/dTree.js';
import { serialiseSTree } from '../ppg/serialise.js';
import { PPG_MIS_ALPHA } from '../ppg/ppgConstants.js';
import type { AABB, STree } from '../ppg/types.js';
import { allocatePPGResources, type FrameResources } from './resourceManager.js';
import type { PipelineSubsystem } from './PipelineSubsystem.js';

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
 * Owns the PPG bootstrap state (enable flag, sTree, scene AABB) and the
 * serialise/upload writers. All methods are safe no-ops when PPG was not
 * enabled (or never initialized).
 */
export class PPGCoordinator implements PipelineSubsystem {
  private readonly _device: GPUDevice;
  private static readonly _FLUX_SCALE = 65536.0;
  private static readonly _DEFAULT_READBACK_INTERVAL_FRAMES = 64;
  private _enabled = false;
  /** CPU-side PPG model (sTree + per-cell dTrees). Allocated at
   *  initialize() when ppgEnabled is true; serialised to GPU buffers per
   *  frame (Phase 1: static empty tree uploaded once). */
  private _sTree: STree | null = null;
  /** Scene-bounds AABB used to initialise the spatial tree root. Set from the
   *  BVH bounds at initialize() time. */
  private _sceneAABB: AABB = { min: [-10, -10, -10], max: [10, 10, 10] };
  private _fluxReadbackBuffer: GPUBuffer | null = null;
  private _fluxReadbackInFlight = false;
  private _lastFluxReadbackFrame = -1;
  /**
   * Reusable zero-fill scratch for clearing the GPU flux accumulators after a
   * refine cycle. Grown on demand to the active-prefix byte count we actually
   * clear (see {@link _mergeFluxAndRefine}); a fresh `Uint32Array` per cycle
   * would churn the GC with a multi-MB allocation every readback window.
   */
  private _fluxZeroScratch: Uint32Array | null = null;
  constructor(device: GPUDevice) {
    this._device = device;
  }

  /** Whether PPG dispatch is live. Mirrors the gate the pipeline forwards
   *  into {@link PassGateOptions.ppgEnabled}. */
  get enabled(): boolean {
    return this._enabled;
  }

  /** MIS mixing weight α (Müller §3.4) the gi-ris RIS source pdf uses for the
   *  guided/cosine mixture `p_src = α·p_guide + (1−α)·p_cos`. Currently the
   *  paper default {@link PPG_MIS_ALPHA}; update training and gi-ris guided
   *  sampling agree on this value through the shared host constant. */
  get mixAlpha(): number {
    return PPG_MIS_ALPHA;
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
  ): void {
    if (!ppgEnabled) {
      this._enabled = false;
      return;
    }
    this._enabled = true;
    // Derive scene bounds from the uploaded BVH if available; the initial
    // single-cell sTree root must cover the rendered scene before adaptive
    // splits begin.
    this._sceneAABB = derivePPGSceneAABB(bvhBuffers);
    this._sTree = buildSTree(this._sceneAABB);
    allocatePPGResources(
      this._device,
      frameResources,
      width,
      height,
      maxSpatialCells !== undefined ? { maxSpatialCells } : undefined,
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
    allocatePPGResources(this._device, frameResources, width, height);
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
    const fluxAtomicsBuf = frameResources.ppg.fluxAtomicsBuf;
    const offsetsBuf = frameResources.ppg.dTreeOffsetsBuf;
    if (!fluxAtomicsBuf || !offsetsBuf) return;
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

    const enc = this._device.createCommandEncoder({ label: 'ppg-flux-readback-copy' });
    enc.copyBufferToBuffer(fluxAtomicsBuf, 0, this._fluxReadbackBuffer, 0, activeBytes);
    this._device.queue.submit([enc.finish()]);

    void this._device.queue.onSubmittedWorkDone()
      .then(async () => {
        if (this._fluxReadbackBuffer == null || this._sTree == null) return;
        await this._fluxReadbackBuffer.mapAsync(GPUMapMode.READ, 0, activeBytes);
        const mapped = this._fluxReadbackBuffer.getMappedRange(0, activeBytes);
        const raw = new Uint32Array(mapped.slice(0));
        this._fluxReadbackBuffer.unmap();
        this._mergeFluxAndRefine(
          raw, frameResources, maxSpatialCells, maxDTreeNodesPerCell,
        );
      })
      .catch((err) => {
        console.warn('[PPGCoordinator] training refine readback failed:', err);
      })
      .finally(() => {
        this._fluxReadbackInFlight = false;
      });
  }

  dispose(): void {
    this._enabled = false;
    this._sTree = null;
    this._fluxReadbackInFlight = false;
    this._lastFluxReadbackFrame = -1;
    this._fluxReadbackBuffer?.destroy();
    this._fluxReadbackBuffer = null;
    this._fluxZeroScratch = null;
  }

  /**
   * W9 — Serialise the CPU sTree + per-cell dTrees and upload to the GPU
   * storage buffers. Called once at init; Phase 2 will call this after each
   * refinement cycle. No-op when PPG is disabled.
   */
  private _uploadTree(frameResources: FrameResources): void {
    if (!this._enabled || !this._sTree) return;
    const ppg = frameResources.ppg;
    if (!ppg.sTreeBuf || !ppg.dTreeBuf || !ppg.dTreeOffsetsBuf) return;
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
    const ppg = frameResources.ppg;
    const fluxAtomicsBuf = ppg.fluxAtomicsBuf;
    const offsetsBuf = ppg.dTreeOffsetsBuf;
    if (!fluxAtomicsBuf || !offsetsBuf) return undefined;
    const maxSpatialCells = Math.max(1, Math.floor(offsetsBuf.size / 4));
    return Math.max(1, Math.floor((fluxAtomicsBuf.size / 4) / maxSpatialCells));
  }

  /**
   * W9 — Pack and upload the update-kernel UBO. Layout (16 bytes):
   *   [0] sampleCount  (u32) — half-res ReSTIR-GI reservoir entries
   *   [1] fluxBudget   (u32) — total atomic slots
   *   [2..3] padding
   */
  private _writeUpdateUBO(
    frameResources: FrameResources,
    width: number,
    height: number,
  ): void {
    if (!this._enabled) return;
    const buf = frameResources.ppg.updateUboBuffer;
    if (!buf) return;
    const fluxAtomics = frameResources.ppg.fluxAtomicsBuf;
    const fluxBudget = fluxAtomics ? Math.floor(fluxAtomics.size / 4) : 0;
    const halfW = Math.max(1, Math.floor(width / 2));
    const halfH = Math.max(1, Math.floor(height / 2));
    const data = new ArrayBuffer(16);
    const u32 = new Uint32Array(data);
    u32[0] = halfW * halfH;
    u32[1] = fluxBudget;
    u32[2] = 0;
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
   * @param maxSpatialCells      Cell capacity of the GPU buffers (= offsets /4).
   * @param maxDTreeNodesPerCell Per-cell slot stride baked into the buffers and
   *                             the update kernel (MAX_DTREE_NODES_PER_CELL).
   */
  private _mergeFluxAndRefine(
    rawFlux: Uint32Array,
    frameResources: FrameResources,
    maxSpatialCells: number,
    maxDTreeNodesPerCell: number,
  ): void {
    const sTree = this._sTree;
    if (!sTree) return;
    const fluxAtomicsBuf = frameResources.ppg.fluxAtomicsBuf;
    if (!fluxAtomicsBuf) return;

    const activeCells = Math.min(sTree.dTrees.length, maxSpatialCells);

    for (let dTreeIdx = 0; dTreeIdx < activeCells; dTreeIdx++) {
      const dTree = sTree.dTrees[dTreeIdx]!;

      // ── Step 1: assign leaf flux from GPU readback ────────────────────────
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
        const flux = (rawFlux[slot] ?? 0) / PPGCoordinator._FLUX_SCALE;
        if (node.isLeaf) {
          node.flux = flux;
          totalFlux += flux;
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
    splitOverflowLeaves(sTree, undefined, maxSpatialCells);

    this._uploadTree(frameResources);
    resetAccumulators(sTree);

    // Reset GPU flux accumulators for the next training window. Only the
    // active prefix was ever written (every other slot is still zero), so we
    // only need to clear that prefix — and we reuse a growable scratch buffer
    // instead of allocating a fresh multi-MB zero array each window.
    const clearU32 = Math.min(
      Math.floor(fluxAtomicsBuf.size / 4),
      Math.max(1, activeCells * maxDTreeNodesPerCell),
    );
    if (this._fluxZeroScratch == null || this._fluxZeroScratch.length < clearU32) {
      this._fluxZeroScratch = new Uint32Array(clearU32);
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
  }
}
