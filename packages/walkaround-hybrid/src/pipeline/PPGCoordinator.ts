/**
 * PPGCoordinator — owns the Practical Path Guiding (Müller 2017) bootstrap
 * state that lives alongside the WebGPU pipeline.
 *
 * Extracted from {@link WalkaroundGPUPipeline} in the 2026-05-18 refactor
 * sweep: the pipeline used to hold `_ppgEnabled`, `_ppgSTree`, `_ppgSceneAABB`
 * plus three UBO/buffer writers (`_uploadPPGTree`, `_writePPGGuideUBO`,
 * `_writePPGUpdateUBO`) as private members. Concentrating them here keeps
 * the orchestrator focused on pass scheduling and gives PPG a single owner
 * for its CPU-side sTree, scene bounds, and serialise / upload lifecycle.
 *
 * Lifecycle: the pipeline constructs one `PPGCoordinator` and forwards
 * `initialize`, `onResize`, `refreshGuideUBO`, and `dispose` calls. When PPG
 * is disabled (host opt-out, or one of the compute pipelines failed to
 * compile) every method is a cheap no-op — `enabled` stays `false`.
 */

import type { SceneBVHBuffers } from '../restir/bvhCompute.js';
import { buildSTree } from '../ppg/sTree.js';
import { serialiseSTree } from '../ppg/serialise.js';
import { PPG_MIS_ALPHA } from '../ppg/ppgConstants.js';
import type { AABB, STree } from '../ppg/types.js';
import { allocatePPGResources, type FrameResources } from './resourceManager.js';

/**
 * W9 — derive a world-space AABB for the PPG sTree from the uploaded BVH data.
 *
 * The walkaround pipeline doesn't surface its scene bounds as a first-class
 * field; we recover them by scanning the BVH position buffer (which the host
 * always uploads, per `restir/bvhCompute.ts`). If the buffer is empty we
 * fall back to a generous default that contains any plausible scene.
 *
 * Phase 1: this AABB is used for two things — the sTree root cell extents
 * (so adaptive splits subdivide the actual scene volume), and the placeholder
 * "scene centre" query position uploaded into the guide UBO (until Phase 2
 * wires a per-pixel surface-position buffer).
 */
function derivePPGSceneAABB(bvh: { bvhPositions: { cpuData: ArrayBuffer; count: number } }): AABB {
  const view = new Float32Array(bvh.bvhPositions.cpuData);
  if (view.length < 4) {
    return { min: [-10, -10, -10], max: [10, 10, 10] };
  }
  // BVH position layout: vec4f per vertex (xyz + packed UV in w). Stride 4.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i + 3 <= view.length; i += 4) {
    const x = view[i]!, y = view[i + 1]!, z = view[i + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return { min: [-10, -10, -10], max: [10, 10, 10] };
  }
  // Pad by 1% to avoid edge-case boundary queries.
  const padX = (maxX - minX) * 0.01 + 1e-3;
  const padY = (maxY - minY) * 0.01 + 1e-3;
  const padZ = (maxZ - minZ) * 0.01 + 1e-3;
  return {
    min: [minX - padX, minY - padY, minZ - padZ],
    max: [maxX + padX, maxY + padY, maxZ + padZ],
  };
}

/**
 * Owns the PPG bootstrap state (enable flag, sTree, scene AABB) and the
 * three serialise/upload writers. All methods are safe no-ops when PPG was
 * not enabled (or never initialized).
 */
export class PPGCoordinator {
  private readonly _device: GPUDevice;
  private _enabled = false;
  /** CPU-side PPG model (sTree + per-cell dTrees). Allocated at
   *  initialize() when ppgEnabled is true; serialised to GPU buffers per
   *  frame (Phase 1: static empty tree uploaded once). */
  private _sTree: STree | null = null;
  /** Scene-bounds AABB carried in the guide UBO so the kernel can map flat
   *  pixel indices to the (placeholder) world-space query position for sTree
   *  descent. Set from the BVH bounds at initialize() time. */
  private _sceneAABB: AABB = { min: [-10, -10, -10], max: [10, 10, 10] };

  constructor(device: GPUDevice) {
    this._device = device;
  }

  /** Whether PPG dispatch is live. Mirrors the gate the pipeline forwards
   *  into {@link PassGateOptions.ppgEnabled}. */
  get enabled(): boolean {
    return this._enabled;
  }

  /**
   * W9 — initialize PPG resources at engine boot.
   *
   * Derives scene bounds from the uploaded BVH, builds a single-cell sTree
   * at those bounds, allocates the PPG GPU buffers via
   * {@link allocatePPGResources}, then uploads the serialised tree and packs
   * both UBOs. No-op when `ppgEnabled` is false — leaves `enabled` at false.
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
    frameCount: number,
  ): void {
    if (!ppgEnabled) {
      this._enabled = false;
      return;
    }
    this._enabled = true;
    // Derive scene bounds from the uploaded BVH if available — for Phase 1
    // we use a generous default that contains any plausible walkaround
    // scene. The world-space query position is currently the scene centre
    // (see ppgGuide.wgsl.ts), so the exact bound doesn't drive correctness
    // until Phase 2 wires a per-pixel position buffer.
    this._sceneAABB = derivePPGSceneAABB(bvhBuffers);
    this._sTree = buildSTree(this._sceneAABB);
    allocatePPGResources(this._device, frameResources, width, height);
    this._uploadTree(frameResources);
    this._writeGuideUBO(frameResources, width, height, frameCount);
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
    frameCount: number,
  ): void {
    if (!this._enabled) return;
    allocatePPGResources(this._device, frameResources, width, height);
    this._uploadTree(frameResources);
    this._writeGuideUBO(frameResources, width, height, frameCount);
    this._writeUpdateUBO(frameResources, width, height);
  }

  /**
   * Refresh the guide-kernel UBO so the kernel's per-frame RNG salt (and any
   * future per-frame inputs) stay current. The update UBO is
   * static-per-resolution and is NOT touched here.
   *
   * Called once per frame from {@link WalkaroundGPUPipeline.renderFrame}.
   * No-op when PPG is disabled.
   */
  refreshGuideUBO(
    frameResources: FrameResources,
    width: number,
    height: number,
    frameCount: number,
  ): void {
    if (!this._enabled) return;
    this._writeGuideUBO(frameResources, width, height, frameCount);
  }

  /** PPG owns no destroy()-able buffers of its own — the PPG GPU buffers
   *  live inside `FrameResources.ppg` and are released by
   *  `destroyFrameResources`. `dispose` here exists for API symmetry with
   *  the other state objects extracted in the same refactor. */
  dispose(): void {
    this._enabled = false;
    this._sTree = null;
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
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(this._sTree);
    this._device.queue.writeBuffer(ppg.sTreeBuf, 0, sTreeBuf.buffer, sTreeBuf.byteOffset, sTreeBuf.byteLength);
    this._device.queue.writeBuffer(ppg.dTreeBuf, 0, dTreeBuf.buffer, dTreeBuf.byteOffset, dTreeBuf.byteLength);
    this._device.queue.writeBuffer(ppg.dTreeOffsetsBuf, 0, dTreeOffsets.buffer, dTreeOffsets.byteOffset, dTreeOffsets.byteLength);
  }

  /**
   * W9 — Pack and upload the guide-kernel UBO. Layout matches `PPGGuideUBO`
   * in `ppgGuide.wgsl.ts` (12 × f32 = 48 bytes):
   *   [0]    pixelCount      (u32)
   *   [1]    imgWidth        (u32)
   *   [2]    alpha           (f32, MIS mixing weight)
   *   [3]    frameSeed       (u32)
   *   [4..6] sceneMin xyz    (f32)
   *   [7..9] sceneMax xyz    (f32)
   *   [10..11] padding
   */
  private _writeGuideUBO(
    frameResources: FrameResources,
    width: number,
    height: number,
    frameCount: number,
  ): void {
    if (!this._enabled) return;
    const buf = frameResources.ppg.guideUboBuffer;
    if (!buf) return;
    const data = new ArrayBuffer(48);
    const u32 = new Uint32Array(data);
    const f32 = new Float32Array(data);
    const pixelCount = width * height;
    u32[0] = pixelCount;
    u32[1] = width;
    f32[2] = PPG_MIS_ALPHA;
    u32[3] = frameCount >>> 0;
    f32[4] = this._sceneAABB.min[0];
    f32[5] = this._sceneAABB.min[1];
    f32[6] = this._sceneAABB.min[2];
    f32[7] = this._sceneAABB.max[0];
    f32[8] = this._sceneAABB.max[1];
    f32[9] = this._sceneAABB.max[2];
    u32[10] = 0;
    u32[11] = 0;
    this._device.queue.writeBuffer(buf, 0, data);
  }

  /**
   * W9 — Pack and upload the update-kernel UBO. Layout (16 bytes):
   *   [0] sampleCount  (u32)
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
    const data = new ArrayBuffer(16);
    const u32 = new Uint32Array(data);
    u32[0] = width * height;
    u32[1] = fluxBudget;
    u32[2] = 0;
    u32[3] = 0;
    this._device.queue.writeBuffer(buf, 0, data);
  }
}
