/**
 * ReGIRCoordinator — owns the ReGIR (Boksansky 2021 grid-based reservoirs)
 * bootstrap state that lives alongside the WebGPU pipeline.
 *
 * ReGIR pre-resamples lights into a world-space grid of reservoirs once per
 * frame so ReSTIR-DI's initial-candidate light SELECTION costs O(1) per pixel
 * regardless of light count (instead of an O(log N) light-tree descent per
 * pixel × M candidates). The grid is SEEDED BY THE LIGHT TREE: each cell's WRS
 * draws candidates via the tree's descent at the cell centroid (see
 * `regir.wgsl` + `shared-samplers/lightTree.ts regirBuildSurvivorCPU`).
 *
 * This coordinator resolves the grid geometry (world bounds → cell dims +
 * inverse cell size), computes the grid-region byte count for the COMBINED
 * light-tree + grid storage buffer (so `BvhBufferHost` sizes it before upload),
 * and produces the per-frame {@link RegirUboState}. It mirrors the structure of
 * {@link PPGCoordinator}: every method is a cheap no-op when ReGIR is off.
 *
 * Storage-budget note: the grid lives in the SAME `@group(3)` buffer as the
 * light tree (one binding ⇒ RIS stays at the 16 storage-buffer floor). The
 * grid-build pass binds that buffer read_write in its own bind group and writes
 * only the grid region. See `regir.wgsl` for the full rationale.
 */

import { REGIR_FLOATS_PER_SURVIVOR, LIGHT_TREE_FLOATS_PER_NODE } from '@vitrum/shared-samplers';
import { deriveSceneAABBFromBvhPositions } from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import { REGIR_OFF } from './uboUpdater.js';
import type { RegirUboState } from './uboUpdater.js';
import type { PipelineSubsystem } from './PipelineSubsystem.js';

// Light-tree node stride in floats = `LIGHT_TREE_FLOATS_PER_NODE` (imported from
// shared-samplers — single source of truth; B8 grew it 12→16). The grid region
// of the combined buffer begins immediately after the packed tree nodes.

/** Host-facing ReGIR config (resolved from {@link HybridEngineOptions}). */
export interface ReGIRConfig {
  /** Master gate. `false` ⇒ ReGIR off, RIS uses the light-tree path. */
  readonly enabled: boolean;
  /** Cells per axis (cubic-ish grid; the world bounds set the per-axis cell
   *  size, but the count is shared). Default 16 ⇒ 4096 cells. Clamped ≥ 1. */
  readonly cellsPerAxis: number;
  /** M — WRS candidates drawn per sub-reservoir at grid build. Default 32. */
  readonly candidatesPerCell: number;
  /** K — survivors stored per cell (per-pixel candidate diversity). Default 8. */
  readonly survivorsPerCell: number;
}

/** Resolve a partial host config to the full {@link ReGIRConfig} with defaults. */
export function resolveReGIRConfig(opts?: Partial<ReGIRConfig>): ReGIRConfig {
  return {
    enabled: opts?.enabled ?? false,
    cellsPerAxis: Math.max(1, Math.floor(opts?.cellsPerAxis ?? 16)),
    candidatesPerCell: Math.max(1, Math.floor(opts?.candidatesPerCell ?? 32)),
    survivorsPerCell: Math.max(1, Math.floor(opts?.survivorsPerCell ?? 8)),
  };
}

export class ReGIRCoordinator implements PipelineSubsystem {
  private readonly _config: ReGIRConfig;
  /** True only when the host opted in AND the light tree is live (ReGIR seeds
   *  cells via the tree) AND the grid-build pipeline compiled. */
  private _live = false;
  private _origin: [number, number, number] = [0, 0, 0];
  /** Uniform cubic cell size (max axis span / cellsPerAxis). */
  private _cellSize = 1;
  private _dims: [number, number, number] = [0, 0, 0];
  /** Float offset of the grid region in the combined buffer = nodeCount × LIGHT_TREE_FLOATS_PER_NODE (16). */
  private _gridFloatOffset = 0;

  constructor(config: ReGIRConfig) {
    this._config = config;
  }

  /** Whether ReGIR will dispatch + RIS should sample the grid this frame. */
  get live(): boolean {
    return this._live;
  }

  get config(): ReGIRConfig {
    return this._config;
  }

  /** Total cells in the grid (0 when not live). */
  get cellCount(): number {
    return this._dims[0] * this._dims[1] * this._dims[2];
  }

  /**
   * Byte count of the ReGIR grid region appended to the light-tree buffer. The
   * pipeline passes this to `BvhBufferHost.setRegirGridBytes` BEFORE
   * `uploadInitial`. Returns 0 when ReGIR is disabled by config (the buffer is
   * sized exactly as pre-ReGIR — byte-identical fallback). Uses a uniform grid
   * dim of `cellsPerAxis³` (a conservative upper bound independent of the scene
   * bounds, so the buffer size is stable across `updateEmitters`).
   */
  gridRegionBytes(): number {
    if (!this._config.enabled) return 0;
    const cells = this._config.cellsPerAxis ** 3;
    const floats = cells * this._config.survivorsPerCell * REGIR_FLOATS_PER_SURVIVOR;
    return floats * 4;
  }

  /**
   * Resolve the live grid geometry from the BVH + light-tree node count. Called
   * once at pipeline `initialize` (after the BVH is uploaded). Sets `_live`,
   * `_origin`, `_cellSize`, `_dims`, `_gridFloatOffset`.
   *
   * `gridBuildPipelineReady` is the pipeline's signal that the grid-build
   * compute pipeline compiled (false ⇒ ReGIR stays off even if config asked).
   */
  initialize(bvh: SceneBVHBuffers, gridBuildPipelineReady: boolean): void {
    const lightTreeLive = !!bvh.lightTreeEnabled && (bvh.lightTreeNodeCount ?? 0) > 0;
    this._live = this._config.enabled && lightTreeLive && gridBuildPipelineReady;
    if (!this._live) {
      this._dims = [0, 0, 0];
      this._gridFloatOffset = 0;
      return;
    }
    const aabb = deriveSceneAABBFromBvhPositions(bvh);
    const spanX = aabb.max[0] - aabb.min[0];
    const spanY = aabb.max[1] - aabb.min[1];
    const spanZ = aabb.max[2] - aabb.min[2];
    const maxSpan = Math.max(spanX, spanY, spanZ, 1e-4);
    const N = this._config.cellsPerAxis;
    this._cellSize = maxSpan / N;
    this._origin = [aabb.min[0], aabb.min[1], aabb.min[2]];
    // Per-axis cell count covers the axis span with the shared cubic cell size,
    // clamped to [1, cellsPerAxis] so the grid never exceeds the buffer region
    // sized by `gridRegionBytes` (which uses cellsPerAxis³).
    this._dims = [
      Math.min(N, Math.max(1, Math.ceil(spanX / this._cellSize))),
      Math.min(N, Math.max(1, Math.ceil(spanY / this._cellSize))),
      Math.min(N, Math.max(1, Math.ceil(spanZ / this._cellSize))),
    ];
    this._gridFloatOffset = (bvh.lightTreeNodeCount ?? 0) * LIGHT_TREE_FLOATS_PER_NODE;
  }

  /**
   * Re-derive the grid-region float offset after an emitter rebuild (the light
   * tree's node count may have changed, moving the grid region). Cheap; called
   * from the pipeline's `updateEmitters`. No-op when ReGIR is not live.
   */
  refreshAfterEmitterRebuild(bvh: Pick<SceneBVHBuffers, 'lightTreeNodeCount' | 'lightTreeEnabled'>): void {
    if (!this._config.enabled) return;
    // If the tree became degenerate (< 2 emitters), drop ReGIR for this scene
    // state — RIS falls back to the flat-CDF/tree path until the tree is live.
    const lightTreeLive = !!bvh.lightTreeEnabled && (bvh.lightTreeNodeCount ?? 0) > 0;
    this._live = this._live && lightTreeLive;
    this._gridFloatOffset = (bvh.lightTreeNodeCount ?? 0) * LIGHT_TREE_FLOATS_PER_NODE;
  }

  /** Per-frame UBO state. Returns an OFF state (gate 0) when not live, so RIS
   *  stays on the light-tree path bit-identically. */
  uboState(): RegirUboState {
    if (!this._live) {
      return REGIR_OFF;
    }
    return {
      enabled: true,
      origin: this._origin,
      invCellSize: 1 / this._cellSize,
      dims: this._dims,
      candidatesPerCell: this._config.candidatesPerCell,
      survivorsPerCell: this._config.survivorsPerCell,
      gridFloatOffset: this._gridFloatOffset,
    };
  }

  /**
   * Reset all coordinator state. ReGIRCoordinator owns no GPU resources of its
   * own — the grid data lives in {@link BvhBufferHost}'s combined light-tree +
   * grid buffer, which is released separately by {@link WalkaroundGPUPipeline}.
   * This call drops the CPU-side geometry mirrors and sets `_live` to false,
   * consistent with the no-op state the constructor establishes. Idempotent.
   */
  dispose(): void {
    this._live = false;
    this._origin = [0, 0, 0];
    this._cellSize = 1;
    this._dims = [0, 0, 0];
    this._gridFloatOffset = 0;
  }
}
