/**
 * ReGIRBuildPass — fills the world-space ReGIR light-reservoir grid each frame
 * BEFORE RIS samples it (Boksansky 2021, "Rendering Many Lights with Grid-Based
 * Reservoirs", RTG II ch. 23).
 *
 * Per (cell × survivor) the kernel runs one WRS sub-reservoir over the light
 * tree seeded at the cell centroid (power × proximity target), storing the
 * survivor emitter + its EXACT per-cell selection pmf `q̂_c(e)/Ŝ` into the grid
 * region of the COMBINED light-tree buffer. RIS then divides p̂ by that pmf for
 * an unbiased estimate (same discipline as the light-tree path). See
 * `shaders/regir.wgsl.ts` for the full unbiasedness derivation.
 *
 * Bind group: a DEDICATED group(0) (combined buffer READ_WRITE + emitters +
 * ubo), built per frame from the live buffers. The read_write binding never
 * touches the RIS / shade layouts — RIS binds the SAME buffer read-only at its
 * group(3), so RIS stays at the 16 storage-buffer floor.
 *
 * Gating: `gates()` returns true only when ReGIR is live (host opted in AND the
 * light tree is live AND the grid-build pipeline compiled). When ReGIR is off
 * the pass is never registered (so it never dispatches and the buffer stays
 * read-only everywhere — bit-identical light-tree fallback). The kernel ALSO
 * guards on `ubo.regirEnabled` defensively.
 */

import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import type { BGLCache } from '../bindGroupLayouts.js';
import type { ReGIRCoordinator } from '../ReGIRCoordinator.js';
import { buildRegirBuildBindGroup } from '../bindGroupBuilders.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';

/** Resources the grid-build pass binds — resolved per frame so buffer
 *  re-uploads (emitter rebuild) are picked up without re-registering the pass. */
export interface ReGIRBuildResources {
  /** Combined light-tree + ReGIR-grid buffer (bound read_write here). */
  readonly combinedLightTreeBuffer: GPUBuffer;
  /** Emitter list (for the per-cell q̂_c target). */
  readonly emitterBuffer: GPUBuffer;
  /** WalkaroundUBO (grid geometry + M/K + frameSeed + gate). */
  readonly uboBuffer: GPUBuffer;
}

export class ReGIRBuildPass implements Pass {
  readonly id = 'regir-build' as const;
  /** No DI-pass dependency: the grid only needs the (static) light tree +
   *  emitters + the UBO, all ready before sample-budget. Runs first so RIS
   *  (which depends on sample-budget) sees a fresh grid. */
  readonly dependencies: readonly string[] = [];
  readonly passLabels: readonly PassLabel[] = ['regir-build'];

  private readonly _pipeline: GPUComputePipeline;
  private readonly _coord: ReGIRCoordinator;
  private readonly _resources: () => ReGIRBuildResources;
  private readonly _bglCache: BGLCache;

  constructor(
    pipeline: GPUComputePipeline,
    coord: ReGIRCoordinator,
    bglCache: BGLCache,
    resources: () => ReGIRBuildResources,
  ) {
    this._pipeline = pipeline;
    this._coord = coord;
    this._bglCache = bglCache;
    this._resources = resources;
  }

  gates(): boolean {
    return this._coord.live;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    if (!this._coord.live) return;
    const cells = this._coord.cellCount;
    const k = this._coord.config.survivorsPerCell;
    const total = cells * k;
    if (total === 0) return;

    const res = this._resources();
    const buildBg = (): GPUBindGroup => buildRegirBuildBindGroup(
      ctx.device,
      this._bglCache,
      res.combinedLightTreeBuffer,
      res.emitterBuffer,
      res.uboBuffer,
    );
    const bg = cachedBindGroup(ctx.resourceCache, 'pass:regir-build', [
      res.combinedLightTreeBuffer,
      res.emitterBuffer,
      res.uboBuffer,
    ], buildBg);
    const pass = ctx.encoder.beginComputePass(ctx.computeDesc('regir-build'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    // 1-D dispatch over (cell × survivor); workgroup_size(64,1,1).
    pass.dispatchWorkgroups(Math.ceil(total / 64), 1, 1);
    pass.end();
  }

  dispose(): void {}
}
