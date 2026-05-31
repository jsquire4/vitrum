/**
 * TemporalGIReservoirPass — Sprint 17 GI temporal-reuse pass.
 *
 * Half-resolution (W/2 × H/2) dispatch. group(0) is the dedicated GI reservoir
 * group (current GI reservoir, previous GI reservoir, ubo).
 *
 * GRIS gate (`grisEnabled`) — opt-in via `HybridEngineOptions.restirPtReuse`.
 * When ON, the compile-time pipeline variant (pipelineCompiler `grisOn`) adds a
 * `@group(1)` SHARED scene BVH/TLAS group so the reconnection-visibility ray can
 * traverse the scene; this pass binds it at slot 1 only in that case. When OFF
 * (default) the pipeline is the verbatim Sprint-17 single-group pass and this
 * pass must NOT call `setBindGroup(1, …)` — binding a group the layout does not
 * declare regressed the default render to all-black (f8df9a4).
 */

import { buildTemporalGiBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';

export class TemporalGIReservoirPass implements Pass {
  readonly id = 'gi-temporal' as const;
  readonly dependencies: readonly string[] = ['gi-ris'];
  readonly passLabels: readonly PassLabel[] = ['gi-temporal'];

  private readonly _pipeline: GPUComputePipeline;
  /** GRIS (restirPtReuse) ON ⇒ bind the scene group at @group(1). Must match
   *  the compile-time pipeline layout (see pipelineCompiler `grisOn`). */
  private readonly _grisEnabled: boolean;

  constructor(pipeline: GPUComputePipeline, grisEnabled = false) {
    this._pipeline = pipeline;
    this._grisEnabled = grisEnabled;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, bglCache, resources, sceneBindGroup } = ctx;
    const bg = buildTemporalGiBindGroup(
      device, bglCache,
      resources.restirGI.reservoirGiCurrentBuffer,
      resources.restirGI.reservoirGiPreviousBuffer,
      resources.common.uboBuffer,
    );
    // group(1) — shared scene BVH/TLAS (GRIS reconnection-visibility ray).
    // ONLY when the GRIS pipeline variant is active. Half-res dispatch.
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'gi-temporal', {
      half: true,
      ...(this._grisEnabled ? { extraGroups: [{ slot: 1, group: sceneBindGroup }] } : {}),
    });
  }

  dispose(): void {}
}
