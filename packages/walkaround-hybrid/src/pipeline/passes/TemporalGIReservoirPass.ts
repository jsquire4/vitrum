/**
 * TemporalGIReservoirPass — Sprint 17 GI temporal-reuse pass.
 *
 * Half-resolution (W/2 × H/2) dispatch. group(0) is the dedicated GI reservoir
 * group (current GI reservoir, previous GI reservoir, ubo); group(1) is the
 * SHARED scene BVH/TLAS group, bound (GRIS Phases 1+2) so the reconnection-
 * visibility ray can traverse the scene when `ubo.restirPtReuse == 1`. The
 * scene group is inert on the legacy (gate-off) path.
 */

import { buildTemporalGiBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class TemporalGIReservoirPass implements Pass {
  readonly id = 'gi-temporal' as const;
  readonly dependencies: readonly string[] = ['gi-ris'];
  readonly passLabels: readonly PassLabel[] = ['gi-temporal'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, sceneBindGroup, halfWgX, halfWgY } = ctx;
    const bg = buildTemporalGiBindGroup(
      device, bglCache,
      resources.restirGI.reservoirGiCurrentBuffer,
      resources.restirGI.reservoirGiPreviousBuffer,
      resources.common.uboBuffer,
    );
    const pass = encoder.beginComputePass(computeDesc('gi-temporal'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    // group(1) — shared scene BVH/TLAS (GRIS reconnection-visibility ray).
    pass.setBindGroup(1, sceneBindGroup);
    pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
    pass.end();
  }

  dispose(): void {}
}
