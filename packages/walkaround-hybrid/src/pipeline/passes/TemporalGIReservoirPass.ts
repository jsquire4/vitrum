/**
 * TemporalGIReservoirPass — Sprint 17 GI temporal-reuse pass.
 *
 * Half-resolution (W/2 × H/2) dispatch using a dedicated single-group BGL
 * (current GI reservoir, previous GI reservoir, ubo).
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

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, halfWgX, halfWgY } = ctx;
    const bg = buildTemporalGiBindGroup(
      device, bglCache,
      resources.restirGI.reservoirGiCurrentBuffer,
      resources.restirGI.reservoirGiPreviousBuffer,
      resources.common.uboBuffer,
    );
    const pass = encoder.beginComputePass(computeDesc('gi-temporal'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
    pass.end();
  }

  dispose(): void {}
}
