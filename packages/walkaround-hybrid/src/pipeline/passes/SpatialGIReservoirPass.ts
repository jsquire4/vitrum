/**
 * SpatialGIReservoirPass — Sprint 17 GI spatial-reuse pass (2 ping-pong dispatches).
 *
 * Owns both `gi-spatial-1` and `gi-spatial-2` labels. The shade pass reads
 * the *current* GI reservoir which, after both spatial passes, contains the
 * spatially+temporally fused estimate.
 *
 * Pass 1: current → spatial. Pass 2: spatial → current.
 */

import { buildSpatialGiBindGroup } from '../bindGroupBuilders.js';
import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class SpatialGIReservoirPass implements Pass {
  readonly id = 'gi-spatial-2' as const; // shade depends on this terminal label.
  readonly dependencies: readonly string[] = ['gi-temporal'];
  readonly passLabels: readonly PassLabel[] = ['gi-spatial-1', 'gi-spatial-2'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, halfWgX, halfWgY } = ctx;
    // Pass 1: current → spatial.
    {
      const bg = buildSpatialGiBindGroup(
        device,
        bglCache,
        resources.restirGI.reservoirGiCurrentBuffer,
        resources.restirGI.reservoirGiSpatialBuffer,
        resources.common.uboBuffer,
        'spatial-gi-bg-1',
      );
      const pass = encoder.beginComputePass(computeDesc('gi-spatial-1'));
      pass.setPipeline(this._pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
      pass.end();
    }
    // Pass 2: spatial → current.
    {
      const bg = buildSpatialGiBindGroup(
        device,
        bglCache,
        resources.restirGI.reservoirGiSpatialBuffer,
        resources.restirGI.reservoirGiCurrentBuffer,
        resources.common.uboBuffer,
        'spatial-gi-bg-2',
      );
      const pass = encoder.beginComputePass(computeDesc('gi-spatial-2'));
      pass.setPipeline(this._pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
      pass.end();
    }
  }

  dispose(): void {}
}
