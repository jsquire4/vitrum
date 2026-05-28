/**
 * SpatialGIReservoirPass — Sprint 17 GI spatial-reuse pass (1 or 2 ping-pong dispatches).
 *
 * The shade pass reads the *current* GI reservoir
 * (`reservoirGiCurrentBuffer`), which after this pass contains the
 * spatially+temporally fused estimate.
 *
 * Full (2-pass) routing:
 *   Pass 1: current → spatial. Pass 2: spatial → current.
 * The two dispatches use SEPARATE in/out buffers because the gather reads
 * neighbours from `resIn` while writing the centre pixel to `resOut`; in-place
 * (resIn == resOut) would race a neighbour read against another thread's write.
 *
 * Phase-0 productization — the ping-pong PASS COUNT is host/preset-driven via
 * the constructor `passCount` arg (the spatial NEIGHBOR count `K_SPATIAL_GI=5u`
 * stays a fixed compile-time const). `2` (ultra/high) is full fidelity; `1`
 * (medium/low) does a single `current → spatial` dispatch followed by a buffer
 * copy `spatial → current`, so the result still lands in the buffer shade
 * reads while halving the spatial dispatch cost. The terminal label
 * `gi-spatial-2` is kept for BOTH counts so the `shade` dependency + `id` stay
 * stable; a 1-pass config emits only `['gi-spatial-2']`.
 *
 * R2 — `buildPassLayout` MUST be built with the same `giSpatialPasses` so the
 * timestamp slot layout matches the labels emitted here.
 */

import { buildSpatialGiBindGroup } from '../bindGroupBuilders.js';
import { giSpatialPassLabels } from './passOrder.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class SpatialGIReservoirPass implements Pass {
  readonly id = 'gi-spatial-2' as const; // shade depends on this terminal label.
  readonly dependencies: readonly string[] = ['gi-temporal'];
  readonly passLabels: readonly PassLabel[];

  private readonly _pipeline: GPUComputePipeline;
  private readonly _passCount: 1 | 2;

  constructor(pipeline: GPUComputePipeline, passCount: 1 | 2 = 2) {
    this._pipeline = pipeline;
    this._passCount = passCount;
    this.passLabels = giSpatialPassLabels(passCount);
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, halfWgX, halfWgY } = ctx;
    const current = resources.restirGI.reservoirGiCurrentBuffer;
    const spatial = resources.restirGI.reservoirGiSpatialBuffer;

    if (this._passCount === 2) {
      // Pass 1: current → spatial (label gi-spatial-1).
      {
        const bg = buildSpatialGiBindGroup(
          device, bglCache, current, spatial, resources.common.uboBuffer, 'spatial-gi-bg-1',
        );
        const pass = encoder.beginComputePass(computeDesc('gi-spatial-1'));
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
        pass.end();
      }
      // Pass 2: spatial → current (label gi-spatial-2).
      {
        const bg = buildSpatialGiBindGroup(
          device, bglCache, spatial, current, resources.common.uboBuffer, 'spatial-gi-bg-2',
        );
        const pass = encoder.beginComputePass(computeDesc('gi-spatial-2'));
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
        pass.end();
      }
      return;
    }

    // 1-pass (medium/low): single current → spatial gather, then copy the
    // fused result back into `current` so `shade` (which reads current) sees
    // it. The copy avoids the in-place read/write hazard a single
    // current → current dispatch would introduce. Labelled `gi-spatial-2`
    // (the terminal label the layout + shade dependency expect).
    const bg = buildSpatialGiBindGroup(
      device, bglCache, current, spatial, resources.common.uboBuffer, 'spatial-gi-bg-1of1',
    );
    const pass = encoder.beginComputePass(computeDesc('gi-spatial-2'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(halfWgX, halfWgY, 1);
    pass.end();
    encoder.copyBufferToBuffer(spatial, 0, current, 0, current.size);
  }

  dispose(): void {}
}
