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
 * The canonical generalized-reuse shader binds the shared scene/material group
 * at `@group(1)` to recast receiver material payloads and trace reconnection
 * visibility.
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
import { dispatchSingleBindGroup } from './dispatchHelpers.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';

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
    const { device, bglCache, resources, sceneBindGroup, resourceCache } = ctx;
    const current = resources.restirGI.reservoirGiCurrentBuffer;
    const spatial = resources.restirGI.reservoirGiSpatialBuffer;

    // group(1) — shared scene BVH/TLAS/material atlas. Required by both the
    // default receiver-lobe target and the GRIS reconnection-visibility path.
    const giExtra = { extraGroups: [{ slot: 1, group: sceneBindGroup }] as const };

    // The 7-line dispatch block repeated ×3 across the 2-pass + 1-pass
    // branches, extracted to a local closure. Half-res dispatch.
    const dispatchGiSpatial = (
      inBuf: GPUBuffer,
      outBuf: GPUBuffer,
      label: PassLabel,
      bgLabel: string,
    ): void => {
      const buildBg = (): GPUBindGroup => buildSpatialGiBindGroup(
        device, bglCache, inBuf, outBuf, resources.common.uboBuffer, bgLabel,
      );
      const bg = cachedBindGroup(resourceCache, `pass:gi-spatial:${bgLabel}`, [
        inBuf,
        outBuf,
        resources.common.uboBuffer,
      ], buildBg);
      dispatchSingleBindGroup(ctx, this._pipeline, bg, label, {
        dispatchOverride: {
          wgX: ctx.restirGiWgX,
          wgY: ctx.restirGiWgY,
        },
        ...giExtra,
      });
    };

    if (this._passCount === 2) {
      // Pass 1: current → spatial (label gi-spatial-1).
      dispatchGiSpatial(current, spatial, 'gi-spatial-1', 'spatial-gi-bg-1');
      // Pass 2: spatial → current (label gi-spatial-2).
      dispatchGiSpatial(spatial, current, 'gi-spatial-2', 'spatial-gi-bg-2');
      return;
    }

    // 1-pass (medium/low): single current → spatial gather, then copy the
    // fused result back into `current` so `shade` (which reads current) sees
    // it. The copy avoids the in-place read/write hazard a single
    // current → current dispatch would introduce. Labelled `gi-spatial-2`
    // (the terminal label the layout + shade dependency expect).
    dispatchGiSpatial(current, spatial, 'gi-spatial-2', 'spatial-gi-bg-1of1');
    ctx.encoder.copyBufferToBuffer(spatial, 0, current, 0, current.size);
  }

  dispose(): void {}
}
