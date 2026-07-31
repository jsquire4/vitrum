/**
 * TemporalGIReservoirPass — Sprint 17 GI temporal-reuse pass.
 *
 * Half-resolution (W/2 × H/2) dispatch. group(0) is the dedicated GI reservoir
 * group (current GI reservoir, previous GI reservoir, ubo).
 *
 * group(1) is the shared scene BVH/TLAS/material-atlas group. The canonical
 * generalized-reuse pass binds it for reconnection visibility and mapped
 * environment/alpha support evaluation.
 */

import { buildTemporalGiBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';

export class TemporalGIReservoirPass implements Pass {
  readonly id = 'gi-temporal' as const;
  readonly dependencies: readonly string[];
  readonly passLabels: readonly PassLabel[] = ['gi-temporal'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(
    pipeline: GPUComputePipeline,
    ppgTrainingBeforeReuse = false,
  ) {
    this.dependencies = ppgTrainingBeforeReuse ? ['ppg-update'] : ['gi-ris'];
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, bglCache, resources, sceneBindGroup, resourceCache } = ctx;
    const buildBg = (): GPUBindGroup => buildTemporalGiBindGroup(
      device, bglCache,
      resources.restirGI.reservoirGiCurrentBuffer,
      resources.restirGI.reservoirGiPreviousBuffer,
      resources.common.uboBuffer,
    );
    const bg = cachedBindGroup(resourceCache, 'pass:gi-temporal', [
      resources.restirGI.reservoirGiCurrentBuffer,
      resources.restirGI.reservoirGiPreviousBuffer,
      resources.common.uboBuffer,
    ], buildBg);
    // group(1) — shared scene BVH/TLAS/material atlas. Half-res dispatch.
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'gi-temporal', {
      dispatchOverride: {
        wgX: ctx.restirGiWgX,
        wgY: ctx.restirGiWgY,
      },
      extraGroups: [{ slot: 1, group: sceneBindGroup }],
    });
  }

  dispose(): void {}
}
