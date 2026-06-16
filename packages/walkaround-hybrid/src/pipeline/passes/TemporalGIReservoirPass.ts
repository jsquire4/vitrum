/**
 * TemporalGIReservoirPass — Sprint 17 GI temporal-reuse pass.
 *
 * Half-resolution (W/2 × H/2) dispatch. group(0) is the dedicated GI reservoir
 * group (current GI reservoir, previous GI reservoir, ubo).
 *
 * group(1) is the shared scene BVH/TLAS/material-atlas group. Both default and
 * GRIS variants bind it now: default temporal reuse recasts the current receiver
 * material to evaluate the same rich receiver-lobe p-hat the RIS producer uses,
 * while GRIS additionally uses the group for reconnection visibility.
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

  constructor(pipeline: GPUComputePipeline, _grisEnabled = false) {
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
    const bg = resourceCache?.bindGroup('pass:gi-temporal', [
      resources.restirGI.reservoirGiCurrentBuffer,
      resources.restirGI.reservoirGiPreviousBuffer,
      resources.common.uboBuffer,
    ], buildBg) ?? buildBg();
    // group(1) — shared scene BVH/TLAS/material atlas. Half-res dispatch.
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'gi-temporal', {
      half: true,
      extraGroups: [{ slot: 1, group: sceneBindGroup }],
    });
  }

  dispose(): void {}
}
