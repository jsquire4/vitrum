import {
  buildMotionVectorsBindGroup,
} from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';

export class MotionVectorsPass implements Pass {
  readonly id = 'motion-vectors' as const;
  readonly dependencies: readonly string[] = ['shade'];
  readonly passLabels: readonly PassLabel[] = ['motion-vectors'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, bglCache, resources } = ctx;
    const bg = buildMotionVectorsBindGroup(
      device,
      bglCache,
      resources.common.gNormalDepthTexture.createView(),
      resources.common.motionVectorTexture.createView(),
      resources.common.uboBuffer,
    );
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'motion-vectors');
  }

  dispose(): void {}
}

