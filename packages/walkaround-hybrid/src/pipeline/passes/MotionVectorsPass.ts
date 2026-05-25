import {
  buildMotionVectorsBindGroup,
} from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

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
    const {
      device,
      encoder,
      bglCache,
      resources,
      computeDesc,
      wgX,
      wgY,
    } = ctx;
    const bg = buildMotionVectorsBindGroup(
      device,
      bglCache,
      resources.common.gNormalDepthTexture.createView(),
      resources.common.motionVectorTexture.createView(),
      resources.common.uboBuffer,
    );
    const pass = encoder.beginComputePass(computeDesc('motion-vectors'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {}
}

