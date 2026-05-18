/**
 * ShadePass — re-traces primary ray, evaluates ReSTIR, computes DI + indirect bounce.
 *
 * Reads the final DI reservoir (post-spatial-2) and the final GI reservoir
 * (post-gi-spatial-2). Writes hdrColorTexture (direct), hdrIndirectTexture
 * (indirect), hdrTotalTexture (direct+indirect), gNormalDepthTexture, and
 * the albedoTexture for downstream demodulation.
 */

import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class ShadePass implements Pass {
  readonly id = 'shade' as const;
  readonly dependencies: readonly string[] = ['spatial-2', 'gi-spatial-2'];
  readonly passLabels: readonly PassLabel[] = ['shade'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const {
      encoder,
      computeDesc,
      frameBindGroup,
      sceneBindGroup,
      uboBindGroup,
      hybridLayersBindGroup,
      wgX,
      wgY,
    } = ctx;
    const pass = encoder.beginComputePass(computeDesc('shade'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.setBindGroup(3, hybridLayersBindGroup);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {}
}
