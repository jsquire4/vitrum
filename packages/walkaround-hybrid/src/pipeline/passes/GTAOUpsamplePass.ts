/**
 * GTAOUpsamplePass — Sprint 15 bilateral upsample half-res AO → full-res.
 *
 * Reads aoHalfTexture (gtao output) and gNormalDepthTexture (depth-aware
 * bilateral filtering), writes aoFullTexture which is sampled by shade in
 * the *next* frame (1-frame AO lag, invisible for static cameras).
 */

import { buildGTAOUpsampleBindGroup } from '../bindGroupBuilders.js';
import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class GTAOUpsamplePass implements Pass {
  readonly id = 'gtao-upsample' as const;
  readonly dependencies: readonly string[] = ['gtao'];
  readonly passLabels: readonly PassLabel[] = ['gtao-upsample'];

  constructor(private readonly _pipeline: GPUComputePipeline) {}

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, wgX, wgY } = ctx;
    const bg = buildGTAOUpsampleBindGroup(
      device,
      bglCache,
      resources.gtao.aoHalfTexture.createView(),
      resources.common.gNormalDepthTexture.createView(),
      resources.gtao.aoFullTexture.createView(),
      resources.gtao.gtaoUboBuffer,
    );
    const pass = encoder.beginComputePass(computeDesc('gtao-upsample'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {}
}
