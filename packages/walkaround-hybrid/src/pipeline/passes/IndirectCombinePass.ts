/**
 * IndirectCombinePass — Sprint 18 per-channel direct+indirect combiner.
 *
 * Reads:
 *   - `frameState.denoisedDirect`  (set by the active denoiser dispatch)
 *   - `frameState.denoisedIndirect` (set by `AtrousIndirectPass`)
 *   - `gNormalDepthView` (edge-stop)
 *   - `albedoTexture` (Item 24 / Schied 2017 §4.1 re-modulation)
 *
 * Writes `common.combinedDenoisedTexture` which `TemporalAccumPass` reads
 * in place of the raw direct-denoiser output. Publishes the texture handle
 * via `frameState.combinedDenoised`.
 */

import { buildIndirectCombineBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class IndirectCombinePass implements Pass {
  readonly id = 'indirect-combine' as const;
  readonly dependencies: readonly string[] = ['atrous-indirect-3'];
  readonly passLabels: readonly PassLabel[] = ['indirect-combine'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, wgX16, wgY16, gNormalDepthView, frameState } = ctx;
    const combinedTex = resources.common.combinedDenoisedTexture;
    const bg = buildIndirectCombineBindGroup(
      device, bglCache,
      frameState.denoisedDirect.createView(),
      frameState.denoisedIndirect.createView(),
      gNormalDepthView,
      combinedTex.createView(),
      // Item 24 — re-modulate denoised indirect by albedo (Schied 2017 §4.1).
      resources.common.albedoTexture.createView(),
    );
    const pass = encoder.beginComputePass(computeDesc('indirect-combine'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX16, wgY16, 1);
    pass.end();
    frameState.combinedDenoised = combinedTex;
  }

  dispose(): void {}
}
