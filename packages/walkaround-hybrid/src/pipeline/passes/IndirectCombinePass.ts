/**
 * IndirectCombinePass — Sprint 18 per-channel direct+indirect combiner.
 *
 * Reads:
 *   - `frameState.denoisedDirect`  (set by the active denoiser dispatch)
 *   - `frameState.denoisedIndirect` (set by `AtrousIndirectPass`)
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
import { dispatchSingleBindGroup } from './dispatchHelpers.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';

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
    const { device, bglCache, resources, frameState, resourceCache } = ctx;
    const combinedTex = resources.common.combinedDenoisedTexture;
    // W5-I2 (2026-05-18): gNormalDepthView removed from this call — the
    // indirect-combine shader never read it (declared "for BGL compat,
    // unused"). The PassDispatchContext field stays for the other passes
    // that genuinely use it.
    const buildBg = (): GPUBindGroup => buildIndirectCombineBindGroup(
      device, bglCache,
      resourceCache?.textureView(frameState.denoisedDirect) ?? frameState.denoisedDirect.createView(),
      resourceCache?.textureView(frameState.denoisedIndirect) ?? frameState.denoisedIndirect.createView(),
      resourceCache?.textureView(combinedTex) ?? combinedTex.createView(),
      // Item 24 — re-modulate denoised indirect by albedo (Schied 2017 §4.1).
      resourceCache?.textureView(resources.common.albedoTexture) ?? resources.common.albedoTexture.createView(),
    );
    const bg = cachedBindGroup(resourceCache, 'pass:indirect-combine', [
      frameState.denoisedDirect,
      frameState.denoisedIndirect,
      combinedTex,
      resources.common.albedoTexture,
    ], buildBg);
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'indirect-combine', { wg16: true });
    frameState.combinedDenoised = combinedTex;
  }

  dispose(): void {}
}
