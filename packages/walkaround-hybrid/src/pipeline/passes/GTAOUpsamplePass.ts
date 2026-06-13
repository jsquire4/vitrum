/**
 * GTAOUpsamplePass — Sprint 15 bilateral upsample half-res AO → full-res.
 *
 * Reads aoHalfTexture (gtao output) and gNormalDepthTexture (depth-aware
 * bilateral filtering), writes aoFullTexture which is sampled by shade in
 * the *next* frame (1-frame AO lag, invisible for static cameras).
 */

import { buildGTAOUpsampleBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassGateOptions,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';

export class GTAOUpsamplePass implements Pass {
  readonly id = 'gtao-upsample' as const;
  readonly dependencies: readonly string[] = ['gtao'];
  readonly passLabels: readonly PassLabel[] = ['gtao-upsample'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  /** Phase-0 — gate off in lockstep with {@link GTAOPass}. When GTAO is off,
   *  `aoFullTexture` keeps its init value (1.0 = no occlusion), so shade reads
   *  a correct "AO disabled" signal without this pass running. */
  gates(opts: PassGateOptions): boolean {
    return opts.gtaoEnabled !== false;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, bglCache, resources, resourceCache } = ctx;
    const buildBg = (): GPUBindGroup => buildGTAOUpsampleBindGroup(
      device, bglCache,
      resourceCache?.textureView(resources.gtao.aoHalfTexture) ?? resources.gtao.aoHalfTexture.createView(),
      resourceCache?.textureView(resources.common.gNormalDepthTexture) ?? resources.common.gNormalDepthTexture.createView(),
      resourceCache?.textureView(resources.gtao.aoFullTexture) ?? resources.gtao.aoFullTexture.createView(),
      resources.gtao.gtaoUboBuffer,
    );
    const bg = resourceCache?.bindGroup('pass:gtao-upsample', [
      resources.gtao.aoHalfTexture,
      resources.common.gNormalDepthTexture,
      resources.gtao.aoFullTexture,
      resources.gtao.gtaoUboBuffer,
    ], buildBg) ?? buildBg();
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'gtao-upsample');
  }

  dispose(): void {}
}
