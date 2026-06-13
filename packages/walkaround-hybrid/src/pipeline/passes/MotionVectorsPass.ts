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

/**
 * MotionVectorsPass — computes per-pixel screen-space motion vectors for
 * temporal reprojection (SVGF, BMFR, neural denoiser).
 *
 * Reads `gNormalDepth` (the G-buffer normal + linear depth packed into
 * `rgba32float`) and the current/previous view–projection matrices from the
 * WalkaroundUBO, then writes `motionVector` (`rgba32float`, full-res) — the
 * fractional-pixel displacement between the current and previous frame for
 * each visible surface point.
 */
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
    const { device, bglCache, resources, resourceCache } = ctx;
    const buildBg = (): GPUBindGroup => buildMotionVectorsBindGroup(
      device,
      bglCache,
      resourceCache?.textureView(resources.common.gNormalDepthTexture) ?? resources.common.gNormalDepthTexture.createView(),
      resourceCache?.textureView(resources.common.motionVectorTexture) ?? resources.common.motionVectorTexture.createView(),
      resources.common.uboBuffer,
    );
    const bg = resourceCache?.bindGroup('pass:motion-vectors', [
      resources.common.gNormalDepthTexture,
      resources.common.motionVectorTexture,
      resources.common.uboBuffer,
    ], buildBg) ?? buildBg();
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'motion-vectors');
  }

  dispose(): void {}
}
