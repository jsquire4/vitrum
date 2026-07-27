/**
 * TransparentOitPass — camera-visible alpha blend composition.
 *
 * Runs after `indirect-combine` has produced the opaque/background radiance and
 * before temporal accumulation. The shader walks fractional
 * `alphaMode:'blend'` layers front-to-back and writes the composited result
 * into `common.transparentCompositeTexture`.
 */

import {
  dispatchSharedBindGroupPass,
  type Pass,
  type PassDispatchContext,
  type PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class TransparentOitPass implements Pass {
  readonly id = 'transparent-oit' as const;
  readonly dependencies: readonly string[] = ['indirect-combine'];
  readonly passLabels: readonly PassLabel[] = ['transparent-oit'];

  private readonly _pipeline: GPUComputePipeline;

  constructor(pipeline: GPUComputePipeline) {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { resources, frameState } = ctx;
    const outTex = resources.common.transparentCompositeTexture;
    const oitBg = ctx.buildTransparentOitBindGroup(frameState.combinedDenoised, outTex);

    dispatchSharedBindGroupPass(ctx, this._pipeline, {
      label: 'transparent-oit',
      // The OIT front-to-back layer walk binds its own per-frame composition
      // group (source radiance + output) at slot 3; full-res dispatch.
      extraGroups: [{ slot: 3, group: oitBg }],
    });

    frameState.combinedDenoised = outTex;
  }

  dispose(): void {}
}
