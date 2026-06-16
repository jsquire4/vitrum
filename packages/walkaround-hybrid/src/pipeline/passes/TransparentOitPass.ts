/**
 * TransparentOitPass — camera-visible alpha blend composition.
 *
 * Runs after `indirect-combine` has produced the opaque/background radiance and
 * before temporal accumulation. The shader walks fractional
 * `alphaMode:'blend'` layers front-to-back and writes the composited result
 * into `common.transparentCompositeTexture`.
 */

import { buildTransparentOitBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
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
    const {
      device,
      encoder,
      bglCache,
      resources,
      frameBindGroup,
      sceneBindGroup,
      uboBindGroup,
      frameState,
      computeDesc,
      wgX,
      wgY,
      resourceCache,
    } = ctx;
    const outTex = resources.common.transparentCompositeTexture;
    const buildBg = (): GPUBindGroup => buildTransparentOitBindGroup(
      device,
      bglCache,
      resourceCache?.textureView(frameState.combinedDenoised) ?? frameState.combinedDenoised.createView(),
      resourceCache?.textureView(outTex) ?? outTex.createView(),
    );
    const oitBg = resourceCache?.bindGroup('pass:transparent-oit', [
      frameState.combinedDenoised,
      outTex,
    ], buildBg) ?? buildBg();

    const pass = encoder.beginComputePass(computeDesc('transparent-oit'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, frameBindGroup);
    pass.setBindGroup(1, sceneBindGroup);
    pass.setBindGroup(2, uboBindGroup);
    pass.setBindGroup(3, oitBg);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();

    frameState.combinedDenoised = outTex;
  }

  dispose(): void {}
}
