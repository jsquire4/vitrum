/**
 * CompositePass — final render pass blitting resolvedTexture to the
 * host-supplied swap-chain view.
 *
 * **Render-pass shaped under the same Pass interface.** Begins a
 * `GPURenderPassEncoder` instead of a compute pass, and reads its
 * timestamp-writes struct via the `renderTimestampWrites` helper on
 * {@link PassDispatchContext}. The abstraction is encoder-agnostic — this
 * pass and every compute pass implement the same interface so the
 * orchestrator iterates them uniformly.
 */

import { buildCompositeBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class CompositePass implements Pass {
  readonly id = 'composite' as const;
  readonly dependencies: readonly string[] = ['resolve'];
  readonly passLabels: readonly PassLabel[] = ['composite'];

  constructor(private readonly _pipeline: GPURenderPipeline) {}

  /** Exposed so {@link WalkaroundGPUPipeline.presentLastFrame} can reuse
   *  the compiled composite render pipeline without holding its own copy. */
  get pipeline(): GPURenderPipeline {
    return this._pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, bglCache, resources, inputs, renderTimestampWrites } = ctx;
    const finalTex = resources.common.resolvedTexture;
    const bg = buildCompositeBindGroup(
      device, bglCache,
      finalTex.createView(),
      resources.common.compositeSampler,
    );
    const tsComp = renderTimestampWrites('composite');
    const pass = encoder.beginRenderPass({
      label: 'composite',
      colorAttachments: [{
        view: inputs.swapChainView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      ...(tsComp ? { timestampWrites: tsComp } : {}),
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3, 1, 0, 0); // fullscreen triangle
    pass.end();
  }

  dispose(): void {}
}
