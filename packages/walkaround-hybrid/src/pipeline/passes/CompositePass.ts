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
 *
 * Per-frame tonemap/exposure/outputColorSpace dials are written into a
 * 16-byte CompositeUniforms UBO (see `COMPOSITE_UBO` in uboLayouts.ts) and
 * forwarded via `inputs.composite`. Defaults (mode=0/aces, exposure=1.0,
 * colorSpace=0/srgb) preserve the historical behavior exactly.
 */

import { buildCompositeBindGroup, type UboRef } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { COMPOSITE_UBO } from './uboLayouts.js';
import type { PipelineFrameInputs } from '../pipelineFrameInputs.js';

export class CompositePass implements Pass {
  readonly id = 'composite' as const;
  readonly dependencies: readonly string[] = ['resolve'];
  readonly passLabels: readonly PassLabel[] = ['composite'];

  private _pipeline: GPURenderPipeline;
  private readonly _uboRef: UboRef;

  constructor(pipeline: GPURenderPipeline, uboRef: UboRef) {
    this._pipeline = pipeline;
    this._uboRef = uboRef;
  }

  /** Exposed so {@link WalkaroundGPUPipeline.presentLastFrame} can reuse
   *  the compiled composite render pipeline without holding its own copy. */
  get pipeline(): GPURenderPipeline {
    return this._pipeline;
  }

  /** Swap the format-specialized render pipeline transactionally after the
   *  host changes `FrameInput.swapChainFormat`. */
  setPipeline(pipeline: GPURenderPipeline): void {
    this._pipeline = pipeline;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  /**
   * Publish the exact presentation dials used by either a full frame or the
   * throttled composite-only path.
   */
  writeUniforms(
    device: GPUDevice,
    composite: PipelineFrameInputs['composite'],
  ): boolean {
    const uboBuffer = this._uboRef.buf;
    if (uboBuffer == null) return false;

    // Pack CompositeUniforms: tonemapMode (u32), exposure (f32),
    // outputColorSpace (u32), _pad (u32). Defaults: aces(0), 1.0, srgb(0).
    const compositeUboBytes = new ArrayBuffer(COMPOSITE_UBO.sizeBytes);
    COMPOSITE_UBO.pack(new DataView(compositeUboBytes), 0, {
      tonemapMode:      composite.tonemapMode,
      exposure:         composite.exposure,
      outputColorSpace: composite.outputColorSpace,
      _pad:             0,
    });
    device.queue.writeBuffer(uboBuffer, 0, compositeUboBytes);
    return true;
  }

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, bglCache, resources, inputs, renderTimestampWrites } = ctx;
    if (!this.writeUniforms(device, inputs.composite)) return;
    const uboBuffer = this._uboRef.buf;
    if (uboBuffer == null) return;

    const finalTex = resources.common.resolvedTexture;
    const bg = buildCompositeBindGroup(
      device, bglCache,
      finalTex.createView(),
      uboBuffer,
    );
    const tsComp = renderTimestampWrites('composite');
    const pass = encoder.beginRenderPass({
      label: 'composite',
      colorAttachments: [{
        view: inputs.screen.swapChainView,
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
