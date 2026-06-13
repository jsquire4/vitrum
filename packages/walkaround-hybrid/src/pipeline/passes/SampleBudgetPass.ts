/**
 * SampleBudgetPass — Sprint 9 adaptive-sampling tier classifier.
 *
 * Reads the previous-frame Welford variance buffer and writes a per-pixel
 * tier texture (1 = converged, 2 = medium, 4 = high noise). The tier
 * output is consumed by `risGi` (gi_tier scales M_GI per pixel) and is
 * available for `ris.wgsl`. The pass runs first in
 * the frame so its r32uint output is available downstream.
 *
 * Audit refs: M2 (thresholds host-overridable), Sprint 9 wire-in notes.
 */

import {
  buildSampleBudgetBindGroup,
  type UboRef,
} from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';
import { SAMPLE_BUDGET_UBO, SAMPLE_COUNT_UBO } from './uboLayouts.js';

export class SampleBudgetPass implements Pass {
  readonly id = 'sample-budget' as const;
  readonly dependencies: readonly string[] = [];
  readonly passLabels: readonly PassLabel[] = ['sample-budget'];

  private readonly _pipeline: GPUComputePipeline;
  private readonly _budgetUboRef: UboRef;
  private readonly _sampleCountUboRef: UboRef;

  constructor(
    pipeline: GPUComputePipeline,
    budgetUboRef: UboRef,
    sampleCountUboRef: UboRef,
  ) {
    this._pipeline = pipeline;
    this._budgetUboRef = budgetUboRef;
    this._sampleCountUboRef = sampleCountUboRef;
  }

  gates(): boolean {
    return true; // always-on standard pipeline pass since Sprint 9.
  }

  async initialize(_ctx: PassInitContext): Promise<void> {
    // UBOs are pipeline-owned and pre-allocated before construction.
  }

  dispatch(ctx: PassDispatchContext): void {
    const { device, inputs, resources } = ctx;

    // Budget uniforms: f32 threshold_low, f32 threshold_high, u32 screenW, u32 screenH (16 bytes).
    // W2-C13 follow-up: identical std140 layout to the prior Float32Array/
    // Uint32Array-aliased write (f32 @0/4, u32 @8/12).
    const budgetBytes = new ArrayBuffer(SAMPLE_BUDGET_UBO.sizeBytes);
    SAMPLE_BUDGET_UBO.pack(new DataView(budgetBytes), 0, {
      thresholdLow:  inputs.gtao.adaptiveSamplingThresholdLow,
      thresholdHigh: inputs.gtao.adaptiveSamplingThresholdHigh,
      screenW:       ctx.width,
      screenH:       ctx.height,
    });
    device.queue.writeBuffer(this._budgetUboRef.buf!, 0, budgetBytes);
    // Sample count uniforms: u32 sampleCount + 3 pad u32 (16 bytes). defineUbo
    // zero-fills the trailing pad to match the prior [count, 0, 0, 0] write.
    const sampleCountBytes = new ArrayBuffer(SAMPLE_COUNT_UBO.sizeBytes);
    SAMPLE_COUNT_UBO.pack(new DataView(sampleCountBytes), 0, {
      sampleCount: Math.max(ctx.frameIndex + 1, 1),
    });
    device.queue.writeBuffer(this._sampleCountUboRef.buf!, 0, sampleCountBytes);
    // Select the freshest Welford variance side.  AtrousVarianceDenoiser
    // ping-pongs between varianceBuffer (ping=0) and varianceBufferAux (ping=1):
    //   welfordPing === 0 → frame N-1 wrote varianceBuffer  → read varianceBuffer
    //   welfordPing === 1 → frame N-1 wrote varianceBufferAux → read varianceBufferAux
    // For denoisers that do not ping-pong, welfordPing is always 0, so
    // varianceBuffer is used (the historical default, unchanged behaviour).
    const varianceTex = ctx.welfordPing === 0
      ? resources.common.varianceBuffer
      : resources.common.varianceBufferAux;
    const buildBg = (): GPUBindGroup => buildSampleBudgetBindGroup(
      device, ctx.bglCache,
      ctx.resourceCache?.textureView(varianceTex) ?? varianceTex.createView(),
      ctx.resourceCache?.textureView(resources.common.tierTexture) ?? resources.common.tierTexture.createView(),
      this._budgetUboRef.buf!,
      this._sampleCountUboRef.buf!,
    );
    const bg = ctx.resourceCache?.bindGroup('pass:sample-budget', [
      varianceTex,
      resources.common.tierTexture,
      this._budgetUboRef.buf,
      this._sampleCountUboRef.buf,
    ], buildBg) ?? buildBg();
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'sample-budget');
  }

  dispose(): void {
    // UBO lifecycle is pipeline-owned (allocated in WalkaroundGPUPipeline.initialize).
  }
}
