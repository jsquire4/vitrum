/**
 * SampleBudgetPass — Sprint 9 adaptive-sampling tier classifier.
 *
 * Reads the previous-frame Welford variance buffer and writes a per-pixel
 * tier texture (1 = converged, 2 = medium, 4 = high noise). The tier
 * output is consumed by `risGi` (gi_tier scales M_GI per pixel) and is
 * available for `ris.wgsl` (currently unused). The pass runs first in
 * the frame so its r32uint output is available downstream.
 *
 * Audit refs: M2 (thresholds host-overridable), Sprint 9 wire-in notes.
 */

import { buildSampleBudgetBindGroup, type UboRef } from '../bindGroupBuilders.js';
import type { Pass, PassDispatchContext, PassInitContext } from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

export class SampleBudgetPass implements Pass {
  readonly id = 'sample-budget' as const;
  readonly dependencies: readonly string[] = [];
  readonly passLabels: readonly PassLabel[] = ['sample-budget'];

  constructor(
    private readonly _pipeline: GPUComputePipeline,
    private readonly _budgetUboRef: UboRef,
    private readonly _sampleCountUboRef: UboRef,
  ) {}

  gates(): boolean {
    return true; // always-on standard pipeline pass since Sprint 9.
  }

  async initialize(_ctx: PassInitContext): Promise<void> {
    // UBOs are pipeline-owned and pre-allocated before construction.
  }

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, inputs, wgX, wgY, computeDesc, resources } = ctx;

    // Budget uniforms: f32 threshold_low, f32 threshold_high, u32 screenW, u32 screenH (16 bytes).
    const budgetBytes = new ArrayBuffer(16);
    const budgetF32 = new Float32Array(budgetBytes);
    const budgetU32 = new Uint32Array(budgetBytes);
    budgetF32[0] = inputs.adaptiveSamplingThresholdLow;
    budgetF32[1] = inputs.adaptiveSamplingThresholdHigh;
    budgetU32[2] = ctx.width;
    budgetU32[3] = ctx.height;
    device.queue.writeBuffer(this._budgetUboRef.buf!, 0, budgetBytes);
    // Sample count uniforms: u32 sampleCount + 3 pad u32 (16 bytes).
    device.queue.writeBuffer(
      this._sampleCountUboRef.buf!,
      0,
      new Uint32Array([Math.max(ctx.frameIndex + 1, 1), 0, 0, 0]),
    );
    const bg = buildSampleBudgetBindGroup(
      device,
      ctx.bglCache,
      resources.common.varianceBuffer.createView(),
      resources.common.tierTexture.createView(),
      this._budgetUboRef.buf!,
      this._sampleCountUboRef.buf!,
    );
    const pass = encoder.beginComputePass(computeDesc('sample-budget'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  dispose(): void {
    // UBO lifecycle is pipeline-owned (allocated in WalkaroundGPUPipeline.initialize).
  }
}
