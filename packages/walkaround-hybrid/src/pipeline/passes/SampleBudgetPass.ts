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

import { defineUbo } from '@vitrum/shared-samplers';
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

// W2-C13 follow-up — SampleBudgetUniforms (2×f32 + 2×u32 = 16 B):
// adaptive-sampling thresholds + screen extent for tier classification.
const SAMPLE_BUDGET_UBO = defineUbo([
  { name: 'thresholdLow',  type: 'f32' },
  { name: 'thresholdHigh', type: 'f32' },
  { name: 'screenW',       type: 'u32' },
  { name: 'screenH',       type: 'u32' },
] as const);
// SampleCountUniforms (1×u32 + 3 trailing pad = 16 B floor): per-frame
// 1-based sample count. defineUbo zero-fills bytes 4..15.
const SAMPLE_COUNT_UBO = defineUbo([
  { name: 'sampleCount', type: 'u32' },
] as const);

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
    const { device, encoder, inputs, wgX, wgY, computeDesc, resources } = ctx;

    // Budget uniforms: f32 threshold_low, f32 threshold_high, u32 screenW, u32 screenH (16 bytes).
    // W2-C13 follow-up: identical std140 layout to the prior Float32Array/
    // Uint32Array-aliased write (f32 @0/4, u32 @8/12).
    const budgetBytes = new ArrayBuffer(SAMPLE_BUDGET_UBO.sizeBytes);
    SAMPLE_BUDGET_UBO.pack(new DataView(budgetBytes), 0, {
      thresholdLow:  inputs.adaptiveSamplingThresholdLow,
      thresholdHigh: inputs.adaptiveSamplingThresholdHigh,
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
    const bg = buildSampleBudgetBindGroup(
      device, ctx.bglCache,
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
