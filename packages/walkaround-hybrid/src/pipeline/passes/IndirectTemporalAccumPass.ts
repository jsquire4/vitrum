/**
 * IndirectTemporalAccumPass — Sprint 18 follow-up, pre-atrous TCBB-clipped
 * temporal accumulator on the raw indirect signal.
 *
 * Inserted between shade and the indirect 4-iter atrous chain. Smooths
 * each frame's reservoir-driven jitter (per-pixel chosen-sample changes)
 * *before* atrous so atrous's chromaticity edge-stop has a coherent
 * signal to converge on.
 *
 * Advances the orchestrator-owned `indirectAccumPingPong` index AFTER the
 * dispatch (matching the legacy ordering inside `renderFrame`).
 *
 * Writes the chosen output texture handle into `frameState.indirectAccumOut`
 * so the downstream `AtrousIndirectPass` reads the right ping-pong slot.
 */

import { buildIndirectTemporalAccumBindGroup } from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import type { PingPongRef } from './passRefs.js';

export class IndirectTemporalAccumPass implements Pass {
  readonly id = 'indirect-temporal-accum' as const;
  readonly dependencies: readonly string[] = ['shade', 'gtao-upsample'];
  readonly passLabels: readonly PassLabel[] = ['indirect-temporal-accum'];

  private readonly _pipeline: GPUComputePipeline;
  /** Orchestrator-owned ping-pong index (0 = ping out / pong prev). */
  private readonly _pingPongRef: PingPongRef;

  constructor(
    pipeline: GPUComputePipeline,
    pingPongRef: PingPongRef,
  ) {
    this._pipeline = pipeline;
    this._pingPongRef = pingPongRef;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, wgX16, wgY16, frameState } = ctx;
    const common = resources.common;
    const indirectAccumOut = this._pingPongRef.value === 0
      ? common.indirectAccumPingTexture
      : common.indirectAccumPongTexture;
    const indirectAccumPrev = this._pingPongRef.value === 0
      ? common.indirectAccumPongTexture
      : common.indirectAccumPingTexture;

    const bg = buildIndirectTemporalAccumBindGroup(
      device, bglCache,
      common.hdrIndirectTexture.createView(),
      indirectAccumPrev.createView(),
      indirectAccumOut.createView(),
    );
    const pass = encoder.beginComputePass(computeDesc('indirect-temporal-accum'));
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX16, wgY16, 1);
    pass.end();

    // Publish the output handle for the downstream atrous chain.
    frameState.indirectAccumOut = indirectAccumOut;
    // Flip the ping-pong AFTER the dispatch — mirrors the legacy ordering.
    this._pingPongRef.value = 1 - this._pingPongRef.value;
  }

  dispose(): void {}
}
