/**
 * TemporalAccumPass — EMA blend of the combined per-channel denoised
 * radiance against the previous-frame accumulator.
 *
 * Reads `frameState.combinedDenoised` (from transparent-oit), reads the
 * previous-frame `readAccum` (from frameState — set by the orchestrator
 * before the loop), writes the next-frame accumulator. Alpha = 1.0 on the
 * first frame (history discarded) and `_temporalAccumAlpha` thereafter
 * (resolved by the orchestrator into `frameState.alpha`).
 *
 * Note: the ping-pong index advance happens in the orchestrator AFTER the
 * pass loop completes — `_accumPingPongIndex` is also referenced before
 * the denoiser dispatch (the orchestrator passes `readAccum` into
 * the denoiser context), so the swap has to bracket the entire frame's
 * downstream-of-shade work to preserve dataflow.
 */

import {
  buildPreparedAccumBindGroup,
  writeAccumUbo,
  type UboRef,
} from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';
import { dispatchSingleBindGroup } from './dispatchHelpers.js';

export class TemporalAccumPass implements Pass {
  readonly id = 'temporalAccum' as const;
  readonly dependencies: readonly string[] = ['transparent-oit'];
  readonly passLabels: readonly PassLabel[] = ['temporalAccum'];

  private readonly _pipeline: GPUComputePipeline;
  private readonly _uboRef: UboRef;

  constructor(
    pipeline: GPUComputePipeline,
    uboRef: UboRef,
  ) {
    this._pipeline = pipeline;
    this._uboRef = uboRef;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, bglCache, frameState, resourceCache } = ctx;
    const ubo = writeAccumUbo(device, this._uboRef, frameState.alpha);
    const buildBg = (): GPUBindGroup => buildPreparedAccumBindGroup(
      device, bglCache, ubo,
      resourceCache?.textureView(frameState.combinedDenoised) ?? frameState.combinedDenoised.createView(),
      resourceCache?.textureView(frameState.readAccum) ?? frameState.readAccum.createView(),
      resourceCache?.textureView(frameState.writeAccum) ?? frameState.writeAccum.createView(),
    );
    const bg = resourceCache?.bindGroup('pass:temporal-accum', [
      this._uboRef,
      frameState.combinedDenoised,
      frameState.readAccum,
      frameState.writeAccum,
    ], buildBg) ?? buildBg();
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'temporalAccum', { wg16: true });
  }

  dispose(): void {}
}
