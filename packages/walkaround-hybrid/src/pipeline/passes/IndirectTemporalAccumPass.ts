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
import { dispatchSingleBindGroup } from './dispatchHelpers.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';
import type { PingPongRef } from './passRefs.js';
import { publishFrameState } from '../FramePublication.js';

export class IndirectTemporalAccumPass implements Pass {
  readonly id = 'indirect-temporal-accum' as const;
  /**
   * `denoiser-adapter` is listed here (alongside `shade` + `gtao-upsample`)
   * so the virtual {@link DenoiserAdapterPass} always dispatches before
   * this pass in the topological order. There is no shared GPU resource
   * between the two — the dependency is purely to preserve the dispatch
   * ordering that pre-W1-R5b's manual two-half loop hard-coded, so the
   * GPU timestamp-query slots still line up with the historic positions
   * emitted by `composePassLabels`. Without it, the topo-sort tiebreaker
   * (lexicographic) happens to produce the same order, but that is too
   * fragile a thing to rely on.
   */
  readonly dependencies: readonly string[] = ['shade', 'gtao-upsample', 'denoiser-adapter'];
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
    const { device, bglCache, resources, frameState, resourceCache } = ctx;
    const common = resources.common;
    const indirectAccumOut = this._pingPongRef.value === 0
      ? common.indirectAccumPingTexture
      : common.indirectAccumPongTexture;
    const indirectAccumPrev = this._pingPongRef.value === 0
      ? common.indirectAccumPongTexture
      : common.indirectAccumPingTexture;

    const buildBg = (): GPUBindGroup => buildIndirectTemporalAccumBindGroup(
      device, bglCache,
      resourceCache?.textureView(common.hdrIndirectTexture) ?? common.hdrIndirectTexture.createView(),
      resourceCache?.textureView(indirectAccumPrev) ?? indirectAccumPrev.createView(),
      resourceCache?.textureView(indirectAccumOut) ?? indirectAccumOut.createView(),
    );
    const bg = cachedBindGroup(resourceCache, 'pass:indirect-temporal-accum', [
      common.hdrIndirectTexture,
      indirectAccumPrev,
      indirectAccumOut,
    ], buildBg);
    dispatchSingleBindGroup(ctx, this._pipeline, bg, 'indirect-temporal-accum', { wg16: true });

    // Publish the output handle for the downstream atrous chain.
    frameState.indirectAccumOut = indirectAccumOut;
    // The encoded target is valid history only after the frame submit is
    // accepted. Standalone pass harnesses (no publication transaction) retain
    // the historical immediate flip.
    const nextPingPong = 1 - this._pingPongRef.value;
    publishFrameState(ctx.publication, () => {
      this._pingPongRef.value = nextPingPong;
    });
  }

  dispose(): void {}
}
