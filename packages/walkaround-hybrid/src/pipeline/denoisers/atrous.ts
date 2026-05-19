/**
 * AtrousDenoiser — legacy 3-iteration à-trous spatial filter.
 *
 * Mirrors the historical `_dispatchAtrousLegacy` behaviour bit-for-bit:
 * three à-trous iterations with step width `1 << iter` (1, 2, 4) on
 * `hdrColorTexture`, ping-ponged between `denoisedPing`/`denoisedPong`.
 *
 * Implementation notes:
 *
 * - The à-trous compute pipeline + its BGL stay in the shared compiler
 *   (`pipelineCompiler.ts` + `bindGroupLayouts.ts::getAtrousBindGroupLayout`)
 *   because the always-on indirect-channel chain
 *   (`WalkaroundGPUPipeline._dispatchAtrousIndirect`) ALSO dispatches
 *   the same compiled pipeline. Forking a denoiser-private copy would
 *   double-compile the identical shader. The shared pipeline is handed
 *   to {@link dispatch} via `DenoiserDispatchContext.sharedAtrousPipeline`.
 *
 * - The per-iter `stepWidth` UBO is owned here; the shared
 *   `buildAtrousBindGroup` helper packs it into a 16-byte buffer behind
 *   a lazy {@link UboRef} so first-frame allocation is amortised.
 */

import { buildAtrousBindGroup, type UboRef } from '../bindGroupBuilders.js';
import {
  DENOISER_PASS_LABELS,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
} from './index.js';

/** Number of à-trous iterations. Matches the legacy hard-coded value. */
const ATROUS_ITERATIONS = 3;

export class AtrousDenoiser implements Denoiser {
  readonly id = 'atrous' as const;
  readonly passLabels = DENOISER_PASS_LABELS['atrous'];

  /** Lazy 16-byte UBO holding `(stepWidth, sigmaN, sigmaZ, sigmaC)`. */
  private readonly _uboRef: UboRef = { buf: undefined };

  async initialize(_ctx: DenoiserInitContext): Promise<void> {
    // No persistent GPU resources to allocate eagerly; the per-iter
    // UBO is lazy-allocated on first dispatch via `_uboRef`.
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture {
    const {
      device,
      encoder,
      resources,
      bglCache,
      gNormalDepthView,
      atrousDirectSigmas,
      wgX16,
      wgY16,
      computeDesc,
      sharedAtrousPipeline,
    } = ctx;
    const common = resources.common;

    // B3a — per-frame direct sigmas from HybridEngineOptions (host
    // override) or Cornell defaults `[128, 5, 0.05]`.
    const sigmas = {
      sigmaN: atrousDirectSigmas[0],
      sigmaZ: atrousDirectSigmas[1],
      sigmaC: atrousDirectSigmas[2],
    };

    let inputTex: GPUTexture = common.hdrColorTexture;
    for (let iter = 0; iter < ATROUS_ITERATIONS; iter++) {
      const stepWidth = 1 << iter;
      const outputTex =
        iter % 2 === 0 ? common.denoisedPingTexture : common.denoisedPongTexture;
      const bgAtrous = buildAtrousBindGroup(
        device, bglCache, this._uboRef,
        inputTex.createView(), outputTex.createView(),
        gNormalDepthView, gNormalDepthView, stepWidth,
        sigmas,
      );
      const label = `atrous-${iter}` as `atrous-${0 | 1 | 2}`;
      const pass = encoder.beginComputePass(computeDesc(label));
      pass.setPipeline(sharedAtrousPipeline);
      pass.setBindGroup(0, bgAtrous);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outputTex;
    }
    return inputTex;
  }

  resize(_w: number, _h: number): void {
    // No persistent GPU resources to reallocate; the per-iter ping-pong
    // textures are owned by FrameResources and resized by the pipeline.
  }

  dispose(): void {
    this._uboRef.buf?.destroy();
    this._uboRef.buf = undefined;
  }
}
