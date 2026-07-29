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
 *   `buildAtrousBindGroup` helper packs each encoded dispatch into a distinct
 *   aligned range behind a lazy {@link UboRef} so first-frame allocation is
 *   amortised without queued writes racing to the final iteration's value.
 */

import {
  ATROUS_UBO_BINDING_STRIDE_BYTES,
  buildPreparedAtrousBindGroup,
  type UboRef,
  writeAtrousUbo,
} from '../bindGroupBuilders.js';
import { runAtrousChain } from '../passes/dispatchHelpers.js';
import { cachedBindGroup } from '../PipelineResourceCache.js';
import type { PassLabel } from '../timestampQueries.js';
import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
  type DenoiserState,
} from './index.js';

/** Number of à-trous iterations. Matches the legacy hard-coded value. */
const ATROUS_ITERATIONS = 3;

export class AtrousDenoiser implements Denoiser {
  readonly id = 'atrous' as const;
  readonly passLabels = DENOISER_PASS_LABELS['atrous'];

  /** Lazy UBO slab holding one aligned `(stepWidth, sigmaN, sigmaZ, sigmaC)` range per iteration. */
  private readonly _uboRef: UboRef = { buf: undefined };

  async initialize(_ctx: DenoiserInitContext): Promise<void> {
    // No persistent GPU resources to allocate eagerly; the per-iter
    // UBO is lazy-allocated on first dispatch via `_uboRef`.
  }

  state(): DenoiserState {
    return DENOISER_READY_STATE;
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
      resourceCache,
    } = ctx;
    const common = resources.common;

    // B3a — per-frame direct sigmas from HybridEngineOptions (host
    // override) or Cornell defaults `[128, 5, 0.05]`.
    const sigmas = {
      sigmaN: atrousDirectSigmas[0],
      sigmaZ: atrousDirectSigmas[1],
      sigmaC: atrousDirectSigmas[2],
    };

    return runAtrousChain(encoder, sharedAtrousPipeline, {
      iterations: ATROUS_ITERATIONS,
      startTex: common.hdrColorTexture,
      pingTex: common.denoisedPingTexture,
      pongTex: common.denoisedPongTexture,
      wgX: wgX16,
      wgY: wgY16,
      computeDesc,
      ...(resourceCache ? { textureViewFor: (texture: GPUTexture) => resourceCache.textureView(texture) } : {}),
      bindGroupFor: (iter, inputView, outputView, inputTex, outputTex) => {
        const byteOffset = iter * ATROUS_UBO_BINDING_STRIDE_BYTES;
        const ubo = writeAtrousUbo(
          device, this._uboRef, 1 << iter, sigmas,
          {
            byteOffset,
            minSizeBytes: ATROUS_ITERATIONS * ATROUS_UBO_BINDING_STRIDE_BYTES,
          },
        );
        const buildBg = (): GPUBindGroup => buildPreparedAtrousBindGroup(
          device, bglCache, ubo,
          inputView, outputView,
          gNormalDepthView, gNormalDepthView,
          byteOffset,
        );
        return cachedBindGroup(resourceCache, `denoiser:atrous:${iter}`, [
          this._uboRef.buf,
          inputTex,
          outputTex,
          resources.common.gNormalDepthTexture,
        ], buildBg);
      },
      labelFor: (iter) => `atrous-${iter}` as PassLabel,
    });
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
