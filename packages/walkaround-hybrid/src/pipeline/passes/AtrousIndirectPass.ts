/**
 * AtrousIndirectPass — Sprint 18 follow-up, indirect-channel à-trous chain.
 *
 * Four à-trous iterations with widening step (1, 2, 4, 8) on the indirect
 * accumulator output, ping-pong between `indirectDenoisedPing/Pong`. Uses
 * the SAME compiled à-trous pipeline (`compiled.atrousPipeline`) as the
 * legacy `AtrousDenoiser`, but with the broader `ATROUS_INDIRECT_SIGMAS`
 * (Schied 2017 §4.3 — wider blurs for the already-temporally-smoothed
 * indirect signal).
 *
 * **Pipeline sharing rationale (kept from the original `_dispatchAtrousIndirect`):**
 * the shared à-trous pipeline lives in `pipelineCompiler.ts` because BOTH
 * the legacy `AtrousDenoiser` AND this always-on indirect chain dispatch
 * the same compiled module. Forking a private compile would double the
 * shader compile cost on every engine boot for zero functional benefit.
 *
 * The per-iter `stepWidth` UBO is OWNED BY THIS PASS (separate `UboRef`
 * from the legacy AtrousDenoiser's UBO) so the two consumers never race
 * on their UBO writes within a single frame.
 *
 * Emits 4 timestamp labels in order: `atrous-indirect-0..3`. The terminal
 * label is the pass's stable `id`, so `IndirectCombinePass` depends on
 * `atrous-indirect-3` deterministically.
 */

import {
  buildAtrousBindGroup,
  type AtrousSigmas,
  type UboRef,
} from '../bindGroupBuilders.js';
import type {
  Pass,
  PassDispatchContext,
  PassInitContext,
} from '../Pass.js';
import type { PassLabel } from '../timestampQueries.js';

const ATROUS_INDIRECT_ITERATIONS = 4;

export class AtrousIndirectPass implements Pass {
  readonly id = 'atrous-indirect-3' as const; // terminal label downstream depends on.
  readonly dependencies: readonly string[] = ['indirect-temporal-accum'];
  readonly passLabels: readonly PassLabel[] = [
    'atrous-indirect-0',
    'atrous-indirect-1',
    'atrous-indirect-2',
    'atrous-indirect-3',
  ];

  /** Shared à-trous pipeline (same handle the AtrousDenoiser uses). */
  private readonly _sharedAtrousPipeline: GPUComputePipeline;
  /** Pass-private UBO ref — kept separate from AtrousDenoiser's UBO so
   *  the two consumers never race on their per-iter sigma writes. */
  private readonly _uboRef: UboRef;

  constructor(
    sharedAtrousPipeline: GPUComputePipeline,
    uboRef: UboRef,
  ) {
    this._sharedAtrousPipeline = sharedAtrousPipeline;
    this._uboRef = uboRef;
  }

  gates(): boolean {
    return true;
  }

  async initialize(_ctx: PassInitContext): Promise<void> {}

  dispatch(ctx: PassDispatchContext): void {
    const { device, encoder, computeDesc, bglCache, resources, wgX16, wgY16, gNormalDepthView, frameState, inputs } = ctx;
    const common = resources.common;
    // B3a — per-frame indirect sigmas from HybridEngineOptions (host
    // override) or Cornell defaults `[32, 20, 0.5]`.
    const sigmas: AtrousSigmas = {
      sigmaN: inputs.atrousIndirectSigmas[0],
      sigmaZ: inputs.atrousIndirectSigmas[1],
      sigmaC: inputs.atrousIndirectSigmas[2],
    };
    let inputTex: GPUTexture = frameState.indirectAccumOut;
    for (let iter = 0; iter < ATROUS_INDIRECT_ITERATIONS; iter++) {
      const stepWidth = 1 << iter;
      const outputTex = iter % 2 === 0
        ? common.indirectDenoisedPingTexture
        : common.indirectDenoisedPongTexture;
      const bg = buildAtrousBindGroup(
        device, bglCache, this._uboRef,
        inputTex.createView(), outputTex.createView(),
        gNormalDepthView, gNormalDepthView, stepWidth,
        sigmas,
      );
      const label = `atrous-indirect-${iter}` as PassLabel;
      const pass = encoder.beginComputePass(computeDesc(label));
      pass.setPipeline(this._sharedAtrousPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outputTex;
    }
    frameState.denoisedIndirect = inputTex;
  }

  dispose(): void {
    this._uboRef.buf?.destroy();
    this._uboRef.buf = undefined;
  }
}
