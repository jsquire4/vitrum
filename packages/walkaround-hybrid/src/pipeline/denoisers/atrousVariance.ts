/**
 * AtrousVarianceDenoiser — Welford-temporal variance estimate + 3 à-trous
 * spatial iterations. The default denoiser for the walkaround-hybrid
 * pipeline.
 *
 * Pass order per frame:
 *   1. welford-temporal      — running mean / M2 update on the total radiance
 *                              (so the tier estimator sees direct + indirect)
 *   2. atrous-variance-variance — convert Welford(mean,M2) to per-pixel σ²
 *   3. N × atrous-variance-atrous — variance-weighted à-trous (Sprint 10a)
 *
 * Encapsulates everything the legacy `_dispatchAtrousVariance` orchestration
 * touched: the three compute pipelines, their shader compiles
 * (welford-temporal, atrous-variance kernel), the welford ping-pong index,
 * and the per-pass UBOs. The pipelines use `layout: 'auto'` so the BGLs
 * are derived from the compiled module — no entry in `bindGroupLayouts.ts`.
 *
 * Refs: Schied et al. 2017 §4 (variance estimate), Sprint 10a; complexity
 * sweep 2026-05-17 Theme B (denoiser dispatch encapsulation).
 */

import {
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
  packAtrousVarianceAtrousUniforms,
  packAtrousVarianceVarianceUniforms,
} from '@vitrum/shared-denoisers';
import { defineUbo } from '@vitrum/shared-samplers';

// W2-C13 follow-up — WelfordTemporalUBO (2×u32 active + min-binding pad
// to 16 B). Mirrors `struct WelfordTemporalUBO` in welfordTemporal.wgsl.ts.
const WELFORD_TEMPORAL_UBO = defineUbo([
  { name: 'sampleN',    type: 'u32' },
  { name: 'forceReset', type: 'u32' },
] as const);
import { WELFORD_TEMPORAL_MODULE } from '../../shaders/welfordTemporal.wgsl.js';
import { composeWgsl } from '../wgslComposer.js';
import { ATROUS_VARIANCE_MODULE, WGSL_MODULES } from '../wgslModules.js';
import type { UboRef } from '../bindGroupBuilders.js';
import { buildAtrousVarianceAtrousBindGroup } from '../bindGroupBuilders.js';
import { checkShaderCompile } from '../shaderUtils.js';
import { runAtrousChain } from '../passes/dispatchHelpers.js';
import type { PassLabel } from '../timestampQueries.js';
import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
  type DenoiserState,
} from './index.js';

const ATROUS_VARIANCE_ATROUS_UBO_BINDING_STRIDE_BYTES = 256;

/**
 * Welford BG builder — pre-W1-R3 lived in `bindGroupBuilders.ts`. The
 * pipeline uses `layout: 'auto'` so the BGL source is the pipeline
 * itself. Local to the denoiser since this is the only consumer.
 */
function buildWelfordBindGroup(
  device: GPUDevice,
  welfordPipeline: GPUComputePipeline,
  hdrTotal: GPUTextureView,
  welfordRead: GPUTextureView,
  welfordWrite: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'welford-bg',
    layout: welfordPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrTotal },
      { binding: 1, resource: welfordRead },
      { binding: 2, resource: welfordWrite },
      { binding: 3, resource: { buffer: ubo } },
    ],
  });
}

/**
 * The variance kernel reads only inputColor (0), varianceIn (5), and writes
 * varianceOut (6) + reads varUBO (7). Bindings 1..4 are declared in the
 * WGSL but unreferenced by the kernel body — Dawn's `layout: 'auto'`
 * drops unreferenced bindings, so attempting to bind them yields "binding
 * index N not present in the bind group layout" and the whole command
 * buffer is rejected.
 */
function buildAtrousVarianceVarianceBindGroup(
  device: GPUDevice,
  variancePipeline: GPUComputePipeline,
  hdrColor: GPUTextureView,
  welfordWrite: GPUTextureView,
  varianceEstimate: GPUTextureView,
  ubo: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: 'atrous-variance-variance-bg',
    layout: variancePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrColor },
      { binding: 5, resource: welfordWrite },
      { binding: 6, resource: varianceEstimate },
      { binding: 7, resource: { buffer: ubo } },
    ],
  });
}


export class AtrousVarianceDenoiser implements Denoiser {
  readonly id = 'atrous-variance' as const;
  readonly passLabels = DENOISER_PASS_LABELS['atrous-variance'];

  // ── GPU pipelines (compiled in initialize) ──────────────────────────────
  private _welfordPipeline!: GPUComputePipeline;
  private _variancePipeline!: GPUComputePipeline;
  private _atrousPipeline!: GPUComputePipeline;

  // ── Per-pass UBOs (eagerly allocated in initialize) ─────────────────────
  private readonly _welfordUboRef: UboRef = { buf: undefined };
  private readonly _varianceUboRef: UboRef = { buf: undefined };
  private readonly _atrousUboRef: UboRef = { buf: undefined };

  /** Ping-pong index for the Welford variance buffer (0 = read main, write aux). */
  private _welfordPing = 0;

  /**
   * Return the current Welford ping-pong index so the SampleBudgetPass can
   * bind the freshest side (the side written by the previous frame).
   *
   * When ping === 0: freshest data is in `varianceBuffer` (written by frame N-1).
   * When ping === 1: freshest data is in `varianceBufferAux`.
   *
   * This must be read BEFORE calling dispatch() for the current frame, because
   * dispatch() flips the ping at the end.
   *
   * Exposed as both a method (for internal call sites) and as the `welfordPing`
   * getter required by the {@link Denoiser} interface (D3.4).
   */
  getWelfordPing(): number {
    return this._welfordPing;
  }

  /** D3.4 — {@link Denoiser.welfordPing} interface property. */
  get welfordPing(): number {
    return this._welfordPing;
  }

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    const { device } = ctx;

    // ── Compile shader modules ────────────────────────────────────────────
    // The include-graph handles the self-contained-vs-common-dependent split
    // structurally: WELFORD_TEMPORAL_MODULE declares `requires: ['common']`,
    // while ATROUS_VARIANCE_MODULE declares `requires: []` because the
    // shared-denoisers WGSL fragment ships its own PI/INV_PI/LUM_W/
    // WelfordVariance declarations. The pre-R6 anti-duplication-by-comment
    // is now structural — no hand-rolled prepend, no risk of redeclaration.
    const welfordSM = device.createShaderModule({
      label: 'welford-temporal',
      code: composeWgsl(WELFORD_TEMPORAL_MODULE, WGSL_MODULES),
    });
    const atrousVarianceSM = device.createShaderModule({
      label: 'atrous-variance',
      code: composeWgsl(ATROUS_VARIANCE_MODULE, WGSL_MODULES),
    });

    // Compile checks — mirror the historical pipelineCompiler behaviour.
    for (const [label, sm] of [
      ['welford', welfordSM],
      ['atrous-variance', atrousVarianceSM],
    ] as [string, GPUShaderModule][]) {
      await checkShaderCompile(sm, label);
    }

    // ── Compile pipelines ─────────────────────────────────────────────────
    this._welfordPipeline = await device.createComputePipelineAsync({
      label: 'welford-temporal',
      layout: 'auto',
      compute: { module: welfordSM, entryPoint: 'welfordTemporalMain' },
    });
    this._variancePipeline = await device.createComputePipelineAsync({
      label: 'atrous-variance-variance',
      layout: 'auto',
      compute: { module: atrousVarianceSM, entryPoint: 'svgfVarianceMain' },
    });
    this._atrousPipeline = await device.createComputePipelineAsync({
      label: 'atrous-variance-atrous',
      layout: 'auto',
      compute: { module: atrousVarianceSM, entryPoint: 'svgfAtrousMain' },
    });

    // ── Eager UBO allocation ──────────────────────────────────────────────
    // Each UBO is small (≤32B); allocating up-front avoids first-frame
    // branching in dispatch.
    const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this._welfordUboRef.buf = device.createBuffer({
      label: 'welford-ubo', size: 16, usage: U,
    });
    this._varianceUboRef.buf = device.createBuffer({
      label: 'atrous-variance-variance-ubo',
      size: ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
      usage: U,
    });
    this._atrousUboRef.buf = device.createBuffer({
      label: 'atrous-variance-atrous-ubo',
      size: ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS * ATROUS_VARIANCE_ATROUS_UBO_BINDING_STRIDE_BYTES,
      usage: U,
    });
  }

  state(): DenoiserState {
    return DENOISER_READY_STATE;
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture {
    const {
      device,
      encoder,
      resources,
      gNormalDepthView,
      isMoving,
      wgX16,
      wgY16,
      frameIndex,
      computeDesc,
    } = ctx;
    const common = resources.common;
    const wf = this._welfordPipeline;
    const sv = this._variancePipeline;
    const sa = this._atrousPipeline;

    const welfordRead = this._welfordPing === 0
      ? common.varianceBuffer
      : common.varianceBufferAux;
    const welfordWrite = this._welfordPing === 0
      ? common.varianceBufferAux
      : common.varianceBuffer;

    // welfordUboRef.buf is allocated eagerly in initialize().
    // W2-C13 follow-up: byte-identical to the prior Uint32Array([frameIndex+1,
    // forceReset, 0, 0]) write — defineUbo packs two u32 fields at offsets 0/4
    // and zero-fills the trailing pad to the 16-byte minimum-binding floor.
    const wUboBytes = new ArrayBuffer(WELFORD_TEMPORAL_UBO.sizeBytes);
    WELFORD_TEMPORAL_UBO.pack(new DataView(wUboBytes), 0, {
      sampleN:    frameIndex + 1,
      forceReset: isMoving ? 1 : 0,
    });
    device.queue.writeBuffer(this._welfordUboRef.buf!, 0, wUboBytes);

    const hdrColorView = common.hdrColorTexture.createView();
    // Sprint 18 follow-up — welford reads the total-radiance texture so the
    // variance and the sample-budget tier derived from it cover both direct
    // and indirect channels. Variance + atrous still read hdrColorView
    // (direct-only) so the denoiser sees the channel it is tuned for.
    const hdrTotalView = common.hdrTotalTexture.createView();
    {
      const pass = encoder.beginComputePass(computeDesc('welford-temporal'));
      pass.setPipeline(wf);
      pass.setBindGroup(0, buildWelfordBindGroup(
        device, wf,
        hdrTotalView, welfordRead.createView(), welfordWrite.createView(),
        this._welfordUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    const varUboBytes = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceVarianceUniforms({ frameCount: frameIndex }, varUboBytes, 0);
    device.queue.writeBuffer(this._varianceUboRef.buf!, 0, varUboBytes);

    {
      const pass = encoder.beginComputePass(computeDesc('atrous-variance-variance'));
      pass.setPipeline(sv);
      pass.setBindGroup(0, buildAtrousVarianceVarianceBindGroup(
        device, sv,
        hdrColorView,
        welfordWrite.createView(),
        common.atrousVarianceEstimateTexture.createView(),
        this._varianceUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    const atrousUboBytes = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    const varView = common.atrousVarianceEstimateTexture.createView();
    const denoised = runAtrousChain(encoder, sa, {
      iterations: ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS,
      startTex: common.hdrColorTexture,
      pingTex: common.denoisedPingTexture,
      pongTex: common.denoisedPongTexture,
      wgX: wgX16,
      wgY: wgY16,
      computeDesc,
      // The shared eager UBO is re-packed + re-written each iteration BEFORE
      // its dispatch is encoded into that iteration's aligned byte range.
      bindGroupFor: (iter, inputView, outputView) => {
        const atrousUboByteOffset = iter * ATROUS_VARIANCE_ATROUS_UBO_BINDING_STRIDE_BYTES;
        packAtrousVarianceAtrousUniforms(
          { iteration: iter, ...ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS },
          atrousUboBytes,
          0,
        );
        device.queue.writeBuffer(this._atrousUboRef.buf!, atrousUboByteOffset, atrousUboBytes);
        return buildAtrousVarianceAtrousBindGroup(
          device, sa,
          inputView, outputView,
          gNormalDepthView, varView,
          this._atrousUboRef.buf!,
          `atrous-variance-atrous-bg-${iter}`,
          atrousUboByteOffset,
        );
      },
      labelFor: (iter) => `atrous-variance-atrous-${iter}` as PassLabel,
    });

    // Flip the Welford ping-pong for next frame. The legacy pipeline
    // bumped this AFTER the dispatch returned; mirror that exactly so
    // frame N's `welfordRead` reads the buffer written by frame N-1.
    this._welfordPing = 1 - this._welfordPing;

    return denoised;
  }

  resize(_w: number, _h: number): void {
    // The Welford ping-pong index must reset on resize: the new variance
    // textures (owned by FrameResources) are blank, so the accumulator
    // restarts from scratch.
    this._welfordPing = 0;
  }

  dispose(): void {
    this._welfordUboRef.buf?.destroy();
    this._varianceUboRef.buf?.destroy();
    this._atrousUboRef.buf?.destroy();
    this._welfordUboRef.buf = undefined;
    this._varianceUboRef.buf = undefined;
    this._atrousUboRef.buf = undefined;
  }
}
