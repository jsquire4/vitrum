/**
 * SVGFRealDenoiser — real Schied et al. 2017 SVGF.
 *
 * Pass order per frame:
 *   1. svgf-real-reproj   — bilinear reprojection + disocclusion + EMA (Eq. 1–4)
 *   2. svgf-real-moments  — variance from blended moments (Eq. 5)
 *   3. svgf-real-7x7      — 7×7 spatial fallback for h < 4 pixels (§4.3)
 *   4. N × svgf-real-atrous-K — variance-guided à-trous chain
 *
 * Persistent textures updated each frame (owned by FrameResources.svgf,
 * read/written here via ping-pong):
 *   - svgfHistoryLengthTexture{A,B} — frame counter; reset on disocclusion
 *   - svgfMomentsTexture{A,B}       — E[L], E[L²]
 *   - svgfPrevRadianceTexture{A,B}  — EMA-blended radiance fed back next frame
 *
 * The à-trous chain reads the reproj output (radWrite) rather than the
 * raw hdrColorTexture so high-variance pixels still get spatial smoothing
 * from the chain even when temporal history is thin.
 *
 * All four pipelines use `layout: 'auto'`; the per-iter à-trous UBO is
 * allocated transiently per frame (5 × 16 B = 80 B) and destroyed after
 * `queue.submit()` — owning these is part of the per-frame work.
 */

import {
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  packAtrousVarianceAtrousUniforms,
  packSVGFReprojUniforms,
  SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
  SVGF_REPROJ_DEFAULT_UNIFORMS,
  SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
} from '@vitrum/shared-denoisers';
import { composeWgsl } from '../wgslComposer.js';
import {
  ATROUS_VARIANCE_MODULE,
  SVGF_7X7_SPATIAL_FALLBACK_MODULE,
  SVGF_REPROJECTION_MODULE,
  SVGF_VARIANCE_FROM_MOMENTS_MODULE,
  WGSL_MODULES,
} from '../wgslModules.js';
import type { UboRef } from '../bindGroupBuilders.js';
import { runAtrousChain } from '../passes/dispatchHelpers.js';
import type { PassLabel } from '../timestampQueries.js';
import {
  DENOISER_PASS_LABELS,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
} from './index.js';

export class SVGFRealDenoiser implements Denoiser {
  readonly id = 'svgf-real' as const;
  readonly passLabels = DENOISER_PASS_LABELS['svgf-real'];

  // ── GPU pipelines (compiled in initialize) ──────────────────────────────
  private _reprojPipeline!: GPUComputePipeline;
  private _momentsPipeline!: GPUComputePipeline;
  private _fallbackPipeline!: GPUComputePipeline;
  private _atrousPipeline!: GPUComputePipeline;

  /** Reprojection UBO — packed once at init time, stable per frame. */
  private readonly _reprojUboRef: UboRef = { buf: undefined };

  /** Ping-pong index for history/moments/prevRadiance (0 = A→read, B→write). */
  private _pingPong = 0;

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    const { device } = ctx;

    // ── Compile shader modules ────────────────────────────────────────────
    // All four SVGF kernels are self-contained per their shared-denoisers
    // source — `requires: []` in WGSL_MODULES. The composer still routes
    // them through composeWgsl() for the structural uniformity this gives
    // the include-graph (and so future SVGF modules that DO grow a
    // dependency need only update their `requires` array).
    const reprojSM = device.createShaderModule({
      label: 'svgf-reproj', code: composeWgsl(SVGF_REPROJECTION_MODULE, WGSL_MODULES),
    });
    const momentsSM = device.createShaderModule({
      label: 'svgf-moments', code: composeWgsl(SVGF_VARIANCE_FROM_MOMENTS_MODULE, WGSL_MODULES),
    });
    const fallbackSM = device.createShaderModule({
      label: 'svgf-7x7', code: composeWgsl(SVGF_7X7_SPATIAL_FALLBACK_MODULE, WGSL_MODULES),
    });
    // SVGF-real reuses the atrous-variance kernel for spatial filtering.
    const atrousSM = device.createShaderModule({
      label: 'svgf-real-atrous-variance', code: composeWgsl(ATROUS_VARIANCE_MODULE, WGSL_MODULES),
    });

    for (const [label, sm] of [
      ['svgf-reproj', reprojSM],
      ['svgf-moments', momentsSM],
      ['svgf-7x7', fallbackSM],
      ['svgf-real-atrous-variance', atrousSM],
    ] as [string, GPUShaderModule][]) {
      const info = await sm.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length > 0) {
        console.error(
          `[ReSTIR] Shader compile errors in '${label}':`,
          errors.map((e) => `line ${e.lineNum}: ${e.message}`),
        );
        throw new Error(
          `[ReSTIR] Shader compile error in '${label}': ${errors[0]!.message} (line ${errors[0]!.lineNum})`,
        );
      }
    }

    // ── Compile pipelines ─────────────────────────────────────────────────
    [this._reprojPipeline, this._momentsPipeline, this._fallbackPipeline] =
      await Promise.all([
        device.createComputePipelineAsync({
          label: 'svgf-real-reproj', layout: 'auto',
          compute: { module: reprojSM, entryPoint: 'svgfReprojMain' },
        }),
        device.createComputePipelineAsync({
          label: 'svgf-real-moments', layout: 'auto',
          compute: { module: momentsSM, entryPoint: 'svgfVarianceFromMomentsMain' },
        }),
        device.createComputePipelineAsync({
          label: 'svgf-real-7x7', layout: 'auto',
          compute: { module: fallbackSM, entryPoint: 'svgf7x7FallbackMain' },
        }),
      ]);
    this._atrousPipeline = await device.createComputePipelineAsync({
      label: 'svgf-real-atrous', layout: 'auto',
      compute: { module: atrousSM, entryPoint: 'svgfAtrousMain' },
    });

    // ── Eager reproj UBO allocation + default pack ────────────────────────
    this._reprojUboRef.buf = device.createBuffer({
      label: 'svgf-real-reproj-ubo',
      size: SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const scratch = new ArrayBuffer(SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
    packSVGFReprojUniforms(SVGF_REPROJ_DEFAULT_UNIFORMS, scratch);
    device.queue.writeBuffer(this._reprojUboRef.buf, 0, scratch);
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture {
    const {
      device,
      encoder,
      resources,
      gNormalDepthView,
      wgX16,
      wgY16,
      computeDesc,
    } = ctx;
    const common = resources.common;
    const svgf = resources.svgf;

    // ── Pass 1: Reprojection ─────────────────────────────────────────────
    // Bindings follow svgfReprojection.wgsl.ts binding declarations (0..14).
    // For the walkaround-hybrid pipeline, currDepth + currNormal come from
    // gNormalDepthTexture (.r = depth packed, .xyz = normal packed 0..1).
    // We use gNormalDepthTexture for both curr and prev depth/normal: one-frame
    // lag on the previous-frame G-buffer is acceptable for a real-time engine
    // and avoids allocating a full second G-buffer. Object IDs are not available
    // in the current walkaround pipeline; we bind conservative placeholders
    // (curr=0, prev=1) so obj-id mismatch rejects temporal reuse instead of
    // accepting potentially stale history.
    // Select ping-pong slots: read from A, write to B (or vice versa).
    const histRead = this._pingPong === 0 ? svgf.svgfHistoryLengthTextureA : svgf.svgfHistoryLengthTextureB;
    const histWrite = this._pingPong === 0 ? svgf.svgfHistoryLengthTextureB : svgf.svgfHistoryLengthTextureA;
    const momRead = this._pingPong === 0 ? svgf.svgfMomentsTextureA : svgf.svgfMomentsTextureB;
    const momWrite = this._pingPong === 0 ? svgf.svgfMomentsTextureB : svgf.svgfMomentsTextureA;
    const radRead = this._pingPong === 0 ? svgf.svgfPrevRadianceTextureA : svgf.svgfPrevRadianceTextureB;
    const radWrite = this._pingPong === 0 ? svgf.svgfPrevRadianceTextureB : svgf.svgfPrevRadianceTextureA;

    const reproj = this._reprojPipeline;
    {
      const bg = device.createBindGroup({
        label: 'svgf-real-reproj-bg',
        layout: reproj.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: common.hdrColorTexture.createView() },          // currColor (sampled)
          { binding: 1, resource: radRead.createView() },                          // prevColor (sampled)
          { binding: 2, resource: common.motionVectorTexture.createView() },       // motionVec
          { binding: 3, resource: common.gNormalDepthTexture.createView() },       // currDepth (.r)
          { binding: 4, resource: common.gNormalDepthTexture.createView() },       // currNormal (.xyz 0..1)
          { binding: 5, resource: svgf.svgfObjIdPlaceholderTexture.createView() }, // currObjId (1×1 r32uint, val=0)
          { binding: 6, resource: svgf.svgfPrevNormalDepthTexture.createView() },   // prevDepth
          { binding: 7, resource: svgf.svgfPrevNormalDepthTexture.createView() },   // prevNormal
          { binding: 8, resource: svgf.svgfPrevObjIdPlaceholderTexture.createView() }, // prevObjId (conservative placeholder)
          { binding: 9, resource: histRead.createView() },                         // historyLengthIn
          { binding: 10, resource: momRead.createView() },                         // momentsIn
          { binding: 11, resource: radWrite.createView() },                        // colorOut (storage write)
          { binding: 12, resource: histWrite.createView() },                       // historyOut (storage write)
          { binding: 13, resource: momWrite.createView() },                        // momentsOut (storage write)
          { binding: 14, resource: { buffer: this._reprojUboRef.buf! } },
        ],
      });
      const pass = encoder.beginComputePass(computeDesc('svgf-real-reproj'));
      pass.setPipeline(reproj);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 2: Variance from moments ────────────────────────────────────
    // Reads momWrite (just written by reproj) and histWrite.
    {
      const bg = device.createBindGroup({
        label: 'svgf-real-moments-bg',
        layout: this._momentsPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: momWrite.createView() },
          { binding: 1, resource: histWrite.createView() },
          { binding: 2, resource: svgf.svgfVarianceMomentsIntermedTexture.createView() },
        ],
      });
      const pass = encoder.beginComputePass(computeDesc('svgf-real-moments'));
      pass.setPipeline(this._momentsPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 3: 7×7 spatial fallback ─────────────────────────────────────
    {
      const bg = device.createBindGroup({
        label: 'svgf-real-7x7-bg',
        layout: this._fallbackPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: common.hdrColorTexture.createView() },
          { binding: 1, resource: histWrite.createView() },
          { binding: 2, resource: svgf.svgfVarianceMomentsIntermedTexture.createView() },
          { binding: 3, resource: svgf.svgfVarianceTexture.createView() },
        ],
      });
      const pass = encoder.beginComputePass(computeDesc('svgf-real-7x7'));
      pass.setPipeline(this._fallbackPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // Flip ping-pong for next frame (history, moments, prevRadiance).
    this._pingPong = 1 - this._pingPong;

    // ── Pass 4: À-trous chain (svgfAtrousMain) ───────────────────────────
    // Feed the EMA-blended reprojection output (radWrite) as the starting color.
    // Ping-pong with denoisedPing/Pong as usual.
    const sa = this._atrousPipeline;
    const atrousUboBytes = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    const varView = svgf.svgfVarianceTexture.createView();
    // Collect transient per-iteration UBOs so they can be destroyed after submit().
    // GPUBuffer is a GPU resource — it must be explicitly destroyed; GC does not
    // release GPU memory. The pipeline calls this denoiser's `cleanupAfterSubmit()`
    // hook after `device.queue.submit()` has drained the encoder.
    this._pendingTransientUbos = [];
    const denoised = runAtrousChain(encoder, sa, {
      iterations: SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
      startTex: radWrite,
      pingTex: common.denoisedPingTexture,
      pongTex: common.denoisedPongTexture,
      wgX: wgX16,
      wgY: wgY16,
      computeDesc,
      // Each iteration allocates its own transient UBO (so the pipeline can
      // release them after submit) and assembles the verbatim 6-binding
      // layout. Identical JS ordering to the prior loop.
      bindGroupFor: (iter, inputView, outputView) => {
        packAtrousVarianceAtrousUniforms(
          { iteration: iter, ...ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS },
          atrousUboBytes,
          0,
        );
        // One 16-byte UBO per atrous iteration (5 × 16 = 80 bytes total per frame).
        const iterUbo = device.createBuffer({
          label: `svgf-real-atrous-ubo-${iter}`,
          size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(iterUbo, 0, atrousUboBytes);
        this._pendingTransientUbos.push(iterUbo);
        return device.createBindGroup({
          label: `svgf-real-atrous-bg-${iter}`,
          layout: sa.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: inputView },
            { binding: 1, resource: outputView },
            { binding: 2, resource: gNormalDepthView },
            { binding: 3, resource: gNormalDepthView },
            { binding: 4, resource: varView },
            { binding: 5, resource: { buffer: iterUbo } },
          ],
        });
      },
      labelFor: (iter) => `svgf-real-atrous-${iter}` as PassLabel,
    });
    // Publish current-frame normal+depth for next-frame reprojection checks.
    encoder.copyTextureToTexture(
      { texture: common.gNormalDepthTexture },
      { texture: svgf.svgfPrevNormalDepthTexture },
      {
        width: common.gNormalDepthTexture.width,
        height: common.gNormalDepthTexture.height,
        depthOrArrayLayers: 1,
      },
    );
    return denoised;
  }

  // ── Transient per-frame UBO bookkeeping ─────────────────────────────────

  /** Per-iter à-trous UBOs created in the current dispatch; the pipeline
   *  calls {@link cleanupAfterSubmit} after `queue.submit()` to drain
   *  this list and destroy each entry. */
  private _pendingTransientUbos: GPUBuffer[] = [];

  /** Free the transient per-iter UBOs allocated in the most recent
   *  {@link dispatch}. Called by the pipeline after `queue.submit()` so
   *  the GPU queue retains the command buffer's reference while host-side
   *  handles are released. */
  cleanupAfterSubmit(): void {
    for (const ubo of this._pendingTransientUbos) ubo.destroy();
    this._pendingTransientUbos = [];
  }

  resize(_w: number, _h: number): void {
    // The new persistent SVGF textures (owned by FrameResources) are
    // blank, so the ping-pong index must restart at 0 to avoid reading
    // garbage from the (about-to-be-written) "B" slot first.
    this._pingPong = 0;
  }

  dispose(): void {
    this._reprojUboRef.buf?.destroy();
    this._reprojUboRef.buf = undefined;
    for (const ubo of this._pendingTransientUbos) ubo.destroy();
    this._pendingTransientUbos = [];
  }
}
