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
 * All four pipelines use `layout: 'auto'`; the à-trous chain owns one
 * persistent UBO per iteration to avoid per-frame GPUBuffer churn.
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
  private readonly _atrousUboRefs: UboRef[] = Array.from(
    { length: SVGF_REAL_DEFAULT_ATROUS_ITERATIONS },
    () => ({ buf: undefined }),
  );

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
      await checkShaderCompile(sm, label);
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

    const atrousUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    for (let i = 0; i < this._atrousUboRefs.length; i += 1) {
      this._atrousUboRefs[i]!.buf = device.createBuffer({
        label: `svgf-real-atrous-ubo-${i}`,
        size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
        usage: atrousUsage,
      });
    }
  }

  state(): DenoiserState {
    return DENOISER_READY_STATE;
  }

  /** Select between two ping-pong resources: returns `a` when the current
   *  ping-pong index is 0, `b` when it is 1. Collapses the repeated
   *  `this._pingPong === 0 ? x : y` ternaries throughout dispatch(). */
  private _selectPingPong<T>(a: T, b: T): T {
    return this._pingPong === 0 ? a : b;
  }

  private _buildReprojBindGroup(
    device: GPUDevice,
    common: DenoiserDispatchContext['resources']['common'],
    svgf: DenoiserDispatchContext['resources']['svgf'],
    radRead: GPUTexture,
    radWrite: GPUTexture,
    histRead: GPUTexture,
    histWrite: GPUTexture,
    momRead: GPUTexture,
    momWrite: GPUTexture,
    currObjIdTexture: GPUTexture,
    prevObjIdTexture: GPUTexture,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: 'svgf-real-reproj-bg',
      layout: this._reprojPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: common.hdrColorTexture.createView() },          // currColor (sampled)
        { binding: 1, resource: radRead.createView() },                          // prevColor (sampled)
        { binding: 2, resource: common.motionVectorTexture.createView() },       // motionVec
        { binding: 3, resource: common.gNormalDepthTexture.createView() },       // currDepth (.r)
        { binding: 4, resource: common.gNormalDepthTexture.createView() },       // currNormal (.xyz 0..1)
        { binding: 5, resource: currObjIdTexture.createView() },                 // currObjId
        { binding: 6, resource: svgf.svgfPrevNormalDepthTexture.createView() },   // prevDepth
        { binding: 7, resource: svgf.svgfPrevNormalDepthTexture.createView() },   // prevNormal
        { binding: 8, resource: prevObjIdTexture.createView() },                 // prevObjId
        { binding: 9, resource: histRead.createView() },                         // historyLengthIn
        { binding: 10, resource: momRead.createView() },                         // momentsIn
        { binding: 11, resource: radWrite.createView() },                        // colorOut (storage write)
        { binding: 12, resource: histWrite.createView() },                       // historyOut (storage write)
        { binding: 13, resource: momWrite.createView() },                        // momentsOut (storage write)
        { binding: 14, resource: { buffer: this._reprojUboRef.buf! } },
      ],
    });
  }

  private _buildMomentsBindGroup(
    device: GPUDevice,
    svgf: DenoiserDispatchContext['resources']['svgf'],
    momWrite: GPUTexture,
    histWrite: GPUTexture,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: 'svgf-real-moments-bg',
      layout: this._momentsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: momWrite.createView() },
        { binding: 1, resource: histWrite.createView() },
        { binding: 2, resource: svgf.svgfVarianceMomentsIntermedTexture.createView() },
      ],
    });
  }

  private _buildFallbackBindGroup(
    device: GPUDevice,
    common: DenoiserDispatchContext['resources']['common'],
    svgf: DenoiserDispatchContext['resources']['svgf'],
    histWrite: GPUTexture,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: 'svgf-real-7x7-bg',
      layout: this._fallbackPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: common.hdrColorTexture.createView() },
        { binding: 1, resource: histWrite.createView() },
        { binding: 2, resource: svgf.svgfVarianceMomentsIntermedTexture.createView() },
        { binding: 3, resource: svgf.svgfVarianceTexture.createView() },
      ],
    });
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
    // We use the current shade-authored G-buffer for curr depth/normal and the
    // previous-frame snapshot copied at the end of this dispatch for prev
    // depth/normal. Object IDs follow the same lifecycle: shade writes the
    // current full-res r32uint ID, reprojection reads the previous full-res
    // r32uint ID, then this dispatch copies current → previous for frame N+1.
    // Select ping-pong slots: read from A, write to B (or vice versa).
    const histRead = this._selectPingPong(svgf.svgfHistoryLengthTextureA, svgf.svgfHistoryLengthTextureB);
    const histWrite = this._selectPingPong(svgf.svgfHistoryLengthTextureB, svgf.svgfHistoryLengthTextureA);
    const momRead = this._selectPingPong(svgf.svgfMomentsTextureA, svgf.svgfMomentsTextureB);
    const momWrite = this._selectPingPong(svgf.svgfMomentsTextureB, svgf.svgfMomentsTextureA);
    const radRead = this._selectPingPong(svgf.svgfPrevRadianceTextureA, svgf.svgfPrevRadianceTextureB);
    const radWrite = this._selectPingPong(svgf.svgfPrevRadianceTextureB, svgf.svgfPrevRadianceTextureA);
    const hasRealObjectIdHistory =
      svgf.svgfCurrentObjectIdTexture != null && svgf.svgfPreviousObjectIdTexture != null;
    const currObjIdTexture = hasRealObjectIdHistory
      ? svgf.svgfCurrentObjectIdTexture
      : svgf.svgfObjIdPlaceholderTexture;
    const prevObjIdTexture = hasRealObjectIdHistory
      ? svgf.svgfPreviousObjectIdTexture
      : svgf.svgfPrevObjIdPlaceholderTexture;

    {
      const bg = this._buildReprojBindGroup(
        device, common, svgf,
        radRead, radWrite,
        histRead, histWrite,
        momRead, momWrite,
        currObjIdTexture, prevObjIdTexture,
      );
      const pass = encoder.beginComputePass(computeDesc('svgf-real-reproj'));
      pass.setPipeline(this._reprojPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 2: Variance from moments ────────────────────────────────────
    // Reads momWrite (just written by reproj) and histWrite.
    {
      const bg = this._buildMomentsBindGroup(device, svgf, momWrite, histWrite);
      const pass = encoder.beginComputePass(computeDesc('svgf-real-moments'));
      pass.setPipeline(this._momentsPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 3: 7×7 spatial fallback ─────────────────────────────────────
    {
      const bg = this._buildFallbackBindGroup(device, common, svgf, histWrite);
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
    const denoised = runAtrousChain(encoder, sa, {
      iterations: SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
      startTex: radWrite,
      pingTex: common.denoisedPingTexture,
      pongTex: common.denoisedPongTexture,
      wgX: wgX16,
      wgY: wgY16,
      computeDesc,
      // Each iteration has a separate persistent UBO, so encoded dispatches
      // keep their own constants while avoiding per-frame GPUBuffer churn.
      bindGroupFor: (iter, inputView, outputView) => {
        packAtrousVarianceAtrousUniforms(
          { iteration: iter, ...ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS },
          atrousUboBytes,
          0,
        );
        const iterUbo = this._atrousUboRefs[iter]?.buf;
        if (iterUbo == null) {
          throw new Error(`SVGFRealDenoiser: missing atrous UBO for iteration ${iter}`);
        }
        device.queue.writeBuffer(iterUbo, 0, atrousUboBytes);
        return buildAtrousVarianceAtrousBindGroup(
          device, sa,
          inputView, outputView,
          gNormalDepthView, varView,
          iterUbo,
          `svgf-real-atrous-bg-${iter}`,
        );
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
    if (hasRealObjectIdHistory) {
      encoder.copyTextureToTexture(
        { texture: currObjIdTexture },
        { texture: prevObjIdTexture },
        {
          width: currObjIdTexture.width,
          height: currObjIdTexture.height,
          depthOrArrayLayers: 1,
        },
      );
    }
    return denoised;
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
    for (const ref of this._atrousUboRefs) {
      ref.buf?.destroy();
      ref.buf = undefined;
    }
  }
}
