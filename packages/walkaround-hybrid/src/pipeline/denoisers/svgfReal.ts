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
 *   - svgfPrevRadianceTexture{A,B}  — first-wavelet color fed back next frame
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
  packAtrousVarianceAtrousUniforms,
  packSVGFReprojUniforms,
  SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
  SVGF_REPROJ_DEFAULT_UNIFORMS,
  SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
} from '@vitrum/shared-denoisers';
import { composeWgsl } from '../wgslComposer.js';
import {
  SVGF_7X7_SPATIAL_FALLBACK_MODULE,
  SVGF_REAL_ATROUS_MODULE,
  SVGF_REPROJECTION_MODULE,
  SVGF_VARIANCE_FROM_MOMENTS_MODULE,
  WGSL_MODULES,
} from '../wgslModules.js';
import type { UboRef } from '../bindGroupBuilders.js';
import { buildAtrousVarianceAtrousBindGroup } from '../bindGroupBuilders.js';
import { checkShaderCompile } from '../shaderUtils.js';
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
import { publishFrameState } from '../FramePublication.js';
import { shouldResetDenoiserHistory } from './historyReset.js';

type TextureViewFor = (texture: GPUTexture) => GPUTextureView;

export class SVGFRealDenoiser implements Denoiser {
  readonly id = 'svgf-real' as const;
  readonly passLabels = DENOISER_PASS_LABELS['svgf-real'];

  // ── GPU pipelines (compiled in initialize) ──────────────────────────────
  private _reprojPipeline!: GPUComputePipeline;
  private _momentsPipeline!: GPUComputePipeline;
  private _fallbackPipeline!: GPUComputePipeline;
  private _atrousPipeline!: GPUComputePipeline;

  /** Reprojection UBO — stable shape; forceReset may be re-packed per frame. */
  private readonly _reprojUboRef: UboRef = { buf: undefined };
  private readonly _atrousUboRefs: UboRef[] = Array.from(
    { length: SVGF_REAL_DEFAULT_ATROUS_ITERATIONS },
    () => ({ buf: undefined }),
  );
  private _lastForceReset = -1;

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
    const atrousSM = device.createShaderModule({
      label: 'svgf-real-atrous', code: composeWgsl(SVGF_REAL_ATROUS_MODULE, WGSL_MODULES),
    });

    for (const [label, sm] of [
      ['svgf-reproj', reprojSM],
      ['svgf-moments', momentsSM],
      ['svgf-7x7', fallbackSM],
      ['svgf-real-atrous', atrousSM],
    ] as [string, GPUShaderModule][]) {
      await checkShaderCompile(sm, label);
    }

    // ── Compile pipelines ─────────────────────────────────────────────────
    // Keep every candidate pipeline local until the matching UBO cohort is also
    // complete. Reinitialization may target a different device; publishing only
    // part of that generation would pair new-device pipelines with old-device
    // buffers after any later failure.
    const [reprojPipeline, momentsPipeline, fallbackPipeline] = await Promise.all([
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
    const atrousPipeline = await device.createComputePipelineAsync({
      label: 'svgf-real-atrous', layout: 'auto',
      compute: { module: atrousSM, entryPoint: 'svgfRealAtrousMain' },
    });

    // ── Eager reproj UBO allocation + default pack ────────────────────────
    // createBuffer may fail at any allocation. Keep the candidate cohort local
    // until it is complete so a partial initialize cannot leak buffers or
    // replace a previously-live cohort on reinitialize.
    let reprojUbo: GPUBuffer | undefined;
    const atrousUbos: GPUBuffer[] = [];
    try {
      reprojUbo = device.createBuffer({
        label: 'svgf-real-reproj-ubo',
        size: SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const reprojScratch = new ArrayBuffer(SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
      packSVGFReprojUniforms({
        ...SVGF_REPROJ_DEFAULT_UNIFORMS,
        forceReset: 0,
      }, reprojScratch);
      device.queue.writeBuffer(reprojUbo, 0, reprojScratch);

      const atrousUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
      for (let i = 0; i < this._atrousUboRefs.length; i += 1) {
        atrousUbos.push(device.createBuffer({
          label: `svgf-real-atrous-ubo-${i}`,
          size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
          usage: atrousUsage,
        }));
      }
    } catch (error) {
      try { reprojUbo?.destroy(); } catch { /* preserve the allocation error */ }
      for (const buffer of atrousUbos) {
        try { buffer.destroy(); } catch { /* preserve the allocation error */ }
      }
      throw error;
    }

    const previousReprojUbo = this._reprojUboRef.buf;
    const previousAtrousUbos = this._atrousUboRefs.map((ref) => ref.buf);
    this._reprojPipeline = reprojPipeline;
    this._momentsPipeline = momentsPipeline;
    this._fallbackPipeline = fallbackPipeline;
    this._atrousPipeline = atrousPipeline;
    this._reprojUboRef.buf = reprojUbo;
    for (let i = 0; i < this._atrousUboRefs.length; i += 1) {
      this._atrousUboRefs[i]!.buf = atrousUbos[i];
    }
    this._lastForceReset = 0;
    this._pingPong = 0;

    // Retire the old cohort only after the complete replacement is published.
    try { previousReprojUbo?.destroy(); } catch { /* replacement remains live */ }
    for (const buffer of previousAtrousUbos) {
      try { buffer?.destroy(); } catch { /* replacement remains live */ }
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

  private _writeReprojUniforms(device: GPUDevice, forceReset: number): void {
    if (this._lastForceReset === forceReset) return;
    const scratch = new ArrayBuffer(SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
    packSVGFReprojUniforms({
      ...SVGF_REPROJ_DEFAULT_UNIFORMS,
      forceReset,
    }, scratch);
    device.queue.writeBuffer(this._reprojUboRef.buf!, 0, scratch);
    this._lastForceReset = forceReset;
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
    textureViewFor: TextureViewFor,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: 'svgf-real-reproj-bg',
      layout: this._reprojPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textureViewFor(common.hdrColorTexture) },        // currColor (sampled)
        { binding: 1, resource: textureViewFor(radRead) },                       // prevColor (sampled)
        { binding: 2, resource: textureViewFor(common.motionVectorTexture) },    // motionVec
        { binding: 3, resource: textureViewFor(common.gNormalDepthTexture) },    // currDepth (.r)
        { binding: 4, resource: textureViewFor(common.gNormalDepthTexture) },    // currNormal (.xyz 0..1)
        { binding: 5, resource: textureViewFor(currObjIdTexture) },              // currObjId
        { binding: 6, resource: textureViewFor(svgf.svgfPrevNormalDepthTexture) }, // prevDepth
        { binding: 7, resource: textureViewFor(svgf.svgfPrevNormalDepthTexture) }, // prevNormal
        { binding: 8, resource: textureViewFor(prevObjIdTexture) },              // prevObjId
        { binding: 9, resource: textureViewFor(histRead) },                      // historyLengthIn
        { binding: 10, resource: textureViewFor(momRead) },                      // momentsIn
        { binding: 11, resource: textureViewFor(radWrite) },                     // colorOut (storage write)
        { binding: 12, resource: textureViewFor(histWrite) },                    // historyOut (storage write)
        { binding: 13, resource: textureViewFor(momWrite) },                     // momentsOut (storage write)
        { binding: 14, resource: { buffer: this._reprojUboRef.buf! } },
      ],
    });
  }

  private _buildMomentsBindGroup(
    device: GPUDevice,
    svgf: DenoiserDispatchContext['resources']['svgf'],
    momWrite: GPUTexture,
    histWrite: GPUTexture,
    textureViewFor: TextureViewFor,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: 'svgf-real-moments-bg',
      layout: this._momentsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textureViewFor(momWrite) },
        { binding: 1, resource: textureViewFor(histWrite) },
        { binding: 2, resource: textureViewFor(svgf.svgfVarianceMomentsIntermedTexture) },
      ],
    });
  }

  private _buildFallbackBindGroup(
    device: GPUDevice,
    common: DenoiserDispatchContext['resources']['common'],
    svgf: DenoiserDispatchContext['resources']['svgf'],
    histWrite: GPUTexture,
    textureViewFor: TextureViewFor,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: 'svgf-real-7x7-bg',
      layout: this._fallbackPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: textureViewFor(common.hdrColorTexture) },
        { binding: 1, resource: textureViewFor(histWrite) },
        { binding: 2, resource: textureViewFor(svgf.svgfVarianceMomentsIntermedTexture) },
        { binding: 3, resource: textureViewFor(svgf.svgfVarianceTexture) },
        { binding: 4, resource: textureViewFor(common.gNormalDepthTexture) },
        { binding: 5, resource: textureViewFor(common.gNormalDepthTexture) },
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
      resourceCache,
      isMoving,
      atrousDirectSigmas,
    } = ctx;
    const common = resources.common;
    const svgf = resources.svgf;
    const textureViewFor: TextureViewFor = resourceCache
      ? (texture) => resourceCache.textureView(texture)
      : (texture) => texture.createView();
    const forceReset = shouldResetDenoiserHistory(ctx.frameIndex, isMoving) ? 1 : 0;
    this._writeReprojUniforms(device, forceReset);

    // ── Pass 1: Reprojection ─────────────────────────────────────────────
    // Bindings follow svgfReprojection.wgsl.ts binding declarations (0..14).
    // For the walkaround-hybrid pipeline, currDepth + currNormal come from
    // gNormalDepthTexture (.w = signed depth, .xyz = normal packed 0..1).
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
    const currObjIdTexture = svgf.svgfCurrentObjectIdTexture;
    const prevObjIdTexture = svgf.svgfPreviousObjectIdTexture;

    {
      const buildBg = (): GPUBindGroup => this._buildReprojBindGroup(
        device, common, svgf,
        radRead, radWrite,
        histRead, histWrite,
        momRead, momWrite,
        currObjIdTexture, prevObjIdTexture,
        textureViewFor,
      );
      const bg = cachedBindGroup(resourceCache, 'denoiser:svgf-real:reproj', [
        common.hdrColorTexture,
        radRead,
        common.motionVectorTexture,
        common.gNormalDepthTexture,
        currObjIdTexture,
        svgf.svgfPrevNormalDepthTexture,
        prevObjIdTexture,
        histRead,
        momRead,
        radWrite,
        histWrite,
        momWrite,
        this._reprojUboRef.buf,
      ], buildBg);
      const pass = encoder.beginComputePass(computeDesc('svgf-real-reproj'));
      pass.setPipeline(this._reprojPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 2: Variance from moments ────────────────────────────────────
    // Reads momWrite (just written by reproj) and histWrite.
    {
      const buildBg = (): GPUBindGroup => this._buildMomentsBindGroup(
        device,
        svgf,
        momWrite,
        histWrite,
        textureViewFor,
      );
      const bg = cachedBindGroup(resourceCache, 'denoiser:svgf-real:moments', [
        momWrite,
        histWrite,
        svgf.svgfVarianceMomentsIntermedTexture,
      ], buildBg);
      const pass = encoder.beginComputePass(computeDesc('svgf-real-moments'));
      pass.setPipeline(this._momentsPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 3: 7×7 spatial fallback ─────────────────────────────────────
    {
      const buildBg = (): GPUBindGroup => this._buildFallbackBindGroup(
        device,
        common,
        svgf,
        histWrite,
        textureViewFor,
      );
      const bg = cachedBindGroup(resourceCache, 'denoiser:svgf-real:fallback', [
        common.hdrColorTexture,
        histWrite,
        svgf.svgfVarianceMomentsIntermedTexture,
        svgf.svgfVarianceTexture,
        common.gNormalDepthTexture,
      ], buildBg);
      const pass = encoder.beginComputePass(computeDesc('svgf-real-7x7'));
      pass.setPipeline(this._fallbackPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // History/moments/radiance become the next read side only after the frame
    // submission is accepted. The wavelet chain below replaces radWrite's
    // reprojection output with its first filtered level before that happens.
    const nextPingPong = 1 - this._pingPong;
    publishFrameState(ctx.publication, () => {
      this._pingPong = nextPingPong;
    });

    // ── Pass 4: variance-propagating real-SVGF à-trous chain ─────────────
    // Feed the EMA-blended reprojection output (radWrite) as the starting color.
    // Ping-pong with denoisedPing/Pong as usual.
    const sa = this._atrousPipeline;
    const atrousUboBytes = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    const varView = resourceCache?.textureView(svgf.svgfVarianceTexture)
      ?? svgf.svgfVarianceTexture.createView();
    const denoised = runAtrousChain(encoder, sa, {
      iterations: SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
      startTex: radWrite,
      pingTex: common.denoisedPingTexture,
      pongTex: common.denoisedPongTexture,
      wgX: wgX16,
      wgY: wgY16,
      computeDesc,
      ...(resourceCache ? { textureViewFor: (texture: GPUTexture) => resourceCache.textureView(texture) } : {}),
      // Each iteration has a separate persistent UBO, so encoded dispatches
      // keep their own constants while avoiding per-frame GPUBuffer churn.
      bindGroupFor: (iter, inputView, outputView, inputTex, outputTex) => {
        packAtrousVarianceAtrousUniforms(
          {
            iteration: iter,
            sigmaNormal: atrousDirectSigmas[0],
            sigmaDepth: atrousDirectSigmas[1],
            sigmaColor: atrousDirectSigmas[2],
          },
          atrousUboBytes,
          0,
        );
        const iterUbo = this._atrousUboRefs[iter]?.buf;
        if (iterUbo == null) {
          throw new Error(`SVGFRealDenoiser: missing atrous UBO for iteration ${iter}`);
        }
        device.queue.writeBuffer(iterUbo, 0, atrousUboBytes);
        const buildBg = (): GPUBindGroup => buildAtrousVarianceAtrousBindGroup(
          device, sa,
          inputView, outputView,
          gNormalDepthView, varView,
          iterUbo,
          `svgf-real-atrous-bg-${iter}`,
        );
        return cachedBindGroup(resourceCache, `denoiser:svgf-real:atrous:${iter}`, [
          iterUbo,
          inputTex,
          outputTex,
          common.gNormalDepthTexture,
          svgf.svgfVarianceTexture,
        ], buildBg);
      },
      // Schied 2017 §4.3 feeds the first à-trous level—not the raw temporal
      // reprojection—back as next frame's color history. Preserve that level
      // in the write-side history texture before later wavelet levels reuse
      // the denoised ping-pong pair.
      afterIteration: (iter, _inputTex, outputTex) => {
        if (iter !== 0) return;
        encoder.copyTextureToTexture(
          { texture: outputTex },
          { texture: radWrite },
          {
            width: outputTex.width,
            height: outputTex.height,
            depthOrArrayLayers: 1,
          },
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
    encoder.copyTextureToTexture(
      { texture: currObjIdTexture },
      { texture: prevObjIdTexture },
      {
        width: currObjIdTexture.width,
        height: currObjIdTexture.height,
        depthOrArrayLayers: 1,
      },
    );
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
    this._lastForceReset = -1;
    this._pingPong = 0;
  }
}
