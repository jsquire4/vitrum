/**
 * NeuralDenoiser registry entry.
 *
 * Enabled iff the pipeline supplied a pre-initialized InferenceGraph.
 *
 * Dispatch path:
 *   1) pack current frame textures (noisy/albedo/normalDepth) into three
 *      f32 storage buffers
 *   2) run InferenceGraph on those buffers
 *   3) unpack denoised RGB buffer back into an rgba16float output texture
 *
 * The InferenceGraph itself remains owned by pipeline lifecycle code.
 */

import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
  type DenoiserState,
} from './index.js';
import type { EngineWarning } from '@vitrum/core';
import type { InferenceGraph } from '../../neural/InferenceGraph.js';
import type { ModelWeights } from '../../neural/weights.js';
import { NEURAL_PACK_WGSL } from '../../shaders/neuralPack.wgsl.js';
import { NEURAL_UNPACK_WGSL } from '../../shaders/neuralUnpack.wgsl.js';

export class NeuralDenoiser implements Denoiser {
  readonly id = 'neural' as const;
  readonly disabled: boolean;
  /** The 2 passes this denoiser dispatches: `neural-pack` (input-pack) +
   *  `neural-unpack` (output-unpack). The slot allocator inspects this even
   *  for `disabled` entries when buildPassLayout sizes the querySet. */
  readonly passLabels = DENOISER_PASS_LABELS['neural'];
  private readonly _inferenceGraph: InferenceGraph | undefined;
  private readonly _modelWeights: ModelWeights | undefined;
  private _width = 0;
  private _height = 0;
  /** Dimensions the InferenceGraph is currently initialized with. resize()
   *  schedules an async graph reinitialize when model weights are available;
   *  dispatch falls back while that reinitialize is in flight so the graph never
   *  receives buffers sized for a different resolution. */
  private _graphW = 0;
  private _graphH = 0;
  private _loggedSizeMismatch = false;
  private _loggedDispatchFailure = false;
  private _loggedGraphReinitFailure = false;
  private _graphReinitGeneration = 0;
  private _graphReinitPromise: Promise<void> | null = null;
  private _graphReinitChain: Promise<void> = Promise.resolve();
  private _graphReinitReason: string | null = null;
  private _disposed = false;

  private _device: GPUDevice | null = null;
  private _packPipeline: GPUComputePipeline | null = null;
  private _unpackPipeline: GPUComputePipeline | null = null;
  private _packParamsBuf: GPUBuffer | null = null;
  private _unpackParamsBuf: GPUBuffer | null = null;

  /** The four GPU tensor buffers + output texture, grouped so they are
   *  allocated, checked, and destroyed together.  Null until the first
   *  `_allocTensorBuffers` call (i.e. until `initialize` or `dispatch`). */
  private _tensorBuffers: {
    noisyBuf: GPUBuffer;
    albedoBuf: GPUBuffer;
    normalsBuf: GPUBuffer;
    outputBuf: GPUBuffer;
    outputTex: GPUTexture;
    width: number;
    height: number;
  } | null = null;

  private _lastFallbackReason: string | null = null;
  private readonly _onWarning: ((warning: EngineWarning) => void) | null;

  constructor(options?: {
    inferenceGraph?: InferenceGraph;
    modelWeights?: ModelWeights;
    onWarning?: (warning: EngineWarning) => void;
  }) {
    this._inferenceGraph = options?.inferenceGraph;
    this._modelWeights = options?.modelWeights;
    this._onWarning = options?.onWarning ?? null;
    this.disabled = this._inferenceGraph === undefined;
  }

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    this._disposed = false;
    if (this._inferenceGraph == null) {
      this._lastFallbackReason = 'inference graph not supplied';
      return;
    }
    this._width = ctx.width;
    this._height = ctx.height;
    // Record the InferenceGraph's fixed dims — these never change.
    this._graphW = ctx.width;
    this._graphH = ctx.height;
    const device = ctx.device;
    this._device = device;

    const packSM = device.createShaderModule({
      label: 'neural-denoiser-pack',
      code: NEURAL_PACK_WGSL,
    });
    const unpackSM = device.createShaderModule({
      label: 'neural-denoiser-unpack',
      code: NEURAL_UNPACK_WGSL,
    });
    this._packPipeline = await device.createComputePipelineAsync({
      label: 'neural-denoiser-pack-pipeline',
      layout: 'auto',
      compute: { module: packSM, entryPoint: 'main' },
    });
    this._unpackPipeline = await device.createComputePipelineAsync({
      label: 'neural-denoiser-unpack-pipeline',
      layout: 'auto',
      compute: { module: unpackSM, entryPoint: 'main' },
    });

    this._packParamsBuf = device.createBuffer({
      label: 'neural-denoiser-pack-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._unpackParamsBuf = device.createBuffer({
      label: 'neural-denoiser-unpack-params',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._allocTensorBuffers(device, ctx.width, ctx.height);
    this._lastFallbackReason = null;
  }

  state(): DenoiserState {
    if (this.disabled) {
      return { status: 'fallback', reason: 'inference graph not supplied' };
    }
    if (this._graphReinitPromise != null) {
      return {
        status: 'warming-up',
        reason: this._graphReinitReason ?? 'neural graph reinitializing for resized output',
      };
    }
    if (this._lastFallbackReason != null) {
      return { status: 'fallback', reason: this._lastFallbackReason };
    }
    if (
      this._device == null ||
      this._packPipeline == null ||
      this._unpackPipeline == null ||
      this._packParamsBuf == null ||
      this._unpackParamsBuf == null
    ) {
      return { status: 'fallback', reason: 'neural denoiser is not initialized' };
    }
    if (this._tensorBuffers == null) {
      return { status: 'fallback', reason: 'neural tensor buffers are not allocated' };
    }
    return DENOISER_READY_STATE;
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture | null {
    if (this._inferenceGraph == null) {
      this._lastFallbackReason = 'inference graph not supplied';
      return null;
    }
    if (
      this._packPipeline == null ||
      this._unpackPipeline == null ||
      this._packParamsBuf == null ||
      this._unpackParamsBuf == null
    ) {
      this._lastFallbackReason = 'neural denoiser is not initialized';
      return ctx.resources.common.hdrColorTexture;
    }
    if (this._graphReinitPromise != null) {
      this._lastFallbackReason = this._graphReinitReason ?? 'neural graph reinitializing for resized output';
      return ctx.resources.common.hdrColorTexture;
    }
    // Check against the graph's own dims (_graphW/_graphH), NOT just the
    // current pack/unpack buffer dims. If resize() was not called, but dispatch
    // arrives at a new size and weights are available, start the same in-place
    // graph reinitialize path instead of requiring engine recreation.
    if (ctx.width !== this._graphW || ctx.height !== this._graphH) {
      if (this._device != null && this._modelWeights != null) {
        this._scheduleGraphReinitialize(ctx.width, ctx.height);
        this._lastFallbackReason = this._graphReinitReason;
        return ctx.resources.common.hdrColorTexture;
      }
      this._lastFallbackReason =
        `size changed from ${this._graphW}x${this._graphH} to ` +
        `${ctx.width}x${ctx.height}; recreate engine to resize neural denoiser`;
      if (!this._loggedSizeMismatch) {
        this._loggedSizeMismatch = true;
        this._warnFallback({
          code: 'walkaround-hybrid.neural-size-mismatch-fallback',
          message:
            `[NeuralDenoiser] size changed from ${this._graphW}x${this._graphH} ` +
            `to ${ctx.width}x${ctx.height}; falling back to hdrColorTexture. ` +
            `No model weights were retained for in-place graph reinitialization.`,
          details: {
            previousWidth: this._graphW,
            previousHeight: this._graphH,
            width: ctx.width,
            height: ctx.height,
            fallback: 'hdrColorTexture',
            missing: 'retained model weights',
          },
        });
      }
      return ctx.resources.common.hdrColorTexture;
    }
    const device = ctx.device;
    this._allocTensorBuffers(device, ctx.width, ctx.height);
    const tb = this._tensorBuffers;
    if (tb == null) {
      this._lastFallbackReason = 'neural tensor buffers are not allocated';
      return ctx.resources.common.hdrColorTexture;
    }
    this._lastFallbackReason = null;
    const packPipeline = this._packPipeline;
    const unpackPipeline = this._unpackPipeline;
    const packParamsBuf = this._packParamsBuf;
    const unpackParamsBuf = this._unpackParamsBuf;
    const pixelCount = ctx.width * ctx.height;
    const params = new Uint32Array([ctx.width, ctx.height, pixelCount, 0]);
    device.queue.writeBuffer(packParamsBuf, 0, params);
    device.queue.writeBuffer(unpackParamsBuf, 0, params);

    const buildPackBg = (): GPUBindGroup => device.createBindGroup({
      label: 'neural-denoiser-pack-bg',
      layout: packPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: ctx.resourceCache?.textureView(ctx.resources.common.hdrColorTexture) ?? ctx.resources.common.hdrColorTexture.createView() },
        { binding: 1, resource: ctx.resourceCache?.textureView(ctx.resources.common.albedoTexture) ?? ctx.resources.common.albedoTexture.createView() },
        { binding: 2, resource: ctx.resourceCache?.textureView(ctx.resources.common.gNormalDepthTexture) ?? ctx.resources.common.gNormalDepthTexture.createView() },
        { binding: 3, resource: { buffer: tb.noisyBuf } },
        { binding: 4, resource: { buffer: tb.albedoBuf } },
        { binding: 5, resource: { buffer: tb.normalsBuf } },
        { binding: 6, resource: { buffer: packParamsBuf } },
      ],
    });
    const packBG = ctx.resourceCache?.bindGroup('denoiser:neural:pack', [
      ctx.resources.common.hdrColorTexture,
      ctx.resources.common.albedoTexture,
      ctx.resources.common.gNormalDepthTexture,
      tb.noisyBuf,
      tb.albedoBuf,
      tb.normalsBuf,
      packParamsBuf,
    ], buildPackBg) ?? buildPackBg();
    {
      const pass = ctx.encoder.beginComputePass(ctx.computeDesc('neural-pack'));
      pass.setPipeline(packPipeline);
      pass.setBindGroup(0, packBG);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / 256), 1, 1);
      pass.end();
    }

    try {
      this._inferenceGraph.run(
        tb.noisyBuf,
        tb.albedoBuf,
        tb.normalsBuf,
        tb.outputBuf,
        ctx.encoder,
      );
    } catch (err) {
      const reason = `inference graph dispatch failed: ${errorMessage(err)}`;
      this._lastFallbackReason = reason;
      if (!this._loggedDispatchFailure) {
        this._loggedDispatchFailure = true;
        this._warnFallback({
          code: 'walkaround-hybrid.neural-dispatch-failed',
          message: `[NeuralDenoiser] ${reason}; falling back to hdrColorTexture.`,
          details: {
            reason,
            width: ctx.width,
            height: ctx.height,
            fallback: 'hdrColorTexture',
          },
          raw: err,
        });
      }
      return ctx.resources.common.hdrColorTexture;
    }

    const buildUnpackBg = (): GPUBindGroup => device.createBindGroup({
      label: 'neural-denoiser-unpack-bg',
      layout: unpackPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tb.outputBuf } },
        { binding: 1, resource: ctx.resourceCache?.textureView(tb.outputTex) ?? tb.outputTex.createView() },
        { binding: 2, resource: { buffer: unpackParamsBuf } },
      ],
    });
    const unpackBG = ctx.resourceCache?.bindGroup('denoiser:neural:unpack', [
      tb.outputBuf,
      tb.outputTex,
      unpackParamsBuf,
    ], buildUnpackBg) ?? buildUnpackBg();
    {
      const pass = ctx.encoder.beginComputePass(ctx.computeDesc('neural-unpack'));
      pass.setPipeline(unpackPipeline);
      pass.setBindGroup(0, unpackBG);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / 256), 1, 1);
      pass.end();
    }
    return tb.outputTex;
  }

  resize(w: number, h: number): void {
    // Issue 1 fix: resize must leave the denoiser in a consistent allocated
    // state, not a torn-down intermediate state. If a device is available
    // (initialize has been called), _allocTensorBuffers handles both teardown +
    // realloc atomically and the denoiser is immediately ready to dispatch.
    // If no device yet (resize called before initialize), just update the
    // target dimensions so initialize uses the new size.
    this._width = w;
    this._height = h;
    this._loggedSizeMismatch = false;
    this._loggedDispatchFailure = false;
    this._lastFallbackReason = null;
    if (this._device != null) {
      this._allocTensorBuffers(this._device, w, h);
      if (this._modelWeights != null) {
        this._scheduleGraphReinitialize(w, h);
      }
    }
    // When _device is null (pre-initialize), buffers are already null and
    // _allocTensorBuffers will be called from dispatch/initialize — no torn-down
    // window because there was nothing allocated to begin with.
  }

  dispose(): void {
    this._disposed = true;
    this._graphReinitGeneration++;
    this._graphReinitPromise = null;
    this._graphReinitReason = null;
    this._destroyTensorBuffers();
    this._packParamsBuf?.destroy();
    this._unpackParamsBuf?.destroy();
    this._packParamsBuf = null;
    this._unpackParamsBuf = null;
    this._packPipeline = null;
    this._unpackPipeline = null;
    this._device = null;
    this._lastFallbackReason = 'neural denoiser has been disposed';
  }

  private _scheduleGraphReinitialize(w: number, h: number): void {
    if (
      this._inferenceGraph == null ||
      this._modelWeights == null ||
      this._device == null ||
      (this._graphReinitPromise == null && this._graphW === w && this._graphH === h)
    ) {
      return;
    }

    const generation = ++this._graphReinitGeneration;
    const graph = this._inferenceGraph;
    const weights = this._modelWeights;
    const device = this._device;
    const reason = `neural graph reinitializing for ${w}x${h}`;
    this._graphReinitReason = reason;
    this._lastFallbackReason = reason;

    const run = this._graphReinitChain
      .catch(() => undefined)
      .then(async () => {
        if (this._disposed || generation !== this._graphReinitGeneration) return;
        await graph.initialize(device, weights, w, h);
        if (this._disposed) {
          graph.dispose();
          return;
        }
        if (generation !== this._graphReinitGeneration) return;
        this._graphW = w;
        this._graphH = h;
        this._lastFallbackReason = null;
        this._graphReinitReason = null;
        this._loggedSizeMismatch = false;
        this._loggedGraphReinitFailure = false;
      })
      .catch((err: unknown) => {
        if (this._disposed || generation !== this._graphReinitGeneration) return;
        const failureReason = `neural graph resize reinitialization failed: ${errorMessage(err)}`;
        this._lastFallbackReason = failureReason;
        this._graphReinitReason = null;
        if (!this._loggedGraphReinitFailure) {
          this._loggedGraphReinitFailure = true;
          this._warnFallback({
            code: 'walkaround-hybrid.neural-resize-reinit-failed',
            message: `[NeuralDenoiser] ${failureReason}; falling back to hdrColorTexture.`,
            method: 'resize',
            details: {
              reason: failureReason,
              width: w,
              height: h,
              fallback: 'hdrColorTexture',
            },
            raw: err,
          });
        }
      })
      .finally(() => {
        if (generation === this._graphReinitGeneration) {
          this._graphReinitPromise = null;
        }
      });

    this._graphReinitPromise = run;
    this._graphReinitChain = run.catch(() => undefined);
  }

  /** Destroy the current `_tensorBuffers` record (if any) and null it out. */
  private _destroyTensorBuffers(): void {
    if (this._tensorBuffers == null) return;
    this._tensorBuffers.outputTex.destroy();
    this._tensorBuffers.noisyBuf.destroy();
    this._tensorBuffers.albedoBuf.destroy();
    this._tensorBuffers.normalsBuf.destroy();
    this._tensorBuffers.outputBuf.destroy();
    this._tensorBuffers = null;
  }

  /**
   * Allocate (or reallocate) the four tensor GPU buffers + output texture
   * for dimensions `(w, h)`. No-ops when the current record already matches.
   * Mirrors `bmfr.ts _allocHistory` in shape.
   */
  private _allocTensorBuffers(device: GPUDevice, w: number, h: number): void {
    if (
      this._tensorBuffers != null &&
      this._tensorBuffers.width === w &&
      this._tensorBuffers.height === h
    ) {
      return;
    }
    this._destroyTensorBuffers();
    const pixelCount = w * h;
    const bytes = Math.max(4, pixelCount * 3 * 4);
    const mkStorage = (label: string) =>
      device.createBuffer({
        label,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    this._tensorBuffers = {
      noisyBuf:   mkStorage('neural-denoiser-noisy'),
      albedoBuf:  mkStorage('neural-denoiser-albedo'),
      normalsBuf: mkStorage('neural-denoiser-normals'),
      outputBuf:  mkStorage('neural-denoiser-output'),
      outputTex:  device.createTexture({
        label: 'neural-denoiser-output-texture',
        size: [w, h],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }),
      width:  w,
      height: h,
    };
  }

  private _warnFallback(warning: {
    readonly code: string;
    readonly message: string;
    readonly method?: string;
    readonly details?: Readonly<Record<string, unknown>>;
    readonly raw?: unknown;
  }): void {
    const routed: EngineWarning = {
      code: warning.code,
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: warning.method ?? 'renderFrame',
      message: warning.message,
      ...(warning.details !== undefined ? { details: warning.details } : {}),
      ...(warning.raw !== undefined ? { raw: warning.raw } : {}),
    };
    if (this._onWarning) {
      try {
        this._onWarning(routed);
      } catch {
        // Host warning callbacks must not break denoiser fallback.
      }
      return;
    }
    console.warn(routed.message);
  }
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
