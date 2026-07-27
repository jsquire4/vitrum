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
import { publishFrameState } from '../FramePublication.js';
import type { EngineWarning } from '@vitrum/core';
import type { InferenceGraph } from '../../neural/InferenceGraph.js';
import type { ModelWeights } from '../../neural/weights.js';
import { buildNeuralPackWgsl } from '../../shaders/neuralPack.wgsl.js';
import { buildNeuralUnpackWgsl } from '../../shaders/neuralUnpack.wgsl.js';
import { preprocessingContractForCheckpoint } from '../../neural/preprocessing.js';
import { withNeuralGpuErrorScopes } from '../../neural/gpuValidation.js';
import {
  WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
  assertWalkaroundNeuralDenoiserShape,
  walkaroundNeuralDenoiserShapeError,
} from '../../neural/shapeContract.js';
import {
  NEURAL_F32_TENSOR_STORAGE,
  resolveNeuralTensorStorage,
  type NeuralTensorStorageContract,
} from '../../neural/tensorPrecision.js';


type NeuralLifecycleState = 'idle' | 'initializing' | 'ready' | 'failed' | 'disposed';

interface NeuralTensorBuffers {
  noisyBuf: GPUBuffer;
  albedoBuf: GPUBuffer;
  normalsBuf: GPUBuffer;
  outputBuf: GPUBuffer;
  outputTex: GPUTexture;
  width: number;
  height: number;
  storage: NeuralTensorStorageContract;
}

interface NeuralWrapperCandidate extends NeuralTensorBuffers {
  packPipeline: GPUComputePipeline;
  unpackPipeline: GPUComputePipeline;
  packParamsBuf: GPUBuffer;
  unpackParamsBuf: GPUBuffer;
}
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
  private _loggedUnsupportedShape = false;
  private _graphReinitGeneration = 0;
  private _graphReinitPromise: Promise<void> | null = null;
  private _graphReinitChain: Promise<void> = Promise.resolve();
  private _graphReinitReason: string | null = null;
  private _lifecycleState: NeuralLifecycleState = 'idle';
  private _resizeFailure: {
    readonly width: number;
    readonly height: number;
    readonly reason: string;
  } | null = null;
  private _lifecycleGeneration = 0;
  private _failureReason: string | null = null;
  private _disposed = false;

  private _device: GPUDevice | null = null;
  private _packPipeline: GPUComputePipeline | null = null;
  private _unpackPipeline: GPUComputePipeline | null = null;
  private _packParamsBuf: GPUBuffer | null = null;
  private _unpackParamsBuf: GPUBuffer | null = null;

  /** The four GPU tensor buffers + output texture, grouped so they are
   *  allocated, checked, and destroyed together.  Null until the first
   *  `_allocTensorBuffers` call (i.e. until `initialize` or `dispatch`). */
  private _tensorBuffers: NeuralTensorBuffers | null = null;

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
    if (this._lifecycleState === 'disposed') {
      throw new Error('[NeuralDenoiser] cannot initialize after dispose');
    }
    if (this._inferenceGraph == null) {
      this._lastFallbackReason = 'inference graph not supplied';
      return;
    }
    assertWalkaroundNeuralDenoiserShape(ctx.width, ctx.height);

    const generation = ++this._lifecycleGeneration;
    const hadReadyGeneration = this._lifecycleState === 'ready';
    this._lifecycleState = 'initializing';
    this._failureReason = null;

    try {
      const candidate = await withNeuralGpuErrorScopes(
        ctx.device,
        `NeuralDenoiser ${ctx.width}x${ctx.height} generation ${generation}`,
        () => this._buildWrapperCandidate(ctx.device, ctx.width, ctx.height),
        disposeWrapperCandidate,
      );
      if (this._isDisposed() || generation !== this._lifecycleGeneration) {
        disposeWrapperCandidate(candidate);
        throw new Error(`[NeuralDenoiser] generation ${generation} was superseded before publication`);
      }
      const graphOwns = (this._inferenceGraph as InferenceGraph & {
        owns?: (device: GPUDevice, width: number, height: number) => boolean;
      }).owns;
      if (typeof graphOwns === 'function' &&
          !graphOwns.call(this._inferenceGraph, ctx.device, ctx.width, ctx.height)) {
        disposeWrapperCandidate(candidate);
        throw new Error('[NeuralDenoiser] supplied inference graph does not own the initialization device and dimensions');
      }
      if (candidate.storage.precision !== this._inferenceGraph.tensorStorage.precision) {
        disposeWrapperCandidate(candidate);
        throw new Error('[NeuralDenoiser] wrapper and inference graph resolved different tensor precision');
      }

      const previousTensors = this._tensorBuffers;
      const previousPackParams = this._packParamsBuf;
      const previousUnpackParams = this._unpackParamsBuf;
      this._device = ctx.device;
      this._width = ctx.width;
      this._height = ctx.height;
      this._graphW = ctx.width;
      this._graphH = ctx.height;
      this._packPipeline = candidate.packPipeline;
      this._unpackPipeline = candidate.unpackPipeline;
      this._packParamsBuf = candidate.packParamsBuf;
      this._unpackParamsBuf = candidate.unpackParamsBuf;
      this._tensorBuffers = candidate;
      this._lastFallbackReason = null;
      this._failureReason = null;
      this._lifecycleState = 'ready';
      this._resizeFailure = null;

      destroyTensorBuffers(previousTensors);
      destroyBuffer(previousPackParams);
      destroyBuffer(previousUnpackParams);
    } catch (err) {
      if (generation === this._lifecycleGeneration && !this._isDisposed()) {
        this._failureReason = `neural denoiser initialization failed: ${errorMessage(err)}`;
        this._lastFallbackReason = this._failureReason;
        this._lifecycleState = hadReadyGeneration ? 'ready' : 'failed';
      }
      throw err;
    }
  }

  state(): DenoiserState {
    if (this.disabled) {
      return { status: 'fallback', reason: 'inference graph not supplied' };
    }
    if (this._lifecycleState === 'disposed') {
      return { status: 'failed', reason: 'neural denoiser has been disposed', retryable: false };
    }
    if (this._lifecycleState === 'initializing' || this._graphReinitPromise != null) {
      return {
        status: 'warming-up',
        reason: this._graphReinitReason ?? 'neural denoiser initializing',
      };
    }
    if (this._lifecycleState === 'failed') {
      return {
        status: 'failed',
        reason: this._failureReason ?? 'neural denoiser initialization failed',
        retryable: false,
      };
    }
    if (
      this._lifecycleState !== 'ready' ||
      this._device == null ||
      this._packPipeline == null ||
      this._unpackPipeline == null ||
      this._packParamsBuf == null ||
      this._unpackParamsBuf == null ||
      this._tensorBuffers == null
    ) {
      return { status: 'failed', reason: 'neural denoiser is not initialized', retryable: true };
    }
    return DENOISER_READY_STATE;
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture | null {
    if (this._inferenceGraph == null) {
      this._lastFallbackReason = 'inference graph not supplied';
      return null;
    }
    if (this._lifecycleState === 'failed' || this._failureReason != null) {
      throw new Error(this._failureReason ?? 'neural denoiser is in a failed state');
    }
    if (this._lifecycleState === 'disposed') {
      throw new Error('neural denoiser has been disposed');
    }
    if (this._device != null && ctx.device !== this._device) {
      throw this._recordFailure('neural dispatch device does not match the initialized device');
    }
    if (walkaroundNeuralDenoiserShapeError(ctx.width, ctx.height) != null) {
      this._setUnsupportedShapeFailure(ctx.width, ctx.height, 'renderFrame');
      throw this._recordFailure(this._lastFallbackReason ?? 'unsupported neural render size');
    }

    if (
      this._lifecycleState !== 'ready' ||
      this._packPipeline == null ||
      this._unpackPipeline == null ||
      this._packParamsBuf == null ||
      this._unpackParamsBuf == null
    ) {
      throw this._recordFailure('neural denoiser is not initialized');
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
      if (this._resizeFailure?.width === ctx.width && this._resizeFailure.height === ctx.height) {
        throw new Error(`[NeuralDenoiser] ${this._resizeFailure.reason}`);
      }
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
        this._warn({
          code: 'walkaround-hybrid.neural-size-mismatch-failed',
          message:
            `[NeuralDenoiser] size changed from ${this._graphW}x${this._graphH} ` +
            `to ${ctx.width}x${ctx.height}; neural mode is now failed because ` +
            `no model weights were retained for graph reinitialization.`,
          details: {
            previousWidth: this._graphW,
            previousHeight: this._graphH,
            width: ctx.width,
            height: ctx.height,
            state: 'failed',
            missing: 'retained model weights',
          },
        });
      }
      throw this._recordFailure(this._lastFallbackReason);
    }
    const device = ctx.device;
    const tb = this._tensorBuffers;
    if (tb == null) {
      throw this._recordFailure('neural tensor buffers are not allocated');
    }
    if (tb.width !== ctx.width || tb.height !== ctx.height) {
      throw this._recordFailure(
        `neural tensor dimensions ${tb.width}x${tb.height} do not match dispatch ${ctx.width}x${ctx.height}`,
      );
    }
    // A successful encode does not clear fallback telemetry until the frame is
    // actually accepted. If a later pass/finish/submit fails, retry retains the
    // prior diagnostic and the same graph generation.
    publishFrameState(ctx.publication, () => {
      this._lastFallbackReason = null;
    });
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
        this._warn({
          code: 'walkaround-hybrid.neural-dispatch-failed',
          message: `[NeuralDenoiser] ${reason}; neural mode is now failed.`,
          details: {
            reason,
            width: ctx.width,
            height: ctx.height,
            state: 'failed',
          },
          raw: err,
        });
      }
      throw this._recordFailure(reason);
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
    if (this._lifecycleState === 'disposed') return;
    this._width = w;
    this._resizeFailure = null;
    this._height = h;
    this._loggedSizeMismatch = false;
    this._loggedDispatchFailure = false;
    this._loggedUnsupportedShape = false;

    const shapeError = walkaroundNeuralDenoiserShapeError(w, h);
    if (shapeError != null) {
      this._graphReinitGeneration++;
      this._graphReinitPromise = null;
      this._graphReinitReason = null;
      this._setUnsupportedShapeFailure(w, h, 'resize');
      this._recordFailure(this._lastFallbackReason ?? shapeError);
      return;
    }

    if (this._device == null) return;
    if (
      this._graphReinitPromise == null &&
      this._graphW === w &&
      this._graphH === h &&
      this._tensorBuffers?.width === w &&
      this._tensorBuffers.height === h
    ) {
      return;
    }
    if (this._modelWeights == null) {
      this._recordFailure(
        `cannot resize neural graph from ${this._graphW}x${this._graphH} to ${w}x${h} without retained model weights`,
      );
      return;
    }
    this._scheduleGraphReinitialize(w, h);
  }

  dispose(): void {
    this._disposed = true;
    this._lifecycleGeneration++;
    this._lifecycleState = 'disposed';
    this._failureReason = 'neural denoiser has been disposed';
    this._graphReinitGeneration++;
    this._graphReinitPromise = null;
    this._graphReinitReason = null;
    this._destroyTensorBuffers();
    this._resizeFailure = null;
    destroyBuffer(this._packParamsBuf);
    destroyBuffer(this._unpackParamsBuf);
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

    let candidate: NeuralTensorBuffers | null = null;
    const run = this._graphReinitChain
      .catch(() => undefined)
      .then(async () => {
        if (this._disposed || generation !== this._graphReinitGeneration) return;
        candidate = await withNeuralGpuErrorScopes(
          device,
          `NeuralDenoiser resize ${w}x${h} generation ${generation}`,
          () => this._createTensorBuffers(device, w, h),
          destroyTensorBuffers,
        );
        await graph.initialize(device, weights, w, h);
        if (this._disposed || generation !== this._graphReinitGeneration) {
          destroyTensorBuffers(candidate);
          candidate = null;
          return;
        }
        if (!graph.owns(device, w, h)) {
          throw new Error('candidate inference graph failed device/dimension ownership validation');
        }

        const previous = this._tensorBuffers;
        this._tensorBuffers = candidate;
        candidate = null;
        this._graphW = w;
        this._graphH = h;
        this._width = w;
        this._height = h;
        this._lastFallbackReason = null;
        this._failureReason = null;
        this._lifecycleState = 'ready';
        this._graphReinitReason = null;
        this._loggedSizeMismatch = false;
        this._resizeFailure = null;
        this._loggedGraphReinitFailure = false;
        destroyTensorBuffers(previous);
      })
      .catch((err: unknown) => {
        destroyTensorBuffers(candidate);
        candidate = null;
        if (this._disposed || generation !== this._graphReinitGeneration) return;
        const failureReason = `neural graph resize reinitialization failed: ${errorMessage(err)}`;
        this._resizeFailure = { width: w, height: h, reason: failureReason };
        this._lastFallbackReason = failureReason;
        this._failureReason = null;
        this._lifecycleState = 'ready';
        this._graphReinitReason = null;
        if (!this._loggedGraphReinitFailure) {
          this._loggedGraphReinitFailure = true;
          this._warn({
            code: 'walkaround-hybrid.neural-resize-reinit-failed',
            message: `[NeuralDenoiser] ${failureReason}; the previous neural generation remains ready.`,
            method: 'resize',
            details: {
              reason: failureReason,
              width: w,
              height: h,
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

  private _isDisposed(): boolean {
    return this._lifecycleState === 'disposed';
  }


  private _recordFailure(reason: string): Error {
    this._failureReason = reason;
    this._lastFallbackReason = reason;
    this._lifecycleState = 'failed';
    return new Error(`[NeuralDenoiser] ${reason}`);
  }
  private async _buildWrapperCandidate(
    device: GPUDevice,
    width: number,
    height: number,
  ): Promise<NeuralWrapperCandidate> {
    let packParamsBuf: GPUBuffer | null = null;
    let unpackParamsBuf: GPUBuffer | null = null;
    let tensors: NeuralTensorBuffers | null = null;
    try {
      const storage = this._tensorStorageForDevice(device);
      const packSM = device.createShaderModule({
        label: 'neural-denoiser-pack',
        code: buildNeuralPackWgsl(preprocessingContractForCheckpoint(this._modelWeights?.checkpoint), storage),
      });
      const unpackSM = device.createShaderModule({
        label: 'neural-denoiser-unpack',
        code: buildNeuralUnpackWgsl(preprocessingContractForCheckpoint(this._modelWeights?.checkpoint), storage),
      });
      const packPipeline = await device.createComputePipelineAsync({
        label: 'neural-denoiser-pack-pipeline',
        layout: 'auto',
        compute: { module: packSM, entryPoint: 'main' },
      });
      const unpackPipeline = await device.createComputePipelineAsync({
        label: 'neural-denoiser-unpack-pipeline',
        layout: 'auto',
        compute: { module: unpackSM, entryPoint: 'main' },
      });
      packParamsBuf = device.createBuffer({
        label: 'neural-denoiser-pack-params',
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      unpackParamsBuf = device.createBuffer({
        label: 'neural-denoiser-unpack-params',
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      tensors = this._createTensorBuffers(device, width, height);
      return {
        ...tensors,
        packPipeline,
        unpackPipeline,
        packParamsBuf,
        unpackParamsBuf,
      };
    } catch (err) {
      destroyTensorBuffers(tensors);
      destroyBuffer(packParamsBuf);
      destroyBuffer(unpackParamsBuf);
      throw err;
    }
  }

  private _createTensorBuffers(
    device: GPUDevice,
    w: number,
    h: number,
  ): NeuralTensorBuffers {
    assertWalkaroundNeuralDenoiserShape(w, h);
    const pixelCount = w * h;
    const storage = this._tensorStorageForDevice(device);
    const bytes = Math.max(4, pixelCount * 3 * storage.bytesPerScalar);
    const createdBuffers: GPUBuffer[] = [];
    let outputTex: GPUTexture | null = null;
    const mkStorage = (label: string): GPUBuffer => {
      const buffer = device.createBuffer({
        label,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      createdBuffers.push(buffer);
      return buffer;
    };
    try {
      const noisyBuf = mkStorage('neural-denoiser-noisy');
      const albedoBuf = mkStorage('neural-denoiser-albedo');
      const normalsBuf = mkStorage('neural-denoiser-normals');
      const outputBuf = mkStorage('neural-denoiser-output');
      outputTex = device.createTexture({
        label: 'neural-denoiser-output-texture',
        size: [w, h],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      return { noisyBuf, albedoBuf, normalsBuf, outputBuf, outputTex, width: w, height: h, storage };
    } catch (err) {
      try { outputTex?.destroy(); } catch { /* teardown is best-effort */ }
      for (const buffer of createdBuffers) destroyBuffer(buffer);
      throw err;
    }
  }

  private _tensorStorageForDevice(device: GPUDevice): NeuralTensorStorageContract {
    if (this._inferenceGraph != null) {
      return this._inferenceGraph.tensorStorage;
    }
    if (this._modelWeights != null) {
      return resolveNeuralTensorStorage(device, this._modelWeights);
    }
    return NEURAL_F32_TENSOR_STORAGE;
  }

  /** Destroy the current `_tensorBuffers` record (if any) and null it out. */
  private _destroyTensorBuffers(): void {
    destroyTensorBuffers(this._tensorBuffers);
    this._tensorBuffers = null;
  }

  /**
   * Allocate (or reallocate) the four tensor GPU buffers + output texture
   * for dimensions `(w, h)`. No-ops when the current record already matches.
   * Mirrors `bmfr.ts _allocHistory` in shape.
   */
  private _allocTensorBuffers(device: GPUDevice, w: number, h: number): void {
    assertWalkaroundNeuralDenoiserShape(w, h);
    if (
      this._tensorBuffers != null &&
      this._tensorBuffers.width === w &&
      this._tensorBuffers.height === h
    ) {
      return;
    }
    const next = this._createTensorBuffers(device, w, h);
    const previous = this._tensorBuffers;
    this._tensorBuffers = next;
    destroyTensorBuffers(previous);
  }

  private _setUnsupportedShapeFailure(
    width: number,
    height: number,
    method: 'renderFrame' | 'resize',
  ): void {
    const shapeError = walkaroundNeuralDenoiserShapeError(width, height) ?? 'unknown shape error';
    const reason = `unsupported neural internal render size ${width}x${height}: ${shapeError}`;
    this._lastFallbackReason = reason;
    if (this._loggedUnsupportedShape) return;
    this._loggedUnsupportedShape = true;
    this._warn({
      code: 'walkaround-hybrid.neural-unsupported-shape-failed',
      message: `[NeuralDenoiser] ${reason}; neural mode is now failed without allocating neural tensors.`,
      method,
      details: {
        width,
        height,
        requirement: WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
        state: 'failed',
        neuralAllocationAttempted: false,
      },
    });
  }


  private _warn(warning: {
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
        // Host warning callbacks must not break the denoiser lifecycle.
      }
      return;
    }
    console.warn(routed.message);
  }
}

function destroyBuffer(buffer: GPUBuffer | null): void {
  if (buffer == null) return;
  try { buffer.destroy(); } catch { /* teardown is best-effort */ }
}

function destroyTensorBuffers(tensors: NeuralTensorBuffers | null): void {
  if (tensors == null) return;
  try { tensors.outputTex.destroy(); } catch { /* teardown is best-effort */ }
  destroyBuffer(tensors.noisyBuf);
  destroyBuffer(tensors.albedoBuf);
  destroyBuffer(tensors.normalsBuf);
  destroyBuffer(tensors.outputBuf);
}

function disposeWrapperCandidate(candidate: NeuralWrapperCandidate): void {
  destroyTensorBuffers(candidate);
  destroyBuffer(candidate.packParamsBuf);
  destroyBuffer(candidate.unpackParamsBuf);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
