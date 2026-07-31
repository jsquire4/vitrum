/**
 * OIDNFinalDenoiser — Intel Open Image Denoise "final" pass via the
 * `@vitrum/shared-denoisers` `oidnBridge` (ONNX Runtime Web).
 *
 * Wired by W11 — see `plan/premium-grade-refactor-20260517.md §W11`.
 * Before W11 this was a `disabled: true` placeholder; the registry
 * rejected it at `lookup()` so a host could not select it.
 *
 * Layering — the bridge is CPU-side ONNX inference (5–20 MB lazy import
 * on first run; native ORT execution providers chosen per Decision 11:
 * 'webnn' → 'webgpu' → 'wasm'). It is **not** a real-time per-frame
 * filter — typical inference on the OIDN UNet runs at ~50–200 ms per
 * frame even on integrated GPUs, far above the 16 ms budget. The
 * intended use is one-shot denoising of a converged path-traced frame.
 *
 * Per-frame model — `dispatch` is synchronous and the Denoiser contract
 * requires it to return immediately, but OIDN inference is async. We
 * resolve that asymmetry by treating the inference as a background task:
 *
 *   1. On each `dispatch` call, if no inference is in flight, kick off a
 *      background pipeline:
 *        a. after the primary frame is submitted, enqueue a second ordered
 *           submission that copies hdrColor + albedo + normal-depth into
 *           transient readback buffers.
 *        b. await mapAsync on the three readback buffers.
 *        c. unpack rgba16float → Float32Array RGB (HxWx3), decode normal
 *           from `xyz*2 - 1`.
 *        d. await `denoiseFinal({ color, normal, albedo })`.
 *        e. Pad the Float32Array RGB back to RGBA + queue.writeTexture
 *           into the owned `_denoisedOutputTexture`.
 *        f. Clear the `_inFlight` flag.
 *
 *   2. `dispatch` returns the most recently completed denoised texture
 *      (or `ctx.resources.common.hdrColorTexture` on the first frame
 *      before any inference has completed). This is the "stale by N
 *      frames" trade — inevitable for a 50–200 ms operation in a
 *      ≥30 FPS pipeline.
 *
 * Errors during the background pipeline (ORT load failure, model file
 * missing, malformed model output) are caught and logged; the in-flight
 * flag clears so the next dispatch retries. A failed cycle returns raw HDR
 * rather than presenting an older denoised frame as current; successful
 * retry restores the denoised output.
 *
 * GPU memory budget: 1 owned full-res rgba16float output texture
 * (≈16 MB at 1080p) + 3 transient readback buffers allocated lazily on
 * first dispatch (also ≈16 MB each at 1080p).
 */

import {
  acquireOIDNSession,
  denoiseFinal,
  deriveOidnState,
  type OIDNDenoiseInputs,
  type OIDNSessionLease,
} from '@vitrum/shared-denoisers';
import type { EngineWarning } from '@vitrum/core';
import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
  type DenoiserState,
} from './index.js';
import type { PreparedSceneMutation } from '../../SceneMutationTransaction.js';
import { commitPreparedDenoiserResize } from './resizeTransaction.js';
import { publishFrameState } from '../FramePublication.js';
import { shouldResetDenoiserHistory } from './historyReset.js';

/**
 * Construction-time configuration for {@link OIDNFinalDenoiser}.
 *
 * Forwarded from `HybridEngineOptions.extensions['walkaround-hybrid'].oidnModelUrl`
 * (and, if specified, the matching execution-provider override). The
 * factory in `registerBuiltinDenoisers` constructs the denoiser with
 * these options when the engine is built with `denoiser: 'oidn-final'`.
 *
 * When `modelUrl` is undefined the denoiser is registered as a `disabled`
 * placeholder so {@link import('./index.js').DenoiserRegistry.ids} still
 * enumerates the slot for diagnostics — `lookup('oidn-final')` will throw
 * the canonical "registered but disabled" error pointing the host at the
 * `extensions['walkaround-hybrid'].oidnModelUrl` config key.
 */
export interface OIDNFinalDenoiserOptions {
  /**
   * URL or path to the bundled OIDN ONNX model. Host is responsible for
   * making the file available (bundled asset or fetched from a CDN).
   * Common convention: `'/models/oidn_rt_hdr_alb_nrm.onnx'` for the
   * albedo+normal+color UNet, or `'/models/oidn_rt_hdr.onnx'` for the
   * color-only variant.
   *
   * When undefined, the denoiser registers as `disabled: true` and any
   * `denoiser: 'oidn-final'` lookup throws — see class doc above.
   */
  readonly modelUrl?: string;

  /**
   * Optional override of the ONNX Runtime Web execution providers,
   * forwarded verbatim to {@link denoiseFinal}. Default order is
   * `['webnn', 'webgpu', 'wasm']` (Decision 11). Pass a single-element
   * array (`['wasm']`) to pin a provider for deterministic testing.
   */
  readonly executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;

  /**
   * Structured warning sink forwarded from HybridEngine.onWarning.
   *
   * OIDN inference runs asynchronously after dispatch returns, so a failed
   * inference cannot throw through renderFrame. The sink gives hosts the same
   * machine-readable degradation surface as the realtime denoisers while the
   * denoiser keeps returning the stale/HDR fallback texture.
   */
  readonly onWarning?: (warning: EngineWarning) => void;
}

import {
  alignedTextureCopyBytesPerRow,
  rgba16fBufferToRgbF32,
  rgbF32ToRgba16fRowAligned,
} from '@vitrum/shared-denoisers';

function errorReason(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Initial attempt plus two bounded automatic retries. Persistent failures
 * require engine recreation instead of performing full GPU readback + ONNX
 * inference on every render frame forever. */
const OIDN_MAX_AUTOMATIC_ATTEMPTS = 3;

export class OIDNFinalDenoiser implements Denoiser {
  readonly id = 'oidn-final' as const;
  readonly passLabels = DENOISER_PASS_LABELS['oidn-final'];
  /** Set to `true` only when the registration site supplied no
   *  `modelUrl`. The registry then rejects `lookup('oidn-final')` with
   *  the canonical "registered but disabled" error, and the late-bound
   *  initialize/dispatch path is never reached. */
  readonly disabled: boolean;

  private readonly _modelUrl: string;
  private readonly _executionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;

  // ── Lazy GPU state (allocated in `initialize`) ──────────────────────────
  private _device: GPUDevice | null = null;
  private _width = 0;
  private _height = 0;
  /** Owned full-res output texture — the result of the most recent OIDN
   *  inference. `dispatch` returns this once at least one inference has
   *  completed; before that it returns the raw HDR signal. */
  private _denoisedOutputTexture: GPUTexture | null = null;
  /** True once at least one inference has populated the output texture.
   *  While `false`, `dispatch` falls back to `ctx.resources.common.hdrColorTexture`. */
  private _haveDenoisedOutput = false;
  /** True while a background readback+inference+upload chain is in flight.
   *  Each `dispatch` call short-circuits if this is set, so concurrent
   *  kicks are not allowed. */
  private _inFlight = false;
  /** Frame inputs awaiting the primary frame submission. The post-submit hook
   *  consumes this exactly once and records copies in a separate encoder. */
  private _pendingReadback: DenoiserDispatchContext | null = null;
  /** True while initialize() is preloading the ONNX runtime/session. */
  private _warmupInFlight = false;
  /** Last async preload/inference failure, surfaced via state() until retry. */
  private _lastFailureReason: string | null = null;
  private _lastFailureRetryable = true;
  private _consecutiveFailureCount = 0;
  private readonly _onWarning: ((warning: EngineWarning) => void) | null;
  /** Disposed-flag — set in `dispose`. The background chain checks this
   *  after every await to bail out (and skip writes to destroyed textures). */
  private _disposed = false;
  /** Incremented on every resize(). The background inference chain captures
   *  this at the point of dispatch and checks it before writing the result
   *  back — if the generation has changed, the texture is a different size
   *  and the write would be a stale-size validation error. */
  private _resizeGeneration = 0;
  /**
   * Accumulation/content cohort. Camera motion and every frame-zero
   * accumulation reset (scene, lighting, material, or geometry mutation) bump
   * this value. Async work from an older cohort is discarded before upload so
   * a pre-mutation image can never replace the current frame.
   */
  private _contentGeneration = 0;
  /** Explicit ownership claim on the bridge's shared model session. */
  private _sessionLease: OIDNSessionLease | null = null;
  /** Cancels an initialize candidate when dispose wins an async race. */
  private _lifecycleGeneration = 0;
  /** Keep the lease alive until this denoiser's in-flight call has settled. */
  private _releaseSessionWhenIdle = false;

  constructor(opts?: OIDNFinalDenoiserOptions) {
    // No-modelUrl construction registers as a `disabled` placeholder.
    // The HybridEngine constructor validates up-front, so reaching
    // initialize/dispatch on a placeholder would already be a bug — but
    // we keep the late-bound throw as a defence-in-depth.
    if (opts?.modelUrl === undefined || opts.modelUrl.length === 0) {
      this.disabled = true;
      this._modelUrl = '';
      this._executionProviders = undefined;
      this._onWarning = opts?.onWarning ?? null;
      return;
    }
    this.disabled = false;
    this._modelUrl = opts.modelUrl;
    this._executionProviders = opts.executionProviders;
    this._onWarning = opts.onWarning ?? null;
  }

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    if (this.disabled) {
      throw new Error(
        `[OIDNFinalDenoiser] cannot initialize a placeholder instance (no modelUrl supplied). ` +
          `Pass HybridEngineOptions.extensions['walkaround-hybrid'].oidnModelUrl to enable.`,
      );
    }
    if (this._warmupInFlight) {
      throw new Error('[OIDNFinalDenoiser] initialize is already in progress.');
    }
    if (this._inFlight) {
      throw new Error('[OIDNFinalDenoiser] cannot initialize while inference is in flight.');
    }
    const lifecycleGeneration = ++this._lifecycleGeneration;
    this._disposed = false;
    const previousTexture = this._denoisedOutputTexture;
    const previousLease = this._sessionLease;
    let candidateTexture: GPUTexture | null = null;
    let candidateLease: OIDNSessionLease | null = null;

    // Pre-warm the ONNX runtime + session so the first dispatch doesn't
    // pay the ~500 ms — 5 s "first run" cost on top of the inference cost.
    // The bridge caches the InferenceSession by modelUrl so subsequent
    // denoiseFinal calls skip session creation.
    this._warmupInFlight = true;
    try {
      candidateTexture = OIDNFinalDenoiser._createOutputTexture(
        ctx.device,
        ctx.width,
        ctx.height,
      );
      if (candidateTexture === previousTexture) {
        // A replacement must be a distinct ownership object. Preserve the live
        // texture if a malformed device/mock aliases createTexture().
        candidateTexture = null;
        throw new Error('[OIDNFinalDenoiser] initialize candidate aliased the live output texture.');
      }
      candidateLease = await acquireOIDNSession({
        modelUrl: this._modelUrl,
        ...(this._executionProviders !== undefined
          ? { executionProviders: this._executionProviders }
          : {}),
      });
      if (this._disposed || this._lifecycleGeneration !== lifecycleGeneration) {
        throw new Error('[OIDNFinalDenoiser] initialize was cancelled by dispose.');
      }

      // Publish only after both GPU allocation and model acquisition succeed.
      this._device = ctx.device;
      this._width = ctx.width;
      this._height = ctx.height;
      this._denoisedOutputTexture = candidateTexture;
      this._sessionLease = candidateLease;
      this._resizeGeneration++;
      this._pendingReadback = null;
      this._haveDenoisedOutput = false;
      this._lastFailureReason = null;
      this._lastFailureRetryable = true;
      this._consecutiveFailureCount = 0;
      this._releaseSessionWhenIdle = false;
      candidateTexture = null;
      candidateLease = null;

      try { previousTexture?.destroy(); } catch { /* retired candidate */ }
      try { previousLease?.release(); } catch { /* best-effort retirement */ }
    } catch (err) {
      try { candidateTexture?.destroy(); } catch { /* partial allocation */ }
      try { candidateLease?.release(); } catch { /* partial acquisition */ }
      if (
        !this._disposed &&
        this._lifecycleGeneration === lifecycleGeneration &&
        previousTexture == null
      ) {
        this._lastFailureReason = `OIDN preload failed: ${errorReason(err)}`;
        this._lastFailureRetryable = true;
      }
      throw err;
    } finally {
      this._warmupInFlight = false;
    }
  }

  state(): DenoiserState {
    if (this.disabled) {
      return { status: 'fallback', reason: 'OIDN modelUrl not supplied' };
    }
    if (this._disposed) {
      return { status: 'fallback', reason: 'OIDN denoiser has been disposed' };
    }
    if (this._lastFailureReason != null) {
      return {
        status: 'failed',
        reason: this._lastFailureReason,
        retryable: this._lastFailureRetryable,
      };
    }
    if (this._warmupInFlight) {
      return { status: 'warming-up', reason: 'preloading OIDN model' };
    }
    if (this._device == null || this._denoisedOutputTexture == null) {
      return { status: 'warming-up', reason: 'OIDN denoiser is not initialized' };
    }
    // Reuse the shared OIDN status ladder for the in-flight / ready / fallback
    // tail (single source of truth with the pt-webgl2 / pt-webgpu wrappers via
    // `deriveOidnState`). The disabled / disposed / failed / warmup /
    // not-initialized branches above are wh-local ORCHESTRATION and have
    // already short-circuited, so `lastError` is null here. wh keeps its own
    // reason strings (pinned by denoiserState.test) — only the STATUS value is
    // shared. `_haveDenoisedOutput` is wh's cohort-completed flag.
    const derived = deriveOidnState({
      lastError: null,
      inFlight: this._inFlight,
      haveCompleted: this._haveDenoisedOutput,
    });
    switch (derived.status) {
      case 'in-flight':
        return { status: 'in-flight', reason: 'OIDN inference cycle in flight' };
      case 'ready':
        return DENOISER_READY_STATE;
      default:
        return { status: 'fallback', reason: 'waiting for first OIDN output' };
    }
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture | null {
    if (!this._denoisedOutputTexture || !this._device) return null;

    if (shouldResetDenoiserHistory(ctx.frameIndex, ctx.isMoving)) {
      this._contentGeneration++;
      this._pendingReadback = null;
      this._haveDenoisedOutput = false;
    }

    // Stage the background readback + inference + upload chain when idle.
    // afterFrameSubmit() records the copies only after the frame encoder has
    // been submitted, then relies on same-queue ordering before mapAsync
    // observes the copied bytes.
    if (
      !this._inFlight &&
      this._pendingReadback == null &&
      !this._disposed &&
      this._consecutiveFailureCount < OIDN_MAX_AUTOMATIC_ATTEMPTS
    ) {
      publishFrameState(ctx.publication, () => {
        // Re-check lifecycle state at the accepted boundary; resize/dispose are
        // synchronous but this keeps the ownership rule explicit.
        if (
          !this._inFlight &&
          this._pendingReadback == null &&
          !this._disposed &&
          this._consecutiveFailureCount < OIDN_MAX_AUTOMATIC_ATTEMPTS
        ) {
          this._pendingReadback = ctx;
        }
      });
    }

    // Return the latest completed output only while the denoiser is healthy.
    // Before the first success, and after any failed cycle, return raw HDR so
    // downstream passes never mistake a stale denoised frame for this frame.
    if (this._haveDenoisedOutput && this._lastFailureReason == null) {
      return this._denoisedOutputTexture;
    }
    return ctx.resources.common.hdrColorTexture;
  }

  afterFrameSubmit(): void {
    const ctx = this._pendingReadback;
    this._pendingReadback = null;
    if (ctx == null || this._inFlight || this._disposed || this._device == null) return;

    this._inFlight = true;
    void this._runInferenceCycle(ctx).finally(() => {
      this._inFlight = false;
      if (this._releaseSessionWhenIdle) {
        this._releaseSessionLease();
      }
    });
  }

  /**
   * Background pipeline: GPU readback → CPU decode → OIDN inference →
   * GPU upload.  Delegates the three async stages to the private helpers
   * below so each concern can be read and tested in isolation.
   *
   * The primary frame submit happens before this method is entered. Stage 1
   * submits the readback copies on the same queue; queue ordering guarantees
   * they observe the frame's texture writes before mapAsync resolves.
   */
  private async _runInferenceCycle(ctx: DenoiserDispatchContext): Promise<void> {
    const device = this._device!;
    const W = ctx.width;
    const H = ctx.height;
    // Capture the generation at dispatch time. If resize() is called before
    // the async chain reaches the writeTexture, the generation will have
    // changed and we abort instead of writing old-size data into the
    // new (different-size) texture (WebGPU validation error / stale frame).
    const dispatchGeneration = this._resizeGeneration;
    const contentGeneration = this._contentGeneration;

    let readbacks: ReturnType<OIDNFinalDenoiser['_readbackTextures']> | null = null;

    try {
      readbacks = this._readbackTextures(device, ctx, W, H);
      const { colorReadback, albedoReadback, normalReadback, bytesPerRow } = readbacks;
      const { color, albedo, normal } = await this._decodeReadbacks(
        colorReadback,
        albedoReadback,
        normalReadback,
        bytesPerRow,
        W,
        H,
      );

      if (this._disposed || this._contentGeneration !== contentGeneration) return;

      const inputs: OIDNDenoiseInputs = { color, albedo, normal, width: W, height: H };
      const denoised = await denoiseFinal(inputs, {
        modelUrl: this._modelUrl,
        ...(this._executionProviders !== undefined
          ? { executionProviders: this._executionProviders }
          : {}),
      });

      if (
        this._disposed ||
        this._contentGeneration !== contentGeneration ||
        !this._denoisedOutputTexture
      ) return;
      // Abort if resize() was called while we were awaiting — the output
      // texture is now a different size, so writing W×H into it is wrong.
      if (this._resizeGeneration !== dispatchGeneration) return;

      this._uploadResult(device, denoised, W, H);
      this._haveDenoisedOutput = true;
      this._lastFailureReason = null;
      this._lastFailureRetryable = true;
      this._consecutiveFailureCount = 0;
    } catch (err) {
      if (this._disposed || this._contentGeneration !== contentGeneration) return;
      // Swallow + report. Dispatch falls back to raw HDR until a later retry
      // succeeds. Hosts receive both a structured warning and the denoiser
      // state transition (`failed`, retryable) through FrameStats.
      const reason = `OIDN inference cycle failed: ${errorReason(err)}`;
      this._lastFailureReason = reason;
      this._consecutiveFailureCount += 1;
      this._lastFailureRetryable =
        this._consecutiveFailureCount < OIDN_MAX_AUTOMATIC_ATTEMPTS;
      this._warnInferenceFailure(reason, err, W, H);
    } finally {
      // Release the readback buffers — they're transient per-cycle.
      try {
        readbacks?.colorReadback.destroy();
      } catch {
        /* already destroyed */
      }
      try {
        readbacks?.albedoReadback.destroy();
      } catch {
        /* already destroyed */
      }
      try {
        readbacks?.normalReadback.destroy();
      } catch {
        /* already destroyed */
      }
    }
  }

  /**
   * Stage 1 — Allocate three transient readback buffers and submit their
   * `copyTextureToBuffer` commands after the primary frame submission.
   * Same-queue ordering makes the copies observe the completed frame writes.
   * WebGPU requires `bytesPerRow` to be a multiple of 256.
   */
  private _readbackTextures(
    device: GPUDevice,
    ctx: DenoiserDispatchContext,
    W: number,
    H: number,
  ): {
    colorReadback: GPUBuffer;
    albedoReadback: GPUBuffer;
    normalReadback: GPUBuffer;
    bytesPerRow: number;
  } {
    const bytesPerRow = alignedTextureCopyBytesPerRow(W, 8); // rgba16float = 8 B / texel
    const readSize = bytesPerRow * H;
    const common = ctx.resources.common;
    const encoder = device.createCommandEncoder({ label: 'oidn-post-submit-readback' });

    let colorReadback: GPUBuffer | null = null;
    let albedoReadback: GPUBuffer | null = null;
    let normalReadback: GPUBuffer | null = null;
    try {
      colorReadback = device.createBuffer({
        label: 'oidn-readback-color',
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      albedoReadback = device.createBuffer({
        label: 'oidn-readback-albedo',
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      normalReadback = device.createBuffer({
        label: 'oidn-readback-normal',
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      encoder.copyTextureToBuffer(
        { texture: common.hdrColorTexture },
        { buffer: colorReadback, bytesPerRow },
        { width: W, height: H, depthOrArrayLayers: 1 },
      );
      encoder.copyTextureToBuffer(
        { texture: common.albedoTexture },
        { buffer: albedoReadback, bytesPerRow },
        { width: W, height: H, depthOrArrayLayers: 1 },
      );
      encoder.copyTextureToBuffer(
        { texture: common.gNormalDepthTexture },
        { buffer: normalReadback, bytesPerRow },
        { width: W, height: H, depthOrArrayLayers: 1 },
      );
      device.queue.submit([encoder.finish()]);

      return { colorReadback, albedoReadback, normalReadback, bytesPerRow };
    } catch (err) {
      try {
        colorReadback?.destroy();
      } catch {
        /* already destroyed */
      }
      try {
        albedoReadback?.destroy();
      } catch {
        /* already destroyed */
      }
      try {
        normalReadback?.destroy();
      } catch {
        /* already destroyed */
      }
      throw err;
    }
  }

  /**
   * Stage 2 — `mapAsync` all three readback buffers, unpack rgba16float →
   * Float32 RGB, then unmap so the buffers can be destroyed while the OIDN
   * inference awaits.
   *
   * gNormalDepth packs normals as `xyz*0.5+0.5`; the decode lambda converts
   * back to `[-1, 1]` for the OIDN normal input.
   */
  private async _decodeReadbacks(
    colorReadback: GPUBuffer,
    albedoReadback: GPUBuffer,
    normalReadback: GPUBuffer,
    bytesPerRow: number,
    W: number,
    H: number,
  ): Promise<{ color: Float32Array; albedo: Float32Array; normal: Float32Array }> {
    // mapAsync resolves once the GPU has finished writing the copy,
    // i.e. once the ordered readback submission has completed.
    await Promise.all([
      colorReadback.mapAsync(GPUMapMode.READ),
      albedoReadback.mapAsync(GPUMapMode.READ),
      normalReadback.mapAsync(GPUMapMode.READ),
    ]);

    const color = rgba16fBufferToRgbF32(colorReadback.getMappedRange().slice(0), bytesPerRow, W, H);
    const albedo = rgba16fBufferToRgbF32(
      albedoReadback.getMappedRange().slice(0),
      bytesPerRow,
      W,
      H,
    );
    // gNormalDepth packs normals as xyz*0.5+0.5 in rgb; decode back to [-1,1].
    const normal = rgba16fBufferToRgbF32(
      normalReadback.getMappedRange().slice(0),
      bytesPerRow,
      W,
      H,
      (r, g, b) => [r * 2 - 1, g * 2 - 1, b * 2 - 1],
    );

    // Unmap before the caller awaits OIDN inference so the buffers don't
    // block destruction if the inference run takes a while.
    colorReadback.unmap();
    albedoReadback.unmap();
    normalReadback.unmap();

    return { color, albedo, normal };
  }

  /**
   * Stage 3 — Pad the denoised Float32 RGB result to rgba16float and
   * upload it to the owned `_denoisedOutputTexture` via `queue.writeTexture`.
   */
  private _uploadResult(device: GPUDevice, denoised: Float32Array, W: number, H: number): void {
    const { buffer, bytesPerRow: uploadBpr } = rgbF32ToRgba16fRowAligned(denoised, W, H);
    device.queue.writeTexture(
      { texture: this._denoisedOutputTexture! },
      buffer,
      { offset: 0, bytesPerRow: uploadBpr },
      { width: W, height: H, depthOrArrayLayers: 1 },
    );
  }

  private _warnInferenceFailure(reason: string, err: unknown, width: number, height: number): void {
    const warning: EngineWarning = {
      code: 'walkaround-hybrid.oidn-final-inference-failed',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'renderFrame',
      message: this._lastFailureRetryable
        ? `[OIDNFinalDenoiser] ${reason}; falling back to hdrColorTexture and retrying (bounded attempt ${this._consecutiveFailureCount + 1}/${OIDN_MAX_AUTOMATIC_ATTEMPTS}).`
        : `[OIDNFinalDenoiser] ${reason}; falling back to hdrColorTexture. Automatic retry budget exhausted; recreate the engine after correcting the model/provider configuration.`,
      details: {
        reason,
        modelUrl: this._modelUrl,
        width,
        height,
        fallback: 'hdrColorTexture',
        retryable: this._lastFailureRetryable,
        attempt: this._consecutiveFailureCount,
        maxAttempts: OIDN_MAX_AUTOMATIC_ATTEMPTS,
      },
      raw: err,
    };
    if (this._onWarning != null) {
      try {
        this._onWarning(warning);
      } catch {
        // Host warning callbacks must not break the denoiser retry path.
      }
      return;
    }
    console.error('[OIDNFinalDenoiser] inference cycle failed:', err);
  }

  prepareResize(width: number, height: number): PreparedSceneMutation {
    // Allocate before publication so a failed resize leaves the live output
    // texture, dimensions, generation, and in-flight inference state intact.
    const previous = this._denoisedOutputTexture;
    let replacement = previous;
    if (previous != null && this._device != null) {
      replacement = OIDNFinalDenoiser._createOutputTexture(this._device, width, height);
      if (replacement === previous) {
        throw new Error('OIDNFinalDenoiser.resize: replacement aliased the live output texture.');
      }
    }

    const previousWidth = this._width;
    const previousHeight = this._height;
    const previousResizeGeneration = this._resizeGeneration;
    const previousContentGeneration = this._contentGeneration;
    const previousPendingReadback = this._pendingReadback;
    const previousHaveDenoisedOutput = this._haveDenoisedOutput;
    let committed = false;
    let retired = false;
    let candidateOwned = replacement !== previous;
    return {
      commit: () => {
        if (committed) return;
        this._width = width;
        this._height = height;
        this._resizeGeneration = previousResizeGeneration + 1;
        this._contentGeneration = previousContentGeneration + 1;
        this._pendingReadback = null;
        this._haveDenoisedOutput = false;
        this._denoisedOutputTexture = replacement;
        committed = true;
      },
      rollback: () => {
        if (retired) return;
        if (committed) {
          this._width = previousWidth;
          this._height = previousHeight;
          this._resizeGeneration = previousResizeGeneration;
          this._contentGeneration = previousContentGeneration;
          this._pendingReadback = previousPendingReadback;
          this._haveDenoisedOutput = previousHaveDenoisedOutput;
          this._denoisedOutputTexture = previous;
          committed = false;
        }
        if (candidateOwned) {
          replacement?.destroy();
          candidateOwned = false;
        }
      },
      finalize: () => {
        if (!committed || retired) return;
        retired = true;
        candidateOwned = false;
        if (replacement !== previous) {
          try {
            previous?.destroy();
          } catch {
            // A retired texture must not invalidate the published generation.
          }
        }
      },
    };
  }

  resize(width: number, height: number): void {
    commitPreparedDenoiserResize(this.prepareResize(width, height));
  }

  /** Full-res rgba16float owned output texture (STORAGE|TEXTURE|COPY_DST|COPY_SRC).
   *  Single source of truth for the descriptor shared by `initialize` + `resize`. */
  private static _createOutputTexture(
    device: GPUDevice,
    width: number,
    height: number,
  ): GPUTexture {
    return device.createTexture({
      label: 'oidn-final-denoised-output',
      size: [width, height],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    });
  }

  dispose(): void {
    this._lifecycleGeneration++;
    this._disposed = true;
    this._pendingReadback = null;
    this._lastFailureReason = null;
    if (this._denoisedOutputTexture) {
      try {
        this._denoisedOutputTexture.destroy();
      } catch {
        /* ignore */
      }
      this._denoisedOutputTexture = null;
    }
    this._haveDenoisedOutput = false;
    if (this._inFlight) {
      this._releaseSessionWhenIdle = true;
    } else {
      this._releaseSessionLease();
    }
    this._device = null;
  }

  private _releaseSessionLease(): void {
    this._releaseSessionWhenIdle = false;
    const lease = this._sessionLease;
    this._sessionLease = null;
    try { lease?.release(); } catch { /* disposal must not throw */ }
  }
}
