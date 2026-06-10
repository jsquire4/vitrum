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
 *        a. encoder.copyTextureToBuffer for hdrColor + albedo + normal-depth
 *           (we share `ctx.encoder` so the copy is folded into the same
 *           queue.submit as the frame's compute work — no extra submit).
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
 * flag clears so the next dispatch retries. The output texture stays
 * stale until a successful inference lands.
 *
 * GPU memory budget: 1 owned full-res rgba16float output texture
 * (≈16 MB at 1080p) + 3 transient readback buffers allocated lazily on
 * first dispatch (also ≈16 MB each at 1080p).
 */

import {
  preloadOIDNModel,
  denoiseFinal,
  releaseOIDNCacheEntry,
  type OIDNDenoiseInputs,
} from '@vitrum/shared-denoisers';
import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
  type DenoiserDispatchContext,
  type DenoiserInitContext,
  type DenoiserState,
} from './index.js';

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
}

// Local aliases routed to the canonical half-float and row-alignment helpers
// in @vitrum/shared-denoisers. Earlier revisions inlined these (~40 LOC); the
// canonical versions are byte-identical with one tiny exception:
// `alignedTextureCopyBytesPerRow(width, bpp)` does not impose the `max(256, …)`
// lower bound — for any width > 0 the result is already ≥ 256, so behaviour is
// equivalent. `decoupledF32` ensures the local call sites read like the
// inline originals.
import {
  float16BitsToFloat32 as f16ToF32,
  float32ToFloat16Bits as f32ToF16,
} from '@vitrum/shared-denoisers';
import { alignedTextureCopyBytesPerRow } from '@vitrum/shared-denoisers';

/**
 * Read 4 channels of a row-major rgba16float buffer into a Float32 RGB
 * (3-channel) layout matching {@link OIDNDenoiseInputs.color}. `decode`
 * is applied per-pixel post-extraction — used by the normal channel to
 * convert from `[0, 1]` packed normals back to `[-1, 1]`.
 */
function rgba16fBufferToRgbF32(
  src: ArrayBuffer,
  bytesPerRow: number,
  width: number,
  height: number,
  decode?: (r: number, g: number, b: number) => [number, number, number],
): Float32Array {
  const dst = new Float32Array(width * height * 3);
  const view = new DataView(src);
  for (let y = 0; y < height; y++) {
    const rowOff = y * bytesPerRow;
    for (let x = 0; x < width; x++) {
      const texOff = rowOff + x * 8; // 4 channels × 2 bytes
      const r = f16ToF32(view.getUint16(texOff,     true));
      const g = f16ToF32(view.getUint16(texOff + 2, true));
      const b = f16ToF32(view.getUint16(texOff + 4, true));
      const [or, og, ob] = decode ? decode(r, g, b) : [r, g, b];
      const dstIdx = (y * width + x) * 3;
      dst[dstIdx    ] = or;
      dst[dstIdx + 1] = og;
      dst[dstIdx + 2] = ob;
    }
  }
  return dst;
}

/**
 * Pack a Float32 RGB (HxWx3) buffer into rgba16float layout suitable
 * for `queue.writeTexture` into a `rgba16float` storage texture. The
 * alpha channel is set to 1.0.
 */
function rgbF32ToRgba16fRowAligned(
  src: Float32Array,
  width: number,
  height: number,
): { buffer: ArrayBuffer; bytesPerRow: number } {
  const bytesPerRow = alignedTextureCopyBytesPerRow(width, 8);
  // Allocate as ArrayBuffer (not ArrayBufferLike via `new Uint8Array(N).buffer`)
  // so the return type stays narrow enough for GPUAllowSharedBufferSource —
  // TS 5.5+ widens `Uint8Array<...>` to `Uint8Array<ArrayBufferLike>` which
  // GPUQueue.writeTexture's strict overload rejects.
  const buf = new ArrayBuffer(bytesPerRow * height);
  const view = new DataView(buf);
  const oneF16 = f32ToF16(1.0);
  for (let y = 0; y < height; y++) {
    const rowOff = y * bytesPerRow;
    for (let x = 0; x < width; x++) {
      const srcIdx = (y * width + x) * 3;
      const texOff = rowOff + x * 8;
      view.setUint16(texOff,     f32ToF16(src[srcIdx    ] ?? 0), true);
      view.setUint16(texOff + 2, f32ToF16(src[srcIdx + 1] ?? 0), true);
      view.setUint16(texOff + 4, f32ToF16(src[srcIdx + 2] ?? 0), true);
      view.setUint16(texOff + 6, oneF16,                           true);
    }
  }
  return { buffer: buf, bytesPerRow };
}

function errorReason(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

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
  /** True while initialize() is preloading the ONNX runtime/session. */
  private _warmupInFlight = false;
  /** Last async preload/inference failure, surfaced via state() until retry. */
  private _lastFailureReason: string | null = null;
  private _lastFailureRetryable = true;
  /** Disposed-flag — set in `dispose`. The background chain checks this
   *  after every await to bail out (and skip writes to destroyed textures). */
  private _disposed = false;
  /** Incremented on every resize(). The background inference chain captures
   *  this at the point of dispatch and checks it before writing the result
   *  back — if the generation has changed, the texture is a different size
   *  and the write would be a stale-size validation error. */
  private _resizeGeneration = 0;

  constructor(opts?: OIDNFinalDenoiserOptions) {
    // No-modelUrl construction registers as a `disabled` placeholder.
    // The HybridEngine constructor validates up-front, so reaching
    // initialize/dispatch on a placeholder would already be a bug — but
    // we keep the late-bound throw as a defence-in-depth.
    if (opts?.modelUrl === undefined || opts.modelUrl.length === 0) {
      this.disabled = true;
      this._modelUrl = '';
      this._executionProviders = undefined;
      return;
    }
    this.disabled = false;
    this._modelUrl = opts.modelUrl;
    this._executionProviders = opts.executionProviders;
  }

  async initialize(ctx: DenoiserInitContext): Promise<void> {
    if (this.disabled) {
      throw new Error(
        `[OIDNFinalDenoiser] cannot initialize a placeholder instance (no modelUrl supplied). ` +
        `Pass HybridEngineOptions.extensions['walkaround-hybrid'].oidnModelUrl to enable.`,
      );
    }
    this._device = ctx.device;
    this._width = ctx.width;
    this._height = ctx.height;
    this._disposed = false;
    this._lastFailureReason = null;

    // Owned output texture — full-res rgba16float, same layout / usage as
    // the SVGF-real / atrous-variance ping-pongs so downstream
    // temporalAccum can sample it identically.
    this._denoisedOutputTexture = ctx.device.createTexture({
      label: 'oidn-final-denoised-output',
      size: [ctx.width, ctx.height],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    });

    // Pre-warm the ONNX runtime + session so the first dispatch doesn't
    // pay the ~500 ms — 5 s "first run" cost on top of the inference cost.
    // The bridge caches the InferenceSession by modelUrl so subsequent
    // denoiseFinal calls skip session creation.
    this._warmupInFlight = true;
    try {
      await preloadOIDNModel({
        modelUrl: this._modelUrl,
        ...(this._executionProviders !== undefined
          ? { executionProviders: this._executionProviders }
          : {}),
      });
    } catch (err) {
      this._lastFailureReason = `OIDN preload failed: ${errorReason(err)}`;
      this._lastFailureRetryable = false;
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
    if (this._inFlight) {
      return { status: 'in-flight', reason: 'OIDN inference cycle in flight' };
    }
    if (this._device == null || this._denoisedOutputTexture == null) {
      return { status: 'warming-up', reason: 'OIDN denoiser is not initialized' };
    }
    if (!this._haveDenoisedOutput) {
      return { status: 'fallback', reason: 'waiting for first OIDN output' };
    }
    return DENOISER_READY_STATE;
  }

  dispatch(ctx: DenoiserDispatchContext): GPUTexture | null {
    if (!this._denoisedOutputTexture || !this._device) return null;

    // Kick off the background readback + inference + upload chain when no
    // inference is currently in flight. The chain reuses `ctx.encoder` for
    // the readback copy so the copy lands inside the same queue.submit as
    // the frame's compute work — no extra submission, no GPU stall.
    if (!this._inFlight && !this._disposed) {
      this._inFlight = true;
      this._lastFailureReason = null;
      void this._runInferenceCycle(ctx).finally(() => {
        this._inFlight = false;
      });
    }

    // Always return the owned output (stale by ≥1 frame). On the very
    // first frame — before any inference has completed — fall back to the
    // raw HDR target so downstream temporalAccum has a valid signal.
    if (this._haveDenoisedOutput) {
      return this._denoisedOutputTexture;
    }
    return ctx.resources.common.hdrColorTexture;
  }

  /**
   * Background pipeline: GPU readback → CPU decode → OIDN inference →
   * GPU upload. All four steps run async (the await boundaries are
   * explicit). Awaiting a queue submission is implicit — we let
   * `encoder` finish + submit normally as part of the frame's loop, then
   * map the readback buffers on the queue once that submit completes.
   */
  private async _runInferenceCycle(ctx: DenoiserDispatchContext): Promise<void> {
    const device = this._device!;
    const common = ctx.resources.common;
    const W = ctx.width;
    const H = ctx.height;
    // Capture the generation at dispatch time. If resize() is called before
    // the async chain reaches the writeTexture, the generation will have
    // changed and we abort instead of writing old-size data into the
    // new (different-size) texture (WebGPU validation error / stale frame).
    const dispatchGeneration = this._resizeGeneration;

    // Allocate transient readback buffers + queue the copies into the
    // current frame's encoder. WebGPU requires bytesPerRow to be a
    // multiple of 256 in copyTextureToBuffer, hence the alignment.
    const bytesPerRow = alignedTextureCopyBytesPerRow(W, 8); // rgba16float = 8 B / texel
    const readSize = bytesPerRow * H;

    const colorReadback = device.createBuffer({
      label: 'oidn-readback-color',
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const albedoReadback = device.createBuffer({
      label: 'oidn-readback-albedo',
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const normalReadback = device.createBuffer({
      label: 'oidn-readback-normal',
      size: readSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    ctx.encoder.copyTextureToBuffer(
      { texture: common.hdrColorTexture },
      { buffer: colorReadback, bytesPerRow },
      { width: W, height: H, depthOrArrayLayers: 1 },
    );
    ctx.encoder.copyTextureToBuffer(
      { texture: common.albedoTexture },
      { buffer: albedoReadback, bytesPerRow },
      { width: W, height: H, depthOrArrayLayers: 1 },
    );
    ctx.encoder.copyTextureToBuffer(
      { texture: common.gNormalDepthTexture },
      { buffer: normalReadback, bytesPerRow },
      { width: W, height: H, depthOrArrayLayers: 1 },
    );

    try {
      // mapAsync resolves once the GPU has finished writing the copy,
      // i.e. once the pipeline's queue.submit has flushed.
      await Promise.all([
        colorReadback.mapAsync(GPUMapMode.READ),
        albedoReadback.mapAsync(GPUMapMode.READ),
        normalReadback.mapAsync(GPUMapMode.READ),
      ]);

      if (this._disposed) return;

      const color = rgba16fBufferToRgbF32(
        colorReadback.getMappedRange().slice(0), bytesPerRow, W, H,
      );
      const albedo = rgba16fBufferToRgbF32(
        albedoReadback.getMappedRange().slice(0), bytesPerRow, W, H,
      );
      // gNormalDepth packs normals as xyz*0.5+0.5 in rgb; decode back to [-1,1].
      const normal = rgba16fBufferToRgbF32(
        normalReadback.getMappedRange().slice(0), bytesPerRow, W, H,
        (r, g, b) => [r * 2 - 1, g * 2 - 1, b * 2 - 1],
      );

      // Unmap before the await so the buffers don't block destruction
      // if the OIDN run takes a while.
      colorReadback.unmap();
      albedoReadback.unmap();
      normalReadback.unmap();

      const inputs: OIDNDenoiseInputs = {
        color,
        albedo,
        normal,
        width: W,
        height: H,
      };
      const denoised = await denoiseFinal(inputs, {
        modelUrl: this._modelUrl,
        ...(this._executionProviders !== undefined
          ? { executionProviders: this._executionProviders }
          : {}),
      });

      if (this._disposed || !this._denoisedOutputTexture) return;
      // Abort if resize() was called while we were awaiting — the output
      // texture is now a different size, so writing W×H into it is wrong.
      if (this._resizeGeneration !== dispatchGeneration) return;

      // Pad RGB → RGBA16F and upload back to the owned output texture.
      const { buffer, bytesPerRow: uploadBpr } = rgbF32ToRgba16fRowAligned(
        denoised, W, H,
      );
      device.queue.writeTexture(
        { texture: this._denoisedOutputTexture },
        buffer,
        { offset: 0, bytesPerRow: uploadBpr },
        { width: W, height: H, depthOrArrayLayers: 1 },
      );
      this._haveDenoisedOutput = true;
      this._lastFailureReason = null;
    } catch (err) {
      // Swallow + log. The stale output texture remains visible; the next
      // dispatch will retry. Hosts can detect persistent failure by
      // observing that `_haveDenoisedOutput` never flips true (no public
      // surface for this yet — W11 follow-up could expose a status hook).
      this._lastFailureReason = `OIDN inference cycle failed: ${errorReason(err)}`;
      this._lastFailureRetryable = true;
      console.error('[OIDNFinalDenoiser] inference cycle failed:', err);
    } finally {
      // Release the readback buffers — they're transient per-cycle.
      try { colorReadback.destroy(); } catch { /* already destroyed */ }
      try { albedoReadback.destroy(); } catch { /* already destroyed */ }
      try { normalReadback.destroy(); } catch { /* already destroyed */ }
    }
  }

  resize(width: number, height: number): void {
    // Tear down + reallocate the output texture at the new size. Bump the
    // generation counter so any in-flight inference cycle that captured
    // the previous generation aborts its writeTexture instead of writing
    // old-size data into the new (different-size) texture (which would be
    // a WebGPU validation error or a stale partial frame).
    this._width = width;
    this._height = height;
    this._resizeGeneration++;
    this._haveDenoisedOutput = false;
    if (this._denoisedOutputTexture && this._device) {
      try { this._denoisedOutputTexture.destroy(); } catch { /* ignore */ }
      this._denoisedOutputTexture = this._device.createTexture({
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
  }

  dispose(): void {
    this._disposed = true;
    this._warmupInFlight = false;
    this._inFlight = false;
    this._lastFailureReason = null;
    if (this._denoisedOutputTexture) {
      try { this._denoisedOutputTexture.destroy(); } catch { /* ignore */ }
      this._denoisedOutputTexture = null;
    }
    this._haveDenoisedOutput = false;
    if (!this.disabled && this._modelUrl.length > 0) {
      releaseOIDNCacheEntry({
        modelUrl: this._modelUrl,
        ...(this._executionProviders !== undefined
          ? { executionProviders: this._executionProviders }
          : {}),
      });
    }
    this._device = null;
  }
}
