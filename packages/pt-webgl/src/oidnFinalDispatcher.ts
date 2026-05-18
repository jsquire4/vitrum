/**
 * oidnFinalDispatcher.ts — internal kick-and-return state machine for the
 * `denoiser: 'oidn-final'` mode in {@link PTEngineWebGL2}.
 *
 * Wired by W11 follow-up (see plan/premium-grade-refactor-20260517.md §W11).
 * The HybridEngine W11 wire (feat/w11-oidn-wire on the walkaround pipeline)
 * is the sibling implementation; this module brings the same `'oidn-final'`
 * denoiser mode to the pt-webgl converged-frame path.
 *
 * ## Why a separate module
 *
 * The engine's render loop is synchronous (the host drives it from RAF and
 * reads the FrameOutput on the same tick). OIDN inference, by contrast, is
 * an async ONNX Runtime Web call that typically takes 50–500 ms on a 1080p
 * frame even on integrated GPUs. The two cadences don't compose without an
 * intermediary, so this dispatcher owns the bridge:
 *
 *  - The engine calls {@link OIDNFinalDispatcher.kickIfReady} once per frame
 *    when {@link FrameOutput.isConverged} flips true. The dispatcher reads
 *    back the HDR accumulator (sync GL readback — the converged frame is
 *    the only one we pay this cost on), spawns an async denoiseFinal()
 *    call, and stores the resulting Float32 RGB once the promise resolves.
 *
 *  - The engine queries {@link OIDNFinalDispatcher.getLatestDenoised} on
 *    every frame thereafter; the dispatcher returns the latest completed
 *    denoised image (or null while the first inference is still in flight).
 *
 * ## Color-only mode
 *
 * The current pt-webgl fork allocates `WebGLRenderTarget` (not MRT) for the
 * primary accumulator. Although `PhysicalPathTracingMaterial`'s shader
 * declares `gNormalDepth` and `gAlbedo` MRT outputs, the host-side render
 * target only captures location 0 (primary radiance); locations 1 and 2 are
 * harmlessly ignored by the driver. So this dispatcher runs OIDN in
 * color-only mode (matching the `oidn_rt_hdr.onnx` model variant) — no
 * albedo / normal aux inputs are read back.
 *
 * Exposing aux buffers at the pt-webgl host level requires fork changes to
 * `PathTracingRenderer` (allocate WebGLMultipleRenderTargets, plumb getters
 * for the new attachments). That is a separate scope and is intentionally
 * out of band for this wire — the OIDN bridge accepts color-only inputs
 * cleanly.
 *
 * ## Re-kick policy
 *
 * Once a converged-frame inference completes, the dispatcher will NOT
 * re-kick on subsequent converged frames unless the engine calls
 * {@link OIDNFinalDispatcher.invalidate} (typically: scene mutated, camera
 * moved → engine.reset() → invalidate). This avoids burning 100s of ms per
 * frame re-denoising a stable image while the user is just panning the UI.
 */

import type { WebGLRenderer, WebGLRenderTarget } from 'three';
import { readAccumulationRgbFloat } from './readbackHdr.js';

/**
 * Construction-time configuration for {@link OIDNFinalDispatcher}.
 *
 * `modelUrl` is required. `executionProviders`, if provided, overrides the
 * shared-denoisers default (`['webnn', 'webgpu', 'wasm']`) — useful for
 * pinning a specific provider in deterministic tests.
 */
export interface OIDNFinalDispatcherOptions {
  /**
   * URL or path to the bundled OIDN ONNX model file (e.g.
   * `'/models/oidn_rt_hdr.onnx'`). The host owns provisioning the file.
   * Required; the dispatcher throws at construction if absent.
   */
  readonly modelUrl: string;

  /**
   * Optional override of the ONNX Runtime Web execution providers, in
   * priority order. Forwarded verbatim to `denoiseFinal()`.
   */
  readonly executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
}

/**
 * Latest completed denoised image, returned by
 * {@link OIDNFinalDispatcher.getLatestDenoised}.
 *
 * Layout matches `OIDNDenoiseInputs.color`: row-major, interleaved RGB
 * (`index = (row * width + col) * 3 + channel`), 3 channels.
 */
export interface DenoisedFrame {
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;
}

/**
 * Minimal surface of the `@vitrum/shared-denoisers` OIDN bridge that this
 * dispatcher depends on. Extracted to an interface so tests can supply a
 * mock without coupling to the live ONNX Runtime Web import chain.
 */
export interface OIDNBridgeLike {
  readonly denoiseFinal: (
    inputs: {
      color: Float32Array;
      normal?: Float32Array;
      albedo?: Float32Array;
      width: number;
      height: number;
    },
    opts: {
      modelUrl: string;
      executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
    },
  ) => Promise<Float32Array>;
  readonly preloadOIDNModel?: (opts: {
    modelUrl: string;
    executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  }) => Promise<void>;
  readonly clearOIDNCache?: () => void;
}

/**
 * Wraps the bridge import in a typed lazy loader so the dispatcher remains
 * test-friendly. Production callers pass `undefined` (the dispatcher resolves
 * the import lazily via dynamic `import('@vitrum/shared-denoisers')`); tests
 * pass a synthetic bridge implementing {@link OIDNBridgeLike}.
 */
export type OIDNBridgeLoader = () => Promise<OIDNBridgeLike>;

const _defaultLoader: OIDNBridgeLoader = async () => {
  // Dynamic import keeps the bridge module's onnxruntime-web peer dep
  // out of the synchronous pt-webgl bundle path — hosts that never select
  // 'oidn-final' don't pay the bridge's module-load cost.
  const mod = await import('@vitrum/shared-denoisers');
  return {
    denoiseFinal: mod.denoiseFinal,
    preloadOIDNModel: mod.preloadOIDNModel,
    clearOIDNCache: mod.clearOIDNCache,
  };
};

export class OIDNFinalDispatcher {
  readonly #modelUrl: string;
  readonly #executionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  readonly #loader: OIDNBridgeLoader;

  /** True while an OIDN inference promise is unresolved. Re-kick attempts
   *  during this window are no-ops. */
  #inFlight = false;
  /** True once at least one inference has completed for the current
   *  invalidation-cohort. Reset to false on {@link invalidate}. */
  #haveCompleted = false;
  /** Latest completed denoised image. Null until the first inference of
   *  the current cohort resolves, or after {@link invalidate}. */
  #latest: DenoisedFrame | null = null;
  /** Set on {@link dispose}. After dispose the dispatcher refuses to
   *  kick, refuses to store new results, and clears the bridge cache. */
  #disposed = false;
  /** Cached bridge — resolved on first kick; subsequent kicks skip the
   *  import. Module-level cache on the bridge side keeps the
   *  InferenceSession warm across cycles. */
  #bridge: OIDNBridgeLike | null = null;

  constructor(opts: OIDNFinalDispatcherOptions, loader?: OIDNBridgeLoader) {
    if (opts.modelUrl === undefined || opts.modelUrl.length === 0) {
      throw new Error(
        '[OIDNFinalDispatcher] modelUrl is required. ' +
          "Pass EngineOptions.extensions['vitrum.ptWebgl.oidnModelUrl'] when constructing the engine with denoiser: 'oidn-final'.",
      );
    }
    this.#modelUrl = opts.modelUrl;
    this.#executionProviders = opts.executionProviders;
    this.#loader = loader ?? _defaultLoader;
  }

  /**
   * Returns the most recently completed denoised image for the current
   * invalidation cohort, or null when no inference has yet completed
   * since the last {@link invalidate} call.
   */
  getLatestDenoised(): DenoisedFrame | null {
    return this.#latest;
  }

  /** True iff an inference is currently unresolved. Diagnostic only. */
  isInFlight(): boolean {
    return this.#inFlight;
  }

  /**
   * Clear the latest result and arm the dispatcher to re-kick on the next
   * {@link kickIfReady} call. The engine calls this on
   * {@link PTEngineWebGL2.reset}, {@link PTEngineWebGL2.setScene}, and
   * {@link PTEngineWebGL2.updateEnvironment} — any state change that
   * invalidates the accumulator also invalidates the denoised cache.
   *
   * An in-flight inference is allowed to complete, but the result is
   * dropped on resolve (the bumped {@link _cohortId} catches the race).
   */
  invalidate(): void {
    this.#cohortId += 1;
    this.#haveCompleted = false;
    this.#latest = null;
  }

  /**
   * Cohort token: every {@link invalidate} call bumps this; inferences in
   * flight at bump time discard their result on resolve. Prevents a stale
   * inference from polluting the post-invalidation cohort.
   */
  #cohortId = 0;

  /**
   * Synchronous "kick the OIDN pipeline if needed" entrypoint. Called once
   * per frame from {@link PTEngineWebGL2.renderFrame} immediately after
   * the sample loop, when the frame is reported converged.
   *
   * Behavior:
   *  - If the dispatcher is disposed, this is a no-op.
   *  - If an inference is already in flight, this is a no-op.
   *  - If an inference has already completed for the current cohort, this
   *    is a no-op (the cached result is reused).
   *  - Otherwise: synchronously reads back the HDR accumulator
   *    (`readAccumulationRgbFloat`) and spawns an async inference. Returns
   *    immediately — the result lands asynchronously and is available via
   *    {@link getLatestDenoised} once resolved.
   *
   * The sync readback happens inside this call (one
   * `gl.readPixels` — typically 5–30 ms at 1080p). The async portion is
   * the OIDN inference itself; the dispatcher swallows its own errors and
   * logs to console.warn so the engine's render loop keeps running.
   */
  kickIfReady(
    renderer: WebGLRenderer,
    target: WebGLRenderTarget,
    width: number,
    height: number,
    divideByAlpha: boolean,
  ): void {
    if (this.#disposed) return;
    if (this.#inFlight) return;
    if (this.#haveCompleted) return;
    if (width <= 0 || height <= 0) return;

    // Snapshot the cohort BEFORE the readback so a concurrent invalidate
    // call between readback and resolve drops the result cleanly.
    const cohortAtKick = this.#cohortId;

    // Sync GL readback — one readPixels into a Float32Array. We pay this
    // ~5–30 ms at 1080p exactly once per converged cohort, which is the
    // honest cost of bridging the GL-host / ONNX-CPU divide.
    let color: Float32Array;
    try {
      color = readAccumulationRgbFloat(renderer, target, width, height, divideByAlpha);
    } catch (err) {
      console.warn('[OIDNFinalDispatcher] readPixels failed — skipping denoise', err);
      return;
    }

    this.#inFlight = true;
    void this.#runInference(color, width, height, cohortAtKick).finally(() => {
      this.#inFlight = false;
    });
  }

  async #runInference(
    color: Float32Array,
    width: number,
    height: number,
    cohortAtKick: number,
  ): Promise<void> {
    try {
      if (this.#bridge == null) {
        this.#bridge = await this.#loader();
      }
      if (this.#disposed || this.#cohortId !== cohortAtKick) return;
      const opts = this.#executionProviders !== undefined
        ? { modelUrl: this.#modelUrl, executionProviders: this.#executionProviders }
        : { modelUrl: this.#modelUrl };
      const denoised = await this.#bridge.denoiseFinal(
        { color, width, height },
        opts,
      );
      if (this.#disposed || this.#cohortId !== cohortAtKick) return;
      this.#latest = { rgb: denoised, width, height };
      this.#haveCompleted = true;
    } catch (err) {
      console.warn(
        '[OIDNFinalDispatcher] denoiseFinal failed — install onnxruntime-web ' +
          'and verify the OIDN ONNX model URL is reachable.',
        err,
      );
    }
  }

  /**
   * Release any cached state. Calls the bridge's `clearOIDNCache` so the
   * ONNX InferenceSession is freed.
   *
   * After dispose, future {@link kickIfReady} calls are no-ops and
   * {@link getLatestDenoised} continues to return whatever was last
   * resolved (so a host that disposed the engine after a successful
   * denoise can still save the result).
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#bridge?.clearOIDNCache != null) {
      try {
        this.#bridge.clearOIDNCache();
      } catch {
        /* swallow — disposal must not throw */
      }
    }
  }
}
