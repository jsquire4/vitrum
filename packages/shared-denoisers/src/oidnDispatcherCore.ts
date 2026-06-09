/**
 * oidnDispatcherCore.ts — shared cohort state machine for converged-backend
 * `denoiser: 'oidn-final'` dispatchers.
 *
 * ## Motivation
 *
 * Two near-identical dispatchers previously lived in separate packages:
 *   - `packages/pt-webgpu/src/denoise/oidnFinalDispatcher.ts`
 *
 * They shared the interfaces, the `_defaultLoader` body, the cohort/async
 * state machine (private fields + invalidate/getLatestDenoised/isInFlight/
 * dispose), and the background inference cycle. This module extracts exactly
 * that shared core.
 *
 * ## What each backend keeps locally
 *
 * The two backends differ in two observable ways:
 *
 *  (a) **Readback input type.** pt-webgl does a synchronous
 *      `gl.readPixels` before even entering the async cycle; pt-webgpu does
 *      an async GPU → CPU readback inside the cycle. Both surfaces are unified
 *      behind a `ReadbackFn` callback: `(input: TInput) => Promise<ReadbackResult | null>`.
 *
 *  (b) **preloadOIDNModel on bridge init.** pt-webgpu calls
 *      `bridge.preloadOIDNModel` when the bridge is first loaded; pt-webgl
 *      does not. This behavioral difference is preserved via the
 *      `preloadOnBridgeInit` constructor option.
 *
 * ## Contract stability
 *
 * The public methods (invalidate / getLatestDenoised / isInFlight / dispose /
 * kickIfReady) behave exactly as before for each backend.
 */

/**
 * Construction-time configuration for {@link OIDNDispatcherCore} (and the
 * per-backend {@link OIDNFinalDispatcher} wrappers).
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
 * {@link OIDNDispatcherCore.getLatestDenoised}.
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
 * Minimal surface of the `@vitrum/shared-denoisers` OIDN bridge that the
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
  readonly releaseOIDNCacheEntry?: (opts: {
    modelUrl: string;
    executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  }) => void;
  /** @deprecated Prefer {@link releaseOIDNCacheEntry} for per-engine dispose. */
  readonly clearOIDNCache?: () => void;
}

/**
 * Wraps the bridge import in a typed lazy loader so the dispatcher remains
 * test-friendly. Production callers pass `undefined` (the dispatcher resolves
 * the import lazily via dynamic `import('@vitrum/shared-denoisers')`); tests
 * pass a synthetic bridge implementing {@link OIDNBridgeLike}.
 */
export type OIDNBridgeLoader = () => Promise<OIDNBridgeLike>;

export const _defaultLoader: OIDNBridgeLoader = async () => {
  // Dynamic import keeps the bridge module's onnxruntime-web peer dep
  // out of the synchronous bundle path — hosts that never select
  // 'oidn-final' don't pay the bridge's module-load cost.
  const mod = await import('./oidnBridge.js');
  return {
    denoiseFinal: mod.denoiseFinal,
    preloadOIDNModel: mod.preloadOIDNModel,
    releaseOIDNCacheEntry: mod.releaseOIDNCacheEntry,
    clearOIDNCache: mod.clearOIDNCache,
  };
};

/**
 * The normalised result type that a backend-supplied readback callback must
 * resolve to. Mirrors the shape passed to `bridge.denoiseFinal`.
 *
 * Return `null` to abort the inference cycle for this kick (readback failed
 * or was skipped — the dispatcher clears `#inFlight` and leaves
 * `#haveCompleted` false so the next kick attempt will retry).
 */
export interface ReadbackResult {
  readonly color: Float32Array;
  readonly albedo?: Float32Array;
  readonly normal?: Float32Array;
  readonly width: number;
  readonly height: number;
}

/**
 * Type of the async callback supplied by each backend.
 *
 * `TInput` is the backend-specific readback source:
 *  - pt-webgl: a pre-read `{ color: Float32Array; width: number; height: number }`
 *    (sync GL readback already done by `kickIfReady`)
 *  - pt-webgpu: `{ device: GPUDevice; sources: OidnTextureSources; width: number; height: number }`
 */
export type ReadbackFn<TInput> = (input: TInput) => Promise<ReadbackResult | null>;

/**
 * Options for constructing an {@link OIDNDispatcherCore}.
 */
export interface OIDNDispatcherCoreOptions<TInput> {
  /** OIDN model URL + optional execution-provider override. */
  readonly dispatcherOptions: OIDNFinalDispatcherOptions;
  /** Bridge loader (production callers omit this; tests pass a mock). */
  readonly loader?: OIDNBridgeLoader;
  /**
   * Backend-specific readback callback. Called inside the async inference
   * cycle with the input value passed to `kickIfReady`. A `null` return
   * aborts the cycle for this cohort tick.
   */
  readonly readback: ReadbackFn<TInput>;
  /**
   * If `true`, the dispatcher calls `bridge.preloadOIDNModel` when the
   * bridge is first loaded (before the first `denoiseFinal` call). pt-webgpu
   * sets this to `true`; pt-webgl sets it to `false`.
   *
   * This preserves the behavioral difference between the two backends.
   */
  readonly preloadOnBridgeInit: boolean;
}

/**
 * Generic cohort state machine shared by both OIDN final-pass dispatchers.
 *
 * `TInput` is the backend-specific kick-site input value (see
 * {@link OIDNDispatcherCoreOptions.readback} for details).
 *
 * @example
 * ```ts
 * // pt-webgpu thin wrapper:
 * const core = new OIDNDispatcherCore({
 *   dispatcherOptions: opts,
 *   loader,
 *   readback: async ({ device, sources, width, height }) =>
 *     readOidnInputsFromTextures(device, sources, width, height),
 *   preloadOnBridgeInit: true,
 * });
 * core.kickIfReady({ device, sources, width, height });
 * ```
 */
export class OIDNDispatcherCore<TInput> {
  readonly #modelUrl: string;
  readonly #executionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  readonly #loader: OIDNBridgeLoader;
  readonly #readback: ReadbackFn<TInput>;
  readonly #preloadOnBridgeInit: boolean;

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
  /** Cohort token: every {@link invalidate} call bumps this; inferences in
   *  flight at bump time discard their result on resolve. Prevents a stale
   *  inference from polluting the post-invalidation cohort. */
  #cohortId = 0;

  constructor(opts: OIDNDispatcherCoreOptions<TInput>) {
    this.#modelUrl = opts.dispatcherOptions.modelUrl;
    this.#executionProviders = opts.dispatcherOptions.executionProviders;
    this.#loader = opts.loader ?? _defaultLoader;
    this.#readback = opts.readback;
    this.#preloadOnBridgeInit = opts.preloadOnBridgeInit;
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
   * {@link kickIfReady} call. The engine calls this on reset / setScene /
   * updateEnvironment — any state change that invalidates the accumulator
   * also invalidates the denoised cache.
   *
   * An in-flight inference is allowed to complete, but the result is
   * dropped on resolve (the bumped {@link #cohortId} catches the race).
   */
  invalidate(): void {
    this.#cohortId += 1;
    this.#haveCompleted = false;
    this.#latest = null;
  }

  /**
   * Synchronous "kick the OIDN pipeline if needed" entrypoint.
   *
   * Behavior:
   *  - If the dispatcher is disposed, this is a no-op.
   *  - If an inference is already in flight, this is a no-op.
   *  - If an inference has already completed for the current cohort, this
   *    is a no-op (the cached result is reused).
   *  - If `width <= 0 || height <= 0`, this is a no-op.
   *  - Otherwise: sets `#inFlight = true` and spawns an async cycle.
   *    Returns immediately — the result lands asynchronously and is
   *    available via {@link getLatestDenoised} once resolved.
   *
   * The `input` value is forwarded verbatim to the `readback` callback
   * supplied at construction. For pt-webgl the readback is synchronous
   * (performed before calling this method); for pt-webgpu it is an async
   * GPU read.
   *
   * @param input  Backend-specific input value for the readback callback.
   * @param width  Frame width (pixels). Must be > 0.
   * @param height Frame height (pixels). Must be > 0.
   */
  kickIfReady(input: TInput, width: number, height: number): void {
    if (this.#disposed) return;
    if (this.#inFlight) return;
    if (this.#haveCompleted) return;
    if (width <= 0 || height <= 0) return;

    const cohortAtKick = this.#cohortId;
    this.#inFlight = true;
    void this.#runCycle(input, width, height, cohortAtKick).finally(() => {
      this.#inFlight = false;
    });
  }

  async #runCycle(
    input: TInput,
    width: number,
    height: number,
    cohortAtKick: number,
  ): Promise<void> {
    try {
      // Step 1 — backend readback (may be sync-pre-done or async GPU readback).
      const readback = await this.#readback(input);
      if (readback === null) return;
      if (this.#disposed || this.#cohortId !== cohortAtKick) return;

      // Step 2 — lazy bridge load (+ optional preload on first init).
      if (this.#bridge == null) {
        this.#bridge = await this.#loader();
        if (this.#preloadOnBridgeInit && this.#bridge.preloadOIDNModel != null) {
          await this.#bridge.preloadOIDNModel({
            modelUrl: this.#modelUrl,
            ...(this.#executionProviders !== undefined
              ? { executionProviders: this.#executionProviders }
              : {}),
          });
        }
      }
      if (this.#disposed || this.#cohortId !== cohortAtKick) return;

      // Step 3 — inference.
      const opts =
        this.#executionProviders !== undefined
          ? { modelUrl: this.#modelUrl, executionProviders: this.#executionProviders }
          : { modelUrl: this.#modelUrl };

      const denoised = await this.#bridge.denoiseFinal(
        {
          color: readback.color,
          width: readback.width,
          height: readback.height,
          ...(readback.albedo !== undefined ? { albedo: readback.albedo } : {}),
          ...(readback.normal !== undefined ? { normal: readback.normal } : {}),
        },
        opts,
      );

      if (this.#disposed || this.#cohortId !== cohortAtKick) return;
      this.#latest = { rgb: denoised, width, height };
      this.#haveCompleted = true;
    } catch (err) {
      console.warn('[OIDNDispatcherCore] readback or denoiseFinal failed', err);
    }
  }

  /**
   * Release this engine's cached ONNX session entry (model URL + EP tuple).
   * Falls back to global `clearOIDNCache` only when the bridge lacks ref-count API.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const opts =
      this.#executionProviders !== undefined
        ? { modelUrl: this.#modelUrl, executionProviders: this.#executionProviders }
        : { modelUrl: this.#modelUrl };
    try {
      if (this.#bridge?.releaseOIDNCacheEntry != null) {
        this.#bridge.releaseOIDNCacheEntry(opts);
      } else if (this.#bridge?.clearOIDNCache != null) {
        this.#bridge.clearOIDNCache();
      }
    } catch {
      /* swallow — disposal must not throw */
    }
  }
}
