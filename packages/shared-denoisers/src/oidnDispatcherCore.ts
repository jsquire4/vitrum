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
  /** Acquire one engine lease on the shared session. */
  readonly acquireOIDNSession?: (opts: {
    modelUrl: string;
    executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  }) => Promise<OIDNSessionLeaseLike>;
  readonly releaseOIDNCacheEntry?: (opts: {
    modelUrl: string;
    executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  }) => void;
  /** @deprecated Prefer {@link releaseOIDNCacheEntry} for per-engine dispose. */
  readonly clearOIDNCache?: () => void;
}

export interface OIDNSessionLeaseLike {
  release(): void;
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
    acquireOIDNSession: mod.acquireOIDNSession,
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
 * Status derived from the shared cohort state machine, for population into a
 * backend's denoiser-state telemetry surface.
 *
 * The `'warming-up'` member exists so the walkaround-hybrid `OIDNFinalDenoiser`
 * (whose richer `DenoiserState` includes an async model-preload/warmup phase it
 * orchestrates locally) can share the *same* status vocabulary as the two
 * converged backends. The core state machine itself never PRODUCES
 * `'warming-up'` from its own fields — it has no warmup concept; the member is
 * present in the union so wh can layer its warmup phase on top of
 * {@link OIDNDispatcherCore.deriveState} without a separate enum.
 */
export interface OIDNDerivedState {
  readonly status: 'ready' | 'warming-up' | 'in-flight' | 'fallback' | 'failed';
  readonly reason: string | null;
  readonly retryable?: boolean;
}

/**
 * Inputs to {@link deriveOidnState} — the three cohort-state facts the status
 * ladder branches on. Extracting them lets consumers that don't own an
 * {@link OIDNDispatcherCore} instance (the walkaround-hybrid `OIDNFinalDenoiser`,
 * which drives the bridge directly and tracks its own in-flight orchestration)
 * reuse the SAME status mapping without duplicating the ladder.
 */
export interface OIDNDerivedStateInputs {
  /** Message of the last unrecovered inference error, or null when healthy. */
  readonly lastError: string | null;
  /** True while an inference cycle is currently unresolved. */
  readonly inFlight: boolean;
  /** True once at least one inference has completed for the current cohort. */
  readonly haveCompleted: boolean;
  /** Whether another automatic attempt remains after a failure. */
  readonly retryable?: boolean;
}

/**
 * Pure status-ladder shared by every OIDN dispatcher/denoiser. Single source of
 * truth for the previously byte-identical `getState()` ladders (pt-webgl2,
 * pt-webgpu) AND the tail of walkaround-hybrid's richer `state()`:
 *
 *  - `'failed'`    — `lastError` set; `retryable` reports whether the bounded
 *                    automatic-attempt budget still has capacity.
 *  - `'in-flight'` — an inference cycle is running.
 *  - `'ready'`     — the last cycle succeeded and a result is available.
 *  - `'fallback'`  — no inference has completed yet (first frame / post-invalidate).
 *
 * Never returns `'warming-up'` — that is a caller-local phase (wh's async model
 * preload) layered on top of this ladder; the union member exists so wh can
 * share the vocabulary.
 */
export function deriveOidnState(inputs: OIDNDerivedStateInputs): OIDNDerivedState {
  if (inputs.lastError !== null) {
    return {
      status: 'failed',
      reason: inputs.lastError,
      retryable: inputs.retryable ?? true,
    };
  }
  if (inputs.inFlight) {
    return { status: 'in-flight', reason: null };
  }
  if (inputs.haveCompleted) {
    return { status: 'ready', reason: null };
  }
  return { status: 'fallback', reason: 'waiting for first OIDN inference' };
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

function validateRgbBuffer(
  label: string,
  value: Float32Array,
  width: number,
  height: number,
): void {
  if (!(value instanceof Float32Array)) {
    throw new TypeError(`${label}: expected Float32Array`);
  }
  const expected = width * height * 3;
  if (!Number.isSafeInteger(expected) || expected <= 0) {
    throw new RangeError(`${label}: invalid dimensions ${width}×${height}`);
  }
  if (value.length !== expected) {
    throw new RangeError(`${label}: expected ${expected} RGB floats, got ${value.length}`);
  }
  for (let i = 0; i < value.length; i += 1) {
    if (!Number.isFinite(value[i])) {
      throw new RangeError(`${label}: non-finite value at index ${i}`);
    }
  }
}

function validateReadback(
  readback: ReadbackResult,
  requestedWidth: number,
  requestedHeight: number,
): void {
  if (!Number.isSafeInteger(readback.width) || !Number.isSafeInteger(readback.height) ||
      readback.width <= 0 || readback.height <= 0) {
    throw new RangeError(
      `OIDN readback: invalid dimensions ${readback.width}×${readback.height}`,
    );
  }
  if (readback.width !== requestedWidth || readback.height !== requestedHeight) {
    throw new RangeError(
      `OIDN readback: dimensions ${readback.width}×${readback.height} do not match ` +
        `requested ${requestedWidth}×${requestedHeight}`,
    );
  }
  validateRgbBuffer('OIDN readback color', readback.color, readback.width, readback.height);
  if (readback.albedo !== undefined) {
    validateRgbBuffer('OIDN readback albedo', readback.albedo, readback.width, readback.height);
  }
  if (readback.normal !== undefined) {
    validateRgbBuffer('OIDN readback normal', readback.normal, readback.width, readback.height);
  }
}

function validatePositiveSafeInteger(label: string, value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
  return value;
}

function validateFiniteNonNegative(label: string, value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be a finite non-negative number`);
  }
  return value;
}

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
   * This preserves the behavioral difference between the two backends. The
   * request is honoured whenever the bridge exposes `preloadOIDNModel`,
   * independently of whether it also exposes `acquireOIDNSession` — the
   * preload runs first, then the lease is acquired.
   */
  readonly preloadOnBridgeInit: boolean;
  /**
   * Optional error callback. Invoked once per distinct error message when
   * the inference cycle catches an exception (readback failure, ORT load
   * failure, model mismatch, etc.). The same error message is NOT repeated
   * on every subsequent failing cohort — only when the message changes from
   * the previous failure.
   *
   * The existing `console.warn` once-per-distinct-error behaviour is
   * preserved regardless of whether this callback is supplied; the callback
   * fires IN ADDITION to the warn.
   */
  readonly onError?: (err: unknown) => void;
  /** Called only after an accepted result is published; observer failures are isolated. */
  readonly onComplete?: (frame: DenoisedFrame) => void;
  /**
   * Automatic retry policy for failed readback/session/inference cycles.
   * Defaults to three total attempts with 1s/2s exponential delays. This
   * prevents a persistent model URL or provider failure from performing a
   * full readback and session construction on every render frame forever.
   *
   * The clock hook and zero delays are intended for deterministic tests.
   */
  readonly retryPolicy?: {
    readonly maxAttempts?: number;
    readonly initialDelayMs?: number;
    readonly maxDelayMs?: number;
    readonly now?: () => number;
  };
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
  readonly #onError: ((err: unknown) => void) | undefined;
  readonly #onComplete: ((frame: DenoisedFrame) => void) | undefined;
  readonly #maxAttempts: number;
  readonly #initialRetryDelayMs: number;
  readonly #maxRetryDelayMs: number;
  readonly #now: () => number;

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
  /** Per-engine ownership of the shared bridge session. */
  #sessionLease: OIDNSessionLeaseLike | null = null;
  /** Disposal waits for this core's in-flight cycle before dropping ownership. */
  #releaseBridgeWhenIdle = false;
  /** Cohort token: every {@link invalidate} call bumps this; inferences in
   *  flight at bump time discard their result on resolve. Prevents a stale
   *  inference from polluting the post-invalidation cohort. */
  #cohortId = 0;
  /**
   * The string message of the last caught error, or null when the last
   * cycle succeeded (or no cycle has run yet). Used to suppress repeated
   * `console.warn` calls for the same error message. Cleared on a
   * successful inference.
   *
   * Exposed via {@link getLastError} so the engine can surface it into
   * `FrameStats.denoiserState.reason` without polling private state.
   */
  #lastErrorMessage: string | null = null;
  /** Consecutive failed cycles. A successful result resets this to zero. */
  #failureCount = 0;
  /** Earliest clock time at which another automatic attempt may begin. */
  #retryNotBeforeMs = 0;
  /** True after the bounded automatic-attempt budget is exhausted. */
  #retryLatched = false;

  constructor(opts: OIDNDispatcherCoreOptions<TInput>) {
    this.#modelUrl = opts.dispatcherOptions.modelUrl;
    this.#executionProviders = opts.dispatcherOptions.executionProviders;
    this.#loader = opts.loader ?? _defaultLoader;
    this.#readback = opts.readback;
    this.#preloadOnBridgeInit = opts.preloadOnBridgeInit;
    this.#onError = opts.onError;
    this.#onComplete = opts.onComplete;
    const retry = opts.retryPolicy;
    this.#maxAttempts = validatePositiveSafeInteger(
      'OIDNDispatcherCore retryPolicy.maxAttempts',
      retry?.maxAttempts ?? 3,
    );
    this.#initialRetryDelayMs = validateFiniteNonNegative(
      'OIDNDispatcherCore retryPolicy.initialDelayMs',
      retry?.initialDelayMs ?? 1_000,
    );
    this.#maxRetryDelayMs = validateFiniteNonNegative(
      'OIDNDispatcherCore retryPolicy.maxDelayMs',
      retry?.maxDelayMs ?? Math.max(30_000, this.#initialRetryDelayMs),
    );
    if (this.#maxRetryDelayMs < this.#initialRetryDelayMs) {
      throw new RangeError(
        'OIDNDispatcherCore retryPolicy.maxDelayMs must be greater than or equal to ' +
          'retryPolicy.initialDelayMs',
      );
    }
    if (retry?.now !== undefined && typeof retry.now !== 'function') {
      throw new TypeError('OIDNDispatcherCore retryPolicy.now must be a function');
    }
    this.#now = retry?.now ?? (() => globalThis.performance?.now() ?? Date.now());
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
   * The string message of the most recent unrecovered inference error, or
   * `null` when no error has occurred (or after a successful inference
   * clears the previous error). Cleared on each successful inference.
   *
   * Intended for engine telemetry: the engine reads this once per frame
   * to populate `FrameStats.denoiserState.reason` without accessing
   * private fields.
   */
  getLastError(): string | null {
    return this.#lastErrorMessage;
  }

  /**
   * Derive the backend's public denoiser status from the shared cohort state
   * machine. Single source of truth for the previously byte-identical
   * `getState()` ladders in the pt-webgl2 and pt-webgpu wrappers:
   *
   *  - `'failed'`    — the last cycle threw; `reason` = error message;
   *                    `retryable` is false once the attempt budget is exhausted.
   *  - `'in-flight'` — an async inference cycle is currently running.
   *  - `'ready'`     — the last cycle succeeded and a result is available.
   *  - `'fallback'`  — no inference has completed yet (first frame / post-invalidate).
   *
   * Never returns `'warming-up'`: that member of {@link OIDNDerivedState} exists
   * only so wh's warmup-aware wrapper can share the union (see the type doc).
   */
  deriveState(): OIDNDerivedState {
    return deriveOidnState({
      lastError: this.#lastErrorMessage,
      inFlight: this.#inFlight,
      haveCompleted: this.#latest !== null,
      retryable: !this.#retryLatched,
    });
  }

  /**
   * Clear the latest result for a new accumulator cohort. If the automatic
   * retry budget still has capacity, the dispatcher may re-kick on a later
   * {@link kickIfReady} call. Invalidation deliberately preserves failure
   * count, backoff, and an exhausted retry latch: ordinary per-frame resets
   * cannot bypass the bounded-attempt policy. Recovery after exhaustion
   * requires disposing and recreating the dispatcher.
   *
   * The engine calls this on reset / setScene / updateEnvironment — any state
   * change that invalidates the accumulator also invalidates the denoised cache.
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
    if (this.#retryLatched) return;
    if (width <= 0 || height <= 0) return;
    const now = this.#readNow();
    if (this.#lastErrorMessage !== null && now < this.#retryNotBeforeMs) return;

    const cohortAtKick = this.#cohortId;
    this.#inFlight = true;
    void this.#runCycle(input, width, height, cohortAtKick).finally(() => {
      this.#inFlight = false;
      if (this.#releaseBridgeWhenIdle) {
        this.#releaseBridgeOwnership();
      }
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
      validateReadback(readback, width, height);

      // Step 2 — lazy bridge load (+ optional preload on first init).
      if (this.#bridge == null) {
        const candidateBridge = await this.#loader();
        let candidateLease: OIDNSessionLeaseLike | null = null;
        const bridgeOpts = {
          modelUrl: this.#modelUrl,
          ...(this.#executionProviders !== undefined
            ? { executionProviders: this.#executionProviders }
            : {}),
        };
        // Invariant: `preloadOnBridgeInit` is an independent host request, not a
        // fallback for bridges that lack a lease API. A lease is an ownership
        // claim; the bridge interface does not require `acquireOIDNSession` to
        // load the model, so gating the preload behind the absence of that
        // optional method silently drops an explicitly requested warm-up (the
        // shipped loader always supplies `acquireOIDNSession`, which made the
        // option unreachable and erased the pt-webgpu/pt-webgl2 difference).
        // Preload runs BEFORE lease acquisition so a rejecting preload can never
        // strand an acquired lease that was never published to `#sessionLease`.
        if (this.#preloadOnBridgeInit && candidateBridge.preloadOIDNModel != null) {
          await candidateBridge.preloadOIDNModel(bridgeOpts);
        }
        if (candidateBridge.acquireOIDNSession != null) {
          candidateLease = await candidateBridge.acquireOIDNSession(bridgeOpts);
        }
        // Candidate-first publication: a failed loader/acquire leaves no
        // partially published bridge. Dispose during acquisition releases the
        // just-created lease immediately instead of reviving this core.
        if (this.#disposed) {
          try {
            if (candidateLease != null) {
              candidateLease.release();
            } else if (candidateBridge.releaseOIDNCacheEntry != null) {
              candidateBridge.releaseOIDNCacheEntry(bridgeOpts);
            } else {
              candidateBridge.clearOIDNCache?.();
            }
          } catch { /* disposal is best-effort */ }
          return;
        }
        this.#bridge = candidateBridge;
        this.#sessionLease = candidateLease;
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
      validateRgbBuffer('OIDN output', denoised, readback.width, readback.height);
      const completed = { rgb: denoised, width: readback.width, height: readback.height };
      this.#latest = completed;
      this.#haveCompleted = true;
      // Clear the error state after a successful inference so that
      // `getLastError()` reflects the current health of the dispatcher.
      this.#lastErrorMessage = null;
      this.#failureCount = 0;
      this.#retryNotBeforeMs = 0;
      this.#retryLatched = false;
      if (this.#onComplete !== undefined) {
        try {
          this.#onComplete(completed);
        } catch {
          // Completion observers must not corrupt accepted inference state.
        }
      }
    } catch (err) {
      // A disposed dispatcher or invalidated cohort must not publish a late
      // failure into the replacement lifecycle or notify a host that no longer
      // owns this work.
      if (this.#disposed || this.#cohortId !== cohortAtKick) return;
      this.#recordError(err, 'readback or denoiseFinal failed');
      this.#failureCount += 1;
      this.#retryLatched = this.#failureCount >= this.#maxAttempts;
      if (!this.#retryLatched) {
        const exponent = Math.max(0, this.#failureCount - 1);
        const delay = Math.min(
          this.#maxRetryDelayMs,
          this.#initialRetryDelayMs * (2 ** exponent),
        );
        try {
          const retryNotBeforeMs = this.#readNow() + delay;
          if (!Number.isFinite(retryNotBeforeMs)) {
            throw new RangeError(
              'OIDNDispatcherCore retry policy produced a non-finite retry deadline',
            );
          }
          this.#retryNotBeforeMs = retryNotBeforeMs;
        } catch (clockError) {
          // A broken host clock must not turn a failed OIDN setup into either
          // an unhandled rejection or an expensive per-frame retry loop.
          this.#recordError(clockError, 'retry clock failed');
          this.#retryLatched = true;
          this.#retryNotBeforeMs = 0;
        }
      }
    }
  }

  #readNow(): number {
    const now = this.#now();
    if (!Number.isFinite(now) || now < 0) {
      throw new RangeError(
        'OIDNDispatcherCore retryPolicy.now() must return a finite non-negative number',
      );
    }
    return now;
  }

  #recordError(err: unknown, context: string): void {
    const msg = err instanceof Error ? err.message : String(err);
    // Suppress repeated notifications for an identical error message.
    if (msg !== this.#lastErrorMessage) {
      console.warn(`[OIDNDispatcherCore] ${context}`, err);
      if (this.#onError !== undefined) {
        try { this.#onError(err); } catch { /* onError must not propagate */ }
      }
    }
    this.#lastErrorMessage = msg;
  }

  /**
   * Release this engine's cached ONNX session entry (model URL + EP tuple).
   * Falls back to global `clearOIDNCache` only when the bridge lacks ref-count API.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#inFlight) {
      this.#releaseBridgeWhenIdle = true;
    } else {
      this.#releaseBridgeOwnership();
    }
  }

  #releaseBridgeOwnership(): void {
    this.#releaseBridgeWhenIdle = false;
    const lease = this.#sessionLease;
    this.#sessionLease = null;
    try {
      if (lease != null) {
        lease.release();
      } else if (this.#bridge?.releaseOIDNCacheEntry != null) {
        const opts =
          this.#executionProviders !== undefined
            ? { modelUrl: this.#modelUrl, executionProviders: this.#executionProviders }
            : { modelUrl: this.#modelUrl };
        this.#bridge.releaseOIDNCacheEntry(opts);
      } else if (this.#bridge?.clearOIDNCache != null) {
        this.#bridge.clearOIDNCache();
      }
    } catch {
      /* swallow — disposal must not throw */
    }
  }
}
