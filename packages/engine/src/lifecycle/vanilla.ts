// Vanilla lifecycle helper: pairs a canvas + camera + scene to a vitrum
// engine, runs the host RAF loop, and tears everything down on dispose.
// Intended for non-React hosts. The React component in
// ../react/VitrumCanvas.tsx is a thin wrapper around this.
//
// Survives the engine-contract gotchas:
//   - Visibility: pauses the RAF loop on document.visibilityState !== 'visible'.
//   - Resize: tracks canvas dimensions via ResizeObserver; pushes the
//     latest viewport into FrameInput.viewport every frame (the engine
//     contract carries viewport per frame, not as engine state).
//   - prevView/prevProj: tracks the last frame's camera matrices and
//     forwards them so ReSTIR temporal reuse + motion vectors work
//     out-of-box (a host that omits these silently degrades to no-temporal).
//   - Idempotent dispose.

import type { CapturedFrame, CaptureFrameOptions, Engine, EngineError, FrameInput, FrameStats, ProgressStats, Mat4 } from '@vitrum/core';
import { asBackendTexture, asBackendTextureFormat, asMat4 } from '@vitrum/core';
import { createEngine, type CreateEngineErrorEvent, type CreateEngineOptions } from '../createEngine.js';
import type { GIStatePersistable } from '../idempotentDispose.js';
import type { GIStateSnapshot } from '@vitrum/walkaround-hybrid';
import type { EngineWithBackendId } from '../createEngineInternals.js';

// ────────────────────────────────────────────────────────────────────────────
// A2 — WebGPU swap-chain detection + per-frame view acquisition.
//
// HybridEngine.renderFrame requires `FrameInput.swapChainView` to be a fresh
// per-frame `GPUTextureView`; when absent it returns a "skip" frame
// (HybridEngine.ts:979). createEngine() configures the canvas's WebGPU
// context during walkaround backend construction; here we just observe and
// re-derive the per-frame view.
//
// Extracted as a top-level helper so the unit test can pin the FrameInput
// shape (with vs without WebGPU context) without a DOM/RAF harness.

/** Output of {@link detectWebGPUSwapChain}. Both fields null ⇒ WebGL2/host
 *  is not WebGPU-backed (or canvas lacks a WebGPU context); attachVitrum
 *  will then leave `FrameInput.swapChainView` undefined.
 *  File-local — callers consume the type structurally via inference. */
interface WebGPUSwapChainInfo {
  readonly context: GPUCanvasContext | null;
  readonly format: GPUTextureFormat | undefined;
}

/** Detect whether the canvas currently has a WebGPU context (configured by
 *  createEngine's walkaround backend constructor) and recover its format.
 *  Returns `{ context: null, format: undefined }` for WebGL2 hosts or for test
 *  environments without WebGPU support.
 *
 *  @internal Exported for unit-test access. */
export function detectWebGPUSwapChain(canvas: HTMLCanvasElement): WebGPUSwapChainInfo {
  try {
    const ctx = canvas.getContext('webgpu');
    if (ctx == null) return { context: null, format: undefined };
    const cfg = (ctx as unknown as {
      getConfiguration?: () => { format?: GPUTextureFormat } | null;
    }).getConfiguration?.();
    const format = cfg?.format
      ?? ((typeof navigator !== 'undefined' && 'gpu' in navigator
        ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
            .getPreferredCanvasFormat?.()
        : undefined)
        ?? ('bgra8unorm'));
    return { context: ctx, format };
  } catch {
    return { context: null, format: undefined };
  }
}

/** Resolve a quality option (value or getter) to its current value.
 *  Called inside the rAF tick so live-getters (`() => ref.current`) propagate
 *  on the very next frame without engine recreation.
 *
 *  @internal Exported for unit-test access. */
export function resolveQualityOption(
  q: AttachVitrumOptions['quality'],
): NonNullable<FrameInput['quality']> | undefined {
  if (q == null) return undefined;
  return typeof q === 'function'
    ? (q)()
    : (q);
}

/** Per WebGPU spec, `getCurrentTexture()` MUST be called inside the rAF
 *  tick and the resulting view is single-use (do NOT cache across frames).
 *  Returns undefined when the context is null (WebGL host) or acquisition
 *  throws (canvas size zero, context lost) — caller leaves
 *  `FrameInput.swapChainView` undefined so HybridEngine skips the frame.
 *
 *  @internal Exported for unit-test access. */
export function acquireSwapChainView(
  ctx: GPUCanvasContext | null,
  onError?: (error: unknown) => void,
): GPUTextureView | undefined {
  if (ctx == null) return undefined;
  try {
    return ctx.getCurrentTexture().createView();
  } catch (err) {
    onError?.(err);
    return undefined;
  }
}

/** Convert CSS-pixel canvas dimensions to the physical viewport contract. */
export function toPhysicalViewport(
  cssWidth: number,
  cssHeight: number,
  dpr: number,
): Pick<FrameInput['viewport'], 'width' | 'height' | 'devicePixelRatio'> {
  const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  return {
    width: Math.max(1, Math.floor(cssWidth * safeDpr)),
    height: Math.max(1, Math.floor(cssHeight * safeDpr)),
    devicePixelRatio: safeDpr,
  };
}

export interface ComposeAttachVitrumFrameInputOptions {
  readonly viewMatrix: Mat4;
  readonly projMatrix: Mat4;
  readonly cameraPosition: readonly [number, number, number];
  readonly prevViewMatrix?: Mat4;
  readonly prevProjMatrix?: Mat4;
  readonly viewport: FrameInput['viewport'];
  readonly frameIndex: number;
  readonly quality?: NonNullable<FrameInput['quality']>;
  readonly swapChainView?: GPUTextureView;
  readonly swapChainFormat?: GPUTextureFormat;
}

/** Compose the exact FrameInput shape used by attachVitrum's RAF tick. */
export function composeAttachVitrumFrameInput(opts: ComposeAttachVitrumFrameInputOptions): FrameInput {
  return {
    viewMatrix: opts.viewMatrix,
    projMatrix: opts.projMatrix,
    cameraPosition: [opts.cameraPosition[0], opts.cameraPosition[1], opts.cameraPosition[2]],
    ...(opts.prevViewMatrix ? { prevViewMatrix: opts.prevViewMatrix } : {}),
    ...(opts.prevProjMatrix ? { prevProjMatrix: opts.prevProjMatrix } : {}),
    viewport: opts.viewport,
    frameIndex: opts.frameIndex,
    frameSeed: (opts.frameIndex * 1664525 + 1013904223) >>> 0,
    ...(opts.quality ? { quality: opts.quality } : {}),
    ...(opts.swapChainView != null
      ? {
          swapChainView: asBackendTexture<'webgpu', GPUTextureView>(opts.swapChainView),
          ...(opts.swapChainFormat != null
            ? {
                swapChainFormat: asBackendTextureFormat<'webgpu', GPUTextureFormat>(
                  opts.swapChainFormat,
                ),
              }
            : {}),
        }
      : {}),
  };
}

/** Structural camera contract for the RAF loop.
 *
 * The loop only reads `updateMatrixWorld()`, `matrixWorldInverse.elements`,
 * `projectionMatrix.elements`, and `position.{x,y,z}` — a real
 * `THREE.PerspectiveCamera` / `THREE.OrthographicCamera` satisfies this
 * structurally, so existing callers are unaffected. Defined locally so this
 * non-React entrypoint does not carry a THREE value-import. */
export interface CameraLike {
  updateMatrixWorld(): void;
  readonly matrixWorldInverse: { readonly elements: ArrayLike<number> };
  readonly projectionMatrix: { readonly elements: ArrayLike<number> };
  readonly position: { readonly x: number; readonly y: number; readonly z: number };
}

/**
 * Per-frame quality option for {@link AttachVitrumOptions.quality}.
 *
 * Pass a value for static quality, or a `() => value | undefined` getter for
 * live propagation: the RAF tick invokes it every frame, so React refs /
 * mutable state can swap quality without engine recreation.  React's
 * `<VitrumCanvas>` uses the getter form internally so its `quality` prop
 * propagates without remount.
 *
 * D1.5 — named type replaces the inline anonymous union on the options field.
 */
export type QualityOption =
  | NonNullable<FrameInput['quality']>
  | (() => NonNullable<FrameInput['quality']> | undefined);

export interface AttachVitrumSceneControllerPlaybackOptions {
  /** Forwarded to `sceneController.advance(..., { loop })`. Default true. */
  readonly loop?: boolean;
}

export type AttachVitrumSceneControllerPlayback =
  | boolean
  | AttachVitrumSceneControllerPlaybackOptions;

export interface AttachVitrumOptions extends Omit<CreateEngineOptions, 'scene'> {
  /**
   * Optional already-constructed engine for callers that prepared the scene
   * through a higher-level bridge before handing off to the lifecycle loop.
   * The supplied engine must already represent `scene`; attachVitrum will own
   * its RAF/resize/error subscriptions and dispose it with the handle. Fatal
   * auto-recreate still falls back to `createEngine()` using the latest tracked
   * scene.
   */
  readonly engine?: EngineWithBackendId;
  /**
   * Runtime scene controller to re-target whenever the lifecycle swaps engines
   * (for example after fatal device-loss auto-recreate). Structural on purpose:
   * glTF is the first consumer, but the lifecycle helper does not need adapter
   * runtime imports.
   */
  readonly sceneController?: AttachVitrumSceneController;
  /**
   * Opt-in RAF-driven playback for scene controllers attached through this
   * lifecycle helper. When enabled, attachVitrum calls
   * `sceneController.advance(deltaSeconds, { engine, loop })` once per frame.
   * Default: false, so hosts that own their own animation clocks keep full
   * control.
   */
  readonly sceneControllerPlayback?: AttachVitrumSceneControllerPlayback;
  /** Engine-level GPU/runtime error callback. Receives errors from the
   *  underlying engine's `onError` subscription (device-lost, GPU validation
   *  errors, WebGL context-lost).  Distinct from `onError` in
   *  {@link CreateEngineOptions}, which covers engine-construction failures:
   *  this callback fires for runtime GPU errors AFTER construction succeeds.
   *
   *  `fatal: true` means the engine is in `'error'` state — the host should
   *  call `handle.dispose()` and recreate.  Non-fatal errors are informational;
   *  rendering continues.
   *
   *  When {@link autoRecreateOnDeviceLoss} is enabled, this callback fires
   *  BEFORE the auto-recreate logic runs so the host always sees every error
   *  regardless of whether the engine is about to be reconstructed.  It also
   *  fires for the final error if the retry cap is exceeded. */
  readonly onEngineError?: (error: EngineError) => void;
  /**
   * Automatically recreate the engine after a fatal device-lost or
   * context-lost error (`EngineError.kind === 'device-lost'` or
   * `'context-lost'`).
   *
   * When `true`, on a fatal loss event:
   *  1. `onEngineError` is called first (host sees the event regardless).
   *  2. The RAF loop is stopped.
   *  3. If the engine exposes `exportGIState` (walkaround-hybrid backend),
   *     the DDGI probe state is exported before the engine is torn down.
   *     Export/import failures are reported through `onError` as recoverable
   *     lifecycle errors; recreate still proceeds best-effort.
   *  4. The engine is disposed.
   *  5. `createEngine` is called again with the original options, minting a
   *     fresh GPU device automatically.
   *  6. If GI state was exported and the new engine exposes `importGIState`,
   *     it is imported — warm DDGI probes survive the recreate.
   *  7. The RAF loop resumes.
   *
   * Retries are capped at **2 attempts within 30 seconds**.  If the cap is
   * exceeded, the final `onEngineError` is delivered with `fatal: true` and
   * the loop stays stopped (same as if `autoRecreateOnDeviceLoss` were false).
   *
   * `onEngineError` receives every event — before auto-recreate and on the
   * final failure after the cap.
   *
   * Default: `false`.
   */
  readonly autoRecreateOnDeviceLoss?: boolean;
  /** Scene description in the host-agnostic @vitrum/core contract. */
  readonly scene: CreateEngineOptions['scene'];
  /** Camera the engine reads viewMatrix / projMatrix / position from every
   *  frame. The host mutates this camera (orbit controls, scripted animation)
   *  and the helper pushes the latest matrices into renderFrame. A real
   *  `THREE.PerspectiveCamera` or `THREE.OrthographicCamera` satisfies
   *  {@link CameraLike} structurally. */
  readonly camera: CameraLike;
  /** Per-frame quality dials. Honoured if non-null; otherwise the engine's
   *  defaults apply.
   *
   *  Pass a value for static quality, or a `() => quality | undefined`
   *  getter for live propagation (the RAF tick invokes it every frame, so
   *  React refs / mutable state can swap quality without engine recreation).
   *  React's `<VitrumCanvas>` uses the getter form internally so its
   *  `quality` prop propagates without remount. */
  readonly quality?: QualityOption;
  /** Frame-level callback. Convenience over engine.onFrame() — fires after
   *  every successful renderFrame() with the engine's FrameStats. */
  readonly onFrame?: (stats: FrameStats) => void;
  /** Progress callback (PT spp accumulation). Convenience over
   *  engine.onProgress(); fires only when the underlying engine emits. */
  readonly onProgress?: (progress: ProgressStats) => void;
  /** When true (default), pauses the RAF loop while
   *  document.visibilityState !== 'visible'. Set false for hosts that
   *  need to keep accumulating SPP in a backgrounded tab. */
  readonly pauseOnHidden?: boolean;
}

export interface AttachVitrumSceneController {
  readonly animations?: readonly unknown[];
  attachEngine(engine: EngineWithBackendId, options?: { readonly setScene?: boolean }): void;
  advance?(
    deltaSeconds: number,
    options?: { readonly engine?: EngineWithBackendId; readonly loop?: boolean },
  ): unknown;
}

export interface AttachVitrumHandle {
  /** The underlying Engine. Hosts can subscribe to telemetry or call
   *  reset() / pause() / resume() directly via this handle. */
  readonly engine: Engine;
  /** Backend selected by createEngine for the currently attached engine.
   *  Auto-recreate keeps the same stable handle object, so this is exposed as a
   *  live getter and updates if a recreated engine lands on a different backend. */
  readonly backendId: EngineWithBackendId['backendId'];
  /** Stop the RAF loop, disconnect the ResizeObserver, unsubscribe from
   *  document.visibilitychange, and dispose the engine. Idempotent. */
  dispose(): void;
  /**
   * Passthrough to `engine.captureFrame(opts)`. Returns `null` when the
   * engine has not yet rendered a frame or when the underlying engine does
   * not implement `captureFrame`. See {@link Engine.captureFrame} for the
   * full contract, source-texture documentation, and pipeline-stall warning.
   */
  captureFrame(opts?: CaptureFrameOptions): Promise<CapturedFrame | null>;
}

/** @internal — maximum number of auto-recreate attempts within the window. */
export const AUTO_RECREATE_MAX_ATTEMPTS = 2;
/** @internal — window duration (ms) within which retry attempts are counted. */
export const AUTO_RECREATE_WINDOW_MS = 30_000;

/**
 * Canonical builder: forwards the lifecycle-level `AttachVitrumOptions` into
 * `createEngine`.  The spread was duplicated verbatim at initial construction
 * (line ~275) and in the auto-recreate path (line ~465); centralising it here
 * makes the forwarding semantics explicit and impossible to diverge.
 *
 * D1.5 — onError/onEngineError unification: `onError` (construction-phase)
 * flows through normally.  `onEngineError` (runtime GPU errors) is a separate
 * channel and is NOT forwarded to createEngine — it is wired on the returned
 * engine via engine.onError() after construction.  Both public fields are kept
 * accepted for back-compat; this function handles only the construction-phase
 * channel.
 *
 * @internal
 */
function buildEngineFromOpts(
  opts: AttachVitrumOptions,
  scene: AttachVitrumOptions['scene'] = opts.scene,
): Promise<EngineWithBackendId> {
  return createEngine({
    canvas: opts.canvas,
    scene,
    ...(opts.prefer != null ? { prefer: opts.prefer } : {}),
    ...(opts.gltfAsset != null ? { gltfAsset: opts.gltfAsset } : {}),
    ...(opts.advanced != null ? { advanced: opts.advanced } : {}),
    ...(opts.advancedBackend != null ? { advancedBackend: opts.advancedBackend } : {}),
    ...(opts.advancedByBackend != null ? { advancedByBackend: opts.advancedByBackend } : {}),
    ...(opts.debug != null ? { debug: opts.debug } : {}),
    ...(opts.onAdapterProfile != null ? { onAdapterProfile: opts.onAdapterProfile } : {}),
    ...(opts.onError != null ? { onError: opts.onError } : {}),
    ...(opts.onWarning != null ? { onWarning: opts.onWarning } : {}),
  });
}

export async function attachVitrum(opts: AttachVitrumOptions): Promise<AttachVitrumHandle> {
  const reportError = (error: unknown, event: CreateEngineErrorEvent): void => {
    try {
      opts.onError?.(error, event);
    } catch { /* host error callback must not propagate — ignore */ }
  };

  // ── Resize state ─────────────────────────────────────────────────────────
  //
  // H30 — synchronous initial sizing: set canvas.width/height from CSS client
  // size × DPR BEFORE engine construction so WebGPU walkaround can allocate its
  // swapchain/history resources at the real visible size. Without this, the
  // constructor sees the browser's default 300×150 backing store and only catches
  // up after a later ResizeObserver tick.
  //
  // A4 — generic PT engines honour viewport-per-frame, but HybridEngine
  // (WebGPU walkaround) does not — its DDGI atlas / ReSTIR reservoirs /
  // history textures / accumulation buffer are sized at construction and
  // can only be resized via `setSize(w, h)`. Call it when available so
  // WebGPU hosts using attachVitrum get correct resize behaviour out of box.
  let viewportDpr = (typeof window !== 'undefined' ? window.devicePixelRatio : null) ?? 1;
  const safeDprInitial = Number.isFinite(viewportDpr) && viewportDpr > 0 ? viewportDpr : 1;
  // Seed from CSS clientWidth/Height × DPR, falling back to the canvas backing
  // store dimensions when the element has no layout (e.g. test environments
  // where clientWidth is always 0).
  const initCssW = opts.canvas.clientWidth > 0 ? opts.canvas.clientWidth : opts.canvas.width;
  const initCssH = opts.canvas.clientHeight > 0 ? opts.canvas.clientHeight : opts.canvas.height;
  const initW = Math.max(1, Math.floor(initCssW * safeDprInitial));
  const initH = Math.max(1, Math.floor(initCssH * safeDprInitial));
  // Synchronously set the backing store so the first renderFrame sees the
  // correct physical size (matches what ResizeObserver will maintain thereafter).
  opts.canvas.width = initW;
  opts.canvas.height = initH;
  let viewportW = initW;
  let viewportH = initH;

  // ── Initial engine construction ─────────────────────────────────────────
  // `engine` is mutable so the auto-recreate path can swap it in place while
  // the stable handle facade (returned to the host) continues to work.
  let currentScene = opts.scene;
  const trackEngineScene = (target: EngineWithBackendId): void => {
    const setScene = target.setScene.bind(target);
    target.setScene = (scene) => {
      setScene(scene);
      currentScene = scene;
    };
  };
  const attachSceneController = (target: EngineWithBackendId): void => {
    if (opts.sceneController == null) return;
    try {
      opts.sceneController.attachEngine(target, { setScene: false });
    } catch (err) {
      reportError(err, { phase: 'attach:scene-controller', backend: target.backendId, recoverable: true });
    }
  };
  let engine = opts.engine ?? await buildEngineFromOpts(opts, currentScene);
  trackEngineScene(engine);
  attachSceneController(engine);

  const sceneControllerPlayback = opts.sceneControllerPlayback;
  let lastSceneControllerFrameMs: number | null = null;
  const advanceSceneController = (nowMs: number): void => {
    const controller = opts.sceneController;
    if (!isSceneControllerPlaybackEnabled(sceneControllerPlayback)) return;
    if (controller?.advance == null) return;
    if (controller.animations != null && controller.animations.length === 0) return;

    if (lastSceneControllerFrameMs == null) {
      lastSceneControllerFrameMs = nowMs;
      return;
    }
    const deltaSeconds = Math.max(0, (nowMs - lastSceneControllerFrameMs) / 1000);
    lastSceneControllerFrameMs = nowMs;
    if (deltaSeconds === 0) return;

    controller.advance(deltaSeconds, {
      engine,
      loop: sceneControllerPlaybackLoop(sceneControllerPlayback),
    });
  };

  // ── ResizeObserver ───────────────────────────────────────────────────────
  let resizeObserver: ResizeObserver | undefined;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver((entries) => {
      viewportDpr = (typeof window !== 'undefined' ? window.devicePixelRatio : null) ?? 1;
      for (const entry of entries) {
        const cr = entry.contentRect;
        const viewport = toPhysicalViewport(cr.width, cr.height, viewportDpr);
        viewportW = viewport.width;
        viewportH = viewport.height;
      }
      // H30 — update the canvas backing store to match the new CSS size × DPR so
      // the swapchain textures (WebGL drawingBuffer, WebGPU canvas texture) are
      // sized correctly after the resize.  Without this the backing store stays at
      // the initial size while the engine's internal targets resize, causing
      // viewport/swapchain mismatches until the next createEngine call.
      opts.canvas.width = viewportW;
      opts.canvas.height = viewportH;
      // A4 — Only backends with `presentationMode === 'swapchain-required'`
      // (walkaround-hybrid / WebGPU) need explicit `setSize()` on resize;
      // offscreen-texture backends (pt-webgl2, pt-webgpu) honour
      // `FrameInput.viewport` per-frame and declare `setSize` absent.
      if (engine.capabilities.presentationMode === 'swapchain-required') {
        try {
          engine.setSize?.(viewportW, viewportH);
        } catch (err) {
          reportError(err, { phase: 'attach:resize', recoverable: true });
        }
      }
    });
    resizeObserver.observe(opts.canvas);
  }

  // ── WebGPU swap-chain detection ──────────────────────────────────────────
  // A2 — WebGPU backend detection. createEngine's walkaround constructor
  // configured the canvas's WebGPU context; we re-observe it here so the
  // RAF tick can acquire a fresh GPUTextureView per frame and pass it as
  // FrameInput.swapChainView. WebGL hosts return { context: null } and the
  // tick omits the swap-chain fields.
  let webgpuSwapChain = detectWebGPUSwapChain(opts.canvas);

  // ── Pause-on-hidden ──────────────────────────────────────────────────────
  const pauseOnHidden = opts.pauseOnHidden ?? true;
  let visibilityHandler: (() => void) | undefined;
  if (pauseOnHidden && typeof document !== 'undefined') {
    visibilityHandler = () => {
      if (document.visibilityState === 'hidden') {
        try { engine.pause(); } catch { /* best-effort pause — ignore */ }
      } else {
        try { engine.resume(); } catch { /* best-effort resume — ignore */ }
      }
    };
    document.addEventListener('visibilitychange', visibilityHandler);
    // Apply initial state.
    visibilityHandler();
  }

  // ── Telemetry subscriptions ──────────────────────────────────────────────
  // Subscriptions are set up through a helper so they can be re-subscribed
  // after auto-recreate.  The host-facing callbacks are stable closures that
  // read opts.onFrame / opts.onProgress / opts.onEngineError by value at
  // call time (not captured once).
  let unsubFrame: (() => void) | undefined;
  let unsubProgress: (() => void) | undefined;
  let unsubEngineError: (() => void) | undefined;

  const subscribeToEngine = (): void => {
    unsubFrame?.();
    unsubProgress?.();
    unsubEngineError?.();
    unsubFrame = opts.onFrame && engine.onFrame
      ? engine.onFrame(opts.onFrame)
      : undefined;
    unsubProgress = opts.onProgress && engine.onProgress
      ? engine.onProgress(opts.onProgress)
      : undefined;
    unsubEngineError = engine.onError
      ? engine.onError((err) => {
          // Always deliver to the host first, before any internal handling.
          try { opts.onEngineError?.(err); } catch { /* host error callback must not propagate — ignore */ }
          if (err.fatal) handleFatalEngineError(err);
        })
      : undefined;
  };
  subscribeToEngine();

  // ── Auto-recreate state machine ──────────────────────────────────────────
  // Retry budget: at most AUTO_RECREATE_MAX_ATTEMPTS within AUTO_RECREATE_WINDOW_MS.
  //
  // State is grouped into a named object so its ownership and mutation surface
  // are explicit.  The functions close over the outer `engine`, `stopped`,
  // `rafHandle`, `disposed`, `unsubFrame/Progress/Error`, `subscribeToEngine`,
  // and `tick` bindings — those remain outer-scope because they are shared with
  // the RAF loop and dispose path.
  const autoRecreateMachine = {
    times: [] as number[],
    recreating: false,
  };

  const handleFatalEngineError = (err: EngineError): void => {
    if (!opts.autoRecreateOnDeviceLoss) return;
    if (!isLossKind(err.kind)) return;
    if (disposed || autoRecreateMachine.recreating) return;

    const now = Date.now();
    // Prune attempts outside the window.
    while (autoRecreateMachine.times.length > 0 && now - autoRecreateMachine.times[0]! > AUTO_RECREATE_WINDOW_MS) {
      autoRecreateMachine.times.shift();
    }
    if (autoRecreateMachine.times.length >= AUTO_RECREATE_MAX_ATTEMPTS) {
      // Cap exceeded — final error already delivered above; stay stopped.
      console.error(
        `[attachVitrum] auto-recreate cap (${AUTO_RECREATE_MAX_ATTEMPTS} attempts / ` +
        `${AUTO_RECREATE_WINDOW_MS}ms) exceeded; RAF loop will not be restarted.`,
      );
      return;
    }

    autoRecreateMachine.recreating = true;
    autoRecreateMachine.times.push(now);

    // Fire-and-forget; errors inside recreate are surface-logged.
    void performAutoRecreate();
  };

  const performAutoRecreate = async (): Promise<void> => {
    // 1. Stop the RAF loop while we recreate.
    stopped = true;
    if (rafHandle != null) {
      cancelAnimationFrame(rafHandle);
      rafHandle = null;
    }

    // 2. Export GI state before dispose (walkaround-hybrid backend only).
    const giEngine = engine as Engine & Partial<GIStatePersistable>;
    let savedGI: GIStateSnapshot | null = null;
    try {
      if (typeof giEngine.exportGIState === 'function') {
        savedGI = await giEngine.exportGIState();
      }
    } catch (err) {
      reportError(err, {
        phase: 'attach:gi-export',
        backend: engine.backendId,
        recoverable: true,
      });
      // Best-effort; proceed without GI state.
    }

    // Snapshot the backend-retained scene immediately before teardown. The
    // lifecycle tracks explicit handle.engine.setScene(...) calls, but whole-
    // primitive/controller fast paths can mutate the backend scene through
    // add/remove/update routes without passing through that wrapper.
    try {
      const liveScene = typeof engine.getScene === 'function' ? engine.getScene() : null;
      if (liveScene != null) currentScene = liveScene;
    } catch (err) {
      reportError(err, {
        phase: 'attach:auto-recreate',
        backend: engine.backendId,
        recoverable: true,
      });
    }

    // 3. Dispose the broken engine.
    unsubFrame?.();
    unsubProgress?.();
    unsubEngineError?.();
    unsubFrame = undefined;
    unsubProgress = undefined;
    unsubEngineError = undefined;
    try { engine.dispose(); } catch { /* best-effort cleanup before recreate — ignore */ }

    // 4. Recreate.
    try {
      engine = await buildEngineFromOpts(opts, currentScene);
      trackEngineScene(engine);
      attachSceneController(engine);
      lastSceneControllerFrameMs = null;
      webgpuSwapChain = detectWebGPUSwapChain(opts.canvas);
    } catch (createErr) {
      reportError(createErr, {
        phase: 'attach:auto-recreate',
        recoverable: false,
      });
      console.error('[attachVitrum] auto-recreate: createEngine failed:', createErr);
      autoRecreateMachine.recreating = false;
      return;
    }

    // 5. Re-subscribe telemetry on the new engine.
    subscribeToEngine();

    // 6. Restore GI state if available.
    if (savedGI != null) {
      try {
        const newGiEngine = engine as Engine & Partial<GIStatePersistable>;
        if (typeof newGiEngine.importGIState === 'function') {
          newGiEngine.importGIState(savedGI);
        }
      } catch (err) {
        reportError(err, {
          phase: 'attach:gi-import',
          backend: engine.backendId,
          recoverable: true,
        });
        // Best-effort.
      }
    }

    // 7. Resume the RAF loop (unless the handle was disposed while we were recreating).
    autoRecreateMachine.recreating = false;
    if (!disposed) {
      stopped = false;
      rafHandle = requestAnimationFrame(tick);
    }
  };

  // ── RAF loop ─────────────────────────────────────────────────────────────
  let frameIndex = 0;
  let prevView: Mat4 | undefined;
  let prevProj: Mat4 | undefined;
  let rafHandle: number | null = null;
  let stopped = false;
  // H31-d — consecutive-throw counter. After RAF_SELF_STOP_THRESHOLD consecutive
  // renderFrame throws the loop stops itself and reports a non-recoverable error.
  // The counter resets on any successful renderFrame call.
  const RAF_SELF_STOP_THRESHOLD = 5;
  let consecutiveThrows = 0;

  const tick = (now: number = currentFrameTimeMs()): void => {
    if (stopped) return;
    rafHandle = requestAnimationFrame(tick);
    try {
      advanceSceneController(now);
    } catch (err) {
      reportError(err, { phase: 'attach:scene-controller', backend: engine.backendId, recoverable: true });
      console.error('[attachVitrum] sceneController.advance threw:', err);
    }
    opts.camera.updateMatrixWorld();
    const view = asMat4(new Float32Array(opts.camera.matrixWorldInverse.elements));
    const proj = asMat4(new Float32Array(opts.camera.projectionMatrix.elements));
    // A2 — acquire the per-frame swap-chain view for WebGPU backends.
    const swapChainView = acquireSwapChainView(webgpuSwapChain.context, (err) => {
      reportError(err, { phase: 'attach:swapchain', recoverable: true });
    });
    const quality = resolveQualityOption(opts.quality);

    const input = composeAttachVitrumFrameInput({
      viewMatrix: view,
      projMatrix: proj,
      cameraPosition: [opts.camera.position.x, opts.camera.position.y, opts.camera.position.z],
      ...(prevView ? { prevViewMatrix: prevView } : {}),
      ...(prevProj ? { prevProjMatrix: prevProj } : {}),
      viewport: { width: viewportW, height: viewportH, devicePixelRatio: viewportDpr },
      frameIndex,
      ...(quality ? { quality } : {}),
      ...(swapChainView != null
        ? {
            swapChainView,
            ...(webgpuSwapChain.format != null ? { swapChainFormat: webgpuSwapChain.format } : {}),
          }
        : {}),
    });
    try {
      engine.renderFrame(input);
      // Reset the consecutive-throw counter on any successful frame.
      consecutiveThrows = 0;
    } catch (err) {
      consecutiveThrows++;
      if (consecutiveThrows >= RAF_SELF_STOP_THRESHOLD) {
        // H31-d: engine is persistently broken — stop the loop and report
        // non-recoverable so the host knows the engine has halted.
        stopped = true;
        if (rafHandle != null) {
          cancelAnimationFrame(rafHandle);
          rafHandle = null;
        }
        reportError(err, { phase: 'attach:renderFrame', recoverable: false });
        console.error(
          `[attachVitrum] renderFrame threw ${RAF_SELF_STOP_THRESHOLD} consecutive times; ` +
          'RAF loop stopped. Dispose and recreate the engine to recover.',
          err,
        );
        return;
      }
      // Recoverable: surface the error but keep the loop alive.
      reportError(err, { phase: 'attach:renderFrame', recoverable: true });
      console.error('[attachVitrum] renderFrame threw:', err);
    }
    prevView = view;
    prevProj = proj;
    frameIndex++;
  };

  rafHandle = requestAnimationFrame(tick);

  let disposed = false;
  return {
    get engine() { return engine; },
    get backendId() { return engine.backendId; },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      stopped = true;
      if (rafHandle != null) cancelAnimationFrame(rafHandle);
      if (visibilityHandler && typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', visibilityHandler);
      }
      try { resizeObserver?.disconnect(); } catch { /* best-effort cleanup — ignore */ }
      try { unsubFrame?.(); } catch { /* best-effort cleanup — ignore */ }
      try { unsubProgress?.(); } catch { /* best-effort cleanup — ignore */ }
      try { unsubEngineError?.(); } catch { /* best-effort cleanup — ignore */ }
      try { engine.dispose(); } catch { /* best-effort cleanup — ignore */ }
    },
    captureFrame: (captureOpts?: CaptureFrameOptions) => {
      if (typeof engine.captureFrame !== 'function') return Promise.resolve(null);
      return engine.captureFrame(captureOpts);
    },
  };
}

/** Return true when the given error kind represents a device/context loss. */
function isLossKind(kind: EngineError['kind']): boolean {
  return kind === 'device-lost' || kind === 'context-lost';
}

function isSceneControllerPlaybackEnabled(
  playback: AttachVitrumSceneControllerPlayback | undefined,
): playback is true | AttachVitrumSceneControllerPlaybackOptions {
  return playback === true || (typeof playback === 'object' && playback != null);
}

function sceneControllerPlaybackLoop(
  playback: true | AttachVitrumSceneControllerPlaybackOptions,
): boolean {
  return typeof playback === 'object' && playback.loop !== undefined
    ? playback.loop
    : true;
}

function currentFrameTimeMs(): number {
  const perf = globalThis.performance;
  const now = perf?.now?.();
  return Number.isFinite(now) ? now : Date.now();
}
