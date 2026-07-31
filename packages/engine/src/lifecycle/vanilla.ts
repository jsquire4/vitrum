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

import type { CapturedFrame, CaptureFrameOptions, Engine, EngineError, EngineWarning, FrameInput, FrameOutput, FrameStats, ProgressStats, Mat4 } from '@vitrum/core';
import { asBackendTexture, asBackendTextureFormat, asMat4 } from '@vitrum/core';
import { createEngine, type CreateEngineErrorEvent, type CreateEngineOptions } from '../createEngine.js';
import { createOffscreenPresenter, type OffscreenPresenter } from '../presentOffscreen.js';
import type { GIStatePersistable } from '../idempotentDispose.js';
import type { GIStateSnapshot } from '@vitrum/walkaround-hybrid';
import type { RuntimeEngineWithBackendId as EngineWithBackendId } from '../createEngineInternals.js';

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
    try {
      onError?.(err);
    } catch {
      // Host error callbacks must not break best-effort swap-chain acquisition.
    }
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

export type AttachVitrumAutoRecreateReason = 'device-lost' | 'context-lost';

export interface AttachVitrumRecreateEngineContext {
  readonly scene: AttachVitrumOptions['scene'];
  readonly previousBackendId: EngineWithBackendId['backendId'];
  readonly reason: AttachVitrumAutoRecreateReason;
  readonly attempt: number;
}

export type AttachVitrumRecreateEngineFactory = (
  context: AttachVitrumRecreateEngineContext,
) => Promise<EngineWithBackendId>;

/**
 * Emitted when fatal-loss recovery succeeds on a different backend or backend
 * profile. Auto selection is intentionally allowed to fall back as host/device
 * availability changes, but that semantic change is never silent.
 */
export interface AttachVitrumBackendChangedEvent {
  readonly previousBackendId: EngineWithBackendId['backendId'];
  readonly backendId: EngineWithBackendId['backendId'];
  readonly previousBackendProfileId: EngineWithBackendId['backendProfileId'];
  readonly backendProfileId: EngineWithBackendId['backendProfileId'];
  readonly previousProfileId: EngineWithBackendId['profileId'];
  readonly profileId: EngineWithBackendId['profileId'];
  readonly reason: AttachVitrumAutoRecreateReason;
  readonly attempt: number;
}

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
   * auto-recreate uses `recreateEngine` when supplied, otherwise it falls back
   * to `createEngine()` using the latest tracked scene.
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
   *  4. For a WebGL context loss, recreation waits for the canvas's
   *     `webglcontextrestored` event; no resources are created against a
   *     still-lost context.
   *  5. The engine is disposed.
   *  6. `createEngine` is called again with the original options, minting a
   *     fresh GPU device automatically.
   *  7. If GI state was exported and the new engine exposes `importGIState`,
   *     it is imported — warm DDGI probes survive the recreate.
   *  8. The RAF loop resumes.
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
  /**
   * Optional factory for fatal auto-recreate when the initial engine came from
   * a higher-level facade that `createEngine()` cannot reproduce by itself
   * (for example the progressive walkaround->pt-webgpu coordinator).
   * Normal attachVitrum callers should omit this and use the default
   * `createEngine()` recreation path.
   */
  readonly recreateEngine?: AttachVitrumRecreateEngineFactory;
  /**
   * Called after fatal-loss recovery installs an engine whose backend or
   * resolved profile differs from the engine that was lost. A matching
   * structured warning is also emitted through `onWarning`.
   */
  readonly onBackendChanged?: (event: AttachVitrumBackendChangedEvent) => void;
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
  readonly engine: EngineWithBackendId;
  /** Backend selected by createEngine for the currently attached engine.
   *  Auto-recreate keeps the same stable handle object, so this is exposed as a
   *  live getter and updates if a recreated engine lands on a different backend. */
  readonly backendId: EngineWithBackendId['backendId'];
  /** Live resolved backend profile; updates after auto-recreate. */
  readonly backendProfileId: EngineWithBackendId['backendProfileId'];
  /** Live legacy-compatible profile identity; forwarded without inference. */
  readonly profileId: EngineWithBackendId['profileId'];
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


interface WebGlContextRestorationWait {
  readonly promise: Promise<void>;
  cancel(): void;
}

/**
 * Return a cancellable wait only when this canvas currently owns a genuinely
 * lost WebGL2 context. Synthetic lifecycle tests and non-WebGL backends have no
 * lost context and continue directly to recreation.
 */
function waitForLostWebGlContextRestoration(
  canvas: HTMLCanvasElement,
): WebGlContextRestorationWait | null {
  let gl: WebGL2RenderingContext | null;
  try {
    gl = canvas.getContext('webgl2');
  } catch {
    return null;
  }
  if (gl == null || typeof gl.isContextLost !== 'function' || !gl.isContextLost()) {
    return null;
  }

  let settled = false;
  let resolveWait!: () => void;
  const finish = (): void => {
    if (settled) return;
    settled = true;
    canvas.removeEventListener('webglcontextrestored', finish);
    resolveWait();
  };
  const promise = new Promise<void>((resolve) => {
    resolveWait = resolve;
    canvas.addEventListener('webglcontextrestored', finish, { once: true });
  });
  return { promise, cancel: finish };
}
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
  const reportWarning = (warning: EngineWarning): void => {
    console.warn(warning.message);
    try {
      opts.onWarning?.(warning);
    } catch { /* host warning callback must not propagate — ignore */ }
  };

  // ── Resize state ─────────────────────────────────────────────────────────
  //
  // H30 — synchronous initial sizing: set canvas.width/height from CSS client
  // size × DPR BEFORE engine construction so WebGPU walkaround can allocate its
  // swapchain/history resources at the real visible size. Without this, the
  // constructor sees the browser's default 300×150 backing store and only catches
  // up after a later ResizeObserver tick.
  //
  // Every backend honours viewport-per-frame. HybridEngine additionally exposes
  // `setSize(w, h)` so attachVitrum can eagerly publish replacement screen-space
  // resources from ResizeObserver instead of deferring that allocation to the
  // next renderFrame call.
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
  let resizeObserver: ResizeObserver | undefined;
  let visibilityHandler: (() => void) | undefined;
  let unsubFrame: (() => void) | undefined;
  let unsubProgress: (() => void) | undefined;
  let unsubEngineError: (() => void) | undefined;
  let subscribingToEngine = false;
  let fatalErrorDeferredDuringSubscribe: EngineError | undefined;
  let presenter: OffscreenPresenter | undefined;
  let presenterDevice: GPUDevice | undefined;
  let presenterContext: GPUCanvasContext | undefined;
  let rafHandle: number | null = null;
  let stopped = false;
  let disposed = false;
  let autoRecreateController:
    | {
        handleFatalEngineError: (err: EngineError) => void;
        cancelPendingWait: () => void;
      }
    | undefined;

  /**
   * One idempotent teardown path serves normal disposal and initialization
   * rollback. In particular, every failure after engine construction releases
   * partial DOM observers/listeners, partial telemetry subscriptions, RAF
   * ownership, presentation resources, and the engine before attach rejects.
   */
  const disposeLifecycle = (): void => {
    if (disposed) return;
    disposed = true;
    stopped = true;
    if (rafHandle != null) {
      try { cancelAnimationFrame(rafHandle); } catch { /* best-effort cleanup — ignore */ }
      rafHandle = null;
    }
    if (visibilityHandler && typeof document !== 'undefined') {
      try {
        document.removeEventListener('visibilitychange', visibilityHandler);
      } catch {
        // Best-effort cleanup for partially initialized hosts.
      }
    }
    try { resizeObserver?.disconnect(); } catch { /* best-effort cleanup — ignore */ }
    try { unsubFrame?.(); } catch { /* best-effort cleanup — ignore */ }
    try { unsubProgress?.(); } catch { /* best-effort cleanup — ignore */ }
    try { unsubEngineError?.(); } catch { /* best-effort cleanup — ignore */ }
    unsubFrame = undefined;
    unsubProgress = undefined;
    unsubEngineError = undefined;
    try { presenter?.dispose(); } catch { /* best-effort cleanup — ignore */ }
    presenter = undefined;
    presenterDevice = undefined;
    autoRecreateController?.cancelPendingWait();
    try { engine.dispose(); } catch { /* best-effort cleanup — ignore */ }
  };

  try {
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
      // `setSize` itself is the resize capability. Presentation mode does not
      // imply it: the progressive facade is swapchain-optional but must fan the
      // new dimensions out to both of its phases, while ordinary offscreen
      // backends simply omit the method and continue to use FrameInput.viewport.
      try {
        engine.setSize?.(viewportW, viewportH);
      } catch (err) {
        reportError(err, { phase: 'attach:resize', recoverable: true });
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
  const subscribeToEngine = (): void => {
    unsubFrame?.();
    unsubProgress?.();
    unsubEngineError?.();
    unsubFrame = undefined;
    unsubProgress = undefined;
    unsubEngineError = undefined;
    let completed = false;
    subscribingToEngine = true;
    try {
      unsubFrame = opts.onFrame && engine.onFrame
        ? engine.onFrame(opts.onFrame)
        : undefined;
      if (disposed) return;
      unsubProgress = opts.onProgress && engine.onProgress
        ? engine.onProgress(opts.onProgress)
        : undefined;
      if (disposed) return;
      unsubEngineError = engine.onError
        ? engine.onError((err) => {
            // Always deliver to the host first, before any internal handling.
            try { opts.onEngineError?.(err); } catch { /* host error callback must not propagate — ignore */ }
            if (!err.fatal) return;
            // Custom engines are allowed to invoke a subscription callback
            // synchronously. Defer recovery until the unsubscriber has been
            // published, otherwise recovery can dispose the engine before this
            // subscription exists and strand the late callback.
            if (subscribingToEngine) {
              fatalErrorDeferredDuringSubscribe ??= err;
            } else {
              autoRecreateController?.handleFatalEngineError(err);
            }
          })
        : undefined;
      completed = true;
    } finally {
      subscribingToEngine = false;
      const deferredFatal = fatalErrorDeferredDuringSubscribe;
      fatalErrorDeferredDuringSubscribe = undefined;
      if (!completed || disposed) {
        try { unsubFrame?.(); } catch { /* best-effort cleanup — ignore */ }
        try { unsubProgress?.(); } catch { /* best-effort cleanup — ignore */ }
        try { unsubEngineError?.(); } catch { /* best-effort cleanup — ignore */ }
        unsubFrame = undefined;
        unsubProgress = undefined;
        unsubEngineError = undefined;
      } else if (deferredFatal != null) {
        autoRecreateController?.handleFatalEngineError(deferredFatal);
      }
    }
  };

  // ── Auto-recreate controller ─────────────────────────────────────────────
  // The fatal-loss state machine (retry budget, GI-state save/restore, teardown,
  // recreate) is grouped into one named controller so its ownership and mutation
  // surface are explicit and it can be reasoned about as a unit. It still closes
  // over the outer `engine`, `stopped`, `rafHandle`, `disposed`,
  // `presenter*`, `unsubFrame/Progress/Error`, `subscribeToEngine`, and `tick`
  // bindings by lexical scope — those remain outer-scope because they are shared
  // with the RAF loop and dispose path. Constructed after `subscribeToEngine`'s
  // definition (its onError callback references the controller lazily at
  // fire-time) and before the first `subscribeToEngine()` invocation.
  const createAutoRecreateController = (): {
    handleFatalEngineError: (err: EngineError) => void;
    cancelPendingWait: () => void;
  } => {
  // Retry budget: at most AUTO_RECREATE_MAX_ATTEMPTS within AUTO_RECREATE_WINDOW_MS.
  const autoRecreateMachine = {
    times: [] as number[],
    recreating: false,
    cancelContextRestorationWait: null as (() => void) | null,
    pendingFatalError: null as EngineError | null,
  };

  const handleFatalEngineError = (err: EngineError): void => {
    if (!opts.autoRecreateOnDeviceLoss) return;
    if (!isLossKind(err.kind)) return;
    if (disposed) return;
    if (autoRecreateMachine.recreating) {
      // A newly created custom engine may synchronously report loss while its
      // telemetry subscriptions are installed. Preserve that signal and run a
      // fresh bounded recovery after the current attempt unwinds.
      autoRecreateMachine.pendingFatalError ??= err;
      return;
    }

    const now = Date.now();
    // Prune attempts outside the window.
    while (autoRecreateMachine.times.length > 0 && now - autoRecreateMachine.times[0]! > AUTO_RECREATE_WINDOW_MS) {
      autoRecreateMachine.times.shift();
    }
    if (autoRecreateMachine.times.length >= AUTO_RECREATE_MAX_ATTEMPTS) {
      // V1-3 — Cap exceeded: the final error was already delivered above, but
      // the loop and the fatal engine still leak. Mirror the H31-d self-stop
      // teardown: stop the loop, cancel the pending RAF, unsubscribe telemetry,
      // and dispose the broken engine. Without this the RAF keeps re-throwing
      // for ~5 frames (until the self-stop threshold) and the engine is never
      // torn down, leaking its GPU resources.
      stopped = true;
      if (rafHandle != null) {
        cancelAnimationFrame(rafHandle);
        rafHandle = null;
      }
      unsubFrame?.();
      unsubProgress?.();
      unsubEngineError?.();
      unsubFrame = undefined;
      unsubProgress = undefined;
      unsubEngineError = undefined;
      try { presenter?.dispose(); } catch { /* best-effort cleanup after cap — ignore */ }
      presenter = undefined;
      presenterDevice = undefined;
      try { engine.dispose(); } catch { /* best-effort cleanup after cap — ignore */ }
      console.error(
        `[attachVitrum] auto-recreate cap (${AUTO_RECREATE_MAX_ATTEMPTS} attempts / ` +
        `${AUTO_RECREATE_WINDOW_MS}ms) exceeded; RAF loop stopped and engine disposed.`,
      );
      return;
    }

    autoRecreateMachine.recreating = true;
    autoRecreateMachine.times.push(now);

    // Fire-and-forget; errors inside recreate are surface-logged.
    void performAutoRecreate(err);
  };

  const performAutoRecreate = async (lossError: EngineError): Promise<void> => {
    // 1. Stop the RAF loop while we recreate.
    stopped = true;
    if (rafHandle != null) {
      cancelAnimationFrame(rafHandle);
      rafHandle = null;
    }
    const previousBackendId = engine.backendId;
    const previousBackendProfileId = engine.backendProfileId;
    const previousProfileId = engine.profileId;
    const recreateAttempt = autoRecreateMachine.times.length;
    const recreateReason: AttachVitrumAutoRecreateReason = lossError.kind === 'context-lost'
      ? 'context-lost'
      : 'device-lost';

    // 2. Export GI state before dispose (walkaround-hybrid backend only).
    const giEngine = engine as Engine & Partial<GIStatePersistable>;
    let savedGI: GIStateSnapshot | null = null;
    try {
      const pendingGIState = giEngine.exportGIState?.();
      if (pendingGIState != null) {
        savedGI = await pendingGIState;
      }
    } catch (err) {
      reportError(err, {
        phase: 'attach:gi-export',
        backend: engine.backendId,
        recoverable: true,
      });
      // Best-effort; proceed without GI state.
    }
    if (disposed) {
      autoRecreateMachine.recreating = false;
      return;
    }

    // A WebGL context is unusable until the platform restores it. Rebuilding
    // immediately can return the same still-lost context from getContext() and
    // then allocate an engine whose every GL call is invalid. Keep the old
    // engine alive long enough for its preventDefault()-enabled restoration
    // listener to do its job, then tear it down and recreate.
    if (recreateReason === 'context-lost') {
      const restoration = waitForLostWebGlContextRestoration(opts.canvas);
      if (restoration != null) {
        autoRecreateMachine.cancelContextRestorationWait = () => restoration.cancel();
        await restoration.promise;
        autoRecreateMachine.cancelContextRestorationWait = null;
        if (disposed) {
          autoRecreateMachine.recreating = false;
          return;
        }
      }
    }

    // Snapshot the backend-retained scene immediately before teardown. The
    // lifecycle tracks explicit handle.engine.setScene(...) calls, but whole-
    // primitive/controller fast paths can mutate the backend scene through
    // add/remove/update routes without passing through that wrapper.
    //
    // The two snapshot-unavailable warnings (getScene() returned null vs. the
    // engine has no getScene()) are byte-identical except for the middle clause,
    // so they share one emitter to keep the code + details shape single-source.
    const warnSceneSnapshotUnavailable = (becauseClause: string): void => {
      reportWarning({
        code: 'attachVitrum.auto-recreate-scene-snapshot-unavailable',
        backend: engine.backendId,
        phase: 'lifecycle',
        method: 'attachVitrum',
        message:
          '[attachVitrum] auto-recreate could not snapshot the backend-retained live scene because ' +
          becauseClause +
          '; recreating from the latest scene seen by attachVitrum.',
        details: {
          backendId: engine.backendId,
          fallback: 'tracked-scene',
        },
      });
    };
    try {
      if (typeof engine.getScene === 'function') {
        const liveScene = engine.getScene();
        if (liveScene != null) {
          currentScene = liveScene;
        } else {
          warnSceneSnapshotUnavailable('getScene() returned null');
        }
      } else {
        warnSceneSnapshotUnavailable('the current engine does not implement getScene()');
      }
    } catch (err) {
      reportError(err, {
        phase: 'attach:auto-recreate',
        backend: engine.backendId,
        recoverable: true,
      });
    }
    // Snapshot warnings/errors are host callbacks and may synchronously dispose
    // the stable handle. Disposal is terminal: do not invoke a recreate factory
    // after the host has explicitly ended this lifecycle.
    if (disposed) {
      autoRecreateMachine.recreating = false;
      return;
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
    let recreatedCandidate: EngineWithBackendId | undefined;
    try {
      const recreated = opts.recreateEngine != null
        ? await opts.recreateEngine({
            scene: currentScene,
            previousBackendId,
            reason: recreateReason,
            attempt: recreateAttempt,
          })
        : await buildEngineFromOpts(opts, currentScene);
      recreatedCandidate = recreated;
      // V1-2 — a dispose() that arrived while we were awaiting the (async)
      // recreate would otherwise be lost: `disposed` was only re-checked at
      // step 7 (post-install), so the freshly-minted engine got installed +
      // subscribed and never torn down. Re-check here, before install, and if
      // the handle was disposed mid-recreate, dispose the late engine and
      // reset the recreate state machine instead of installing it.
      if (disposed) {
        try { recreated.dispose(); } catch { /* best-effort cleanup of late engine — ignore */ }
        autoRecreateMachine.recreating = false;
        return;
      }
      trackEngineScene(recreated);
      attachSceneController(recreated);
      if (disposed) {
        try { recreated.dispose(); } catch { /* best-effort cleanup of late engine — ignore */ }
        recreatedCandidate = undefined;
        autoRecreateMachine.recreating = false;
        return;
      }
      engine = recreated;
      recreatedCandidate = undefined;
      lastSceneControllerFrameMs = null;
      webgpuSwapChain = detectWebGPUSwapChain(opts.canvas);

      if (
        engine.backendId !== previousBackendId ||
        engine.backendProfileId !== previousBackendProfileId ||
        engine.profileId !== previousProfileId
      ) {
        const event: AttachVitrumBackendChangedEvent = {
          previousBackendId,
          backendId: engine.backendId,
          previousBackendProfileId,
          backendProfileId: engine.backendProfileId,
          previousProfileId,
          profileId: engine.profileId,
          reason: recreateReason,
          attempt: recreateAttempt,
        };
        reportWarning({
          code: 'attachVitrum.auto-recreate-backend-changed',
          backend: engine.backendId,
          phase: 'lifecycle',
          method: 'attachVitrum',
          message:
            `[attachVitrum] auto-recreate changed backend/profile from ` +
            `${previousBackendId}/${String(previousBackendProfileId)} to ` +
            `${engine.backendId}/${String(engine.backendProfileId)}.`,
          details: { ...event },
        });
        // Warning delivery is a host callback boundary. If the host disposes
        // the stable handle there, disposal is terminal and no later lifecycle
        // callback may run against the now-disposed replacement engine.
        if (disposed) {
          autoRecreateMachine.recreating = false;
          return;
        }
        try {
          opts.onBackendChanged?.(event);
        } catch {
          // Host lifecycle callbacks must not break a successful recovery.
        }
      }
      if (disposed) {
        autoRecreateMachine.recreating = false;
        return;
      }
    } catch (createErr) {
      try { recreatedCandidate?.dispose(); } catch { /* best-effort cleanup — ignore */ }
      reportError(createErr, {
        phase: 'attach:auto-recreate',
        recoverable: false,
      });
      console.error('[attachVitrum] auto-recreate: createEngine failed:', createErr);
      autoRecreateMachine.recreating = false;
      return;
    }

    // 5. Re-subscribe telemetry on the new engine.
    try {
      subscribeToEngine();
    } catch (subscriptionError) {
      reportError(subscriptionError, {
        phase: 'attach:auto-recreate',
        backend: engine.backendId,
        recoverable: false,
      });
      console.error(
        '[attachVitrum] auto-recreate: telemetry subscription failed:',
        subscriptionError,
      );
      autoRecreateMachine.pendingFatalError = null;
      autoRecreateMachine.recreating = false;
      // subscribeToEngine rolls back partial subscription handles in its
      // finally block. End the stable lifecycle as well so the installed
      // replacement engine, DOM hooks, presenter, and RAF ownership cannot be
      // stranded in a stopped-but-live state.
      disposeLifecycle();
      return;
    }
    if (disposed) {
      autoRecreateMachine.recreating = false;
      return;
    }

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
    const pendingFatalError = autoRecreateMachine.pendingFatalError;
    autoRecreateMachine.pendingFatalError = null;
    if (!disposed && pendingFatalError != null) {
      handleFatalEngineError(pendingFatalError);
      return;
    }
    if (!disposed) {
      stopped = false;
      rafHandle = requestAnimationFrame(tick);
    }
  };

    return {
      handleFatalEngineError,
      cancelPendingWait: () => {
        autoRecreateMachine.cancelContextRestorationWait?.();
        autoRecreateMachine.cancelContextRestorationWait = null;
        autoRecreateMachine.pendingFatalError = null;
      },
    };
  };
  autoRecreateController = createAutoRecreateController();

  // ── Offscreen-backend presentation ───────────────────────────────────────
  // V1-1 / R2 — offscreen-texture backends (pt-webgpu, and the progressive
  // walkaround→PT facade once it hands off to pt-webgpu) render to an internal
  // GPUTexture and never present to the canvas themselves. When the active engine
  // exposes `getPresentationSource()` returning a `{ device, texture }` for the
  // frame, blit it onto the canvas's WebGPU context so an auto-selected pt-webgpu
  // engine is not black and the progressive handoff doesn't freeze on the last
  // realtime frame. Swapchain-required backends (walkaround) present themselves and
  // return null (or omit the method) → no presenter is constructed for them.
  //
  // The presenter is lazily built on the first non-null source and keyed to that
  // source's device: an auto-recreate that mints a new device rebuilds it. Its
  // context is the canvas's `webgpu` context (the same instance the shared-device
  // progressive facade configured), reused across frames.
  const presentOffscreenFrame = (output: FrameOutput): void => {
    if (typeof engine.getPresentationSource !== 'function') return;
    if (output.kind !== 'rendered') return;
    let source: { device: unknown; texture: unknown } | null;
    try {
      source = engine.getPresentationSource();
    } catch (err) {
      reportError(err, { phase: 'attach:present', recoverable: true });
      return;
    }
    if (source == null) return;
    const device = source.device as GPUDevice | undefined;
    const texture = source.texture as GPUTexture | undefined;
    if (device == null || texture == null) return;
    try {
      if (presenter == null || presenterDevice !== device) {
        // (Re)build the presenter for this device. A prior presenter (from an
        // auto-recreated device) is dropped; the render pipeline/sampler GC with it.
        presenter?.dispose();
        const ctx = presenterContext ?? (opts.canvas.getContext('webgpu')) ?? undefined;
        if (ctx == null) return;
        presenterContext = ctx;
        presenter = createOffscreenPresenter({ device, context: ctx });
        presenterDevice = device;
      }
      presenter.present(texture);
    } catch (err) {
      reportError(err, { phase: 'attach:present', recoverable: true });
    }
  };

  // ── RAF loop ─────────────────────────────────────────────────────────────
  let frameIndex = 0;
  let prevView: Mat4 | undefined;
  let prevProj: Mat4 | undefined;
  // H31-d — consecutive-throw counter. After RAF_SELF_STOP_THRESHOLD consecutive
  // renderFrame throws the loop stops itself and reports a non-recoverable error.
  // The counter resets on any successful renderFrame call.
  const RAF_SELF_STOP_THRESHOLD = 5;
  let consecutiveThrows = 0;

  const handleFrameFailure = (
    err: unknown,
    phase: 'attach:frame-preparation' | 'attach:renderFrame',
    operation: 'frame preparation' | 'renderFrame',
  ): boolean => {
    consecutiveThrows++;
    if (consecutiveThrows >= RAF_SELF_STOP_THRESHOLD) {
      stopped = true;
      if (rafHandle != null) {
        cancelAnimationFrame(rafHandle);
        rafHandle = null;
      }
      reportError(err, { phase, recoverable: false });
      console.error(
        `[attachVitrum] ${operation} threw ${RAF_SELF_STOP_THRESHOLD} consecutive times; ` +
        'RAF loop stopped. Dispose and recreate the engine to recover.',
        err,
      );
      return true;
    }
    reportError(err, { phase, recoverable: true });
    console.error(`[attachVitrum] ${operation} threw:`, err);
    return false;
  };

  const tick = (now: number = currentFrameTimeMs()): void => {
    if (stopped) return;
    rafHandle = requestAnimationFrame(tick);
    try {
      advanceSceneController(now);
    } catch (err) {
      reportError(err, { phase: 'attach:scene-controller', backend: engine.backendId, recoverable: true });
      console.error('[attachVitrum] sceneController.advance threw:', err);
    }
    if (stopped || disposed) return;

    let view: Mat4;
    let proj: Mat4;
    let input: FrameInput;
    try {
      opts.camera.updateMatrixWorld();
      view = asMat4(new Float32Array(opts.camera.matrixWorldInverse.elements));
      proj = asMat4(new Float32Array(opts.camera.projectionMatrix.elements));
      // A2 — acquire the per-frame swap-chain view for WebGPU backends.
      const swapChainView = acquireSwapChainView(webgpuSwapChain.context, (err) => {
        reportError(err, { phase: 'attach:swapchain', recoverable: true });
      });
      const quality = resolveQualityOption(opts.quality);
      if (stopped || disposed) return;

      input = composeAttachVitrumFrameInput({
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
    } catch (err) {
      handleFrameFailure(err, 'attach:frame-preparation', 'frame preparation');
      return;
    }
    try {
      const output = engine.renderFrame(input);
      // Reset the consecutive-throw counter on any successful frame.
      consecutiveThrows = 0;
      // V1-1 / R2 — present the FrameOutput for offscreen-texture backends. The
      // output is no longer discarded: if the active engine renders offscreen
      // (pt-webgpu, or the progressive facade after handoff) blit its texture to
      // the canvas. A no-op for swapchain backends that present themselves.
      presentOffscreenFrame(output);
      // Temporal host state describes the last frame that actually reached a
      // renderer output. A throttled/paused skip has no matching history image,
      // and a thrown frame likewise cannot become the reprojection predecessor.
      if (output.kind === 'rendered') {
        prevView = view;
        prevProj = proj;
        frameIndex++;
      }
    } catch (err) {
      if (handleFrameFailure(err, 'attach:renderFrame', 'renderFrame')) return;
    }
  };

  // Subscribe only after every callback target (including the RAF tick and
  // recreate controller) is initialized. A synchronous custom-engine
  // subscription callback can therefore never observe a temporal-dead-zone.
  subscribeToEngine();
  // A synchronous fatal callback may have started recovery while subscribing.
  // In that case recovery owns the next RAF; scheduling here as well would let
  // two independent RAF chains run once recovery clears `stopped`.
  if (!stopped && !disposed) {
    rafHandle = requestAnimationFrame(tick);
  }
  } catch (error) {
    disposeLifecycle();
    throw error;
  }

  return {
    get engine() { return engine; },
    get backendId() { return engine.backendId; },
    get backendProfileId() { return engine.backendProfileId; },
    get profileId() { return engine.profileId; },
    dispose: disposeLifecycle,
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
