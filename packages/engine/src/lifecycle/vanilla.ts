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

import * as THREE from 'three';
import type { Engine, Scene, FrameInput, FrameStats, ProgressStats, Mat4 } from '@vitrum/core';
import { asBackendTexture, asBackendTextureFormat, asMat4 } from '@vitrum/core';
import { createEngine, type CreateEngineOptions } from '../createEngine.js';

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

/** Output of {@link detectWebGPUSwapChain}. Both fields null ⇒ WebGL/host
 *  is not WebGPU-backed (or canvas lacks a WebGPU context); attachVitrum
 *  will then leave `FrameInput.swapChainView` undefined.
 *  File-local — callers consume the type structurally via inference.
 *  2026-05-18 dead-code sweep verified zero non-self consumers. */
interface WebGPUSwapChainInfo {
  readonly context: GPUCanvasContext | null;
  readonly format: GPUTextureFormat | undefined;
}

/** Detect whether the canvas currently has a WebGPU context (configured by
 *  createEngine's walkaround backend constructor) and recover its format.
 *  Returns `{ context: null, format: undefined }` for WebGL hosts (where
 *  three.js's WebGLRenderer claims the canvas's `webgl2` context, leaving
 *  `getContext('webgpu')` returning null) or for test environments without
 *  WebGPU support.
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
    ? (q as () => NonNullable<FrameInput['quality']> | undefined)()
    : (q as NonNullable<FrameInput['quality']>);
}

/** Per WebGPU spec, `getCurrentTexture()` MUST be called inside the rAF
 *  tick and the resulting view is single-use (do NOT cache across frames).
 *  Returns undefined when the context is null (WebGL host) or acquisition
 *  throws (canvas size zero, context lost) — caller leaves
 *  `FrameInput.swapChainView` undefined so HybridEngine skips the frame.
 *
 *  @internal Exported for unit-test access. */
export function acquireSwapChainView(ctx: GPUCanvasContext | null): GPUTextureView | undefined {
  if (ctx == null) return undefined;
  try {
    return ctx.getCurrentTexture().createView();
  } catch {
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

export interface AttachVitrumOptions extends Omit<CreateEngineOptions, 'scene'> {
  /** Scene description. Either a vitrum Scene or a THREE.Scene. */
  readonly scene: Scene | THREE.Scene;
  /** THREE camera the engine reads viewMatrix / projMatrix / position from
   *  every frame. The host mutates this camera (orbit controls, scripted
   *  animation) and the helper pushes the latest matrices into renderFrame. */
  readonly camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  /** Per-frame quality dials. Honoured if non-null; otherwise the engine's
   *  defaults apply.
   *
   *  Pass a value for static quality, or a `() => quality | undefined`
   *  getter for live propagation (the RAF tick invokes it every frame, so
   *  React refs / mutable state can swap quality without engine recreation).
   *  React's `<VitrumCanvas>` uses the getter form internally so its
   *  `quality` prop propagates without remount. */
  readonly quality?:
    | NonNullable<FrameInput['quality']>
    | (() => NonNullable<FrameInput['quality']> | undefined);
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

export interface AttachVitrumHandle {
  /** The underlying Engine. Hosts can subscribe to telemetry or call
   *  reset() / pause() / resume() directly via this handle. */
  readonly engine: Engine;
  /** Stop the RAF loop, disconnect the ResizeObserver, unsubscribe from
   *  document.visibilitychange, and dispose the engine. Idempotent. */
  dispose(): void;
}

export async function attachVitrum(opts: AttachVitrumOptions): Promise<AttachVitrumHandle> {
  const engine = await createEngine({
    canvas: opts.canvas,
    scene: opts.scene as Parameters<typeof createEngine>[0]['scene'],
    ...(opts.prefer != null ? { prefer: opts.prefer } : {}),
    ...(opts.advanced != null ? { advanced: opts.advanced } : {}),
    ...(opts.debug != null ? { debug: opts.debug } : {}),
  });

  // Telemetry forwarders. We use the engine's own subscription API rather
  // than wrapping renderFrame because subscribers must run even if the
  // host-supplied onFrame throws (engine swallows callback throws).
  const unsubFrame = opts.onFrame && engine.onFrame
    ? engine.onFrame(opts.onFrame)
    : undefined;
  const unsubProgress = opts.onProgress && engine.onProgress
    ? engine.onProgress(opts.onProgress)
    : undefined;

  // Resize: track the canvas's CSS pixel size + DPR. Renderframe receives
  // the latest values via FrameInput.viewport.
  //
  // A4 — generic PT engines honour viewport-per-frame, but HybridEngine
  // (WebGPU walkaround) does not — its DDGI atlas / ReSTIR reservoirs /
  // history textures / accumulation buffer are sized at construction and
  // can only be resized via `setSize(w, h)`. Call it when available so
  // WebGPU hosts using attachVitrum get correct resize behaviour out of box.
  let viewportW = Math.max(1, Math.floor(opts.canvas.width));
  let viewportH = Math.max(1, Math.floor(opts.canvas.height));
  let viewportDpr = (typeof window !== 'undefined' ? window.devicePixelRatio : null) ?? 1;
  const engineSetSize = engine.setSize;
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
      if (typeof engineSetSize === 'function') {
        try { engineSetSize(viewportW, viewportH); } catch {}
      }
    });
    resizeObserver.observe(opts.canvas);
  }

  // A2 — WebGPU backend detection. createEngine's walkaround constructor
  // configured the canvas's WebGPU context; we re-observe it here so the
  // RAF tick can acquire a fresh GPUTextureView per frame and pass it as
  // FrameInput.swapChainView. WebGL hosts return { context: null } and the
  // tick omits the swap-chain fields.
  const { context: webgpuContext, format: webgpuFormat } = detectWebGPUSwapChain(opts.canvas);

  // Pause-on-hidden. Default ON.
  const pauseOnHidden = opts.pauseOnHidden ?? true;
  let visibilityHandler: (() => void) | undefined;
  if (pauseOnHidden && typeof document !== 'undefined') {
    visibilityHandler = () => {
      if (document.visibilityState === 'hidden') {
        try { engine.pause(); } catch {}
      } else {
        try { engine.resume(); } catch {}
      }
    };
    document.addEventListener('visibilitychange', visibilityHandler);
    // Apply initial state.
    visibilityHandler();
  }

  // RAF loop.
  let frameIndex = 0;
  let prevView: Mat4 | undefined;
  let prevProj: Mat4 | undefined;
  let rafHandle: number | null = null;
  let stopped = false;

  const tick = (): void => {
    if (stopped) return;
    rafHandle = requestAnimationFrame(tick);
    opts.camera.updateMatrixWorld();
    const view = asMat4(new Float32Array(opts.camera.matrixWorldInverse.elements));
    const proj = asMat4(new Float32Array(opts.camera.projectionMatrix.elements));
    // A2 — acquire the per-frame swap-chain view for WebGPU backends.
    const swapChainView = acquireSwapChainView(webgpuContext);
    const quality = resolveQualityOption(opts.quality);

    const input: FrameInput = {
      viewMatrix: view,
      projMatrix: proj,
      cameraPosition: [opts.camera.position.x, opts.camera.position.y, opts.camera.position.z],
      ...(prevView ? { prevViewMatrix: prevView } : {}),
      ...(prevProj ? { prevProjMatrix: prevProj } : {}),
      viewport: { width: viewportW, height: viewportH, devicePixelRatio: viewportDpr },
      frameIndex,
      frameSeed: (frameIndex * 1664525 + 1013904223) >>> 0,
      ...(quality ? { quality } : {}),
      ...(swapChainView != null
        ? {
            swapChainView: asBackendTexture<'webgpu', GPUTextureView>(swapChainView),
            ...(webgpuFormat != null
              ? { swapChainFormat: asBackendTextureFormat<'webgpu', GPUTextureFormat>(webgpuFormat) }
              : {}),
          }
        : {}),
    };
    try {
      engine.renderFrame(input);
    } catch (err) {
      // Surface engine errors but don't kill the loop — the engine has its
      // own error state that the host can observe via engine.state.
      console.error('[attachVitrum] renderFrame threw:', err);
    }
    prevView = view;
    prevProj = proj;
    frameIndex++;
  };

  rafHandle = requestAnimationFrame(tick);

  let disposed = false;
  return {
    engine,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      stopped = true;
      if (rafHandle != null) cancelAnimationFrame(rafHandle);
      if (visibilityHandler && typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', visibilityHandler);
      }
      try { resizeObserver?.disconnect(); } catch {}
      try { unsubFrame?.(); } catch {}
      try { unsubProgress?.(); } catch {}
      try { engine.dispose(); } catch {}
    },
  };
}
