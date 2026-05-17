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
import type { Engine, Scene, FrameInput, FrameStats, Mat4, ProgressStats } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createEngine, type CreateEngineOptions } from '../createEngine.js';

export interface AttachVitrumOptions extends Omit<CreateEngineOptions, 'scene'> {
  /** Scene description. Either a vitrum Scene or a THREE.Scene. */
  readonly scene: Scene | THREE.Scene;
  /** THREE camera the engine reads viewMatrix / projMatrix / position from
   *  every frame. The host mutates this camera (orbit controls, scripted
   *  animation) and the helper pushes the latest matrices into renderFrame. */
  readonly camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  /** Per-frame quality dials. Honoured if non-null; otherwise the engine's
   *  defaults apply. */
  readonly quality?: NonNullable<FrameInput['quality']>;
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
  // the latest values via FrameInput.viewport. Engine never sees this
  // observer — the contract is "viewport per frame".
  let viewportW = Math.max(1, opts.canvas.width);
  let viewportH = Math.max(1, opts.canvas.height);
  let viewportDpr = (typeof window !== 'undefined' ? window.devicePixelRatio : null) ?? 1;
  let resizeObserver: ResizeObserver | undefined;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        viewportW = Math.max(1, Math.floor(cr.width));
        viewportH = Math.max(1, Math.floor(cr.height));
      }
      viewportDpr = (typeof window !== 'undefined' ? window.devicePixelRatio : null) ?? 1;
    });
    resizeObserver.observe(opts.canvas);
  }

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
    const input: FrameInput = {
      viewMatrix: view,
      projMatrix: proj,
      cameraPosition: [opts.camera.position.x, opts.camera.position.y, opts.camera.position.z],
      ...(prevView ? { prevViewMatrix: prevView } : {}),
      ...(prevProj ? { prevProjMatrix: prevProj } : {}),
      viewport: { width: viewportW, height: viewportH, devicePixelRatio: viewportDpr },
      frameIndex,
      frameSeed: (frameIndex * 1664525 + 1013904223) >>> 0,
      ...(opts.quality ? { quality: opts.quality } : {}),
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
