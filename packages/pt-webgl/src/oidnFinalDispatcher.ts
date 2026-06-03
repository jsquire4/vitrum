/**
 * oidnFinalDispatcher.ts — internal kick-and-return state machine for the
 * `denoiser: 'oidn-final'` mode in {@link PTEngineWebGL2}.
 *
 * Wired by W11 follow-up (see plan/premium-grade-refactor-20260517.md §W11).
 * The HybridEngine W11 wire (feat/w11-oidn-wire on the walkaround pipeline)
 * is the sibling implementation; this module brings the same `'oidn-final'`
 * denoiser mode to the pt-webgl converged-frame path.
 *
 * ## Thin wrapper
 *
 * This file is now a thin wrapper over {@link OIDNDispatcherCore} from
 * `@vitrum/shared-denoisers`. The shared core holds the cohort state
 * machine (cohortId / inFlight / haveCompleted / latest / disposed /
 * bridge + methods invalidate / getLatestDenoised / isInFlight / dispose).
 *
 * What this wrapper contributes:
 *  - The WebGL-specific `kickIfReady` signature
 *    (`renderer, target, width, height, divideByAlpha`).
 *  - The synchronous GL readback via `readAccumulationRgbFloat` (one
 *    `gl.readPixels` — the converged-frame cost paid exactly once per
 *    cohort), which is performed BEFORE entering the async cycle.
 *  - `preloadOnBridgeInit: false` (pt-webgl does NOT call preloadOIDNModel
 *    on bridge init, unlike pt-webgpu).
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
import {
  OIDNDispatcherCore,
  oidnDefaultLoader,
} from '@vitrum/shared-denoisers';

// Re-export the shared types so existing importers of this module are unchanged.
export type {
  OIDNFinalDispatcherOptions,
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
} from '@vitrum/shared-denoisers';

import type {
  OIDNFinalDispatcherOptions,
  DenoisedFrame,
  OIDNBridgeLoader,
  ReadbackResult,
} from '@vitrum/shared-denoisers';

/** Input type for the pt-webgl readback callback.
 *  The synchronous GL readback is performed in `kickIfReady` before the async
 *  cycle; the callback merely unpacks the already-read data. */
type WebGLReadbackInput = ReadbackResult;

export class OIDNFinalDispatcher {
  readonly #core: OIDNDispatcherCore<WebGLReadbackInput>;

  constructor(opts: OIDNFinalDispatcherOptions, loader?: OIDNBridgeLoader) {
    if (opts.modelUrl === undefined || opts.modelUrl.length === 0) {
      throw new Error(
        '[OIDNFinalDispatcher] modelUrl is required. ' +
          "Pass oidn: { modelUrl } with denoiser: 'oidn-final'.",
      );
    }
    this.#core = new OIDNDispatcherCore<WebGLReadbackInput>({
      dispatcherOptions: opts,
      loader: loader ?? oidnDefaultLoader,
      // pt-webgl: readback is already done synchronously before kickIfReady
      // hands off to the core; the callback is a pass-through.
      readback: async (input) => input,
      // pt-webgl does NOT call preloadOIDNModel on bridge init (behavioral
      // difference from pt-webgpu — preserved intentionally).
      preloadOnBridgeInit: false,
    });
  }

  /**
   * Returns the most recently completed denoised image for the current
   * invalidation cohort, or null when no inference has yet completed
   * since the last {@link invalidate} call.
   */
  getLatestDenoised(): DenoisedFrame | null {
    return this.#core.getLatestDenoised();
  }

  /** True iff an inference is currently unresolved. Diagnostic only. */
  isInFlight(): boolean {
    return this.#core.isInFlight();
  }

  /**
   * Clear the latest result and arm the dispatcher to re-kick on the next
   * {@link kickIfReady} call. The engine calls this on
   * {@link PTEngineWebGL2.reset}, {@link PTEngineWebGL2.setScene}, and
   * {@link PTEngineWebGL2.updateEnvironment} — any state change that
   * invalidates the accumulator also invalidates the denoised cache.
   *
   * An in-flight inference is allowed to complete, but the result is
   * dropped on resolve (the bumped cohortId in the core catches the race).
   */
  invalidate(): void {
    this.#core.invalidate();
  }

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
    // Guard before paying the readback cost.
    if (width <= 0 || height <= 0) return;

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

    // Hand off to the shared core. The readback is already complete;
    // the callback is a pass-through that just returns the data.
    this.#core.kickIfReady({ color, width, height }, width, height);
  }

  /**
   * Release this engine's cached ONNX session entry (model URL + EP tuple).
   * Falls back to global `clearOIDNCache` only when the bridge lacks ref-count API.
   */
  dispose(): void {
    this.#core.dispose();
  }
}
