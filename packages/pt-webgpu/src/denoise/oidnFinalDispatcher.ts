/**
 * OIDN final-pass dispatcher for {@link PTEngineWebGPU} (`denoiser: 'oidn-final'`).
 *
 * Reads HDR + albedo + canonical packed-normal/depth storage textures through
 * `readOidnInputsFromTextures`; its normal decoder is shared with pt-webgl2 so
 * OIDN receives the same signed world-normal convention from both backends.
 *
 * ## Thin wrapper
 *
 * This file is now a thin wrapper over {@link OIDNDispatcherCore} from
 * `@vitrum/shared-denoisers`. The shared core holds the cohort state machine;
 * this wrapper contributes:
 *  - The WebGPU-specific `kickIfReady` signature (`device, sources, width, height`).
 *  - Wiring `readOidnInputsFromTextures` as the async readback callback.
 *  - `preloadOnBridgeInit: true` (pt-webgpu calls `preloadOIDNModel` on bridge
 *    init; pt-webgl does not — this is the one behavioral difference between
 *    the two backends, preserved intentionally).
 */

import {
  readOidnInputsFromTextures,
  type OidnReadbackFn,
  type OidnTextureSources,
} from './rgba16fReadback.js';

export type { OidnReadbackFn, OidnReadbackResult, OidnTextureSources } from './rgba16fReadback.js';

import {
  OIDNDispatcherCore,
  oidnDefaultLoader,
} from '@vitrum/shared-denoisers';

export interface OIDNFinalDispatcherRuntimeHooks {
  readonly onError?: (err: unknown) => void;
  /**
   * Called only after the shared dispatcher core has validated and accepted a
   * completed RGB frame. The hook must not mutate host-owned GPU state; the
   * WebGPU engine queues the frame here and uploads it on the next render call.
   */
  readonly onComplete?: (frame: DenoisedFrame) => void;
}

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
  OIDNDerivedState,
} from '@vitrum/shared-denoisers';

/** Input type for the pt-webgpu readback callback. */
interface WebGPUReadbackInput {
  readonly device: GPUDevice;
  readonly sources: OidnTextureSources;
  readonly width: number;
  readonly height: number;
}

export class OIDNFinalDispatcher {
  readonly #core: OIDNDispatcherCore<WebGPUReadbackInput>;

  constructor(
    opts: OIDNFinalDispatcherOptions,
    loader?: OIDNBridgeLoader,
    readback?: OidnReadbackFn,
    hooks?: OIDNFinalDispatcherRuntimeHooks,
  ) {
    if (opts.modelUrl === undefined || opts.modelUrl.length === 0) {
      throw new Error(
        '[vitrum/pt-webgpu OIDNFinalDispatcher] modelUrl is required. ' +
          "Pass oidn: { modelUrl } with denoiser: 'oidn-final'.",
      );
    }
    const resolvedReadback = readback ?? readOidnInputsFromTextures;
    this.#core = new OIDNDispatcherCore<WebGPUReadbackInput>({
      dispatcherOptions: opts,
      loader: loader ?? oidnDefaultLoader,
      readback: ({ device, sources, width, height }) =>
        resolvedReadback(device, sources, width, height),
      // pt-webgpu calls preloadOIDNModel on bridge init (behavioral
      // difference from pt-webgl — preserved intentionally).
      preloadOnBridgeInit: true,
      ...(hooks?.onError != null ? { onError: hooks.onError } : {}),
      ...(hooks?.onComplete != null ? { onComplete: hooks.onComplete } : {}),
    });
  }

  getLatestDenoised(): DenoisedFrame | null {
    return this.#core.getLatestDenoised();
  }

  isInFlight(): boolean {
    return this.#core.isInFlight();
  }

  /**
   * Current denoiser state for `FrameStats.denoiserState` population.
   *
   * Delegates to the shared core's single-source status ladder
   * ({@link OIDNDispatcherCore.deriveState}):
   *  - `'in-flight'`  — async inference cycle is running.
   *  - `'failed'`     — last cycle threw; `reason` = error message; retryable.
   *  - `'ready'`      — last cycle succeeded and the result is available.
   *  - `'fallback'`   — no inference has completed yet (first frame).
   *
   * pt-webgpu never enters the union's `'warming-up'` branch (no async
   * model-preload phase distinct from the inference cycle); that member exists
   * for walkaround-hybrid's warmup-aware reuse.
   */
  getState(): OIDNDerivedState {
    return this.#core.deriveState();
  }

  invalidate(): void {
    this.#core.invalidate();
  }

  /**
   * Queue GPU readback + async OIDN when converged. Returns immediately.
   */
  kickIfReady(
    device: GPUDevice,
    sources: OidnTextureSources,
    width: number,
    height: number,
  ): void {
    this.#core.kickIfReady({ device, sources, width, height }, width, height);
  }

  dispose(): void {
    this.#core.dispose();
  }
}
