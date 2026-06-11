/**
 * OIDN final-pass dispatcher for {@link PTEngineWebGPU} (`denoiser: 'oidn-final'`).
 *
 * Unlike pt-webgl (color-only MRT), pt-webgpu reads HDR + albedo + normal-depth
 * storage textures via `readOidnInputsFromTextures` (WG-1 / plan primary-release).
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
   * Derives status from the core state machine:
   *  - `'in-flight'`  — async inference cycle is running.
   *  - `'failed'`     — last cycle threw; `reason` = error message; retryable.
   *  - `'ready'`      — last cycle succeeded and the result is available.
   *  - `'fallback'`   — no inference has completed yet (first frame).
   */
  getState(): { status: 'ready' | 'in-flight' | 'fallback' | 'failed'; reason: string | null; retryable?: boolean } {
    const lastError = this.#core.getLastError();
    if (lastError !== null) {
      return { status: 'failed', reason: lastError, retryable: true };
    }
    if (this.#core.isInFlight()) {
      return { status: 'in-flight', reason: null };
    }
    if (this.#core.getLatestDenoised() !== null) {
      return { status: 'ready', reason: null };
    }
    return { status: 'fallback', reason: 'waiting for first OIDN inference' };
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
