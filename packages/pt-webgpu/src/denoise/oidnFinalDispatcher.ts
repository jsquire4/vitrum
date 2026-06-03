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
  type OidnReadbackResult,
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
