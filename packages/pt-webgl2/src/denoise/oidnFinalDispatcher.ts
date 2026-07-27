import {
  OIDNDispatcherCore,
  oidnDefaultLoader,
} from '@vitrum/shared-denoisers';
import type {
  DenoisedFrame,
  OIDNBridgeLoader,
  OIDNDerivedState,
  OIDNFinalDispatcherOptions,
  ReadbackResult,
} from '@vitrum/shared-denoisers';
import type { WebGlOidnReadbackResult } from './rgba32fReadback.js';

export type {
  DenoisedFrame,
  OIDNBridgeLike,
  OIDNBridgeLoader,
  OIDNFinalDispatcherOptions,
} from '@vitrum/shared-denoisers';

export type OidnReadbackFn = (
  input: WebGlOidnReadbackResult,
) => Promise<ReadbackResult | null> | ReadbackResult | null;

export interface OIDNFinalDispatcherRuntimeHooks {
  readonly onError?: (err: unknown) => void;
  readonly onComplete?: (frame: DenoisedFrame) => void;
}

/**
 * OIDN final-pass dispatcher for `@vitrum/pt-webgl2`.
 *
 * WebGL readback is synchronous, so the engine reads RGBA32F attachments before
 * kicking this dispatcher. The shared core still owns the cohort state machine,
 * async bridge load, error dedupe, cache lifetime, and latest-result surface.
 */
export class OIDNFinalDispatcher {
  readonly #core: OIDNDispatcherCore<WebGlOidnReadbackResult>;

  constructor(
    opts: OIDNFinalDispatcherOptions,
    loader?: OIDNBridgeLoader,
    readback?: OidnReadbackFn,
    hooks?: OIDNFinalDispatcherRuntimeHooks,
  ) {
    if (opts.modelUrl === undefined || opts.modelUrl.length === 0) {
      throw new Error(
        '[vitrum/pt-webgl2 OIDNFinalDispatcher] modelUrl is required. ' +
          "Pass oidn: { modelUrl } with denoiser: 'oidn-final'.",
      );
    }
    const resolvedReadback = readback ?? ((input: WebGlOidnReadbackResult) => input);
    this.#core = new OIDNDispatcherCore<WebGlOidnReadbackResult>({
      dispatcherOptions: opts,
      loader: loader ?? oidnDefaultLoader,
      readback: async (input) => resolvedReadback(input),
      preloadOnBridgeInit: false,
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

  getState(): OIDNDerivedState {
    // Single source of truth: the shared core owns the status ladder
    // (OIDNDispatcherCore.deriveState). This wrapper never enters the
    // 'warming-up' branch of the union (pt-webgl2 has no async model-preload
    // phase); that member exists for walkaround-hybrid's warmup-aware reuse.
    return this.#core.deriveState();
  }

  invalidate(): void {
    this.#core.invalidate();
  }

  kickIfReady(input: WebGlOidnReadbackResult): void {
    this.#core.kickIfReady(input, input.width, input.height);
  }

  dispose(): void {
    this.#core.dispose();
  }
}
