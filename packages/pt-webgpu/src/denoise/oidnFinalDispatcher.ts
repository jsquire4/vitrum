/**
 * OIDN final-pass dispatcher for {@link PTEngineWebGPU} (`denoiser: 'oidn-final'`).
 *
 * Unlike pt-webgl (color-only MRT), pt-webgpu reads HDR + albedo + normal-depth
 * storage textures via `readOidnInputsFromTextures` (WG-1 / plan primary-release).
 */

import {
  readOidnInputsFromTextures,
  type OidnReadbackResult,
  type OidnTextureSources,
} from './rgba16fReadback.js';

export type { OidnReadbackResult, OidnTextureSources } from './rgba16fReadback.js';

export interface OIDNFinalDispatcherOptions {
  readonly modelUrl: string;
  readonly executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
}

export interface DenoisedFrame {
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;
}

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
  readonly releaseOIDNCacheEntry?: (opts: {
    modelUrl: string;
    executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  }) => void;
  readonly clearOIDNCache?: () => void;
}

export type OIDNBridgeLoader = () => Promise<OIDNBridgeLike>;

export type OidnReadbackFn = (
  device: GPUDevice,
  sources: OidnTextureSources,
  width: number,
  height: number,
) => Promise<OidnReadbackResult>;

const _defaultLoader: OIDNBridgeLoader = async () => {
  const mod = await import('@vitrum/shared-denoisers');
  return {
    denoiseFinal: mod.denoiseFinal,
    preloadOIDNModel: mod.preloadOIDNModel,
    releaseOIDNCacheEntry: mod.releaseOIDNCacheEntry,
    clearOIDNCache: mod.clearOIDNCache,
  };
};

export class OIDNFinalDispatcher {
  readonly #modelUrl: string;
  readonly #executionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  readonly #loader: OIDNBridgeLoader;
  readonly #readback: OidnReadbackFn;

  #inFlight = false;
  #haveCompleted = false;
  #latest: DenoisedFrame | null = null;
  #disposed = false;
  #bridge: OIDNBridgeLike | null = null;
  #cohortId = 0;

  constructor(
    opts: OIDNFinalDispatcherOptions,
    loader?: OIDNBridgeLoader,
    readback?: OidnReadbackFn,
  ) {
    if (opts.modelUrl === undefined || opts.modelUrl.length === 0) {
      throw new Error(
        '[vitrum/pt-webgpu OIDNFinalDispatcher] modelUrl is required. ' +
          "Pass extensions['vitrum.ptWebgpu.oidnModelUrl'] with denoiser: 'oidn-final'.",
      );
    }
    this.#modelUrl = opts.modelUrl;
    this.#executionProviders = opts.executionProviders;
    this.#loader = loader ?? _defaultLoader;
    this.#readback = readback ?? readOidnInputsFromTextures;
  }

  getLatestDenoised(): DenoisedFrame | null {
    return this.#latest;
  }

  isInFlight(): boolean {
    return this.#inFlight;
  }

  invalidate(): void {
    this.#cohortId += 1;
    this.#haveCompleted = false;
    this.#latest = null;
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
    if (this.#disposed) return;
    if (this.#inFlight) return;
    if (this.#haveCompleted) return;
    if (width <= 0 || height <= 0) return;

    const cohortAtKick = this.#cohortId;
    this.#inFlight = true;
    void this.#runCycle(device, sources, width, height, cohortAtKick).finally(() => {
      this.#inFlight = false;
    });
  }

  async #runCycle(
    device: GPUDevice,
    sources: OidnTextureSources,
    width: number,
    height: number,
    cohortAtKick: number,
  ): Promise<void> {
    try {
      const readback = await this.#readback(device, sources, width, height);
      if (this.#disposed || this.#cohortId !== cohortAtKick) return;

      if (this.#bridge == null) {
        this.#bridge = await this.#loader();
        if (this.#bridge.preloadOIDNModel != null) {
          await this.#bridge.preloadOIDNModel({
            modelUrl: this.#modelUrl,
            ...(this.#executionProviders !== undefined
              ? { executionProviders: this.#executionProviders }
              : {}),
          });
        }
      }
      if (this.#disposed || this.#cohortId !== cohortAtKick) return;

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
      this.#latest = { rgb: denoised, width, height };
      this.#haveCompleted = true;
    } catch (err) {
      console.warn(
        '[vitrum/pt-webgpu OIDNFinalDispatcher] readback or denoiseFinal failed',
        err,
      );
    }
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const opts =
      this.#executionProviders !== undefined
        ? { modelUrl: this.#modelUrl, executionProviders: this.#executionProviders }
        : { modelUrl: this.#modelUrl };
    try {
      if (this.#bridge?.releaseOIDNCacheEntry != null) {
        this.#bridge.releaseOIDNCacheEntry(opts);
      } else if (this.#bridge?.clearOIDNCache != null) {
        this.#bridge.clearOIDNCache();
      }
    } catch {
      /* swallow */
    }
  }
}
