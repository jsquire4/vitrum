/**
 * Schied 2017 SVGF dispatcher for {@link PTEngineWebGPU} (`denoiser: 'svgf-real'`).
 *
 * Reads HDR + albedo + normal-depth from pt-webgpu aux textures, then runs
 * `runSVGFRealWebGPU` from `@vitrum/shared-denoisers` on the host-owned device.
 */

import { runSVGFRealWebGPU, type SVGFRealWebGPUOptions } from '@vitrum/shared-denoisers';
import {
  readOidnInputsFromTextures,
  type OidnReadbackFn,
  type OidnTextureSources,
} from './rgba16fReadback.js';

export interface DenoisedFrame {
  readonly rgb: Float32Array;
  readonly width: number;
  readonly height: number;
}

export type SVGFRealRunFn = typeof runSVGFRealWebGPU;

export class SVGFRealDispatcher {
  readonly #readback: OidnReadbackFn;
  readonly #runSvgf: SVGFRealRunFn;
  readonly #atrousIterations: number;

  #inFlight = false;
  #haveCompleted = false;
  #latest: DenoisedFrame | null = null;
  #disposed = false;
  #cohortId = 0;
  #prevRadiance: Float32Array | null = null;
  #historyLength: Uint32Array | null = null;
  #moments: Float32Array | null = null;

  constructor(
    opts?: { readonly atrousIterations?: number },
    readback?: OidnReadbackFn,
    runSvgf?: SVGFRealRunFn,
  ) {
    this.#atrousIterations = opts?.atrousIterations ?? 5;
    this.#readback = readback ?? readOidnInputsFromTextures;
    this.#runSvgf = runSvgf ?? runSVGFRealWebGPU;
  }

  getLatestDenoised(): DenoisedFrame | null {
    return this.#latest;
  }

  invalidate(): void {
    this.#cohortId += 1;
    this.#haveCompleted = false;
    this.#latest = null;
    this.#prevRadiance = null;
    this.#historyLength = null;
    this.#moments = null;
  }

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
    if (sources.albedo == null || sources.normalDepth == null) {
      console.warn(
        '[vitrum/pt-webgpu SVGFRealDispatcher] albedo + normalDepth required (full trace tier).',
      );
      return;
    }

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

      const pixelCount = width * height;
      const normals = readback.normal ?? new Float32Array(pixelCount * 3);
      const gbufferNormalsRgb = new Float32Array(pixelCount * 3);
      for (let i = 0; i < pixelCount; i += 1) {
        const si = i * 3;
        gbufferNormalsRgb[si] = ((normals[si] ?? 0) + 1) * 0.5;
        gbufferNormalsRgb[si + 1] = ((normals[si + 1] ?? 0) + 1) * 0.5;
        gbufferNormalsRgb[si + 2] = ((normals[si + 2] ?? 0) + 1) * 0.5;
      }

      const svgfOpts: SVGFRealWebGPUOptions = {
        device,
        reuseSharedWebGpuDevice: false,
        rgb: readback.color,
        width,
        height,
        gbufferNormalsRgb,
        atrousIterations: this.#atrousIterations,
        ...(readback.albedo != null ? { albedoRgb: readback.albedo } : {}),
        ...(this.#prevRadiance != null ? { prevRadianceRgb: this.#prevRadiance } : {}),
        ...(this.#historyLength != null ? { historyLengthIn: this.#historyLength } : {}),
        ...(this.#moments != null ? { momentsIn: this.#moments } : {}),
      };
      const filtered = await this.#runSvgf(svgfOpts);

      if (this.#disposed || this.#cohortId !== cohortAtKick) return;
      this.#prevRadiance = filtered.slice();
      this.#latest = { rgb: filtered, width, height };
      this.#haveCompleted = true;
    } catch (err) {
      console.warn('[vitrum/pt-webgpu SVGFRealDispatcher] readback or SVGF failed', err);
    }
  }

  dispose(): void {
    this.#disposed = true;
    this.#latest = null;
    this.#prevRadiance = null;
    this.#historyLength = null;
    this.#moments = null;
  }
}
