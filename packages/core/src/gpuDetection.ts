/**
 * gpuDetection — fail-fast hardware-GPU detection for the hybrid stage.
 * Adapts the shared `wgpuSupport.probeWebGPU()` Tier-2 result into
 * the `GpuDetection` shape that `probeUpdatePass` and the e2e chroma
 * spec gate consume on `window.__WG__`.
 *
 * The shared `probeWebGPU()` is canonical; this wrapper exists so the
 * `__WG__` global shape (`isWebGPU` + `adapterKind` + lower-cased
 * `adapterVendor` / `adapterArchitecture`) keeps working for the existing
 * chroma precondition gate. New callers should consume `probeWebGPU()`
 * directly via the lib.
 */

import { probeWebGPU, type WgpuAdapterKind } from './wgpuSupport.js';

/**
 * GpuDetection — the canonical `__WG__` shape consumed by every walkaround
 * engine (DDGI / RC / ReSTIR / hybrid) and the e2e chroma spec gate.
 *
 * RC's gl-factory writer publishes `adapterVendor` / `adapterArchitecture`
 * + an optional `adapter: { name }` summary; the other engines publish via
 * `detectGpu()` below. `adapterVendor` / `adapterArchitecture` remain
 * optional because some gates only need `isWebGPU` / `adapterKind`.
 */
export interface GpuDetection {
  /** True if `navigator.gpu` is exposed AND an adapter was successfully obtained. */
  isWebGPU: boolean;
  /**
   * Classified adapter kind from the probe (or `'unknown'` when WebGPU is
   * unavailable). Use `adapterKind !== 'swiftshader'` to gate mounting a
   * hardware-required path.
   */
  adapterKind: WgpuAdapterKind;
  /** GPUAdapterInfo.vendor (lowercased) — '' if unavailable. */
  adapterVendor?: string;
  /** GPUAdapterInfo.architecture (lowercased) — '' if unavailable. */
  adapterArchitecture?: string;
  /** Friendly summary used by RC's gl-factory writer for diagnostic logs. */
  adapter?: { name: string };
}

/** Options for {@link detectGpu}. */
export interface DetectGpuOptions {
  /**
   * When true (default), assigns `window.__WG__` on first successful probe.
   * Set false in workers or tests that must not touch `window`.
   */
  readonly publishToWindow?: boolean;
}

declare global {
  interface Window {
    __WG__?: GpuDetection;
  }
}

let cached: Promise<GpuDetection> | null = null;

/**
 * Detect whether the runtime is on a real hardware GPU. Memoized — safe
 * to call from multiple call sites; only one adapter is actually requested.
 * By default the first call also assigns `window.__WG__` (see
 * {@link DetectGpuOptions.publishToWindow}). Options apply only to the first
 * invocation.
 */
export function detectGpu(options?: DetectGpuOptions): Promise<GpuDetection> {
  if (cached) return cached;
  const publishToWindow = options?.publishToWindow !== false;
  cached = (async () => {
    const probe = await probeWebGPU();
    const adapterKind: WgpuAdapterKind = probe.supported && probe.adapterKind != null
      ? probe.adapterKind
      : 'unknown';
    const result: GpuDetection = {
      isWebGPU: probe.supported,
      adapterKind,
      adapterVendor: (probe.vendor ?? '').toLowerCase(),
      adapterArchitecture: (probe.architecture ?? '').toLowerCase(),
    };
    if (publishToWindow) {
      publish(result);
    }
    return result;
  })();
  return cached;
}

function publish(result: GpuDetection): void {
  if (typeof window !== 'undefined') {
    window.__WG__ = result;
  }
}
