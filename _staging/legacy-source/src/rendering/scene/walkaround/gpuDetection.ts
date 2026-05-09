/**
 * gpuDetection — fail-fast hardware-GPU detection for the hybrid stage.
 * Adapts the shared `lib/wgpuSupport.probeWebGPU()` Tier-2 result into
 * the `GpuDetection` shape that `probeUpdatePass` and the e2e chroma
 * spec gate consume on `window.__WG__`.
 *
 * The shared `probeWebGPU()` is canonical; this wrapper exists so the
 * `__WG__` global shape (`isWebGPU` + `isHardwareGpu` + lower-cased
 * `adapterVendor` / `adapterArchitecture`) keeps working for the
 * existing chroma precondition gate. New callers should consume
 * `probeWebGPU()` directly via the lib.
 */

import { probeWebGPU } from './lib/wgpuSupport';

/**
 * GpuDetection — the canonical `__WG__` shape consumed by every walkaround
 * engine (DDGI / RC / ReSTIR / hybrid) and the e2e chroma spec gate.
 *
 * RC's gl-factory writer publishes `adapterVendor` / `adapterArchitecture`
 * + an optional `adapter: { name }` summary; the other engines publish via
 * `detectGpu()` below. All fields except `isWebGPU` and `isHardwareGpu`
 * are optional because the SwiftShader gate only needs those two.
 */
export interface GpuDetection {
  /** True if `navigator.gpu` is exposed AND an adapter was successfully obtained. */
  isWebGPU: boolean;
  /** True if `isWebGPU` AND the adapter is NOT SwiftShader. */
  isHardwareGpu: boolean;
  /** GPUAdapterInfo.vendor (lowercased) — '' if unavailable. */
  adapterVendor?: string;
  /** GPUAdapterInfo.architecture (lowercased) — '' if unavailable. */
  adapterArchitecture?: string;
  /** Friendly summary used by RC's gl-factory writer for diagnostic logs. */
  adapter?: { name: string };
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
 * The first call also publishes `window.__WG__`.
 */
export function detectGpu(): Promise<GpuDetection> {
  if (cached) return cached;
  cached = (async () => {
    const probe = await probeWebGPU();
    const result: GpuDetection = {
      isWebGPU: probe.supported,
      isHardwareGpu: Boolean(probe.supported && probe.isHardwareGpu !== false),
      adapterVendor: (probe.vendor ?? '').toLowerCase(),
      adapterArchitecture: (probe.architecture ?? '').toLowerCase(),
    };
    publish(result);
    return result;
  })();
  return cached;
}

// getCachedGpuDetection — synchronous accessor for the cached detection.
// Currently unused; de-exported 2026-05-07 sweep. Re-export if a caller
// needs it (active code uses `await detectGpu()` instead).

/**
 * Test-only — reset the memoized detection cache + clear `window.__WG__`.
 *
 * **Production code must never import this.** The double-underscore
 * prefix is the convention for test-only exports throughout this codebase;
 * any lint pre-commit hook should treat `__resetGpuDetectionForTests` as
 * a tripwire if imported outside `*.test.ts` / `*.spec.ts` files.
 *
 * Cost on production: 7 lines of inert code + zero runtime cost (function
 * is never called). Kept in this module rather than a sibling
 * .test-utils.ts file because the cache it resets is module-private; a
 * sibling file would need either a setter export or duplicated cache state.
 */
export function __resetGpuDetectionForTests(): void {
  cached = null;
  if (typeof window !== 'undefined') {
    delete window.__WG__;
  }
}

function publish(result: GpuDetection): void {
  if (typeof window !== 'undefined') {
    window.__WG__ = result;
  }
}
