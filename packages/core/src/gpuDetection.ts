/**
 * gpuDetection — runtime GPU probe + detection for the hybrid stage.
 * Owns `probeWebGPU()` (the Tier-2 adapter probe, including the retry
 * loop for Chromium builds that defer `adapter.info` population) and
 * `detectGpu()` (the memoized `__WG__`-publishing wrapper consumed by
 * `probeUpdatePass` and the e2e chroma spec gate).
 *
 * Pure classifier helpers (`WgpuAdapterKind`, `classifyAdapter`,
 * `isSwiftShaderAdapter`) live in `wgpuSupport.ts` so this file stays
 * cleanly separated: runtime/async logic here, types + pure functions there.
 */

import {
  classifyAdapter,
  type WgpuAdapterKind,
  type WgpuProbeResult,
} from './wgpuSupport.js';

/**
 * Read `adapter.info` with up to N retries (8ms each), tolerating Chromium
 * builds that defer info population. Falls back to the deprecated
 * `requestAdapterInfo()` path on the last attempt for older Chromiums where
 * `adapter.info` is missing entirely.
 */
async function readAdapterInfo(
  adapter: GPUAdapter,
  retries = 4,
): Promise<{ vendor: string; architecture: string }> {
  for (let i = 0; i <= retries; i++) {
    const info = adapter.info;
    const vendor = ((info?.vendor ?? '')).toString();
    const architecture = ((info?.architecture ?? '')).toString();
    if (vendor.length > 0 || architecture.length > 0) {
      return { vendor, architecture };
    }
    if (i === retries) break;
    await new Promise<void>((r) => setTimeout(r, 8));
  }
  // Last-resort fallback: `requestAdapterInfo()` (deprecated; may exist on
  // older Chromiums where the synchronous `.info` getter was missing).
  const legacyAdapter = adapter as GPUAdapter & {
    requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
  };
  if (typeof legacyAdapter.requestAdapterInfo === 'function') {
    try {
      const info = await legacyAdapter.requestAdapterInfo();
      return {
        vendor: (info?.vendor ?? '').toString(),
        architecture: (info?.architecture ?? '').toString(),
      };
    } catch {
      // fall through
    }
  }
  return { vendor: '', architecture: '' };
}

/**
 * Async probe — requests a high-performance adapter and returns limits +
 * features for the spike validation (§14.1) and the 0.75× resolution
 * fallback decision (§11.3).
 *
 * Returns `{ supported: false }` on any failure (no adapter, API absent, etc.)
 * so callers don't need try/catch.
 */
export async function probeWebGPU(): Promise<WgpuProbeResult> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
    return { supported: false };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) return { supported: false };

    const limits: Record<string, number> = {};
    const limitsRecord = adapter.limits as unknown as Record<string, unknown>;
    for (const key of Object.keys(limitsRecord)) {
      const val = limitsRecord[key];
      if (typeof val === 'number') limits[key] = val;
    }

    // Read adapter info with retry — some Chromium builds populate
    // `adapter.info` asynchronously on first read (empty strings until
    // a follow-up frame). The retry handles that race; falsely classifying
    // empty info as software was the bug fixed in an earlier revision.
    const { vendor, architecture } = await readAdapterInfo(adapter);
    const adapterKind = classifyAdapter({ vendor, architecture });

    return {
      supported: true,
      adapterKind,
      // exactOptionalPropertyTypes: spread absent-when-empty to avoid
      // assigning `undefined` to optional string fields explicitly.
      ...(vendor ? { vendor } : {}),
      ...(architecture ? { architecture } : {}),
      features: [...adapter.features].map(String),
      limits,
    };
  } catch {
    return { supported: false };
  }
}

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
