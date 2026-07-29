/**
 * gpuDetection — runtime GPU probe + detection for the hybrid stage.
 * Owns `probeWebGPU()` (the Tier-2 adapter probe, including the retry
 * loop for Chromium builds that defer `adapter.info` population) and
 * `detectGpu()` (the memoized public wrapper).
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
/**
 * The canonical WebGPU `GPUSupportedLimits` member names. Read by name (direct
 * property access) so the limit bag is populated even when the limits object
 * exposes them as NON-ENUMERABLE prototype getters — which is the case in native
 * wgpu bindings (Deno, wgpu-py): `Object.keys(adapter.limits)` returns `[]` there,
 * so an enumeration-only extractor silently produced an EMPTY bag and every
 * downstream tier verdict fell back to the WebGPU defaults (8 buffers / 4 textures
 * — the exact symptom that made `createProgressiveEngine` wrongly reject an
 * RTX-4090 dzn adapter). Browsers DO enumerate, so the by-name read is a superset.
 */
const GPU_LIMIT_NAMES = [
  'maxTextureDimension1D',
  'maxTextureDimension2D',
  'maxTextureDimension3D',
  'maxTextureArrayLayers',
  'maxBindGroups',
  'maxBindGroupsPlusVertexBuffers',
  'maxBindingsPerBindGroup',
  'maxDynamicUniformBuffersPerPipelineLayout',
  'maxDynamicStorageBuffersPerPipelineLayout',
  'maxSampledTexturesPerShaderStage',
  'maxSamplersPerShaderStage',
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxUniformBuffersPerShaderStage',
  'maxUniformBufferBindingSize',
  'maxStorageBufferBindingSize',
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
  'maxVertexBuffers',
  'maxBufferSize',
  'maxVertexAttributes',
  'maxVertexBufferArrayStride',
  'maxInterStageShaderComponents',
  'maxInterStageShaderVariables',
  'maxColorAttachments',
  'maxColorAttachmentBytesPerSample',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
] as const;

/**
 * Extract the numeric limit bag from a `GPUSupportedLimits`-like object in a way
 * that is portable across browsers (enumerable own-keys) AND native wgpu bindings
 * (non-enumerable prototype getters). Reads every canonical {@link GPU_LIMIT_NAMES}
 * member by direct access, then merges any additional enumerable own-keys (so a
 * future / vendor limit the browser exposes is still captured).
 *
 * @internal — shared by `probeWebGPU` (core) and `@vitrum/engine`'s adapterProfile.
 */
export function extractGpuLimits(limitsObj: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (limitsObj == null || typeof limitsObj !== 'object') return out;
  const record = limitsObj as Record<string, unknown>;
  // 1. Canonical names by direct access (works for getter-backed limits).
  for (const key of GPU_LIMIT_NAMES) {
    const val = record[key];
    if (typeof val === 'number' && Number.isFinite(val)) out[key] = val;
  }
  // 2. Any extra enumerable own-keys (browser-exposed future/vendor limits).
  for (const key of Object.keys(record)) {
    if (key in out) continue;
    const val = record[key];
    if (typeof val === 'number' && Number.isFinite(val)) out[key] = val;
  }
  return out;
}

export async function probeWebGPU(): Promise<WgpuProbeResult> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
    return { supported: false };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) return { supported: false };

    const limits = extractGpuLimits(adapter.limits);

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
  } catch (err: unknown) {
    // Carry the underlying error as a human-readable reason so callers can
    // distinguish a transient adapter failure (driver crash, context lost,
    // requestAdapter exception on a new-but-broken GPU) from a genuine
    // no-WebGPU environment (navigator.gpu absent / requestAdapter → null).
    const reason = err instanceof Error
      ? err.message
      : String(err);
    return { supported: false, reason };
  }
}

/**
 * Public, side-effect-free summary returned by {@link detectGpu}.
 */
export interface GpuDetection {
  /** True if `navigator.gpu` is exposed AND an adapter was successfully obtained. */
  readonly isWebGPU: boolean;
  /**
   * Classified adapter kind from the probe (or `'unknown'` when WebGPU is
   * unavailable). Use `adapterKind !== 'swiftshader'` to gate mounting a
   * hardware-required path.
   */
  readonly adapterKind: WgpuAdapterKind;
  /** GPUAdapterInfo.vendor (lowercased) — '' if unavailable. */
  readonly adapterVendor?: string;
  /** GPUAdapterInfo.architecture (lowercased) — '' if unavailable. */
  readonly adapterArchitecture?: string;
}

let cached: Promise<GpuDetection> | null = null;

/**
 * H62 — drop the module-level memo so the NEXT {@link detectGpu} call re-probes.
 *
 * The memo is otherwise permanent for the page lifetime, which is wrong after a
 * device-lost / eGPU unplug / driver reset, and makes per-test isolation
 * impossible.
 */
export function resetGpuDetectionCache(): void {
  cached = null;
}

/**
 * Detect whether the runtime is on a real hardware GPU. Memoized — safe
 * to call from multiple call sites; only one adapter is actually requested.
 * The probe is side-effect-free: callers receive the immutable detection
 * summary directly rather than communicating through a global window slot.
 * After a GPU topology change (device-lost, eGPU unplug) call
 * {@link resetGpuDetectionCache} to force a re-probe.
 */
export function detectGpu(): Promise<GpuDetection> {
  if (cached) return cached;
  cached = (async () => {
    const probe = await probeWebGPU();
    const adapterKind: WgpuAdapterKind = probe.supported && probe.adapterKind != null
      ? probe.adapterKind
      : 'unknown';
    const result: GpuDetection = Object.freeze({
      isWebGPU: probe.supported,
      adapterKind,
      adapterVendor: (probe.vendor ?? '').toLowerCase(),
      adapterArchitecture: (probe.architecture ?? '').toLowerCase(),
    });
    return result;
  })();
  return cached;
}
