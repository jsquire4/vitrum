/**
 * WebGPU feature detection utilities for the walkaround GI pipelines.
 *
 * These helpers are called at mount time to decide whether to use a
 * WebGPU-backed canvas (explore mode) vs. fall back to standard Canvas
 * (editor/verify). Provides a 3-state verdict (HardwareAccelerated /
 * SoftwareEmulated / Unsupported) plus a 4× retry to handle the
 * Chromium adapter-info race.
 *
 * Tier 2 shared GI primitive — consumed by every GI engine (DDGI / RC /
 * ReSTIR / hybrid).
 */

export interface WgpuProbeResult {
  supported: boolean;
  vendor?: string;
  architecture?: string;
  features?: string[];
  limits?: Record<string, number>;
  /**
   * True iff the adapter is real hardware (NVIDIA / AMD / Intel / Apple).
   * False when WebGPU falls back to SwiftShader (`vendor === 'google'`
   * AND `architecture === 'swiftshader'`) — software rasterizer that
   * silently returns wrong perf/visual results.  Option F of the
   * hardware-GPU validation spec gates against this.
   */
  isHardwareGpu?: boolean;
}

/**
 * Returns true when the adapter info indicates SwiftShader software rasterizer.
 *
 * SwiftShader is Chromium's WebGPU CPU fallback.  It identifies as
 * `vendor === 'google'` AND `architecture === 'swiftshader'`.  Validation
 * runs that land on SwiftShader produce honest-looking-but-wrong output
 * (fps, chroma, etc.) — Option F of the hardware-GPU validation spec
 * exists specifically to catch this case at mount time.
 */
export function isSwiftShaderAdapter(info: {
  vendor?: string | null;
  architecture?: string | null;
} | null | undefined): boolean {
  if (!info) return false;
  const vendor = (info.vendor ?? '').toLowerCase();
  const arch = (info.architecture ?? '').toLowerCase();
  return vendor === 'google' && arch === 'swiftshader';
}

/**
 * Three-state hardware-GPU verdict.
 *  - 'hardware'    — adapter info names a real GPU vendor (nvidia, amd, intel, apple, etc.)
 *  - 'swiftshader' — adapter info names Chromium's software rasterizer
 *  - 'unknown'     — adapter exists but reports empty vendor/architecture strings.
 *                     Per WebGPU spec, UAs may filter info to prevent fingerprinting.
 *                     Some Chromium builds defer info population until first device
 *                     creation. Treat as unknown rather than falsely flagging false —
 *                     the post-init re-derivation from `backend.adapter` settles it.
 */
type HardwareVerdict = 'hardware' | 'swiftshader' | 'unknown';

/**
 * Read `adapter.info` with up to N retries (8ms each), tolerating Chromium
 * builds that defer info population. Falls back to the deprecated
 * `requestAdapterInfo()` path on the last attempt for older Chromiums where
 * `adapter.info` is missing entirely.
 */
async function readAdapterInfo(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  adapter: any,
  retries = 4,
): Promise<{ vendor: string; architecture: string }> {
  for (let i = 0; i <= retries; i++) {
    const info = adapter?.info;
    const vendor = ((info?.vendor ?? '') as string).toString();
    const architecture = ((info?.architecture ?? '') as string).toString();
    if (vendor.length > 0 || architecture.length > 0) {
      return { vendor, architecture };
    }
    if (i === retries) break;
    await new Promise<void>((r) => setTimeout(r, 8));
  }
  // Last-resort fallback: `requestAdapterInfo()` (deprecated; may exist on
  // older Chromiums where the synchronous `.info` getter was missing).
  if (typeof adapter?.requestAdapterInfo === 'function') {
    try {
      const info = await adapter.requestAdapterInfo();
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
 * Classify adapter info into one of three verdicts. Empty info → 'unknown'
 * (NOT 'swiftshader' and NOT 'hardware') so callers can defer the decision
 * to a post-init re-read of `backend.adapter`.
 */
function classifyAdapter(info: { vendor: string; architecture: string }): HardwareVerdict {
  if (isSwiftShaderAdapter(info)) return 'swiftshader';
  if (info.vendor.length === 0 && info.architecture.length === 0) return 'unknown';
  return 'hardware';
}

/**
 * Synchronous check — only verifies the API exists. Use this for conditional
 * rendering; use `probeWebGPU()` for full adapter capability detection.
 */
function isWebGPUSupported(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    'gpu' in navigator &&
    navigator.gpu !== undefined &&
    navigator.gpu !== null
  );
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
  if (!isWebGPUSupported()) return { supported: false };

  try {
    const gpu = navigator.gpu as GPURequestAdapterOptions extends never
      ? never
      : typeof navigator.gpu;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const adapter = await (gpu as any).requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) return { supported: false };

    const limits: Record<string, number> = {};
    for (const key of Object.keys(adapter.limits)) {
      const val = (adapter.limits as Record<string, unknown>)[key];
      if (typeof val === 'number') limits[key] = val;
    }

    // Read adapter info with retry — some Chromium builds populate
    // `adapter.info` asynchronously on first read (empty strings until
    // a follow-up frame). The retry handles that race; falsely flagging
    // empty info as `isHardwareGpu: false` was the bug fixed in this revision.
    const { vendor, architecture } = await readAdapterInfo(adapter);
    const verdict = classifyAdapter({ vendor, architecture });
    // Treat 'unknown' as hardware-undetermined: the spike caller (test suite)
    // currently expects a boolean, so map unknown→true (do not falsely refuse
    // a likely-real GPU). The mount-time gate in StudioScene's gl factory
    // re-derives the verdict from `backend.adapter` post-init for the
    // authoritative answer.
    const isHardwareGpu = verdict !== 'swiftshader';

    return {
      supported: true,
      vendor: vendor || undefined,
      architecture: architecture || undefined,
      features: [...adapter.features].map(String),
      limits,
      isHardwareGpu,
    };
  } catch {
    return { supported: false };
  }
}

