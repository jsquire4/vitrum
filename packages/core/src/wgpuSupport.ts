/** Verdict from adapter info — see {@link classifyAdapter}. */
export type WgpuAdapterKind = 'hardware' | 'swiftshader' | 'unknown';

export interface WgpuProbeResult {
  supported: boolean;
  vendor?: string;
  architecture?: string;
  features?: string[];
  limits?: Record<string, number>;
  /**
   * Classified from `adapter.info` (with empty vendor/arch treated as unknown).
   */
  adapterKind?: WgpuAdapterKind;
  /**
   * @deprecated Prefer {@link adapterKind}. Scheduled for removal in Phase 7 /
   * Sprint 1 once host call sites migrate to `adapterKind`. When present,
   * `true` means not SwiftShader (`adapterKind !== 'swiftshader'`), including
   * the fingerprinting `unknown` case where the real GPU is still treated as
   * usable.
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
function classifyAdapter(info: { vendor: string; architecture: string }): WgpuAdapterKind {
  if (isSwiftShaderAdapter(info)) return 'swiftshader';
  if (info.vendor.length === 0 && info.architecture.length === 0) return 'unknown';
  return 'hardware';
}

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
    // a follow-up frame). The retry handles that race; falsely flagging
    // empty info as `isHardwareGpu: false` was the bug fixed in this revision.
    const { vendor, architecture } = await readAdapterInfo(adapter);
    const adapterKind = classifyAdapter({ vendor, architecture });
    // Treat 'unknown' as hardware-undetermined: legacy boolean and mount gates
    // map unknown→true (do not falsely refuse a likely-real GPU).
    const isHardwareGpu = adapterKind !== 'swiftshader';

    return {
      supported: true,
      adapterKind,
      // exactOptionalPropertyTypes: spread absent-when-empty to avoid
      // assigning `undefined` to optional string fields explicitly.
      ...(vendor ? { vendor } : {}),
      ...(architecture ? { architecture } : {}),
      features: [...adapter.features].map(String),
      limits,
      isHardwareGpu,
    };
  } catch {
    return { supported: false };
  }
}
