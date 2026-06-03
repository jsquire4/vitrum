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
export function classifyAdapter(info: { vendor: string; architecture: string }): WgpuAdapterKind {
  if (isSwiftShaderAdapter(info)) return 'swiftshader';
  if (info.vendor.length === 0 && info.architecture.length === 0) return 'unknown';
  return 'hardware';
}
