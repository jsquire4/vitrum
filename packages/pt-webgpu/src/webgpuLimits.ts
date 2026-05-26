/**
 * Device limits for {@link createPTEngine_WebGPU}. Does not raise
 * `maxStorageTexturesPerShaderStage` — the trace pass is buffer-heavy only.
 * Host pages that also run walkaround-hybrid must use
 * `HYBRID_WEBGPU_REQUIRED_LIMITS` from `@vitrum/walkaround-hybrid` instead.
 */
export const PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 23;

export const PT_WEBGPU_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
};

/** Clamp requested limits to what the adapter actually supports. */
export function mergeAdapterRequiredLimits(
  adapter: GPUAdapter,
  required: Record<string, number>,
): Record<string, number> {
  const merged: Record<string, number> = {};
  for (const [key, wanted] of Object.entries(required)) {
    const cap = adapter.limits[key as keyof GPUSupportedLimits];
    merged[key] =
      typeof cap === 'number' && Number.isFinite(cap) ? Math.min(wanted, cap) : wanted;
  }
  return merged;
}
