/**
 * Device limits for {@link createPTEngine_WebGPU}. Does not raise
 * `maxStorageTexturesPerShaderStage` — the trace pass is buffer-heavy only.
 * Host pages that also run walkaround-hybrid must use
 * `HYBRID_WEBGPU_REQUIRED_LIMITS` from `@vitrum/walkaround-hybrid` instead.
 */
export const PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 23;

/** Lite trace pass: bindings 2–8 (7) + read_write accum at 2 → 8 storage buffers. */
export const PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 8;

/** Lite trace pass: output + normalDepth + albedo + variance. */
export const PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE = 4;

export const PT_WEBGPU_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
};

/** Request the highest tier the adapter can satisfy (used by host device acquisition). */
export function ptWebgpuRequiredLimitsForAdapter(
  adapter: GPUAdapter,
): Record<string, number> {
  const maxBuffers = adapter.limits.maxStorageBuffersPerShaderStage;
  const maxTextures = adapter.limits.maxStorageTexturesPerShaderStage;
  if (
    maxBuffers >= PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE &&
    maxTextures >= 5
  ) {
    return PT_WEBGPU_REQUIRED_LIMITS;
  }
  return {
    maxStorageBuffersPerShaderStage: Math.min(
      maxBuffers,
      PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    ),
    maxStorageTexturesPerShaderStage: Math.min(
      maxTextures,
      PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    ),
  };
}

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
