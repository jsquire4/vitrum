/**
 * Device limits for {@link createPTEngine_WebGPU}. Does not raise
 * `maxStorageTexturesPerShaderStage` — the trace pass is buffer-heavy only.
 * Host pages that also run walkaround-hybrid must use
 * `HYBRID_WEBGPU_REQUIRED_LIMITS` from `@vitrum/walkaround-hybrid` instead.
 */
/**
 * Full tier uses 3 bind groups; peak storage buffers in any one group is 10
 * (group 1: analytics + env + area lights).
 */
export const PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP = 10;

/** @deprecated Use {@link PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP}. */
export const PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP;

/** Lite trace pass: bindings 2–8 (7) + read_write accum at 2 → 8 storage buffers. */
export const PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 8;

/** Full tier group 0 uses 5 storage textures (output + G-buffer aux). */
export const PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE = 5;

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
    maxBuffers >= PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP &&
    maxTextures >= PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE
  ) {
    return {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
      maxStorageTexturesPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    };
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
