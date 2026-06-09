/**
 * Device limits for {@link createPTEngine_WebGPU}. Host pages that also run
 * walkaround-hybrid must request the per-key union of these limits and
 * `HYBRID_WEBGPU_REQUIRED_LIMITS` from `@vitrum/walkaround-hybrid`.
 */
/**
 * Full tier uses 4 bind groups (WS2 added group 3); peak storage buffers in any
 * one group is 10 (group 1: analytics + env + area lights). Group 2 carries 7
 * storage buffers (5 TLAS + the BDPT light-path scratch buffer + the BDPT
 * eye-stack scratch buffer); the light-path was an `rgba32float` read_write
 * storage TEXTURE but core WebGPU rejects that format for read_write storage
 * (gpuweb #4651), so it is a storage buffer — one fewer storage texture, one more
 * storage buffer. Group 3 carries three read-only storage buffers (the WS2
 * many-light importance-sampling tree, mesh UVs, and material texture
 * descriptors). The WebGPU device-request contract is per-STAGE, so the
 * exported full-tier request uses the aggregate storage-buffer count below; this
 * per-group peak remains useful for layout audits.
 */
export const PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP = 10;

/** Full-tier storage-buffer bindings visible to the compute stage. */
export const PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 28;

/** Full tier plus the opt-in ReSTIR-PT reuse pre-pass group-0 reservoirs. */
export const PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE + 4;

export const PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;

/** Lite trace pass: bindings 2–8 (7) + read_write accum at 2 → 8 storage buffers. */
export const PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 8;

/**
 * Full tier uses 5 storage textures per stage, all in group 0 (output +
 * G-buffer aux). The former group-2 BDPT light-path storage texture is now a
 * storage buffer (see {@link PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP}), so
 * the per-stage storage-texture total is exactly group 0's 5.
 */
export const PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE = 5;

/** Lite trace pass: output + normalDepth + albedo + variance. */
export const PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE = 4;

export const PT_WEBGPU_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  maxStorageTexturesPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
};

export interface PtWebgpuRequiredLimitOptions {
  /** Include the opt-in ReSTIR-PT reuse reservoir/result storage buffers. */
  readonly restirPtReuse?: boolean;
}

/** Request the highest tier the adapter can satisfy (used by host device acquisition). */
export function ptWebgpuRequiredLimitsForAdapter(
  adapter: GPUAdapter,
  options: PtWebgpuRequiredLimitOptions = {},
): Record<string, number> {
  const maxBuffers = adapter.limits.maxStorageBuffersPerShaderStage;
  const maxTextures = adapter.limits.maxStorageTexturesPerShaderStage;
  const fullBufferFloor = options.restirPtReuse === true
    ? PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE
    : PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  if (
    maxBuffers >= fullBufferFloor &&
    maxTextures >= PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE
  ) {
    return {
      maxStorageBuffersPerShaderStage: fullBufferFloor,
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
