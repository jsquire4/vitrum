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
 * descriptors) + two read_write storage buffers (A4 SPPM: sppmPhotonCells at
 * binding 6, sppmCellCounters at binding 7) + one uniform (sppmStats at binding 8,
 * does NOT count against the storage-buffer limit). The WebGPU device-request
 * contract is per-STAGE, so the exported full-tier request uses the aggregate
 * storage-buffer count below; this per-group peak remains useful for layout audits.
 */
export const PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP = 10;

/**
 * Full-tier storage-buffer bindings visible to the compute stage.
 * A4 (SPPM): +2 for group-3 sppmPhotonCells (binding 6) + sppmCellCounters
 * (binding 7) — both read_write storage. sppmStats (binding 8) is uniform and
 * does NOT count against this limit.
 * 28 → 30. Total: g0(8) + g1(10) + g2(7) + g3(5) = 30.
 */
export const PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 30;

/** Full tier plus the opt-in ReSTIR-PT reuse pre-pass group-0 reservoirs. */
export const PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE + 4;

export const PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;

/** Lite trace pass: bindings 2–8 (7) + read_write accum at 2 → 8 storage buffers. */
export const PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 8;

/**
 * B12 — lite-tier binding-budget proof (fidelity cliff arithmetic, PINNED by the
 * liteTierBindingBudget test). The lite group-0 layout binds exactly these many
 * storage buffers today: accum(2), positions(3), indices(4), triMaterialIds(5),
 * materials(6), bvhNodes(7), normals(8) = 7. Under the lite cap of 8
 * (PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE) that leaves ONE free
 * storage-buffer slot.
 *   • HDRI importance sampling = environmentMapTexels + environmentMapCdf = 2
 *     storage buffers → 7 + 2 = 9 > 8 → does NOT fit as storage buffers (the
 *     texture-packing route — equirect + CDF rows in sampled textures — is the
 *     B12 follow-up; sampled textures do not count against this budget).
 *   • Area-light BSDF MIS = rectAreaLights = 1 storage buffer → 7 + 1 = 8 → fits
 *     exactly but with zero headroom, and needs the same lite-pipeline plumbing
 *     + constrained-hardware GPU validation as the HDRI route.
 * These three constants make the cliff arithmetic explicit and machine-checkable
 * so a future change that frees/consumes a lite storage-buffer slot trips the pin.
 */
export const PT_WEBGPU_LITE_STORAGE_BUFFERS_IN_USE = 7;
/** Storage buffers an HDRI importance sampler would add to the lite layout. */
export const PT_WEBGPU_LITE_HDRI_STORAGE_BUFFERS_NEEDED = 2;
/** Storage buffers area-light BSDF MIS would add to the lite layout. */
export const PT_WEBGPU_LITE_AREA_LIGHT_STORAGE_BUFFERS_NEEDED = 1;

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
