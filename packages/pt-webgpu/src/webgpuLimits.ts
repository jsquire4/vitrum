/**
 * Device limits for {@link createPTEngine_WebGPU}. Host pages that also run
 * walkaround-hybrid must request the per-key union of these limits and
 * `HYBRID_WEBGPU_REQUIRED_LIMITS` from `@vitrum/walkaround-hybrid`.
 */
/**
 * Full tier uses 4 bind groups (WS2 added group 3); peak storage buffers in any
 * one group is 11 (group 1: analytics + env + area lights + directionalLights).
 * N-directional expansion (2026-06-10) added binding 10 = directionalLights to
 * group 1, raising the per-group peak from 10 to 11. Group 2 carries 7
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
export const PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP = 11;

/**
 * Full-tier storage-buffer bindings visible to the compute stage.
 * N-directional (2026-06-10): +1 for group-1 directionalLights (binding 10).
 * Total: g0(8) + g1(11) + g2(7) + g3(5) = 31.
 * @public — public device-limit constant; consumed by host device-acquisition code and tests.
 */
export const PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 31;

/** Full tier plus the opt-in ReSTIR-PT reuse pre-pass group-0 reservoirs. */
export const PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE + 4;

/** @public — back-compat alias for PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE; test consumers reference this name. */
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
 *
 * B12 RESOLUTION (2026-06-10): light data and HDRI env are packed as sampled
 * texture_2d<f32> (bindings 12–14 in group-0). Sampled textures are counted from
 * maxSampledTexturesPerShaderStage (WebGPU baseline ≥ 16), NOT the storage-buffer
 * budget. Post-B12 the lite layout uses:
 *   • 7 storage buffers (unchanged — still 1 free slot, zero headroom).
 *   • 3 sampled textures (new, drawn from a separate ≥16 budget):
 *     – liteEnvTex (binding 12): W×H RGBA32F env radiance + pdf.
 *     – liteEnvCdfTex (binding 13): W×H RGBA32F env marginal/conditional CDF.
 *     – liteLightTex (binding 14): 1×N RGBA32F point/spot/rect-area packed data.
 * The storage-buffer constants below (IN_USE, HDRI_NEEDED, AREA_LIGHT_NEEDED)
 * remain intact so the cliff arithmetic is machine-checkable and any future
 * storage-buffer consumption trips the pin.
 */
export const PT_WEBGPU_LITE_STORAGE_BUFFERS_IN_USE = 7;
/** Storage buffers an HDRI importance sampler *would* add as storage buffers (proof it doesn't fit). */
export const PT_WEBGPU_LITE_HDRI_STORAGE_BUFFERS_NEEDED = 2;
/** Storage buffers area-light BSDF MIS *would* add as storage buffers (proof it barely fits with 0 headroom). */
export const PT_WEBGPU_LITE_AREA_LIGHT_STORAGE_BUFFERS_NEEDED = 1;

/**
 * B12 — sampled textures added to the lite layout (drawn from maxSampledTexturesPerShaderStage,
 * NOT the storage-buffer budget). Value = 3: liteEnvTex + liteEnvCdfTex + liteLightTex.
 * The WebGPU baseline guarantee is maxSampledTexturesPerShaderStage ≥ 16.
 */
export const PT_WEBGPU_LITE_SAMPLED_TEXTURES_IN_USE = 3;
/** WebGPU baseline minimum for maxSampledTexturesPerShaderStage (spec §3.6.2). */
export const PT_WEBGPU_SAMPLED_TEXTURES_BASELINE = 16;

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
