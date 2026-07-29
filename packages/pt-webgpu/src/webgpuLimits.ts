/**
 * Device limits for {@link createPTEngine_WebGPU}. Host pages that also run
 * walkaround-hybrid must request the per-key union of these limits and
 * `HYBRID_WEBGPU_REQUIRED_LIMITS` from `@vitrum/walkaround-hybrid`.
 */
/**
 * Full-tier storage-buffer bindings visible to the compute stage.
 * N-directional (2026-06-10): +1 for group-1 directionalLights (binding 10).
 * D10/H53 (2026-06-14): group 3 is 8, not 5: lightTree, meshUvs,
 * materialTexDescriptors, sppmPhotonCells, sppmCellCounters, sppmPixelStats,
 * meshTangents, meshVertexColors. Total: g0(8) + g1(11) + g2(5) + g3(8) = 32.
 * @public — public device-limit constant; consumed by host device-acquisition code and tests.
 */
export const PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE = 32;

/** Full tier plus the BDPT t=1 atomic RGB camera-splat buffer. */
export const PT_WEBGPU_BDPT_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE + 1;

/**
 * Full tier plus the opt-in ReSTIR-PT reuse group-0 storage bindings.
 *
 * The extended layout declares five compute-visible storage entries:
 * reservoirOut, current, history, result, and spatial output. Some entries
 * alias the same three physical storage buffers in individual bind groups, but
 * WebGPU's per-stage limit counts layout bindings rather than unique resources.
 */
export const PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE + 5;

/** Full tier plus the opt-in CWBVH closest-hit traversal buffers in group 3. */
export const PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE + 5;

/** Full tier plus both opt-in CWBVH closest-hit and ReSTIR-PT reuse buffers. */
export const PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE =
  PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE +
  (
    PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE -
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE
  );

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
 *     – liteLightTex (binding 14): 1×N RGBA32F directional/point/spot/rect-area packed data.
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
 * G-buffer aux). BDPT's cross-pixel camera splats use one additional storage
 * buffer, not a storage texture, so the per-stage storage-texture total stays
 * exactly group 0's 5.
 */
export const PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE = 5;

/** Lite trace pass: output + normalDepth + albedo + variance. */
export const PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE = 4;

export const PT_WEBGPU_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  maxStorageTexturesPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
};

export interface PtWebgpuRequiredLimitOptions {
  /** Include the BDPT t=1 atomic camera-splat buffer. */
  readonly bdpt?: boolean;
  /** Include the opt-in opaque one-edge reconnection storage buffers. */
  readonly oneEdgeReconnectionReuse?: boolean;
  /** @deprecated Compatibility alias for oneEdgeReconnectionReuse. */
  readonly restirPtReuse?: boolean;
  /** Include the opt-in CWBVH closest-hit traversal storage buffers. */
  readonly cwbvhClosest?: boolean;
}

/** Request the highest tier the adapter can satisfy (used by host device acquisition). */
export function ptWebgpuRequiredLimitsForAdapter(
  adapter: GPUAdapter,
  options: PtWebgpuRequiredLimitOptions = {},
): Record<string, number> {
  const maxBuffers = adapter.limits.maxStorageBuffersPerShaderStage;
  const maxTextures = adapter.limits.maxStorageTexturesPerShaderStage;
  const reconnectionReuse =
    (options.oneEdgeReconnectionReuse ?? options.restirPtReuse) === true;
  const reconnectionStorageBindings =
    PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE -
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const cwbvhStorageBindings =
    PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE -
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const bdptStorageBindings =
    PT_WEBGPU_BDPT_REQUIRED_STORAGE_BUFFERS_PER_STAGE -
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const fullBufferFloor =
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE +
    (options.bdpt === true ? bdptStorageBindings : 0) +
    (reconnectionReuse ? reconnectionStorageBindings : 0) +
    (options.cwbvhClosest === true ? cwbvhStorageBindings : 0);
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
