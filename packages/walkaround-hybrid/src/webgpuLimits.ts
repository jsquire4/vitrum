/**
 * Dependency-free WebGPU device-limit authority for the walkaround renderer.
 *
 * Keep this module free of pipeline imports so browser hosts and native
 * validation tools can request the exact production floor without loading the
 * renderer graph or duplicating numeric thresholds.
 */

/**
 * Hard minima derived from the explicit bind-group layouts compiled by the
 * walkaround pipeline:
 *
 * - 8 storage buffers: RIS/shade/GI-RIS packed scene, PPG, and NRC arenas
 * - 7 storage textures: transparent OIT (six frame targets plus OIT output)
 * - 16 sampled textures: NRC GI-RIS frame, scene, UBO, and hybrid layers
 */
export const HYBRID_WEBGPU_REQUIRED_LIMITS: Readonly<Record<string, number>> =
  Object.freeze({
    maxStorageBuffersPerShaderStage: 8,
    maxStorageTexturesPerShaderStage: 7,
    maxSampledTexturesPerShaderStage: 16,
  });

/**
 * Lite currently compiles the same explicit layouts as full. Runtime gates and
 * merged BVH reduce memory/work, but WebGPU validates every declared binding,
 * including bindings unused by the selected shader path.
 */
export const HYBRID_LITE_LIMITS: Readonly<Record<string, number>> =
  HYBRID_WEBGPU_REQUIRED_LIMITS;
