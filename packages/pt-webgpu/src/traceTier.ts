import {
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
} from './webgpuLimits.js';

export type PtWebgpuTraceTier = 'full' | 'lite';

/**
 * Pick full vs lite from adapter limits. Full tier enables TLAS, analytics,
 * HDRI, all emitter buffers, motion/variance aux, and caustics when the adapter
 * can satisfy the aggregate per-stage storage-buffer/texture floor. Lite is
 * only for constrained adapters (for example SwiftShader-class CI devices).
 */
export function selectPtWebgpuTraceTier(device: GPUDevice): PtWebgpuTraceTier {
  const maxBuffers =
    device.limits?.maxStorageBuffersPerShaderStage ??
    PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const maxTextures = device.limits?.maxStorageTexturesPerShaderStage ?? 8;
  if (
    maxBuffers >= PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE &&
    maxTextures >= PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE
  ) {
    return 'full';
  }
  if (
    maxBuffers >= PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE &&
    maxTextures >= PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE
  ) {
    return 'lite';
  }
  throw new Error(
    `pt-webgpu: adapter limits maxStorageBuffersPerShaderStage=${maxBuffers}, ` +
      `maxStorageTexturesPerShaderStage=${maxTextures} are below the lite tier ` +
      `(${PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE} buffers, ` +
      `${PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE} textures)`,
  );
}

/** Auto-detect unless `force` is set; forcing `full` throws when the adapter cannot bind it. */
export function resolvePtWebgpuTraceTier(
  device: GPUDevice,
  force?: PtWebgpuTraceTier,
): PtWebgpuTraceTier {
  if (force === 'lite') {
    // A forced tier is still a capability request, not permission to defer an
    // impossible bind layout to shader/pipeline creation. select validates the
    // lite floor (and may return full on a stronger adapter); then honor the
    // caller's deliberate lower-tier choice.
    selectPtWebgpuTraceTier(device);
    return 'lite';
  }
  if (force === 'full') {
    const tier = selectPtWebgpuTraceTier(device);
    if (tier !== 'full') {
      const maxBuffers = device.limits.maxStorageBuffersPerShaderStage;
      const maxTextures = device.limits.maxStorageTexturesPerShaderStage;
      throw new Error(
        `pt-webgpu: traceTier=full requested but adapter reports ` +
          `maxStorageBuffersPerShaderStage=${maxBuffers}, ` +
          `maxStorageTexturesPerShaderStage=${maxTextures} (need >=${PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE} buffers/stage and >=${PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE} textures). ` +
          'Use a discrete GPU / native browser WebGPU, not SwiftShader-only CI.',
      );
    }
    return 'full';
  }
  return selectPtWebgpuTraceTier(device);
}
