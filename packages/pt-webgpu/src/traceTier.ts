import { PT_WEBGPU_TRACE_LITE_WGSL } from './wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_TRACE_WGSL } from './wgsl/pathTraceBruteforce.wgsl.js';
import {
  PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
} from './webgpuLimits.js';

export type PtWebgpuTraceTier = 'full' | 'lite';

/**
 * Pick full vs lite from adapter limits. Full tier (3 bind groups, ≤10 buffers
 * each) enables TLAS, analytics, HDRI, all emitter buffers, motion/variance aux,
 * and caustics on Chrome-class GPUs (16 buffers/stage).
 * Lite is only for software adapters (e.g. SwiftShader 10/4).
 */
export function selectPtWebgpuTraceTier(device: GPUDevice): PtWebgpuTraceTier {
  const maxBuffers =
    device.limits?.maxStorageBuffersPerShaderStage ??
    PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP;
  const maxTextures = device.limits?.maxStorageTexturesPerShaderStage ?? 8;
  if (
    maxBuffers >= PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP &&
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

export function traceWgslForTier(tier: PtWebgpuTraceTier): string {
  return tier === 'lite' ? PT_WEBGPU_TRACE_LITE_WGSL : PT_WEBGPU_TRACE_WGSL;
}

/** Auto-detect unless `force` is set; forcing `full` throws when the adapter cannot bind it. */
export function resolvePtWebgpuTraceTier(
  device: GPUDevice,
  force?: PtWebgpuTraceTier,
): PtWebgpuTraceTier {
  if (force === 'lite') {
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
          `maxStorageTexturesPerShaderStage=${maxTextures} (need ≥${PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP} buffers/group and ≥${PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE} textures). ` +
          'Use a discrete GPU / native browser WebGPU, not SwiftShader-only CI.',
      );
    }
    return 'full';
  }
  return selectPtWebgpuTraceTier(device);
}
