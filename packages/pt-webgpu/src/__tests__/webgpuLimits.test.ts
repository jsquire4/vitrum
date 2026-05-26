import { describe, expect, it } from 'vitest';
import {
  PT_WEBGPU_REQUIRED_LIMITS,
  PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  mergeAdapterRequiredLimits,
} from '../webgpuLimits.js';

describe('webgpuLimits', () => {
  it('PT limits request storage buffers only (no storage texture bump)', () => {
    expect(PT_WEBGPU_REQUIRED_LIMITS.maxStorageTexturesPerShaderStage).toBeUndefined();
    expect(PT_WEBGPU_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it('mergeAdapterRequiredLimits clamps to adapter caps', () => {
    const adapter = {
      limits: {
        maxStorageBuffersPerShaderStage: 10,
        maxStorageTexturesPerShaderStage: 4,
      },
    } as GPUAdapter;
    const merged = mergeAdapterRequiredLimits(adapter, {
      maxStorageBuffersPerShaderStage: 23,
      maxStorageTexturesPerShaderStage: 8,
    });
    expect(merged.maxStorageBuffersPerShaderStage).toBe(10);
    expect(merged.maxStorageTexturesPerShaderStage).toBe(4);
  });
});
