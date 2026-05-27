import { describe, expect, it } from 'vitest';
import {
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_REQUIRED_LIMITS,
  PT_WEBGPU_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  mergeAdapterRequiredLimits,
  ptWebgpuRequiredLimitsForAdapter,
} from '../webgpuLimits.js';

describe('webgpuLimits', () => {
  it('PT limits request storage buffers only (no storage texture bump)', () => {
    expect(PT_WEBGPU_REQUIRED_LIMITS.maxStorageTexturesPerShaderStage).toBeUndefined();
    expect(PT_WEBGPU_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage).toBe(10);
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

  it('ptWebgpuRequiredLimitsForAdapter picks lite caps on SwiftShader-class adapters', () => {
    const adapter = {
      limits: {
        maxStorageBuffersPerShaderStage: 10,
        maxStorageTexturesPerShaderStage: 4,
      },
    } as GPUAdapter;
    expect(ptWebgpuRequiredLimitsForAdapter(adapter)).toEqual({
      maxStorageBuffersPerShaderStage: PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 4,
    });
  });
});
