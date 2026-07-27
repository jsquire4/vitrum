import { describe, expect, it } from 'vitest';

import {
  assertNrcDeviceCapable,
  NRC_REQUIRED_MAX_BIND_GROUPS,
  NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
  NRC_REQUIRED_WORKGROUP_STORAGE_BYTES,
} from '../WalkaroundGPUPipeline.js';

function limits(overrides: Partial<GPUSupportedLimits> = {}): GPUSupportedLimits {
  return {
    maxBindGroups: 4,
    maxStorageBuffersPerShaderStage: 8,
    maxComputeWorkgroupStorageSize: 16_384,
    ...overrides,
  } as GPUSupportedLimits;
}

function expectLimitFailure(deviceLimits: GPUSupportedLimits, needle: string): void {
  let thrown: unknown;
  try {
    assertNrcDeviceCapable(deviceLimits);
  } catch (err) {
    thrown = err;
  }
  expect(thrown).toBeInstanceOf(TypeError);
  expect((thrown as Error).message).toContain('nrcEnabled requires');
  expect((thrown as Error).message).toContain(needle);
}

describe('NRC device capability gate', () => {
  it('exact required NRC limits pass', () => {
    expect(() => assertNrcDeviceCapable(limits({
      maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS,
      maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
      maxComputeWorkgroupStorageSize: NRC_REQUIRED_WORKGROUP_STORAGE_BYTES,
    }))).not.toThrow();
  });

  it('fits the default WebGPU group/storage-buffer axes', () => {
    expect(() => assertNrcDeviceCapable(limits())).not.toThrow();
    expect(NRC_REQUIRED_WORKGROUP_STORAGE_BYTES).toBe(16_384);
  });

  it('rejects devices that are under the maxBindGroups NRC threshold', () => {
    expectLimitFailure(limits({
      maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS - 1,
      maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
      maxComputeWorkgroupStorageSize: NRC_REQUIRED_WORKGROUP_STORAGE_BYTES,
    }), 'maxBindGroups');
  });

  it('rejects devices that are under the maxStorageBuffersPerShaderStage NRC threshold', () => {
    expectLimitFailure(limits({
      maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS,
      maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE - 1,
      maxComputeWorkgroupStorageSize: NRC_REQUIRED_WORKGROUP_STORAGE_BYTES,
    }), 'maxStorageBuffersPerShaderStage');
  });

  it('rejects devices that are under the maxComputeWorkgroupStorageSize NRC threshold', () => {
    expectLimitFailure(limits({
      maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS,
      maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
      maxComputeWorkgroupStorageSize: NRC_REQUIRED_WORKGROUP_STORAGE_BYTES - 1,
    }), 'maxComputeWorkgroupStorageSize');
  });
});
