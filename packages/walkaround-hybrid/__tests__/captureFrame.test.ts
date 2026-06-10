/**
 * captureFrame — walkaround-hybrid backend.
 *
 * Tests:
 *   1. captureFrame({ colorSpace: 'output' }) rejects with a clear message
 *      (swap-chain constraint: no engine-owned display buffer to read back).
 *   2. captureFrame() returns null before the first frame (getProgressiveSeedTexture
 *      returns null when the pipeline is not yet initialised).
 *
 * Both tests use the HybridEngine class directly with a mock GPUDevice.
 */
import { describe, expect, it, vi } from 'vitest';
import type { HybridEngineOptions } from '../src/HybridEngineOptions.js';
import { createWalkaroundEngine_Hybrid } from '../src/index.js';

// Minimal mock GPUDevice — enough for HybridEngine construction.
function makeStubDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: 8,
      maxSampledTexturesPerShaderStage: 16,
      maxSamplersPerShaderStage: 16,
      maxBindGroupsPlusVertexBuffers: 8,
      maxBindGroups: 4,
      maxUniformBuffersPerShaderStage: 12,
      maxStorageBufferBindingSize: 1 << 30,
      maxUniformBufferBindingSize: 65536,
      maxDynamicStorageBuffersPerPipelineLayout: 0,
      maxDynamicUniformBuffersPerPipelineLayout: 0,
      maxTextureArrayLayers: 256,
      maxTextureDimension2D: 8192,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64,
      maxComputeWorkgroupsPerDimension: 65535,
      maxVertexBuffers: 8,
      maxBufferSize: 1 << 30,
    },
    features: new Set<string>(),
    createBuffer: vi.fn(() => ({
      destroy: vi.fn(),
      getMappedRange: vi.fn(() => new ArrayBuffer(16)),
      unmap: vi.fn(),
      mapAsync: vi.fn(() => Promise.resolve()),
    })),
    createTexture: vi.fn(() => ({
      destroy: vi.fn(),
      createView: vi.fn(() => ({})),
      width: 64, height: 64, depthOrArrayLayers: 1,
    })),
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({})),
    createRenderPipeline: vi.fn(() => ({})),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(), setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(), end: vi.fn(),
      })),
      beginRenderPass: vi.fn(() => ({
        setPipeline: vi.fn(), setBindGroup: vi.fn(),
        draw: vi.fn(), end: vi.fn(),
      })),
      copyTextureToBuffer: vi.fn(),
      clearBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    createQuerySet: vi.fn(() => ({ destroy: vi.fn() })),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
    destroy: vi.fn(),
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(() => Promise.resolve(null)),
  } as unknown as GPUDevice;
}

const MINIMAL_OPTS: Pick<HybridEngineOptions, 'device' | 'width' | 'height'> = {
  device: makeStubDevice(),
  width: 64,
  height: 64,
};

describe('captureFrame walkaround-hybrid', () => {
  it("captureFrame({ colorSpace: 'output' }) rejects with swap-chain message", async () => {
    const engine = await createWalkaroundEngine_Hybrid(MINIMAL_OPTS as HybridEngineOptions);
    await expect(
      engine.captureFrame?.({ colorSpace: 'output' }),
    ).rejects.toThrow(/swap-chain|swap.chain/i);
    engine.dispose();
  });

  it("captureFrame() (linear default) returns null before first frame", async () => {
    const engine = await createWalkaroundEngine_Hybrid(MINIMAL_OPTS as HybridEngineOptions);
    // No renderFrame call — getProgressiveSeedTexture is null → captureFrame is null.
    const frame = await engine.captureFrame?.();
    expect(frame).toBeNull();
    engine.dispose();
  });

  it('engine exposes captureFrame as a function', async () => {
    const engine = await createWalkaroundEngine_Hybrid(MINIMAL_OPTS as HybridEngineOptions);
    expect(typeof engine.captureFrame).toBe('function');
    engine.dispose();
  });
});
