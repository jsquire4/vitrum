import { describe, expect, it, vi } from 'vitest';
import { GpuResources } from '../gpuResources.js';
import { installGpuConstStubs } from './gpuStub.js';

describe('GpuResources texture usage', () => {
  it('marks OIDN aux textures copyable for readback', () => {
    installGpuConstStubs();
    const textureDescs: GPUTextureDescriptor[] = [];
    const encoder = {
      clearBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    };
    const device = {
      createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
      createTexture: vi.fn((desc: GPUTextureDescriptor) => {
        textureDescs.push(desc);
        return { createView: vi.fn(() => ({})), destroy: vi.fn() };
      }),
      createCommandEncoder: vi.fn(() => encoder),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;

    const gpu = new GpuResources(device, 'full', false);
    gpu.ensureAccumResources(4, 4);

    const usageByLabel = new Map(textureDescs.map((desc) => [desc.label, desc.usage]));
    const copySrc = GPUTextureUsage.COPY_SRC;
    expect((usageByLabel.get('vitrum.pt-webgpu.normalDepth')! & copySrc) !== 0).toBe(true);
    expect((usageByLabel.get('vitrum.pt-webgpu.albedo')! & copySrc) !== 0).toBe(true);
  });
});
