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

  it('replaces stale SPPM per-pixel stats buffers with a placeholder when device limits are exceeded', () => {
    installGpuConstStubs();
    const encoder = {
      clearBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    };
    const buffers: Array<{
      label: string | undefined;
      size: number;
      destroy: ReturnType<typeof vi.fn>;
    }> = [];
    const device = {
      limits: { maxBufferSize: 1024, maxStorageBufferBindingSize: 1024 },
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
        const buffer = {
          label: desc.label,
          size: desc.size,
          destroy: vi.fn(),
        };
        buffers.push(buffer);
        return buffer;
      }),
      createCommandEncoder: vi.fn(() => encoder),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const gpu = new GpuResources(device, 'full', false);
    expect(gpu.ensureSppmPixelStatsBuffer(4, 4)).toBe(true);
    const realBuffer = gpu.sppmPixelStatsBuffer as unknown as { label?: string; size: number; destroy: ReturnType<typeof vi.fn> };
    expect(realBuffer.label).toBe('vitrum.pt-webgpu.sppm.pixelStats');
    expect(realBuffer.size).toBe(512);
    expect(gpu.sppmPixelStatsWidth).toBe(4);
    expect(gpu.sppmPixelStatsHeight).toBe(4);

    gpu.pathTraceBindGroup = {} as GPUBindGroup;
    gpu.pathTraceBindGroup3 = {} as GPUBindGroup;

    expect(gpu.ensureSppmPixelStatsBuffer(16, 16)).toBe(false);

    const placeholder = gpu.sppmPixelStatsBuffer as unknown as { label?: string; size: number };
    expect(realBuffer.destroy).toHaveBeenCalledOnce();
    expect(placeholder).not.toBe(realBuffer);
    expect(placeholder.label).toBe('vitrum.pt-webgpu.sppm.pixelStats.placeholder');
    expect(placeholder.size).toBe(64);
    expect(gpu.sppmPixelStatsWidth).toBe(0);
    expect(gpu.sppmPixelStatsHeight).toBe(0);
    expect(gpu.pathTraceBindGroup).toBeNull();
    expect(gpu.pathTraceBindGroup3).toBeNull();
    expect(warnSpy).toHaveBeenCalledOnce();
  });
});
