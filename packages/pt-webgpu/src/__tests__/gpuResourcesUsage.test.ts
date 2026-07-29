import { describe, expect, it, vi } from 'vitest';
import { GpuResources } from '../gpuResources.js';
import { createSizeValidatingGpuDeviceStub, installGpuConstStubs } from './gpuStub.js';

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
    const formatByLabel = new Map(textureDescs.map((desc) => [desc.label, desc.format]));
    const copySrc = GPUTextureUsage.COPY_SRC;
    expect((usageByLabel.get('vitrum.pt-webgpu.normalDepth')! & copySrc) !== 0).toBe(true);
    expect((usageByLabel.get('vitrum.pt-webgpu.albedo')! & copySrc) !== 0).toBe(true);
    expect(formatByLabel.get('vitrum.pt-webgpu.variance')).toBe('r32float');
  });

  it('allocates, clears, reuses, and disposes the BDPT camera-splat buffer', () => {
    installGpuConstStubs();
    const bufferDescs: GPUBufferDescriptor[] = [];
    const buffers = new Map<string, {
      readonly label: string;
      readonly destroy: ReturnType<typeof vi.fn>;
    }>();
    const encoder = {
      clearBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    };
    const device = {
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
        bufferDescs.push(desc);
        const buffer = {
          label: String(desc.label),
          destroy: vi.fn(),
        };
        buffers.set(buffer.label, buffer);
        return buffer;
      }),
      createTexture: vi.fn(() => ({
        createView: vi.fn(() => ({})),
        destroy: vi.fn(),
      })),
      createCommandEncoder: vi.fn(() => encoder),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;

    const gpu = new GpuResources(device, 'full', true);
    expect(gpu.ensureAccumResources(4, 3)).toBe(true);
    const descriptor = bufferDescs.find(
      (desc) => desc.label === 'vitrum.pt-webgpu.bdpt.cameraSplats.buffer',
    );
    expect(descriptor).toMatchObject({
      size: 4 * 3 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    expect(encoder.clearBuffer).toHaveBeenCalledWith(
      gpu.bdptCameraSplatBuffer,
    );

    const allocated = gpu.bdptCameraSplatBuffer;
    expect(gpu.ensureAccumResources(4, 3)).toBe(false);
    expect(gpu.bdptCameraSplatBuffer).toBe(allocated);

    gpu.dispose();
    expect(
      buffers.get('vitrum.pt-webgpu.bdpt.cameraSplats.buffer')?.destroy,
    ).toHaveBeenCalledOnce();
    expect(gpu.bdptCameraSplatBuffer).toBeNull();
  });

  it('preserves a usable SPPM per-pixel stats buffer when a larger request exceeds device limits', () => {
    const { device, buffers } = createSizeValidatingGpuDeviceStub({ maxBufferSize: 1024 });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const gpu = new GpuResources(device, 'full', false);
    expect(gpu.ensureSppmPixelStatsBuffer(4, 4)).toBe(true);
    const realBuffer = gpu.sppm.sppmPixelStatsBuffer as unknown as { label?: string; size: number; destroy: ReturnType<typeof vi.fn> };
    expect(realBuffer.label).toBe('vitrum.pt-webgpu.sppm.pixelStats');
    expect(realBuffer.size).toBe(1024);
    expect(gpu.sppm.sppmPixelStatsWidth).toBe(4);
    expect(gpu.sppm.sppmPixelStatsHeight).toBe(4);

    gpu.pathTraceBindGroup = {} as GPUBindGroup;
    gpu.pathTraceBindGroup3 = {} as GPUBindGroup;

    expect(gpu.ensureSppmPixelStatsBuffer(16, 16)).toBe(false);

    const retained = gpu.sppm.sppmPixelStatsBuffer as unknown as { label?: string; size: number };
    expect(realBuffer.destroy).not.toHaveBeenCalled();
    expect(retained).toBe(realBuffer);
    expect(retained.label).toBe('vitrum.pt-webgpu.sppm.pixelStats');
    expect(retained.size).toBe(1024);
    expect(buffers.every((b) => b.size <= 1024)).toBe(true);
    expect(gpu.sppm.sppmPixelStatsWidth).toBe(4);
    expect(gpu.sppm.sppmPixelStatsHeight).toBe(4);
    expect(gpu.pathTraceBindGroup).not.toBeNull();
    expect(gpu.pathTraceBindGroup3).not.toBeNull();
    expect(warnSpy).toHaveBeenCalledOnce();
  });
});
