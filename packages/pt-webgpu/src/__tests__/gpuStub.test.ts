import { describe, expect, it } from 'vitest';
import { createSizeValidatingGpuDeviceStub } from './gpuStub.js';

describe('D10 size-validating GPU test stub', () => {
  it('rejects invalid and over-limit buffers like a real WebGPU device would', () => {
    const { device } = createSizeValidatingGpuDeviceStub({ maxBufferSize: 64 });
    expect(() => device.createBuffer({ label: 'zero', size: 0, usage: GPUBufferUsage.STORAGE })).toThrow(
      /invalid buffer size/,
    );
    expect(() => device.createBuffer({ label: 'too-big', size: 128, usage: GPUBufferUsage.STORAGE })).toThrow(
      /maxBufferSize=64/,
    );
    expect(() => device.createBuffer({ label: 'ok', size: 64, usage: GPUBufferUsage.STORAGE })).not.toThrow();
  });

  it('rejects buffers without usage flags', () => {
    const { device } = createSizeValidatingGpuDeviceStub();
    expect(() => device.createBuffer({ label: 'bad-usage', size: 16, usage: 0 })).toThrow(/invalid usage 0/);
    expect(() => device.createBuffer({ label: 'fractional-usage', size: 16, usage: 1.5 })).toThrow(
      /invalid usage 1.5/,
    );
    expect(() =>
      device.createBuffer({ label: 'storage-ok', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    ).not.toThrow();
  });

  it('rejects over-limit texture dimensions and array layers', () => {
    const { device } = createSizeValidatingGpuDeviceStub({
      maxTextureDimension2D: 8,
      maxTextureArrayLayers: 2,
    });
    expect(() =>
      device.createTexture({
        label: 'too-wide',
        size: [9, 1, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING,
      }),
    ).toThrow(/maxTextureDimension2D=8/);
    expect(() =>
      device.createTexture({
        label: 'too-many-layers',
        size: [1, 1, 3],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING,
      }),
    ).toThrow(/maxTextureArrayLayers=2/);
    expect(() =>
      device.createTexture({
        label: 'ok',
        size: [8, 8, 2],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING,
      }),
    ).not.toThrow();
  });

  it('validates bind-group buffer ranges, usage flags, and min binding sizes', () => {
    const { device, bindGroups } = createSizeValidatingGpuDeviceStub();
    const uniform = device.createBuffer({ label: 'uniform', size: 64, usage: GPUBufferUsage.UNIFORM });
    const storage = device.createBuffer({ label: 'storage', size: 32, usage: GPUBufferUsage.STORAGE });
    const layout = device.createBindGroupLayout({
      label: 'layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: 32 } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage', minBindingSize: 16 } },
      ],
    });

    expect(() =>
      device.createBindGroup({
        label: 'ok',
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniform, offset: 16, size: 32 } },
          { binding: 1, resource: { buffer: storage, size: 16 } },
        ],
      }),
    ).not.toThrow();
    expect(bindGroups).toHaveLength(1);

    expect(() =>
      device.createBindGroup({
        label: 'uniform-too-small',
        layout,
        entries: [{ binding: 0, resource: { buffer: uniform, size: 16 } }],
      }),
    ).toThrow(/minBindingSize=32/);
    expect(() =>
      device.createBindGroup({
        label: 'uniform-range-overflow',
        layout,
        entries: [{ binding: 0, resource: { buffer: uniform, offset: 48, size: 32 } }],
      }),
    ).toThrow(/exceeds/);
    expect(() =>
      device.createBindGroup({
        label: 'uniform-empty-implicit-range',
        layout,
        entries: [{ binding: 0, resource: { buffer: uniform, offset: 64 } }],
      }),
    ).toThrow(/leaves no bindable range/);
    expect(() =>
      device.createBindGroup({
        label: 'storage-as-uniform',
        layout,
        entries: [{ binding: 0, resource: { buffer: storage, size: 32 } }],
      }),
    ).toThrow(/GPUBufferUsage\.UNIFORM/);
    expect(() =>
      device.createBindGroup({
        label: 'uniform-as-storage',
        layout,
        entries: [{ binding: 1, resource: { buffer: uniform, size: 32 } }],
      }),
    ).toThrow(/GPUBufferUsage\.STORAGE/);
  });
});
