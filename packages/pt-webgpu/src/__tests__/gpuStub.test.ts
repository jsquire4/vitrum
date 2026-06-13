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
});
