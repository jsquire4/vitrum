import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const textureUploadMocks = vi.hoisted(() => {
  const noop = vi.fn();
  const readbackFailure = vi.fn(async () => {
    throw new Error('mock readback failure');
  });

  return {
    fillR16Uint: noop,
    fillRg32f: noop,
    fillRgba32f: noop,
    readRgba16fToRgb: readbackFailure,
    readRgba32fToRgb: readbackFailure,
    uploadInterleavedRgAsRg32f: noop,
    uploadLinearDepthAsRgba32f: noop,
    uploadR16Uint: noop,
    uploadR32f: noop,
    uploadR32Uint: noop,
    uploadRgbAsRgba16f: noop,
    uploadRgbAsRgba32f: noop,
    uploadRgbAsRgba32fPacked: noop,
  };
});

vi.mock('../src/webGpuTextureUpload.js', () => textureUploadMocks);

import { runAtrousVarianceWebGPU } from '../src/atrousVarianceWebGPU.js';
import { runBmfrWebGPU } from '../src/bmfrWebGPU.js';
import { runHdrLuminanceBilateralWebGPU } from '../src/hdrLuminanceBilateralWebGPU.js';
import { runSVGFRealWebGPU } from '../src/svgfRealWebGPU.js';

interface FakeTexture {
  readonly label: string | undefined;
  destroyed: boolean;
  createView: () => unknown;
  destroy: () => void;
}

interface FakeBuffer {
  readonly label: string | undefined;
  destroyed: boolean;
  destroy: () => void;
}

interface FakeDevice {
  readonly textures: FakeTexture[];
  readonly buffers: FakeBuffer[];
  destroyed: boolean;
}

function installWebGpuGlobals(): void {
  vi.stubGlobal('navigator', { gpu: {} });
  vi.stubGlobal('GPUTextureUsage', {
    TEXTURE_BINDING: 1,
    COPY_DST: 2,
    COPY_SRC: 4,
    STORAGE_BINDING: 8,
  });
  vi.stubGlobal('GPUBufferUsage', {
    UNIFORM: 1,
    COPY_DST: 2,
  });
}

function createFakeDevice(): FakeDevice {
  const device = {
    textures: [] as FakeTexture[],
    buffers: [] as FakeBuffer[],
    destroyed: false,
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
    },
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    createTexture: vi.fn((desc?: { label?: string }) => {
      const texture: FakeTexture = {
        label: desc?.label,
        destroyed: false,
        createView: vi.fn(() => ({})),
        destroy: vi.fn(() => {
          texture.destroyed = true;
        }),
      };
      device.textures.push(texture);
      return texture;
    }),
    createBuffer: vi.fn((desc?: { label?: string }) => {
      const buffer: FakeBuffer = {
        label: desc?.label,
        destroyed: false,
        destroy: vi.fn(() => {
          buffer.destroyed = true;
        }),
      };
      device.buffers.push(buffer);
      return buffer;
    }),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(),
        end: vi.fn(),
      })),
      copyTextureToTexture: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    destroy: vi.fn(() => {
      device.destroyed = true;
    }),
  };
  return device;
}

function expectAllTransientsDestroyed(device: FakeDevice): void {
  expect(device.textures.length).toBeGreaterThan(0);
  expect(device.buffers.length).toBeGreaterThan(0);
  expect(device.textures.every((texture) => texture.destroyed)).toBe(true);
  expect(device.buffers.every((buffer) => buffer.destroyed)).toBe(true);
  expect(device.destroyed).toBe(false);
}

describe('one-shot WebGPU denoiser cleanup', () => {
  beforeEach(() => {
    installWebGpuGlobals();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('destroys HDR bilateral resources when readback throws', async () => {
    const device = createFakeDevice();

    await expect(
      runHdrLuminanceBilateralWebGPU({
        device: device as unknown as GPUDevice,
        rgb: new Float32Array([1, 1, 1]),
        width: 1,
        height: 1,
      }),
    ).rejects.toThrow('mock readback failure');

    expect(device.textures).toHaveLength(2);
    expect(device.buffers).toHaveLength(1);
    expectAllTransientsDestroyed(device);
  });

  it('destroys atrous-variance resources when readback throws', async () => {
    const device = createFakeDevice();

    await expect(
      runAtrousVarianceWebGPU({
        device: device as unknown as GPUDevice,
        rgb: new Float32Array([1, 1, 1]),
        width: 1,
        height: 1,
        atrousIterations: 2,
      }),
    ).rejects.toThrow('mock readback failure');

    expect(device.textures).toHaveLength(7);
    expect(device.buffers).toHaveLength(3);
    expectAllTransientsDestroyed(device);
  });

  it('destroys BMFR resources when readback throws', async () => {
    const device = createFakeDevice();

    await expect(
      runBmfrWebGPU({
        device: device as unknown as GPUDevice,
        rgb: new Float32Array([1, 1, 1]),
        worldPosRgb: new Float32Array([0, 0, 0]),
        width: 1,
        height: 1,
      }),
    ).rejects.toThrow('mock readback failure');

    // color/normal/worldpos/history/out + block-fit storage + the UBO.
    expect(device.textures).toHaveLength(5);
    expect(device.buffers).toHaveLength(2);
    expectAllTransientsDestroyed(device);
  });

  it('destroys SVGF-real resources when readback throws', async () => {
    const device = createFakeDevice();

    await expect(
      runSVGFRealWebGPU({
        device: device as unknown as GPUDevice,
        rgb: new Float32Array([1, 1, 1]),
        width: 1,
        height: 1,
        atrousIterations: 2,
      }),
    ).rejects.toThrow('mock readback failure');

    expect(device.textures).toHaveLength(18);
    expect(device.buffers).toHaveLength(3);
    expectAllTransientsDestroyed(device);
  });
});
