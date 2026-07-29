/**
 * Host-side BMFR overlap routing tests. The fit grid may overlap, but each
 * workgroup writes a private coefficient record and a second pass resolves the
 * records onto the pixel grid.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const uploadMocks = vi.hoisted(() => ({
  uploadRgbAsRgba16f: vi.fn(),
  readRgba16fToRgb: vi.fn(async (_d: unknown, _t: unknown, w: number, h: number) =>
    new Float32Array(w * h * 3)),
}));

vi.mock('../src/webGpuTextureUpload.js', () => uploadMocks);

import { runBmfrWebGPU } from '../src/bmfrWebGPU.js';

function createStubDevice(options: { failPipelineAt?: number } = {}) {
  const dispatchCalls: Array<[number, number, number]> = [];
  const bufferDescriptors: GPUBufferDescriptor[] = [];
  const buffers: Array<{ destroy: ReturnType<typeof vi.fn> }> = [];
  let pipelineCount = 0;
  const device = {
    dispatchCalls,
    bufferDescriptors,
    buffers,
    limits: {},
    queue: { writeBuffer: vi.fn(), submit: vi.fn(), writeTexture: vi.fn() },
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => {
      pipelineCount += 1;
      if (pipelineCount === options.failPipelineAt) {
        throw new Error('pipeline failure');
      }
      return { getBindGroupLayout: vi.fn(() => ({})) };
    }),
    createTexture: vi.fn(() => ({
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    })),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      bufferDescriptors.push(descriptor);
      const buffer = { destroy: vi.fn() };
      buffers.push(buffer);
      return buffer;
    }),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn((x: number, y: number, z: number) => {
          dispatchCalls.push([x, y, z]);
        }),
        end: vi.fn(),
      })),
      finish: vi.fn(() => ({})),
    })),
    destroy: vi.fn(),
  };
  return device;
}

beforeEach(() => {
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
    STORAGE: 4,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('BMFR overlap fit + resolve', () => {
  it('blends remodulated history with the current fit in one albedo-demodulated domain', async () => {
    const device = createStubDevice();
    const rho = 0.5;
    const temporalAlpha = 0.2;

    // Identity-reconstruction oracle for one pixel: reproduce the resolve
    // kernel's mix(history, current, alpha) from the actual uploaded arrays.
    uploadMocks.readRgba16fToRgb.mockImplementationOnce(async () => {
      const uploads = uploadMocks.uploadRgbAsRgba16f.mock.calls as unknown as Array<
        [GPUDevice, GPUTexture, Float32Array, number, number]
      >;
      const currentLighting = uploads[0]?.[2];
      const historyLighting = uploads[2]?.[2];
      if (currentLighting == null || historyLighting == null) {
        throw new Error('expected BMFR current and history uploads');
      }
      return new Float32Array([
        historyLighting[0]! * (1 - temporalAlpha) + currentLighting[0]! * temporalAlpha,
        historyLighting[1]! * (1 - temporalAlpha) + currentLighting[1]! * temporalAlpha,
        historyLighting[2]! * (1 - temporalAlpha) + currentLighting[2]! * temporalAlpha,
      ]);
    });

    const result = await runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array([1, 1, 1]),
      historyRgb: new Float32Array([0.5, 0.5, 0.5]),
      albedoRgb: new Float32Array([rho, rho, rho]),
      worldPosRgb: new Float32Array([0, 0, 1]),
      width: 1,
      height: 1,
      temporalAlpha,
    });

    const uploads = uploadMocks.uploadRgbAsRgba16f.mock.calls as unknown as Array<
      [GPUDevice, GPUTexture, Float32Array, number, number]
    >;
    expect(Array.from(uploads[0]![2])).toEqual([2, 2, 2]);
    expect(Array.from(uploads[2]![2])).toEqual([1, 1, 1]);
    // Correct remodulated-domain EMA:
    // (1 - 0.2) * 0.5 history + 0.2 * 1.0 current = 0.6.
    expect(Array.from(result)).toEqual([
      expect.closeTo(0.6, 6),
      expect.closeTo(0.6, 6),
      expect.closeTo(0.6, 6),
    ]);
  });

  it('uses the half-block overlap default and a full pixel resolve grid', async () => {
    const device = createStubDevice();
    const width = 64;
    const height = 64;
    await runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array(width * height * 3),
      worldPosRgb: new Float32Array(width * height * 3),
      width,
      height,
    });

    expect(device.dispatchCalls).toEqual([
      [4, 4, 1],
      [8, 8, 1],
    ]);
    expect(device.bufferDescriptors).toContainEqual(
      expect.objectContaining({
        label: 'bmfr-block-fits',
        size: 4 * 4 * 160,
      }),
    );
  });

  it('supports exact-footprint non-overlapping blocks', async () => {
    const device = createStubDevice();
    const width = 64;
    const height = 64;
    await runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array(width * height * 3),
      worldPosRgb: new Float32Array(width * height * 3),
      width,
      height,
      blockSize: 32,
      blockStride: 32,
    });
    expect(device.dispatchCalls).toEqual([
      [2, 2, 1],
      [8, 8, 1],
    ]);
  });

  it('covers non-divisible dimensions in both passes', async () => {
    const device = createStubDevice();
    const width = 65;
    const height = 33;
    await runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array(width * height * 3),
      worldPosRgb: new Float32Array(width * height * 3),
      width,
      height,
      blockSize: 32,
      blockStride: 16,
    });
    expect(device.dispatchCalls).toEqual([
      [5, 3, 1],
      [9, 5, 1],
    ]);
  });

  it.each([
    ['too-small stride', { blockSize: 32, blockStride: 15 }, /blockStride must be >= 16/],
    ['gapped stride', { blockSize: 32, blockStride: 33 }, /blockStride must be <= 32/],
    ['fractional stride', { blockSize: 32, blockStride: 16.5 }, /blockStride must be an integer/],
    ['fractional block', { blockSize: 31.5 }, /blockSize must be an integer/],
    ['non-finite stride', { blockStride: Number.POSITIVE_INFINITY }, /blockStride must be finite/],
  ])('rejects %s before GPU allocation', async (_label, tuning, pattern) => {
    const device = createStubDevice();
    await expect(runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array(3),
      worldPosRgb: new Float32Array(3),
      width: 1,
      height: 1,
      ...tuning,
    })).rejects.toThrow(pattern);
    expect(device.createTexture).not.toHaveBeenCalled();
    expect(device.createBuffer).not.toHaveBeenCalled();
  });

  it('destroys all allocated resources when command encoding fails', async () => {
    const device = createStubDevice();
    device.createCommandEncoder.mockImplementationOnce(() => {
      throw new Error('encoder failure');
    });
    await expect(runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array(3),
      worldPosRgb: new Float32Array(3),
      width: 1,
      height: 1,
    })).rejects.toThrow(/encoder failure/);
    expect(device.buffers.length).toBe(2);
    for (const buffer of device.buffers) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
    for (const texture of device.createTexture.mock.results) {
      expect(texture.value.destroy).toHaveBeenCalledOnce();
    }
  });
});
