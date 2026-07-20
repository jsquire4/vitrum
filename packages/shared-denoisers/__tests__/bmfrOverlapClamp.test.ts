/**
 * bmfrOverlapClamp.test.ts — R7 (V3-4) regression: BMFR blockStride is clamped
 * to >= blockSize so overlapping workgroups never textureStore the same texels
 * (which would be last-writer-wins nondeterministic across workgroups).
 *
 * Device-stubbed (no real GPU): asserts the dispatched workgroup grid reflects
 * the CLAMPED stride, not the requested overlapping one.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const uploadMocks = vi.hoisted(() => ({
  uploadRgbAsRgba16f: vi.fn(),
  readRgba16fToRgb: vi.fn(async (_d: unknown, _t: unknown, w: number, h: number) =>
    new Float32Array(w * h * 3)),
}));

vi.mock('../src/webGpuTextureUpload.js', () => uploadMocks);

import { runBmfrWebGPU } from '../src/bmfrWebGPU.js';

function createStubDevice() {
  const dispatchCalls: Array<[number, number, number]> = [];
  const device = {
    dispatchCalls,
    queue: { writeBuffer: vi.fn(), submit: vi.fn(), writeTexture: vi.fn() },
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
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
    TEXTURE_BINDING: 1, COPY_DST: 2, COPY_SRC: 4, STORAGE_BINDING: 8,
  });
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('BMFR overlap knob clamped to >= blockSize (V3-4)', () => {
  it('clamps a requested overlapping stride (8) up to blockSize (32)', async () => {
    const device = createStubDevice();
    const W = 64, H = 64;
    await runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array(W * H * 3),
      worldPosRgb: new Float32Array(W * H * 3),
      width: W, height: H,
      blockSize: 32,
      blockStride: 8, // overlap request — must be clamped up to 32
    });
    // With stride clamped to 32: blocksX = ceil(64/32) = 2 (an unclamped
    // stride of 8 would have dispatched ceil(64/8) = 8).
    expect(device.dispatchCalls).toHaveLength(1);
    expect(device.dispatchCalls[0]).toEqual([2, 2, 1]);
  });

  it('leaves a non-overlapping stride (>= blockSize) unchanged', async () => {
    const device = createStubDevice();
    const W = 64, H = 64;
    await runBmfrWebGPU({
      device: device as unknown as GPUDevice,
      rgb: new Float32Array(W * H * 3),
      worldPosRgb: new Float32Array(W * H * 3),
      width: W, height: H,
      blockSize: 32,
      blockStride: 64, // larger than blockSize — passes through
    });
    // blocksX = ceil(64/64) = 1.
    expect(device.dispatchCalls[0]).toEqual([1, 1, 1]);
  });
});
