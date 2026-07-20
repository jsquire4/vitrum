/**
 * svgfChainingReadback.test.ts — R7 regression tests (device-stubbed, no real GPU).
 *
 * Covers:
 *   (V3-3) one-shot SVGF chaining plumbing + return shape:
 *          - prev* inputs (prevLinearDepth/prevNormalsRgb/prevObjectIds/
 *            prevHistoryLength) are uploaded to the PREV textures when present;
 *          - the current-frame fallback (prev == curr) still holds when absent;
 *          - the return is { rgb, momentsOut?, historyLengthOut? }: momentsOut /
 *            historyLengthOut appear ONLY when chainable:true, and they are read
 *            back (i.e. survive) rather than being destroyed before return.
 *   (V3-5) readback try/finally: readRgba32fToRg / readR32UintToU32 destroy the
 *          staging buffer even when mapAsync rejects.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// ── Upload/readback mock: uploads are spies; readbacks return sized arrays. ────
const uploadMocks = vi.hoisted(() => {
  const noop = vi.fn();
  return {
    fillR16Uint: vi.fn(),
    fillRg32f: vi.fn(),
    fillRgba32f: vi.fn(),
    uploadInterleavedRgAsRg32f: vi.fn(),
    uploadLinearDepthAsRgba32f: vi.fn(),
    uploadR16Uint: vi.fn(),
    uploadR32f: vi.fn(),
    uploadR32Uint: vi.fn(),
    uploadRgbAsRgba16f: noop,
    uploadRgbAsRgba32f: vi.fn(),
    uploadRgbAsRgba32fPacked: vi.fn(),
    // Sized readbacks so the run succeeds and we can assert on the result.
    readRgba16fToRgb: vi.fn(async (_d: unknown, _t: unknown, w: number, h: number) =>
      new Float32Array(w * h * 3)),
    readRgba32fToRg: vi.fn(async (_d: unknown, _t: unknown, w: number, h: number) =>
      new Float32Array(w * h * 2)),
    readR32UintToU32: vi.fn(async (_d: unknown, _t: unknown, w: number, h: number) =>
      new Uint32Array(w * h)),
  };
});

vi.mock('../src/webGpuTextureUpload.js', () => uploadMocks);

import { runSVGFRealWebGPU } from '../src/svgfRealWebGPU.js';

// A minimal device stub that records created textures/buffers by label so we
// can map upload calls (which receive the texture object) back to a label.
interface StubTexture { readonly label: string | undefined; destroyed: boolean }

function createStubDevice() {
  const textures: StubTexture[] = [];
  const device = {
    textures,
    queue: { writeBuffer: vi.fn(), submit: vi.fn(), writeTexture: vi.fn() },
    createShaderModule: vi.fn(() => ({})),
    createComputePipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
    createTexture: vi.fn((desc?: { label?: string }) => {
      const t: StubTexture & { createView: () => unknown; destroy: () => void } = {
        label: desc?.label,
        destroyed: false,
        createView: vi.fn(() => ({})),
        destroy: vi.fn(() => { t.destroyed = true; }),
      };
      textures.push(t);
      return t;
    }),
    createBuffer: vi.fn((desc?: { label?: string }) => ({
      label: desc?.label,
      destroy: vi.fn(),
    })),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(), end: vi.fn(),
      })),
      copyTextureToTexture: vi.fn(),
      copyTextureToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    destroy: vi.fn(),
  };
  return device;
}

function labelOfArg(arg: unknown): string | undefined {
  return (arg as StubTexture | undefined)?.label;
}

beforeEach(() => {
  // acquireDenoiseDevice checks navigator.gpu even when an explicit device is
  // passed; stub a present-but-unused gpu object.
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

describe('one-shot SVGF chaining plumbing + return shape (V3-3)', () => {
  const W = 2, H = 2;
  const rgb = new Float32Array(W * H * 3);

  it('returns { rgb } only (no moments/history) when chainable is not set', async () => {
    const device = createStubDevice();
    const out = await runSVGFRealWebGPU({
      device: device as unknown as GPUDevice,
      rgb, width: W, height: H, atrousIterations: 1,
    });
    expect(out.rgb).toBeInstanceOf(Float32Array);
    expect(out.rgb.length).toBe(W * H * 3);
    expect(out.momentsOut).toBeUndefined();
    expect(out.historyLengthOut).toBeUndefined();
    // No chaining → the moments/history readbacks are never submitted.
    expect(uploadMocks.readRgba32fToRg).not.toHaveBeenCalled();
    expect(uploadMocks.readR32UintToU32).not.toHaveBeenCalled();
  });

  it('returns momentsOut + historyLengthOut read back BEFORE teardown when chainable:true', async () => {
    const device = createStubDevice();
    const out = await runSVGFRealWebGPU({
      device: device as unknown as GPUDevice,
      rgb, width: W, height: H, atrousIterations: 1, chainable: true,
    });
    expect(out.rgb.length).toBe(W * H * 3);
    // The reprojection outputs are read back and returned (not destroyed first).
    expect(uploadMocks.readRgba32fToRg).toHaveBeenCalledTimes(1);
    expect(uploadMocks.readR32UintToU32).toHaveBeenCalledTimes(1);
    expect(out.momentsOut).toBeInstanceOf(Float32Array);
    expect(out.momentsOut).toHaveLength(W * H * 2);
    expect(out.historyLengthOut).toBeInstanceOf(Uint32Array);
    expect(out.historyLengthOut).toHaveLength(W * H);
    // The moments-out / hist-out textures still exist at readback time — the
    // readback stub was called with the reprojection-output textures.
    const momReadTex = labelOfArg(uploadMocks.readRgba32fToRg.mock.calls[0]?.[1]);
    const histReadTex = labelOfArg(uploadMocks.readR32UintToU32.mock.calls[0]?.[1]);
    expect(momReadTex).toBe('svgf-mom-out');
    expect(histReadTex).toBe('svgf-hist-out');
  });

  it('uploads prev* inputs to the PREV textures when supplied', async () => {
    const device = createStubDevice();
    const prevDepth = new Float32Array(W * H).fill(0.5);
    const prevObj = new Uint32Array(W * H).fill(7);
    const prevNormals = new Float32Array(W * H * 3).fill(0.25);
    const prevHist = new Uint32Array(W * H).fill(3);
    await runSVGFRealWebGPU({
      device: device as unknown as GPUDevice,
      rgb, width: W, height: H, atrousIterations: 1,
      prevLinearDepth: prevDepth,
      prevObjectIds: prevObj,
      prevNormalsRgb: prevNormals,
      prevHistoryLength: prevHist,
    });

    // prev-depth texture received the prev buffer (not the current fallback).
    const depthCall = uploadMocks.uploadR32f.mock.calls.find(
      (c) => labelOfArg(c[1]) === 'svgf-prev-depth',
    );
    expect(depthCall?.[2]).toBe(prevDepth);

    const objCall = uploadMocks.uploadR32Uint.mock.calls.find(
      (c) => labelOfArg(c[1]) === 'svgf-prev-obj',
    );
    expect(objCall?.[2]).toBe(prevObj);

    const normCall = uploadMocks.uploadRgbAsRgba32fPacked.mock.calls.find(
      (c) => labelOfArg(c[1]) === 'svgf-prev-norm',
    );
    expect(normCall?.[2]).toBe(prevNormals);

    // prevHistoryLength feeds the history-in texture.
    const histCall = uploadMocks.uploadR16Uint.mock.calls.find(
      (c) => labelOfArg(c[1]) === 'svgf-hist-in',
    );
    expect(histCall?.[2]).toBe(prevHist);
  });

  it('falls back to current-frame values for prev textures when prev* absent', async () => {
    const device = createStubDevice();
    const depth = new Float32Array(W * H).fill(0.9);
    await runSVGFRealWebGPU({
      device: device as unknown as GPUDevice,
      rgb, width: W, height: H, atrousIterations: 1,
      linearDepth: depth,
    });
    // prev-depth mirrors curr linearDepth (no prevLinearDepth supplied).
    const prevDepthCall = uploadMocks.uploadR32f.mock.calls.find(
      (c) => labelOfArg(c[1]) === 'svgf-prev-depth',
    );
    expect(prevDepthCall?.[2]).toBe(depth);
  });
});

// ── readback try/finally (V3-5) ───────────────────────────────────────────────

describe('webGpuTextureUpload readback try/finally (V3-5)', () => {
  it('destroys the staging buffer when mapAsync rejects (readRgba32fToRg / readR32UintToU32)', async () => {
    // Import the REAL module (this describe is not affected by the vi.mock above
    // because that mock replaces the module for importers; here we import the
    // real implementation directly via a fresh unmocked resolution).
    const real = await vi.importActual<typeof import('../src/webGpuTextureUpload.js')>(
      '../src/webGpuTextureUpload.js',
    );

    const destroyed: boolean[] = [];
    function deviceWithRejectingMap() {
      return {
        createBuffer: vi.fn(() => {
          const idx = destroyed.push(false) - 1;
          return {
            mapAsync: vi.fn(async () => { throw new Error('mock mapAsync failure'); }),
            getMappedRange: vi.fn(() => new ArrayBuffer(0)),
            unmap: vi.fn(),
            destroy: vi.fn(() => { destroyed[idx] = true; }),
          };
        }),
        createCommandEncoder: vi.fn(() => ({
          copyTextureToBuffer: vi.fn(),
          finish: vi.fn(() => ({})),
        })),
        queue: { submit: vi.fn() },
      } as unknown as GPUDevice;
    }

    vi.stubGlobal('GPUBufferUsage', { COPY_DST: 2, MAP_READ: 1 });
    vi.stubGlobal('GPUMapMode', { READ: 1 });

    await expect(real.readRgba32fToRg(deviceWithRejectingMap(), {} as GPUTexture, 2, 2))
      .rejects.toThrow('mock mapAsync failure');
    await expect(real.readR32UintToU32(deviceWithRejectingMap(), {} as GPUTexture, 2, 2))
      .rejects.toThrow('mock mapAsync failure');

    expect(destroyed).toEqual([true, true]);
  });
});
