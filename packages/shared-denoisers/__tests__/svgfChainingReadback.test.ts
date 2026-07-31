/**
 * svgfChainingReadback.test.ts — R7 regression tests (device-stubbed, no real GPU).
 *
 * Covers:
 *   (V3-3) one-shot SVGF chaining plumbing + return shape:
 *          - prev* inputs (prevLinearDepth/prevNormalsRgb/prevObjectIds/
 *            prevHistoryLength) are uploaded to the PREV textures when present;
 *          - the current-frame fallback (prev == curr) still holds when absent;
 *          - the return is
 *            { rgb, prevRadianceOut?, momentsOut?, historyLengthOut? }:
 *            chaining outputs appear ONLY when chainable:true, and they are
 *            read back (i.e. survive) rather than being destroyed before
 *            return;
 *          - first-wavelet output, rather than the unfiltered temporal result,
 *            is copied into and returned as the next color history.
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
        // Preserve texture identity in bind-group assertions below.
        createView: vi.fn(() => t),
        destroy: vi.fn(() => { t.destroyed = true; }),
      };
      textures.push(t);
      return t;
    }),
    createBuffer: vi.fn((desc?: { label?: string }) => ({
      label: desc?.label,
      destroy: vi.fn(),
    })),
    createBindGroup: vi.fn((desc: unknown) => desc),
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
    expect(out.prevRadianceOut).toBeUndefined();
    expect(out.momentsOut).toBeUndefined();
    expect(out.historyLengthOut).toBeUndefined();
    // No chaining → the history/moments readbacks are never submitted.
    expect(uploadMocks.readRgba16fToRgb).toHaveBeenCalledTimes(1);
    expect(uploadMocks.readRgba32fToRg).not.toHaveBeenCalled();
    expect(uploadMocks.readR32UintToU32).not.toHaveBeenCalled();
  });

  it('binds current noisy color and geometry to the 7x7 fallback', async () => {
    const device = createStubDevice();
    await runSVGFRealWebGPU({
      device: device as unknown as GPUDevice,
      rgb, width: W, height: H, atrousIterations: 1,
    });

    const fallbackCall = device.createBindGroup.mock.calls.find((call) => {
      const entries = (call[0] as { entries?: Array<{ resource?: StubTexture }> }).entries;
      return entries?.[0]?.resource?.label === 'svgf-curr-color'
        && entries.length === 6;
    });
    const entries =
      (fallbackCall?.[0] as { entries: Array<{ resource: StubTexture }> } | undefined)?.entries;
    expect(entries?.map((entry) => entry.resource.label)).toEqual([
      'svgf-curr-color',
      'svgf-hist-out',
      'svgf-var-mom',
      'svgf-var-final',
      'svgf-norm',
      'svgf-depth',
    ]);
  });

  it('returns first-wavelet color + moments + history read back before teardown', async () => {
    const device = createStubDevice();
    const out = await runSVGFRealWebGPU({
      device: device as unknown as GPUDevice,
      rgb, width: W, height: H, atrousIterations: 1, chainable: true,
    });
    expect(out.rgb.length).toBe(W * H * 3);
    // Final output plus first-wavelet history are independently read back.
    expect(uploadMocks.readRgba16fToRgb).toHaveBeenCalledTimes(2);
    expect(out.prevRadianceOut).toBeInstanceOf(Float32Array);
    expect(out.prevRadianceOut).toHaveLength(W * H * 3);
    const finalReadTex = labelOfArg(uploadMocks.readRgba16fToRgb.mock.calls[0]?.[1]);
    const historyReadTex = labelOfArg(uploadMocks.readRgba16fToRgb.mock.calls[1]?.[1]);
    expect(finalReadTex).toBe('svgf-pong');
    expect(historyReadTex).toBe('svgf-color-out');

    // Iteration zero writes pong and immediately preserves that exact level in
    // color-out before any later wavelet level could reuse the ping-pong pair.
    const encoder = device.createCommandEncoder.mock.results[0]?.value as {
      copyTextureToTexture: ReturnType<typeof vi.fn>;
    };
    const historyCopy = encoder.copyTextureToTexture.mock.calls.find(
      (call) =>
        labelOfArg((call[0] as { texture?: StubTexture }).texture) === 'svgf-pong'
        && labelOfArg((call[1] as { texture?: StubTexture }).texture) === 'svgf-color-out',
    );
    expect(historyCopy).toBeDefined();

    // The reprojection outputs are also read back and returned (not destroyed first).
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

  it('demodulates previous radiance with previous rather than current albedo', async () => {
    const device = createStubDevice();
    const current = new Float32Array(W * H * 3).fill(0.2);
    const previous = new Float32Array(W * H * 3).fill(0.2);
    const currentAlbedo = new Float32Array(W * H * 3).fill(0.5);
    const previousAlbedo = new Float32Array(W * H * 3).fill(0.25);

    await runSVGFRealWebGPU({
      device: device as unknown as GPUDevice,
      rgb: current,
      prevRadianceRgb: previous,
      albedoRgb: currentAlbedo,
      prevAlbedoRgb: previousAlbedo,
      width: W,
      height: H,
      atrousIterations: 1,
    });

    const currentCall = uploadMocks.uploadRgbAsRgba16f.mock.calls.find(
      (call) => labelOfArg(call[1]) === 'svgf-curr-color',
    );
    const previousCall = uploadMocks.uploadRgbAsRgba16f.mock.calls.find(
      (call) => labelOfArg(call[1]) === 'svgf-prev-color',
    );
    expect(Array.from(currentCall?.[2] as Float32Array)).toEqual(
      Array.from(new Float32Array(W * H * 3).fill(0.4)),
    );
    expect(Array.from(previousCall?.[2] as Float32Array)).toEqual(
      Array.from(new Float32Array(W * H * 3).fill(0.8)),
    );
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

  it('destroys the staging buffer when command encoding fails before mapAsync', async () => {
    const real = await vi.importActual<typeof import('../src/webGpuTextureUpload.js')>(
      '../src/webGpuTextureUpload.js',
    );
    const destroy = vi.fn();
    const mapAsync = vi.fn();
    const device = {
      createBuffer: vi.fn(() => ({
        mapAsync,
        getMappedRange: vi.fn(),
        unmap: vi.fn(),
        destroy,
      })),
      createCommandEncoder: vi.fn(() => {
        throw new Error('mock encoder failure');
      }),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;

    vi.stubGlobal('GPUBufferUsage', { COPY_DST: 2, MAP_READ: 1 });

    await expect(
      real.readRgba16fToRgb(device, {} as GPUTexture, 2, 2),
    ).rejects.toThrow('mock encoder failure');

    expect(mapAsync).not.toHaveBeenCalled();
    expect(destroy).toHaveBeenCalledTimes(1);
  });
});
