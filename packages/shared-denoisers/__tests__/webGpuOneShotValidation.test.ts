import { afterEach, describe, expect, it, vi } from 'vitest';
import { makeResourceTracker } from '../src/atrousChain.js';
import { runAtrousVarianceWebGPU } from '../src/atrousVarianceWebGPU.js';
import { runBmfrWebGPU } from '../src/bmfrWebGPU.js';
import { runHdrLuminanceBilateralWebGPU } from '../src/hdrLuminanceBilateralWebGPU.js';
import {
  assertSVGFRealWebGPUInputs,
  runSVGFRealWebGPU,
} from '../src/svgfRealWebGPU.js';
import {
  assertFiniteFloat16Slice,
  assertOneShotDeviceLimits,
  assertOneShotDimensions,
} from '../src/webGpuOneShotValidation.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('one-shot input preflight', () => {
  it.each([
    [Number.NaN, 1],
    [Number.POSITIVE_INFINITY, 1],
    [1.5, 1],
    [1, 0],
    [1, -1],
  ])('rejects non-positive/non-integral dimensions (%s x %s)', (width, height) => {
    expect(() => assertOneShotDimensions('test', width, height)).toThrow(
      /positive safe integer/,
    );
  });

  it('rejects non-finite HDR input and tunables before touching a device', async () => {
    const device = { createTexture: vi.fn() } as unknown as GPUDevice;
    await expect(runHdrLuminanceBilateralWebGPU({
      device,
      rgb: new Float32Array([Number.NaN, 0, 0]),
      width: 1,
      height: 1,
    })).rejects.toThrow(/rgb\[0\] must be finite/);
      await expect(runHdrLuminanceBilateralWebGPU({
        device,
        rgb: new Float32Array(3),
        width: 1,
        height: 1,
        sigmaLuminance: Number.POSITIVE_INFINITY,
    })).rejects.toThrow(/sigmaLuminance must be finite/);
    expect(device.createTexture).not.toHaveBeenCalled();
  });

  it('requires BMFR world positions before device acquisition', async () => {
    const device = { createTexture: vi.fn() } as unknown as GPUDevice;
    await expect(runBmfrWebGPU({
      device,
      rgb: new Float32Array(3),
      width: 1,
      height: 1,
    } as Parameters<typeof runBmfrWebGPU>[0])).rejects.toThrow(
      /worldPosRgb is required/,
    );
    expect(device.createTexture).not.toHaveBeenCalled();
  });

  it('rejects incomplete/non-finite atrous and BMFR slices before allocation', async () => {
    const device = { createTexture: vi.fn() } as unknown as GPUDevice;
    await expect(runAtrousVarianceWebGPU({
      device,
      rgb: new Float32Array(12),
      width: 2,
      height: 2,
      welfordMeanM2: new Float32Array(7),
    })).rejects.toThrow(/welfordMeanM2 length/);
      await expect(runBmfrWebGPU({
        device,
        rgb: new Float32Array(3),
        worldPosRgb: new Float32Array(3),
        width: 1,
      height: 1,
      positionScale: Number.NaN,
    })).rejects.toThrow(/positionScale must be finite/);
    expect(device.createTexture).not.toHaveBeenCalled();
  });

  it('validates every SVGF slice and the r16uint history range', async () => {
    expect(() => assertSVGFRealWebGPUInputs({
      rgb: new Float32Array(3),
      width: 1,
      height: 1,
      prevNormalsRgb: new Float32Array(2),
    })).toThrow(/prevNormalsRgb length/);

    expect(() => assertSVGFRealWebGPUInputs({
      rgb: new Float32Array(3),
      width: 1,
      height: 1,
      prevHistoryLength: new Uint32Array([0x10000]),
    })).toThrow(/prevHistoryLength\[0\].*65535/);

    const device = { createTexture: vi.fn() } as unknown as GPUDevice;
    await expect(runSVGFRealWebGPU({
      device,
      rgb: new Float32Array(3),
      width: 1,
      height: 1,
      reprojUniforms: { alphaMin: 1.1 },
    })).rejects.toThrow(/alphaMin must be <= 1/);
    expect(device.createTexture).not.toHaveBeenCalled();
  });

  it('requires matching previous albedo for demodulated previous radiance', () => {
    expect(() => assertSVGFRealWebGPUInputs({
      rgb: new Float32Array(3),
      width: 1,
      height: 1,
      albedoRgb: new Float32Array([0.5, 0.5, 0.5]),
      prevRadianceRgb: new Float32Array([1, 1, 1]),
    })).toThrow(/prevAlbedoRgb.*required/);

    expect(() => assertSVGFRealWebGPUInputs({
      rgb: new Float32Array(3),
      width: 1,
      height: 1,
      albedoRgb: new Float32Array([0.5, 0.5, 0.5]),
      prevRadianceRgb: new Float32Array([1, 1, 1]),
      prevAlbedoRgb: new Float32Array([0.25, 0.25, 0.25]),
    })).not.toThrow();
  });

  it('requires matching BMFR history albedo before device acquisition', async () => {
    const device = { createTexture: vi.fn() } as unknown as GPUDevice;
    const base = {
      device,
      rgb: new Float32Array([1, 1, 1]),
      worldPosRgb: new Float32Array([0, 0, 1]),
      width: 1,
      height: 1,
    };

    await expect(runBmfrWebGPU({
      ...base,
      albedoRgb: new Float32Array([0.5, 0.5, 0.5]),
      historyRgb: new Float32Array([0.25, 0.25, 0.25]),
    })).rejects.toThrow(/historyAlbedoRgb.*required/);

    await expect(runBmfrWebGPU({
      ...base,
      historyAlbedoRgb: new Float32Array([0.25, 0.25, 0.25]),
    })).rejects.toThrow(/historyAlbedoRgb requires albedoRgb/);

    await expect(runBmfrWebGPU({
      ...base,
      albedoRgb: new Float32Array([0.5, 0.5, 0.5]),
      historyAlbedoRgb: new Float32Array([0.25, 0.25, 0.25]),
    })).rejects.toThrow(/historyAlbedoRgb requires historyRgb/);

    expect(device.createTexture).not.toHaveBeenCalled();
  });

  it('rejects finite float32 values that overflow the float16 upload domain', async () => {
    expect(() => assertFiniteFloat16Slice(
      'test',
      'rgb',
      new Float32Array([65504, -65504]),
      2,
    )).not.toThrow();
    expect(() => assertFiniteFloat16Slice(
      'test',
      'rgb',
      new Float32Array([65520]),
      1,
    )).toThrow(/rgb\[0\].*finite float16/);

    const device = { createTexture: vi.fn() } as unknown as GPUDevice;
    await expect(runSVGFRealWebGPU({
      device,
      rgb: new Float32Array([100, 100, 100]),
      albedoRgb: new Float32Array([0, 0, 0]),
      width: 1,
      height: 1,
    })).rejects.toThrow(/rgbForChain\[0\].*finite float16/);
    expect(device.createTexture).not.toHaveBeenCalled();
  });
});

describe('one-shot device-limit and teardown preflight', () => {
  it('checks texture and padded readback-buffer limits on the selected device', () => {
    const textureLimited = {
      limits: { maxTextureDimension2D: 4, maxBufferSize: 1_000_000 },
    } as unknown as GPUDevice;
    expect(() => assertOneShotDeviceLimits(textureLimited, 'test', 5, 1, 8))
      .toThrow(/maxTextureDimension2D/);

    const bufferLimited = {
      limits: { maxTextureDimension2D: 1024, maxBufferSize: 255 },
    } as unknown as GPUDevice;
    expect(() => assertOneShotDeviceLimits(bufferLimited, 'test', 1, 1, 8))
      .toThrow(/maxBufferSize/);
  });

  it('destroys an ephemeral device when actual-device preflight fails', async () => {
    const destroy = vi.fn();
    const device = {
      limits: { maxTextureDimension2D: 1, maxBufferSize: 1_000_000 },
      destroy,
    } as unknown as GPUDevice;
    const requestDevice = vi.fn(async () => device);
    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: vi.fn(async () => ({ requestDevice })),
      },
    });

    await expect(runHdrLuminanceBilateralWebGPU({
      rgb: new Float32Array(2 * 1 * 3),
      width: 2,
      height: 1,
    })).rejects.toThrow(/maxTextureDimension2D/);
    expect(requestDevice).toHaveBeenCalledTimes(1);
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  it('continues cleanup when one resource destroy throws', () => {
    const firstBuffer = { destroy: vi.fn(() => { throw new Error('destroy failed'); }) };
    const secondBuffer = { destroy: vi.fn() };
    const texture = { destroy: vi.fn() };
    const destroyDevice = vi.fn();
    const tracker = makeResourceTracker(destroyDevice);
    tracker.trackBuffer(firstBuffer as unknown as GPUBuffer);
    tracker.trackBuffer(secondBuffer as unknown as GPUBuffer);
    tracker.trackTexture(texture as unknown as GPUTexture);

    expect(() => tracker.dispose()).not.toThrow();
    expect(firstBuffer.destroy).toHaveBeenCalledTimes(1);
    expect(secondBuffer.destroy).toHaveBeenCalledTimes(1);
    expect(texture.destroy).toHaveBeenCalledTimes(1);
    expect(destroyDevice).toHaveBeenCalledTimes(1);
  });

  it('releases earlier allocations when a later texture allocation throws', async () => {
    vi.stubGlobal('GPUTextureUsage', {
      TEXTURE_BINDING: 1,
      COPY_DST: 2,
      COPY_SRC: 4,
      STORAGE_BINDING: 8,
    });
    const firstDestroy = vi.fn();
    let textureCall = 0;
    const device = {
      limits: { maxTextureDimension2D: 8192, maxBufferSize: 1_000_000 },
      createShaderModule: vi.fn(() => ({})),
      createComputePipeline: vi.fn(() => ({
        getBindGroupLayout: vi.fn(() => ({})),
      })),
      createTexture: vi.fn(() => {
        textureCall += 1;
        if (textureCall === 2) throw new Error('mock second allocation failure');
        return {
          createView: vi.fn(() => ({})),
          destroy: firstDestroy,
        };
      }),
      queue: { writeTexture: vi.fn() },
      destroy: vi.fn(),
    } as unknown as GPUDevice;

    await expect(runHdrLuminanceBilateralWebGPU({
      device,
      rgb: new Float32Array(3),
      width: 1,
      height: 1,
    })).rejects.toThrow('mock second allocation failure');

    expect(firstDestroy).toHaveBeenCalledTimes(1);
    expect(device.destroy).not.toHaveBeenCalled();
  });
});
