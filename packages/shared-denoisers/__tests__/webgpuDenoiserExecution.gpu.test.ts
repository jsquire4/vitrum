import { afterAll, describe, expect, it } from 'vitest';
import { disposeSharedWebGPUDevice } from '../src/sharedWebGpuDevice.js';
import { runBmfrWebGPU } from '../src/bmfrWebGPU.js';
import { runSVGFRealWebGPU } from '../src/svgfRealWebGPU.js';
import { runHdrLuminanceBilateralWebGPU } from '../src/hdrLuminanceBilateralWebGPU.js';

const hasWebGpu =
  typeof navigator !== 'undefined' &&
  navigator.gpu != null &&
  typeof navigator.gpu.requestAdapter === 'function';

function expectFinitePrefix(values: Float32Array, count: number): void {
  for (let i = 0; i < Math.min(values.length, count); i++) {
    expect(Number.isFinite(values[i] ?? Number.NaN)).toBe(true);
  }
}

describe.skipIf(!hasWebGpu)('exported WebGPU denoiser execution smokes', () => {
  afterAll(() => {
    disposeSharedWebGPUDevice();
  });

  it('executes runSVGFRealWebGPU and returns an RGB buffer', async () => {
    const width = 8;
    const height = 8;
    const rgb = Float32Array.from({ length: width * height * 3 }, (_, i) => {
      const channel = i % 3;
      return channel === 0 ? 0.5 : channel === 1 ? 0.25 : 0.125;
    });

    const out = await runSVGFRealWebGPU({
      rgb,
      width,
      height,
      atrousIterations: 1,
      reuseSharedWebGpuDevice: true,
    });

    expect(out.rgb.length).toBe(width * height * 3);
    expectFinitePrefix(out.rgb, 12);
  });

  it('executes runHdrLuminanceBilateralWebGPU and returns finite RGB', async () => {
    const width = 8;
    const height = 8;
    const rgb = Float32Array.from({ length: width * height * 3 }, (_, i) => {
      const channel = i % 3;
      return channel === 0 ? 0.4 : channel === 1 ? 0.3 : 0.2;
    });

    const out = await runHdrLuminanceBilateralWebGPU({
      rgb,
      width,
      height,
      sigmaLuminance: 0.06,
      reuseSharedWebGpuDevice: true,
    });

    expect(out.length).toBe(width * height * 3);
    expectFinitePrefix(out, 12);
  });

  it('executes runBmfrWebGPU and returns an RGB buffer', async () => {
    const width = 8;
    const height = 8;
    const rgb = new Float32Array(width * height * 3);
    const worldPosRgb = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pi = y * width + x;
        rgb[pi * 3] = 0.2 + x * 0.01;
        rgb[pi * 3 + 1] = 0.3 + y * 0.01;
        rgb[pi * 3 + 2] = 0.4;
        worldPosRgb[pi * 3] = x / width;
        worldPosRgb[pi * 3 + 1] = y / height;
        worldPosRgb[pi * 3 + 2] = 0;
      }
    }

    const out = await runBmfrWebGPU({
      rgb,
      worldPosRgb,
      width,
      height,
      blockSize: 8,
      blockStride: 8,
      reuseSharedWebGpuDevice: true,
    });

    expect(out.length).toBe(width * height * 3);
    expectFinitePrefix(out, 12);
  });
});

describe('exported WebGPU denoiser availability', () => {
  it.skipIf(hasWebGpu)('runHdrLuminanceBilateralWebGPU throws when WebGPU is missing', async () => {
    await expect(
      runHdrLuminanceBilateralWebGPU({
        rgb: new Float32Array(12),
        width: 2,
        height: 2,
      }),
    ).rejects.toThrow(/WebGPU not available/);
  });

  it.skipIf(hasWebGpu)('runSVGFRealWebGPU throws when WebGPU is missing', async () => {
    await expect(
      runSVGFRealWebGPU({
        rgb: new Float32Array(12),
        width: 2,
        height: 2,
      }),
    ).rejects.toThrow(/WebGPU not available/);
  });

  it.skipIf(hasWebGpu)('runBmfrWebGPU throws when WebGPU is missing', async () => {
    await expect(
      runBmfrWebGPU({
        rgb: new Float32Array(12),
        width: 2,
        height: 2,
      }),
    ).rejects.toThrow(/WebGPU not available/);
  });
});
