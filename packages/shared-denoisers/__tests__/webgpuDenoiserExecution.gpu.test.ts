import { afterAll, describe, expect, it } from 'vitest';
import { disposeSharedWebGPUDevice } from '../src/sharedWebGpuDevice.js';
import { runBmfrWebGPU } from '../src/bmfrWebGPU.js';
import { runSVGFRealWebGPU } from '../src/svgfRealWebGPU.js';
import { runHdrLuminanceBilateralWebGPU } from '../src/hdrLuminanceBilateralWebGPU.js';
import { bmfrCpuOverlapOracle } from './bmfrCpuOverlapOracle.js';

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

  it('matches an independent CPU overlap oracle on non-divisible dimensions', async () => {
    const width = 11;
    const height = 7;
    const rgb = new Float32Array(width * height * 3);
    const worldPosRgb = new Float32Array(width * height * 3);
    const normalsRgb = new Float32Array(width * height * 3);
    const validityW = new Float32Array(width * height).fill(1);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pi = y * width + x;
        const px = x / 5;
        const py = y / 4;
        const pz = (x * y) / 50;
        worldPosRgb[pi * 3] = px;
        worldPosRgb[pi * 3 + 1] = py;
        worldPosRgb[pi * 3 + 2] = pz;
        normalsRgb[pi * 3] = 0.5 + x * 0.005;
        normalsRgb[pi * 3 + 1] = 0.5 + y * 0.007;
        normalsRgb[pi * 3 + 2] = 0.98;
        rgb[pi * 3] = 0.15 + 0.04 * px + 0.02 * py * py;
        rgb[pi * 3 + 1] = 0.25 + 0.03 * py + 0.01 * px * px;
        rgb[pi * 3 + 2] = 0.35 + 0.02 * pz;
      }
    }
    // Exercise the explicit pass-through path inside several overlapping fits.
    validityW[width + 1] = 0;

    const out = await runBmfrWebGPU({
      rgb,
      worldPosRgb,
      validityW,
      gbufferNormalsRgb: normalsRgb,
      width,
      height,
      blockSize: 8,
      blockStride: 4,
      positionScale: 4,
      regularisation: 1e-3,
      reuseSharedWebGpuDevice: true,
    });
    const expected = bmfrCpuOverlapOracle({
      rgb,
      worldPosRgb,
      validityW,
      normalsRgb,
      width,
      height,
      blockSize: 8,
      blockStride: 4,
      positionScale: 4,
      regularisation: 1e-3,
    });

    expect(out.length).toBe(width * height * 3);
    for (let i = 0; i < out.length; i += 1) {
      expect(out[i]).toBeCloseTo(expected[i]!, 2);
    }
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
        worldPosRgb: new Float32Array(12),
        width: 2,
        height: 2,
      }),
    ).rejects.toThrow(/WebGPU not available/);
  });
});
