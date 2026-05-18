import { afterAll, describe, expect, it } from 'vitest';
import { disposeSharedWebGPUDevice, getSharedWebGPUDevice } from '../src/sharedWebGpuDevice.js';
import { runAtrousVarianceWebGPU } from '../src/atrousVarianceWebGPU.js';

const hasWebGpu =
  typeof navigator !== 'undefined' &&
  navigator.gpu != null &&
  typeof navigator.gpu.requestAdapter === 'function';

describe.skipIf(!hasWebGpu)('runAtrousVarianceWebGPU (WebGPU)', () => {
  afterAll(() => {
    disposeSharedWebGPUDevice();
  });

  it('returns RGB buffer matching dimensions', async () => {
    const w = 16;
    const h = 16;
    const rgb = new Float32Array(w * h * 3);
    rgb.fill(0.25);
    rgb[0] = 4;
    rgb[1] = 2;
    rgb[2] = 1;

    const out = await runAtrousVarianceWebGPU({
      rgb,
      width: w,
      height: h,
      frameCount: 0,
      atrousIterations: 5,
    });
    expect(out.length).toBe(w * h * 3);
    expect(Number.isFinite(out[0])).toBe(true);
  });

  it('reuses pooled WebGPU device between getSharedWebGPUDevice and runAtrousVarianceWebGPU', async () => {
    const before = await getSharedWebGPUDevice();
    const w = 8;
    const h = 8;
    const rgb = new Float32Array(w * h * 3).fill(0.1);
    await runAtrousVarianceWebGPU({ rgb, width: w, height: h, frameCount: 0, atrousIterations: 2 });
    const after = await getSharedWebGPUDevice();
    expect(after).toBe(before);
  });
});

describe('runAtrousVarianceWebGPU availability', () => {
  it.skipIf(hasWebGpu)('throws when WebGPU is missing', async () => {
    await expect(
      runAtrousVarianceWebGPU({
        rgb: new Float32Array(12),
        width: 2,
        height: 2,
      }),
    ).rejects.toThrow(/WebGPU not available/);
  });
});
