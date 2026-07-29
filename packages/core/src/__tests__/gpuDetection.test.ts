import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  detectGpu,
  resetGpuDetectionCache,
} from '../gpuDetection.js';

afterEach(() => {
  resetGpuDetectionCache();
  vi.unstubAllGlobals();
});

describe('detectGpu memoized public result', () => {
  it('publishes one immutable summary so callers cannot corrupt later reads', async () => {
    const requestAdapter = vi.fn(async () => ({
      info: {
        vendor: 'Example Vendor',
        architecture: 'Example Architecture',
      },
      limits: {},
      features: new Set<string>(),
    }));
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter },
    });

    const first = await detectGpu();
    expect(Object.isFrozen(first)).toBe(true);
    expect(first).toMatchObject({
      isWebGPU: true,
      adapterKind: 'hardware',
      adapterVendor: 'example vendor',
      adapterArchitecture: 'example architecture',
    });
    expect(() => {
      (first as { isWebGPU: boolean }).isWebGPU = false;
    }).toThrow(TypeError);

    const second = await detectGpu();
    expect(second).toBe(first);
    expect(second.isWebGPU).toBe(true);
    expect(requestAdapter).toHaveBeenCalledTimes(1);
  });
});
