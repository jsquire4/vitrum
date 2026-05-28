import { describe, expect, it, vi } from 'vitest';
import {
  acquireDenoiseDevice,
  disposeSharedWebGPUDevice,
  getSharedWebGPUDevice,
} from '../src/sharedWebGpuDevice.js';

const hasWebGpu =
  typeof navigator !== 'undefined' &&
  navigator.gpu != null &&
  typeof navigator.gpu.requestAdapter === 'function';

describe('sharedWebGpuDevice', () => {
  it('disposeSharedWebGPUDevice is safe when idle', () => {
    disposeSharedWebGPUDevice();
    disposeSharedWebGPUDevice();
    expect(true).toBe(true);
  });

  it('getSharedWebGPUDevice throws when WebGPU is unavailable', async () => {
    if (hasWebGpu) return;
    await expect(getSharedWebGPUDevice()).rejects.toThrow(/WebGPU not available/);
  });
});

describe('acquireDenoiseDevice', () => {
  it('returns the provided device and a no-op dispose when device is passed', async () => {
    // The navigator.gpu guard runs first (matching every dispatcher's original
    // preamble), so a minimal stub is installed regardless of host WebGPU. The
    // explicit-device branch then returns the device verbatim and must NEVER
    // destroy it — the host owns the device lifecycle.
    vi.stubGlobal('navigator', { gpu: {} });
    try {
      const destroy = vi.fn();
      const fakeDevice = { destroy } as unknown as GPUDevice;

      const { device, dispose } = await acquireDenoiseDevice({
        device: fakeDevice,
        errorLabel: 'testAcquire',
      });

      expect(device).toBe(fakeDevice);
      dispose();
      expect(destroy).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('throws with the caller label when WebGPU is unavailable (no explicit device)', async () => {
    if (hasWebGpu) return;
    await expect(
      acquireDenoiseDevice({ errorLabel: 'testAcquire' }),
    ).rejects.toThrow('testAcquire: WebGPU not available');
  });
});
