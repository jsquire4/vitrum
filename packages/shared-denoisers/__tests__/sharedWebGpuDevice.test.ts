import { describe, expect, it, vi } from 'vitest';
import {
  acquireDenoiseDevice,
  disposeSharedWebGPUDevice,
  getSharedWebGPUDevice,
  makePerDevicePipelineCache,
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

describe('makePerDevicePipelineCache', () => {
  it('returns the same instance for repeated calls with the same device', () => {
    const factory = vi.fn((d: GPUDevice) => ({ pipeline: `compiled-for-${(d as unknown as { id: string }).id}` }));
    const getBundle = makePerDevicePipelineCache(factory);

    const deviceA = { id: 'A' } as unknown as GPUDevice;
    const r1 = getBundle(deviceA);
    const r2 = getBundle(deviceA);

    expect(r1).toBe(r2);
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it('calls the factory once per distinct device', () => {
    const factory = vi.fn((d: GPUDevice) => ({ label: (d as unknown as { id: string }).id }));
    const getBundle = makePerDevicePipelineCache(factory);

    const deviceA = { id: 'A' } as unknown as GPUDevice;
    const deviceB = { id: 'B' } as unknown as GPUDevice;

    const ra = getBundle(deviceA);
    const rb = getBundle(deviceB);

    expect(ra).not.toBe(rb);
    expect(ra.label).toBe('A');
    expect(rb.label).toBe('B');
    expect(factory).toHaveBeenCalledTimes(2);
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
