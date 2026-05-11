import { describe, expect, it } from 'vitest';
import { disposeSharedWebGPUDevice, getSharedWebGPUDevice } from '../src/sharedWebGpuDevice.js';

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
