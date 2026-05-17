/**
 * W6-E1 acceptance test — verifies the singleton was renamed from the
 * production-sounding `getSharedWebGPUDevice` to the explicit test-only
 * `getSharedTestWebGPUDevice`, and that the singleton still works for tests
 * that opt in via the new name.
 *
 * Two complementary checks:
 *   1. Module-level export shape — the old name is gone, the new name is
 *      present, and the dispose function follows the same rename. Runs
 *      regardless of WebGPU availability.
 *   2. Live singleton behaviour — repeat calls return the same device when
 *      WebGPU is available. Skipped when navigator.gpu is missing.
 */

import { afterAll, describe, expect, it } from 'vitest';
import * as sharedDevice from '../src/sharedWebGpuDevice.js';
import {
  disposeSharedTestWebGPUDevice,
  getSharedTestWebGPUDevice,
} from '../src/sharedWebGpuDevice.js';

const hasWebGpu =
  typeof navigator !== 'undefined' &&
  navigator.gpu != null &&
  typeof navigator.gpu.requestAdapter === 'function';

describe('W6-E1 singleton rename', () => {
  it('exports getSharedTestWebGPUDevice and disposeSharedTestWebGPUDevice', () => {
    expect(typeof sharedDevice.getSharedTestWebGPUDevice).toBe('function');
    expect(typeof sharedDevice.disposeSharedTestWebGPUDevice).toBe('function');
  });

  it('no longer exports the old production-sounding names', () => {
    expect((sharedDevice as Record<string, unknown>).getSharedWebGPUDevice).toBeUndefined();
    expect((sharedDevice as Record<string, unknown>).disposeSharedWebGPUDevice).toBeUndefined();
  });

  it('throws a clear "WebGPU not available" message when navigator.gpu is missing', async () => {
    if (hasWebGpu) return;
    await expect(getSharedTestWebGPUDevice()).rejects.toThrow(/WebGPU not available/);
  });
});

describe.skipIf(!hasWebGpu)('W6-E1 singleton still works for tests', () => {
  afterAll(() => {
    disposeSharedTestWebGPUDevice();
  });

  it('returns the same device across repeat calls (lazy singleton)', async () => {
    const a = await getSharedTestWebGPUDevice();
    const b = await getSharedTestWebGPUDevice();
    expect(b).toBe(a);
  });

  it('disposeSharedTestWebGPUDevice triggers re-acquisition on next call', async () => {
    const a = await getSharedTestWebGPUDevice();
    disposeSharedTestWebGPUDevice();
    const b = await getSharedTestWebGPUDevice();
    // Either a fresh device (most likely) or the same handle if the previous
    // one was never actually destroyed — both prove dispose is a no-op-safe
    // operation that does not break subsequent gets.
    expect(b).toBeDefined();
    // The cached pointer must have changed, since dispose destroyed `a` and
    // the next get acquired a new device.
    expect(b).not.toBe(a);
  });
});
