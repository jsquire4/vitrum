/**
 * TEST / DEMO CONVENIENCE ONLY — NOT FOR PRODUCTION USE.
 *
 * A lazy process-wide WebGPU device singleton used by the Cornell-box example
 * and the shared-denoisers' GPU test suite so they don't have to plumb a
 * `GPUDevice` through every entry point.
 *
 * Production callers MUST pass an explicit `device: GPUDevice` to the driver
 * entry points (`runAtrousVarianceWebGPU`, `runHdrLuminanceBilateralWebGPU`,
 * `runSVGFRealWebGPU`). Per `CLAUDE.md` key design principle #2 ("the host
 * owns lifecycle"), engines and host applications own the GPU device — this
 * module exists only so tests and the Cornell demo can opt in to a shared
 * device for adapter/device-acquisition latency.
 *
 * To opt in from a driver call, set `reuseSharedWebGpuDevice: true` explicitly.
 * The drivers no longer default this to true (W6-E1, 2026-05-17).
 *
 * Concurrency: `disposeSharedTestWebGPUDevice()` bumps a generation counter.
 * Any `requestAdapter`/`requestDevice` completion that observes a stale
 * generation destroys the fresh device and rejects with a sentinel error;
 * callers retry.
 */

let cachedDevice: GPUDevice | null = null;
let pendingDevice: Promise<GPUDevice> | null = null;
let deviceGeneration = 0;

const SUPERSEDED_MSG = 'getSharedTestWebGPUDevice: superseded by dispose';

/**
 * TEST / DEMO CONVENIENCE — returns a shared high-performance adapter device,
 * creating it on first use. Clears automatically if the device is lost (then
 * the next call recreates).
 *
 * Production code must pass an explicit `device: GPUDevice` to the driver
 * entry points instead of relying on this singleton.
 */
export async function getSharedTestWebGPUDevice(): Promise<GPUDevice> {
  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error('WebGPU not available');
  }

  for (let attempt = 0; attempt < 16; attempt += 1) {
    if (cachedDevice != null) {
      return cachedDevice;
    }

    if (pendingDevice == null) {
      const capturedGen = deviceGeneration;
      pendingDevice = acquireSharedDevice(capturedGen).finally(() => {
        pendingDevice = null;
      });
    }

    try {
      const dev = await pendingDevice;
      if (cachedDevice === dev) {
        return dev;
      }
    } catch (e) {
      if (e instanceof Error && e.message === SUPERSEDED_MSG) {
        continue;
      }
      throw e;
    }
  }

  throw new Error('getSharedTestWebGPUDevice: exhausted retries after concurrent dispose');
}

async function acquireSharedDevice(capturedGen: number): Promise<GPUDevice> {
  const adapter = await navigator.gpu!.requestAdapter({ powerPreference: 'high-performance' });
  if (adapter == null) {
    throw new Error('getSharedTestWebGPUDevice: failed to request GPU adapter');
  }
  const device = await adapter.requestDevice();

  if (capturedGen !== deviceGeneration) {
    device.destroy();
    throw new Error(SUPERSEDED_MSG);
  }

  device.lost.then(() => {
    if (cachedDevice === device) {
      cachedDevice = null;
    }
  });
  cachedDevice = device;
  return device;
}

/**
 * Destroys the cached test/demo device (no-op if none). The next call to
 * `getSharedTestWebGPUDevice()` will recreate it.
 *
 * Does not await in-flight device acquisition; concurrent getters retry.
 */
export function disposeSharedTestWebGPUDevice(): void {
  deviceGeneration += 1;
  if (cachedDevice != null) {
    cachedDevice.destroy();
    cachedDevice = null;
  }
}
