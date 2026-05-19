/**
 * Lazy singleton WebGPU device for short compute passes (atrous-variance, HDR bilateral).
 * Avoids adapter/device teardown latency on repeated denoise shots.
 *
 * Concurrency: `disposeSharedWebGPUDevice()` bumps a generation counter. Any
 * `requestAdapter`/`requestDevice` completion that observes a stale generation
 * destroys the fresh device and rejects with a sentinel error; callers retry.
 */

let cachedDevice: GPUDevice | null = null;
let pendingDevice: Promise<GPUDevice> | null = null;
let deviceGeneration = 0;

const SUPERSEDED_MSG = 'getSharedWebGPUDevice: superseded by dispose';

/**
 * Returns a shared high-performance adapter device, creating it on first use.
 * Clears automatically if the device is lost (then the next call recreates).
 */
export async function getSharedWebGPUDevice(): Promise<GPUDevice> {
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

  throw new Error('getSharedWebGPUDevice: exhausted retries after concurrent dispose');
}

async function acquireSharedDevice(capturedGen: number): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (adapter == null) {
    throw new Error('getSharedWebGPUDevice: failed to request GPU adapter');
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
 * Destroys the cached device (no-op if none). Next compute pass will recreate.
 * Does not await in-flight device acquisition; concurrent getters retry instead.
 */
export function disposeSharedWebGPUDevice(): void {
  deviceGeneration += 1;
  if (cachedDevice != null) {
    cachedDevice.destroy();
    cachedDevice = null;
  }
}
