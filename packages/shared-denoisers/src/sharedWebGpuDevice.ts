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

export interface AcquireDenoiseDeviceOptions {
  /** Explicit device, when supplied. Never destroyed by `dispose`. */
  readonly device?: GPUDevice | undefined;
  /**
   * When true (and no explicit `device`), reuses the process-shared device via
   * getSharedWebGPUDevice (also never destroyed by `dispose`). Otherwise an
   * ephemeral high-performance device is requested per call and destroyed by
   * `dispose`. Default: false.
   */
  readonly reuseSharedWebGpuDevice?: boolean | undefined;
  /**
   * Prefix for thrown error messages (e.g. the caller's function name) so
   * adapter-failure errors stay attributable to the originating dispatcher.
   */
  readonly errorLabel: string;
}

export interface AcquiredDenoiseDevice {
  readonly device: GPUDevice;
  /** Destroys the device iff it was acquired ephemerally; no-op otherwise. */
  readonly dispose: () => void;
}

/**
 * Single source for the one-shot denoiser dispatchers' device-acquisition
 * preamble. Preserves the exact 4-branch selection:
 *   1. explicit `opts.device`            → returned as-is; `dispose` is a no-op.
 *   2. `reuseSharedWebGpuDevice === true` → process-shared device; no-op dispose.
 *   3. otherwise                          → ephemeral high-performance device;
 *                                           `dispose` destroys it.
 * Guards `navigator.gpu` availability up front (throws with the caller's label).
 */
export async function acquireDenoiseDevice(
  opts: AcquireDenoiseDeviceOptions,
): Promise<AcquiredDenoiseDevice> {
  const reuseShared = opts.reuseSharedWebGpuDevice === true && opts.device == null;

  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error(`${opts.errorLabel}: WebGPU not available`);
  }

  if (opts.device != null) {
    return { device: opts.device, dispose: () => {} };
  }
  if (reuseShared) {
    return { device: await getSharedWebGPUDevice(), dispose: () => {} };
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (adapter == null) {
    throw new Error(`${opts.errorLabel}: failed to request GPU adapter`);
  }
  const device = await adapter.requestDevice();
  return { device, dispose: () => { device.destroy(); } };
}
