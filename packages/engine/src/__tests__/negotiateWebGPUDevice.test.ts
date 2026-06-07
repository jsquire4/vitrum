import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
} from '@vitrum/walkaround-hybrid';
import {
  PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
} from '@vitrum/pt-webgpu';
import { negotiateWebGPUDevice } from '../negotiateWebGPUDevice.js';

/** A fake GPUAdapter that records the descriptor passed to requestDevice so we
 *  can assert which `requiredLimits` were requested without a real GPU. No
 *  `queue` ⇒ probeAdapterProfile treats it as a GPUAdapter (reads `.info`). */
function fakeAdapter(
  buf: number,
  tex: number,
  info?: { vendor?: string; architecture?: string },
): { adapter: GPUAdapter; lastDescriptor: () => GPUDeviceDescriptor | undefined; device: GPUDevice } {
  let captured: GPUDeviceDescriptor | undefined;
  const device = { destroy: vi.fn() } as unknown as GPUDevice;
  const adapter = {
    limits: {
      maxStorageBuffersPerShaderStage: buf,
      maxStorageTexturesPerShaderStage: tex,
    },
    ...(info ? { info } : {}),
    requestDevice: vi.fn((descriptor?: GPUDeviceDescriptor) => {
      captured = descriptor;
      return Promise.resolve(device);
    }),
  } as unknown as GPUAdapter;
  return { adapter, lastDescriptor: () => captured, device };
}

/** Install a minimal navigator.gpu so the helper's availability guard +
 *  getPreferredCanvasFormat path resolve. `navigator` is a getter-only property
 *  in the test env, so define it (not assign). Returns a restore fn. */
const ORIG_NAVIGATOR = globalThis.navigator;
function setNavigator(value: unknown): void {
  Object.defineProperty(globalThis, 'navigator', { value, configurable: true });
}
function installNavigatorGpu(
  requestAdapter?: () => Promise<GPUAdapter | null>,
): () => void {
  setNavigator({
    gpu: {
      getPreferredCanvasFormat: () => 'rgba8unorm' as GPUTextureFormat,
      ...(requestAdapter ? { requestAdapter } : {}),
    },
  });
  return () => {
    setNavigator(ORIG_NAVIGATOR);
  };
}

describe('negotiateWebGPUDevice — host-owned device negotiation', () => {
  let restore: (() => void) | null = null;
  afterEach(() => {
    restore?.();
    restore = null;
    vi.restoreAllMocks();
  });

  it('throws a clear error when WebGPU is unavailable', async () => {
    restore = installNavigatorGpu();
    // Strip gpu entirely for this case (navigator present, no .gpu).
    setNavigator({});
    await expect(negotiateWebGPUDevice()).rejects.toThrow(/WebGPU is unavailable/);
  });

  it('throws when no adapter is available (requestAdapter → null)', async () => {
    restore = installNavigatorGpu(() => Promise.resolve(null));
    await expect(negotiateWebGPUDevice()).rejects.toThrow(/returned null/);
  });

  it('reuses a host-supplied adapter and never requests a new one', async () => {
    const { adapter } = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    const requestAdapter = vi.fn(() => Promise.resolve(null as GPUAdapter | null));
    restore = installNavigatorGpu(requestAdapter);

    const result = await negotiateWebGPUDevice({ adapter, target: 'pt-webgpu' });
    expect(result.adapter).toBe(adapter);
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it('returns the host-owned device, preferred format, and profile', async () => {
    const { adapter, device } = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    const result = await negotiateWebGPUDevice({ adapter, target: 'pt-webgpu' });
    expect(result.device).toBe(device);
    expect(result.format).toBe('rgba8unorm');
    // The profile is the same AdapterProfile shape probeAdapterProfile yields.
    expect(result.profile.hasWebGPU).toBe(true);
    expect(result.profile.ptWebgpuTier).toBe('full');
    expect(result.profile.hybridCapable).toBe(true);
  });

  it("target 'pt-webgpu' requests the adapter-aware full tier limits", async () => {
    const fa = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'pt-webgpu' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_FULL_MAX_STORAGE_BUFFERS_PER_GROUP,
    );
    expect(limits?.maxStorageTexturesPerShaderStage).toBe(
      PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    );
  });

  it("target 'pt-webgpu' on a lite adapter requests the lite tier limits", async () => {
    const fa = fakeAdapter(
      PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
      { vendor: 'apple', architecture: 'm1' },
    );
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'pt-webgpu' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it("target 'walkaround-hybrid' requests the FULL hybrid floor on a capable adapter", async () => {
    const fa = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'walkaround-hybrid' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(
      HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage'],
    );
    expect(limits?.maxStorageTexturesPerShaderStage).toBe(
      HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage'],
    );
  });

  it("target 'walkaround-hybrid' requests the LITE floor on a lite-only adapter", async () => {
    const fa = fakeAdapter(
      HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']!,
      HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!,
      { vendor: 'intel', architecture: 'gen12' },
    );
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'walkaround-hybrid' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(
      HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage'],
    );
  });

  it("target 'walkaround-hybrid' throws a gap-naming error below the lite floor", async () => {
    const fa = fakeAdapter(4, 2); // below any hybrid floor
    restore = installNavigatorGpu();
    await expect(
      negotiateWebGPUDevice({ adapter: fa.adapter, target: 'walkaround-hybrid' }),
    ).rejects.toThrow(/cannot run the realtime engine/);
    expect(fa.lastDescriptor()).toBeUndefined(); // never requested a device
  });

  it("target 'progressive' requests the limit union on a capable adapter", async () => {
    const fa = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'progressive' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    // Union of hybrid-full (16/8) and pt-webgpu-full (10/5) → 16/8 dominates.
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(16);
    expect(limits?.maxStorageTexturesPerShaderStage).toBe(8);
  });

  it("target 'progressive' throws a gap-naming error when the adapter can't meet the union", async () => {
    const fa = fakeAdapter(10, 5, { vendor: 'intel', architecture: 'gen12' });
    restore = installNavigatorGpu();
    await expect(
      negotiateWebGPUDevice({ adapter: fa.adapter, target: 'progressive' }),
    ).rejects.toThrow(/device-limit UNION/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it("target 'none' requests a device with no requiredLimits", async () => {
    const fa = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'none' });
    expect(fa.lastDescriptor()?.requiredLimits).toBeUndefined();
  });

  it('explicit requiredLimits override the target and are clamped to the adapter', async () => {
    const fa = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    // Over-ask 999 buffers — must clamp down to the adapter's 16.
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'pt-webgpu',
      requiredLimits: { maxStorageBuffersPerShaderStage: 999 },
    });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(16);
  });

  it('forwards requiredFeatures and label verbatim to requestDevice', async () => {
    const fa = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'pt-webgpu',
      requiredFeatures: ['timestamp-query' as GPUFeatureName],
      label: 'vitrum-test',
    });
    const desc = fa.lastDescriptor();
    expect(desc?.requiredFeatures).toEqual(['timestamp-query']);
    expect(desc?.label).toBe('vitrum-test');
  });

  it('does not destroy or track the returned device (host owns it)', async () => {
    const fa = fakeAdapter(16, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    const result = await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'pt-webgpu' });
    // The helper holds no reference and registers no teardown — the device is
    // untouched (not destroyed) when it returns.
    expect((result.device.destroy as ReturnType<typeof vi.fn>)).not.toHaveBeenCalled();
  });
});
