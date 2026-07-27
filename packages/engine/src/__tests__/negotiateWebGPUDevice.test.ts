import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
  NRC_WEBGPU_REQUIRED_LIMITS,
} from '@vitrum/walkaround-hybrid';
import {
  PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
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
  options: {
    limits?: Readonly<Record<string, number>>;
    features?: readonly GPUFeatureName[];
  } = {},
): { adapter: GPUAdapter; lastDescriptor: () => GPUDeviceDescriptor | undefined; device: GPUDevice } {
  let captured: GPUDeviceDescriptor | undefined;
  const device = { destroy: vi.fn() } as unknown as GPUDevice;
  const adapter = {
    limits: {
      maxStorageBuffersPerShaderStage: buf,
      ...options.limits,
      maxSampledTexturesPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: tex,
    },
    ...(info ? { info } : {}),
    features: new Set(options.features ?? ['timestamp-query']),
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
    const { adapter } = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    const requestAdapter = vi.fn(() => Promise.resolve(null as GPUAdapter | null));
    restore = installNavigatorGpu(requestAdapter);

    const result = await negotiateWebGPUDevice({ adapter, target: 'pt-webgpu' });
    expect(result.adapter).toBe(adapter);
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it('returns the host-owned device, preferred format, and profile', async () => {
    const { adapter, device } = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
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
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'pt-webgpu' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
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
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'progressive' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE);
    expect(limits?.maxStorageTexturesPerShaderStage).toBe(
      HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage'],
    );
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

  it('rejects an explicit required-limit over-ask instead of silently weakening it', async () => {
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await expect(
      negotiateWebGPUDevice({
        adapter: fa.adapter,
        target: 'pt-webgpu',
        requiredLimits: { maxStorageBuffersPerShaderStage: 999 },
      }),
    ).rejects.toThrow(/requires maxStorageBuffersPerShaderStage >= 999/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it('preserves a satisfiable explicit required-limit override exactly', async () => {
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'none',
      requiredLimits: { maxStorageBuffersPerShaderStage: 12 },
    });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(12);
  });

  it('forwards requiredFeatures and label verbatim to requestDevice', async () => {
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'pt-webgpu',
      requiredFeatures: ['timestamp-query'],
      label: 'vitrum-test',
    });
    const desc = fa.lastDescriptor();
    expect(desc?.requiredFeatures).toEqual(['timestamp-query']);
    expect(desc?.label).toBe('vitrum-test');
  });

  it('does not destroy or track the returned device (host owns it)', async () => {
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    const result = await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'pt-webgpu' });
    // The helper holds no reference and registers no teardown — the device is
    // untouched (not destroyed) when it returns.
    expect((result.device.destroy as ReturnType<typeof vi.fn>)).not.toHaveBeenCalled();
  });

  // ── Bug7 fix: restirPtReuse forwarded for target:'progressive' ───────────

  it("Bug7 fix — target 'progressive' with restirPtReuse:true requests the higher buffer floor", async () => {
    // The ReSTIR-PT reuse floor is higher than the regular full-tier floor.
    // This adapter must be capable of the ReSTIR-PT reuse tier.
    const fa = fakeAdapter(PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'progressive', restirPtReuse: true });
    const limits = fa.lastDescriptor()?.requiredLimits;
    // With restirPtReuse:true the limit union must use the higher buffer floor.
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it("Bug7 fix — target 'progressive' without restirPtReuse uses the regular full-tier floor", async () => {
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8, { vendor: 'nvidia', architecture: 'ampere' });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({ adapter: fa.adapter, target: 'progressive' });
    const limits = fa.lastDescriptor()?.requiredLimits;
    expect(limits?.maxStorageBuffersPerShaderStage).toBe(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE);
  });

  it('rejects an invalid runtime target before requesting an adapter', async () => {
    const requestAdapter = vi.fn(() => Promise.resolve(null as GPUAdapter | null));
    restore = installNavigatorGpu(requestAdapter);
    await expect(
      negotiateWebGPUDevice({ target: 'bogus' as never }),
    ).rejects.toThrow(/target must be one of/);
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it("target 'pt-webgpu' rejects adapters below the lite floor before allocation", async () => {
    const fa = fakeAdapter(
      PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE - 1,
      PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    );
    restore = installNavigatorGpu();
    await expect(
      negotiateWebGPUDevice({ adapter: fa.adapter, target: 'pt-webgpu' }),
    ).rejects.toThrow(/below the lite-tier floor/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it.each([NaN, Infinity, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid explicit required-limit value %s',
    async (value) => {
      const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8);
      restore = installNavigatorGpu();
      await expect(
        negotiateWebGPUDevice({
          adapter: fa.adapter,
          target: 'none',
          requiredLimits: { maxStorageBuffersPerShaderStage: value },
        }),
      ).rejects.toThrow(/finite, non-negative safe integer/);
      expect(fa.lastDescriptor()).toBeUndefined();
    },
  );

  it('rejects an unknown explicit required-limit key before allocation', async () => {
    const fa = fakeAdapter(PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE, 8);
    restore = installNavigatorGpu();
    await expect(
      negotiateWebGPUDevice({
        adapter: fa.adapter,
        target: 'none',
        requiredLimits: { imaginaryLimit: 1 },
      }),
    ).rejects.toThrow(/unknown or unreported adapter limit/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it('uses the inverse comparison for minimum-alignment limits', async () => {
    const capable = fakeAdapter(16, 8, undefined, {
      limits: { minUniformBufferOffsetAlignment: 256 },
    });
    const incapable = fakeAdapter(16, 8, undefined, {
      limits: { minUniformBufferOffsetAlignment: 256 },
    });
    restore = installNavigatorGpu();

    await negotiateWebGPUDevice({
      adapter: capable.adapter,
      target: 'none',
      requiredLimits: { minUniformBufferOffsetAlignment: 512 },
    });
    expect(capable.lastDescriptor()?.requiredLimits?.minUniformBufferOffsetAlignment).toBe(512);

    await expect(
      negotiateWebGPUDevice({
        adapter: incapable.adapter,
        target: 'none',
        requiredLimits: { minUniformBufferOffsetAlignment: 128 },
      }),
    ).rejects.toThrow(/requires minUniformBufferOffsetAlignment <= 128/);
    expect(incapable.lastDescriptor()).toBeUndefined();
  });

  it('rejects unsupported required features before device allocation', async () => {
    const fa = fakeAdapter(16, 8, undefined, { features: [] });
    restore = installNavigatorGpu();
    await expect(
      negotiateWebGPUDevice({
        adapter: fa.adapter,
        target: 'none',
        requiredFeatures: ['timestamp-query'],
      }),
    ).rejects.toThrow(/required feature "timestamp-query" is not supported/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it('resolves the preferred format before allocating the device', async () => {
    const fa = fakeAdapter(16, 8);
    restore = () => setNavigator(ORIG_NAVIGATOR);
    setNavigator({ gpu: { getPreferredCanvasFormat: () => { throw new Error('format failed'); } } });
    await expect(
      negotiateWebGPUDevice({ adapter: fa.adapter, target: 'none' }),
    ).rejects.toThrow(/format failed/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it("target 'pt-webgpu' includes the ReSTIR-PT buffer floor when requested", async () => {
    const fa = fakeAdapter(
      PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    );
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'pt-webgpu',
      restirPtReuse: true,
    });
    expect(fa.lastDescriptor()?.requiredLimits?.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it("target 'pt-webgpu' includes the combined CWBVH + ReSTIR-PT floor", async () => {
    const fa = fakeAdapter(
      PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    );
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'pt-webgpu',
      cwbvhClosest: true,
      restirPtReuse: true,
    });
    expect(fa.lastDescriptor()?.requiredLimits?.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it("target 'pt-webgpu' rejects an optional layout the adapter cannot satisfy", async () => {
    const fa = fakeAdapter(
      PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE - 1,
      PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
    );
    restore = installNavigatorGpu();
    await expect(
      negotiateWebGPUDevice({
        adapter: fa.adapter,
        target: 'pt-webgpu',
        cwbvhClosest: true,
      }),
    ).rejects.toThrow(/cannot enable the requested CWBVH closest-hit layout/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it("target 'progressive' includes the combined CWBVH + ReSTIR-PT floor", async () => {
    const fa = fakeAdapter(
      PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage']!,
    );
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'progressive',
      cwbvhClosest: true,
      restirPtReuse: true,
    });
    expect(fa.lastDescriptor()?.requiredLimits?.maxStorageBuffersPerShaderStage).toBe(
      PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
    );
  });

  it("target 'walkaround-hybrid' requests every NRC limit when enabled", async () => {
    const fa = fakeAdapter(64, 8, undefined, {
      limits: {
        maxBindGroups: 8,
        maxComputeWorkgroupStorageSize: 65_536,
      },
    });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'walkaround-hybrid',
      nrcEnabled: true,
    });
    const requested = fa.lastDescriptor()?.requiredLimits as Record<string, number>;
    for (const [key, required] of Object.entries(NRC_WEBGPU_REQUIRED_LIMITS)) {
      expect(requested[key]).toBe(required);
    }
  });

  it('derives NRC trainer storage and shader-f16 from nrcConfig', async () => {
    const fa = fakeAdapter(64, 8, undefined, {
      limits: {
        maxBindGroups: 8,
        maxComputeWorkgroupStorageSize: 65_536,
      },
      features: ['shader-f16'],
    });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'walkaround-hybrid',
      nrcEnabled: true,
      nrcConfig: { width: 64, tileB: 64, useF16: true },
    });
    expect(fa.lastDescriptor()?.requiredLimits?.maxComputeWorkgroupStorageSize)
      .toBe(16_384);
    expect(fa.lastDescriptor()?.requiredFeatures).toEqual(['shader-f16']);
  });

  it('rejects NRC f16 before allocation when shader-f16 is unavailable', async () => {
    const fa = fakeAdapter(64, 8, undefined, {
      limits: {
        maxBindGroups: 8,
        maxComputeWorkgroupStorageSize: 65_536,
      },
    });
    restore = installNavigatorGpu();
    await expect(negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'walkaround-hybrid',
      nrcEnabled: true,
      nrcConfig: { useF16: true },
    })).rejects.toThrow(/required feature "shader-f16"/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });

  it("target 'progressive' includes every NRC limit in the shared union", async () => {
    const fa = fakeAdapter(64, 8, undefined, {
      limits: {
        maxBindGroups: 8,
        maxComputeWorkgroupStorageSize: 65_536,
      },
    });
    restore = installNavigatorGpu();
    await negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'progressive',
      nrcEnabled: true,
    });
    const requested = fa.lastDescriptor()?.requiredLimits as Record<string, number>;
    for (const [key, required] of Object.entries(NRC_WEBGPU_REQUIRED_LIMITS)) {
      expect(requested[key]).toBeGreaterThanOrEqual(required);
    }
  });

  it('rejects NRC negotiation before allocation when an NRC-only limit is missing', async () => {
    const fa = fakeAdapter(64, 8);
    restore = installNavigatorGpu();
    await expect(negotiateWebGPUDevice({
      adapter: fa.adapter,
      target: 'progressive',
      nrcEnabled: true,
    })).rejects.toThrow(/maxBindGroups/);
    expect(fa.lastDescriptor()).toBeUndefined();
  });
});
