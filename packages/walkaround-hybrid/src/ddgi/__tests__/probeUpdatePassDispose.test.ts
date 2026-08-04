/**
 * Bug characterization tests for ProbeUpdatePass resource management.
 *
 * (a) BUG 1 — dispose() resource leak: all 22 GPU buffers and every owned
 *     texture allocated in init() (including the optical-identity cohort,
 *     5 TLAS buffers, and traceParamsBuf) must be destroyed on dispose().
 *
 * (b) BUG 2 — init() re-entry on failed GPU init: if navigator.gpu.requestAdapter
 *     returns null (hard WebGPU failure), _initAttempted must prevent re-issuing
 *     the adapter request on every subsequent runFrame() call.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';

// Install GPUBufferUsage / GPUTextureUsage globals so init() doesn't throw in
// the happy-dom Node environment used by vitest.
installWebGPUPolyfills();

// ─── GPU mock helpers ────────────────────────────────────────────────────────

/** Returns a tracking mock GPUBuffer. Each call to destroy() records itself. Retained for dispose tests. */
function _makeBuffer(destroyedList: GPUBuffer[]): GPUBuffer {
  const buf = {
    size: 16,
    destroy: vi.fn(() => { destroyedList.push(buf); }),
  } as unknown as GPUBuffer;
  return buf;
}

/** Returns a tracking mock GPUTexture. destroy() is tracked. Retained for dispose tests. */
function _makeTexture(destroyedList: GPUTexture[]): GPUTexture {
  const tex = {
    width: 64, height: 64, depthOrArrayLayers: 1,
    destroy: vi.fn(() => { destroyedList.push(tex); }),
  } as unknown as GPUTexture;
  return tex;
}

interface MockDeviceTracking {
  createdBuffers: GPUBuffer[];
  destroyedBuffers: GPUBuffer[];
  createdTextures: GPUTexture[];
  destroyedTextures: GPUTexture[];
}

function deferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((accept) => { resolve = accept; });
  return { promise, resolve };
}

/**
 * Build a mock GPUDevice that tracks every createBuffer and createTexture call.
 * Each allocated resource has a destroy() that records itself in the "destroyed"
 * arrays — so the test can assert that destroyed.length === created.length.
 */
function makeMockDevice(
  tracking: MockDeviceTracking,
  options: {
    readonly failBufferAt?: number;
    readonly maxUniformBufferBindingSize?: number;
  } = {},
): GPUDevice {
  let bufferCreateCount = 0;
  let failureAvailable = true;
  return {
    limits: {
      maxTextureDimension2D: 8192,
      maxTextureArrayLayers: 256,
      ...(options.maxUniformBufferBindingSize == null
        ? {}
        : { maxUniformBufferBindingSize: options.maxUniformBufferBindingSize }),
    },
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
      bufferCreateCount++;
      if (failureAvailable && bufferCreateCount === options.failBufferAt) {
        failureAvailable = false;
        throw new Error(`injected init buffer failure ${bufferCreateCount}`);
      }
      const buf = {
        size: desc.size,
        destroy: vi.fn(() => { tracking.destroyedBuffers.push(buf); }),
      } as unknown as GPUBuffer;
      tracking.createdBuffers.push(buf);
      return buf;
    }),
    createTexture: vi.fn((desc: GPUTextureDescriptor) => {
      const tex = {
        width: (desc.size as GPUExtent3DDict).width ?? 4,
        height: (desc.size as GPUExtent3DDict).height ?? 4,
        depthOrArrayLayers: 1,
        destroy: vi.fn(() => { tracking.destroyedTextures.push(tex); }),
        // Wave 4: createView() is needed for the env-map placeholder created in init().
        createView: vi.fn(() => ({})),
      } as unknown as GPUTexture;
      tracking.createdTextures.push(tex);
      return tex;
    }),
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
    },
    destroy: vi.fn(),
  } as unknown as GPUDevice;
}

// ─── Module-level mocks ──────────────────────────────────────────────────────

// detectGpu — controls whether init() believes WebGPU is available.
vi.mock('@vitrum/core', async (importOriginal) => {
  const original = await importOriginal<typeof import('@vitrum/core')>();
  return {
    ...original,
    detectGpu: vi.fn(async () => ({
      isWebGPU: true,
      adapterKind: 'hardware',
      adapterVendor: 'test',
      adapterArchitecture: 'test',
    })),
  };
});

// Import after mocks are registered.
import { ProbeUpdatePass } from '../probeUpdatePass.js';
import { ProbeGrid } from '../probeGrid.js';
import { SceneBvh } from '@vitrum/shared-bvh';
import { detectGpu } from '@vitrum/core';
import { EMITTER_TRI_STRIDE_BYTES } from '../../restir/emitterList.js';
import { DDGI_MATERIAL_STRIDE_BYTES } from '../probeUpdateMaterials.js';

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('ProbeUpdatePass — dispose() destroys all allocated GPU resources', () => {
  let tracking: MockDeviceTracking;
  let mockDevice: GPUDevice;

  beforeEach(() => {
    tracking = {
      createdBuffers: [],
      destroyedBuffers: [],
      createdTextures: [],
      destroyedTextures: [],
    };
    mockDevice = makeMockDevice(tracking);
    // detectGpu returns "hardware WebGPU available" by default.
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: true,
      adapterKind: 'hardware',
      adapterVendor: 'test',
      adapterArchitecture: 'test',
    });
    vi.mocked(detectGpu).mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('rejects an oversized material uniform before shader compilation or allocation', async () => {
    const maxMaterials = 2;
    const requiredBytes = maxMaterials * DDGI_MATERIAL_STRIDE_BYTES;
    const constrainedDevice = makeMockDevice(tracking, {
      maxUniformBufferBindingSize: requiredBytes - 1,
    });
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid(), { maxMaterials });

    await expect(pass.init({
      backend: { device: constrainedDevice, isWebGPUBackend: true },
    })).rejects.toThrow(
      `[DDGI] material uniform buffer requires ${requiredBytes} bytes ` +
      `(${maxMaterials} materials × ${DDGI_MATERIAL_STRIDE_BYTES} bytes), ` +
      `exceeding device.limits.maxUniformBufferBindingSize (${requiredBytes - 1} bytes).`,
    );

    expect(constrainedDevice.createShaderModule).not.toHaveBeenCalled();
    expect(constrainedDevice.createBuffer).not.toHaveBeenCalled();
    expect(constrainedDevice.createTexture).not.toHaveBeenCalled();
    expect(tracking.createdBuffers).toHaveLength(0);
    expect(tracking.createdTextures).toHaveLength(0);
    expect(constrainedDevice.destroy).not.toHaveBeenCalled();
  });

  /**
   * Calls init() with a renderer stub that supplies the mock device, then
   * calls dispose().  Asserts that every buffer allocated by init() is
   * destroyed — including the optical-identity cohort, the 5 TLAS buffers
   * (tlasNodesBuf, tlasInstIdxBuf, tlasBlasRootsBuf, tlasW2lBuf, tlasL2wBuf),
   * and traceParamsBuf.
   */
  it('destroys every buffer/texture allocated during init() — including the 6 previously-leaked resources', async () => {
    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);

    const rendererAdapter = {
      backend: { device: mockDevice, isWebGPUBackend: true as const },
    };

    const ok = await pass.init(rendererAdapter);
    expect(ok).toBe(true);

    // Buffers known to be allocated by init(): bvhBuf, posBuf, idxBuf, normBuf,
    // matIdBuf, opticalTriangleIdentityBuf,
    // opticalInstanceBoundaryIdBasePlusOneBuf, tlasNodesBuf, tlasInstIdxBuf,
    // tlasBlasRootsBuf, tlasW2lBuf, tlasL2wBuf, traceParamsBuf, materialsBuf,
    // lightsBuf, gridParamsBuf,
    // frameParamsBuf, blendParamsBuf, borderVisUboBuf, rayResultsBuf,
    // activeProbesBuf, emitterTrisBuf = 20 buffers (borderIrrUboBuf removed
    // with the SH irradiance migration — no irradiance border pass).
    // Wave 4: 1 extra texture (env-map placeholder) is now also allocated.
    const buffersAfterInit = tracking.createdBuffers.length;
    expect(buffersAfterInit).toBe(22);
    // Wave 4 — placeholder env texture (rgba16float, 1×1) is created in init().
    expect(tracking.createdTextures.length).toBeGreaterThanOrEqual(1);

    pass.dispose();

    // Every buffer allocated during init must be destroyed. visScratchTex
    // starts null (only allocated on runFrame) so we only check
    // the init-time allocations here.
    expect(tracking.destroyedBuffers.length).toBe(buffersAfterInit);

    // No GPUBuffer may remain undestroyed.
    const undestroyed = tracking.createdBuffers.filter(
      (b) => !tracking.destroyedBuffers.includes(b),
    );
    expect(undestroyed).toHaveLength(0);

    // Wave 4 — the pass-owned env-map placeholder texture must also be destroyed.
    const undestroyedTex = tracking.createdTextures.filter(
      (t) => !tracking.destroyedTextures.includes(t),
    );
    expect(undestroyedTex).toHaveLength(0);
    expect(mockDevice.destroy).not.toHaveBeenCalled();
  });

  it('double dispose() does not throw and does not call destroy() a second time', async () => {
    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);

    const rendererAdapter = {
      backend: { device: mockDevice, isWebGPUBackend: true as const },
    };

    await pass.init(rendererAdapter);
    const countAfterInit = tracking.destroyedBuffers.length;

    pass.dispose();
    const countAfter1 = tracking.destroyedBuffers.length;
    expect(countAfter1).toBeGreaterThan(countAfterInit);

    // Second dispose must be a no-op (no extra destroy calls).
    pass.dispose();
    expect(tracking.destroyedBuffers.length).toBe(countAfter1);
  });

  it('rebinds a fresh pass-owned env placeholder when HDRI is disabled after an external view', async () => {
    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);

    const rendererAdapter = {
      backend: { device: mockDevice, isWebGPUBackend: true as const },
    };

    await pass.init(rendererAdapter);
    const internal = pass as unknown as { _gpu: {
      envMapView: GPUTextureView;
      envMapOwnedByPass: boolean;
      envMapPlaceholderTex: GPUTexture | null;
    }};
    const gpu = internal._gpu;
    const initialPlaceholder = gpu.envMapPlaceholderTex;
    expect(initialPlaceholder).not.toBeNull();
    expect(gpu.envMapOwnedByPass).toBe(true);

    const externalView = {} as GPUTextureView;
    const externalSampler = {} as GPUSampler;
    pass.setEnvironment(externalView, externalSampler, 0.25, 1.5, true);
    expect(gpu.envMapView).toBe(externalView);
    expect(gpu.envMapOwnedByPass).toBe(false);
    expect(gpu.envMapPlaceholderTex).toBeNull();
    expect(tracking.destroyedTextures).toContain(initialPlaceholder);

    const texturesBeforeDisable = tracking.createdTextures.length;
    pass.setEnvironment(null, null, 0, 0, false);
    expect(gpu.envMapView).not.toBe(externalView);
    expect(gpu.envMapOwnedByPass).toBe(true);
    expect(gpu.envMapPlaceholderTex).not.toBeNull();
    expect(tracking.createdTextures.length).toBe(texturesBeforeDisable + 1);
    expect(tracking.createdTextures).toContain(gpu.envMapPlaceholderTex);
  });

  /**
   * Verify that each of the 6 previously-leaking resources is individually
   * destroyed.  The gpu-state object fields are accessed via internal access
   * to confirm by name rather than count alone.
   */
  it('dispose() calls destroy() on each of the 6 previously-omitted resources by name', async () => {
    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);

    const rendererAdapter = {
      backend: { device: mockDevice, isWebGPUBackend: true as const },
    };

    await pass.init(rendererAdapter);

    // Access internal gpu state to get references to the specific buffers.
    const internal = pass as unknown as { _gpu: {
      tlasNodesBuf:     GPUBuffer;
      tlasInstIdxBuf:   GPUBuffer;
      tlasBlasRootsBuf: GPUBuffer;
      tlasW2lBuf:       GPUBuffer;
      tlasL2wBuf:       GPUBuffer;
      traceParamsBuf:   GPUBuffer;
    }};
    const gpu = internal._gpu;
    const sixBuffers = [
      gpu.tlasNodesBuf,
      gpu.tlasInstIdxBuf,
      gpu.tlasBlasRootsBuf,
      gpu.tlasW2lBuf,
      gpu.tlasL2wBuf,
      gpu.traceParamsBuf,
    ];

    pass.dispose();

    for (const buf of sixBuffers) {
      const mockBuf = buf as unknown as { destroy: ReturnType<typeof vi.fn> };
      expect(mockBuf.destroy).toHaveBeenCalledOnce();
    }
  });

  it('allocates DDGI TLAS node placeholder at the BVHNode binding minimum', async () => {
    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);

    const rendererAdapter = {
      backend: { device: mockDevice, isWebGPUBackend: true as const },
    };

    await pass.init(rendererAdapter);

    const internal = pass as unknown as { _gpu: {
      bvhBuf: GPUBuffer;
      tlasNodesBuf: GPUBuffer;
      tlasInstIdxBuf: GPUBuffer;
      tlasW2lBuf: GPUBuffer;
      traceParamsBuf: GPUBuffer;
      emitterTrisBuf: GPUBuffer;
    }};
    const gpu = internal._gpu;
    expect(gpu.bvhBuf.size).toBe(32);
    expect(gpu.tlasNodesBuf.size).toBe(32);
    expect(gpu.tlasInstIdxBuf.size).toBe(16);
    expect(gpu.tlasW2lBuf.size).toBe(16);
    expect(gpu.traceParamsBuf.size).toBe(16);
    expect(gpu.emitterTrisBuf.size).toBe(EMITTER_TRI_STRIDE_BYTES);

    pass.dispose();
  });

  it('cleans a partial init allocation, publishes nothing, and retries successfully', async () => {
    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);
    const failingDevice = makeMockDevice(tracking, { failBufferAt: 10 });
    const rendererAdapter = {
      backend: { device: failingDevice, isWebGPUBackend: true as const },
    };

    await expect(pass.init(rendererAdapter)).rejects.toThrow('injected init buffer failure 10');
    const internal = pass as unknown as {
      _gpu: unknown;
      _initAttempted: boolean;
    };
    expect(internal._gpu).toBeNull();
    expect(internal._initAttempted).toBe(false);
    expect(tracking.createdBuffers).toHaveLength(9);
    expect(tracking.destroyedBuffers).toHaveLength(9);
    expect(tracking.destroyedTextures).toHaveLength(tracking.createdTextures.length);

    await expect(pass.init(rendererAdapter)).resolves.toBe(true);
    expect(internal._gpu).not.toBeNull();
    pass.dispose();
    expect(tracking.destroyedBuffers).toHaveLength(tracking.createdBuffers.length);
    expect(tracking.destroyedTextures).toHaveLength(tracking.createdTextures.length);
  });

  it('coalesces concurrent init callers onto the exact same allocation promise', async () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const device = makeMockDevice(tracking);
    const firstPipeline = deferred<GPUComputePipeline>();
    vi.mocked(device.createComputePipelineAsync)
      .mockImplementationOnce(() => firstPipeline.promise)
      .mockResolvedValue({ getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline);
    const renderer = { backend: { device, isWebGPUBackend: true as const } };

    const first = pass.init(renderer);
    await vi.waitFor(() => {
      expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(1);
    });
    const second = pass.init(renderer);

    expect(second).toBe(first);
    expect(tracking.createdBuffers).toHaveLength(0);
    firstPipeline.resolve({
      getBindGroupLayout: vi.fn(() => ({})),
    } as unknown as GPUComputePipeline);
    await expect(first).resolves.toBe(true);
    // rays + classify/relocate + two blends + visibility-border
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(5);
    expect(tracking.createdBuffers.length).toBeGreaterThan(0);
    pass.dispose();
  });

  it('cannot publish resources when dispose wins while pipeline compilation is pending', async () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const device = makeMockDevice(tracking);
    const firstPipeline = deferred<GPUComputePipeline>();
    vi.mocked(device.createComputePipelineAsync)
      .mockImplementationOnce(() => firstPipeline.promise)
      .mockResolvedValue({ getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline);
    const renderer = { backend: { device, isWebGPUBackend: true as const } };

    const init = pass.init(renderer);
    await vi.waitFor(() => {
      expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(1);
    });
    pass.dispose();
    firstPipeline.resolve({
      getBindGroupLayout: vi.fn(() => ({})),
    } as unknown as GPUComputePipeline);

    await expect(init).resolves.toBe(false);
    expect(tracking.createdBuffers).toHaveLength(0);
    expect(tracking.createdTextures).toHaveLength(0);
    expect((pass as unknown as { _gpu: unknown })._gpu).toBeNull();
    expect(device.destroy).not.toHaveBeenCalled();
  });

  it('destroys a navigator-fallback device only after its pass-owned resources', async () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const fallbackDevice = makeMockDevice(tracking);
    const requestDevice = vi.fn(async () => fallbackDevice);
    vi.stubGlobal('navigator', {
      gpu: { requestAdapter: vi.fn(async () => ({ requestDevice })) },
    });

    await expect(pass.init({})).resolves.toBe(true);
    expect(requestDevice).toHaveBeenCalledTimes(1);
    expect(fallbackDevice.destroy).not.toHaveBeenCalled();

    pass.dispose();

    expect(fallbackDevice.destroy).toHaveBeenCalledTimes(1);
    const deviceDestroyOrder = vi.mocked(fallbackDevice.destroy).mock.invocationCallOrder[0]!;
    const resourceDestroyOrders = [
      ...tracking.createdBuffers,
      ...tracking.createdTextures,
    ].map((resource) => vi.mocked(resource.destroy).mock.invocationCallOrder[0]!);
    expect(Math.max(...resourceDestroyOrders)).toBeLessThan(deviceDestroyOrder);
  });

  it('destroys a navigator-fallback device when compilation fails before publication', async () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid(), {
      onWarning: vi.fn(),
    });
    const fallbackDevice = makeMockDevice(tracking);
    vi.mocked(fallbackDevice.createComputePipelineAsync)
      .mockRejectedValueOnce(new Error('injected fallback compile failure'));
    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: vi.fn(async () => ({
          requestDevice: vi.fn(async () => fallbackDevice),
        })),
      },
    });

    await expect(pass.init({})).resolves.toBe(false);
    expect((pass as unknown as { _gpu: unknown })._gpu).toBeNull();
    expect(fallbackDevice.destroy).toHaveBeenCalledTimes(1);
  });

  it('destroys a navigator-fallback device when dispose wins an init race', async () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const fallbackDevice = makeMockDevice(tracking);
    const firstPipeline = deferred<GPUComputePipeline>();
    vi.mocked(fallbackDevice.createComputePipelineAsync)
      .mockImplementationOnce(() => firstPipeline.promise)
      .mockResolvedValue({ getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline);
    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: vi.fn(async () => ({
          requestDevice: vi.fn(async () => fallbackDevice),
        })),
      },
    });

    const init = pass.init({});
    await vi.waitFor(() => {
      expect(fallbackDevice.createComputePipelineAsync).toHaveBeenCalledTimes(1);
    });
    pass.dispose();
    firstPipeline.resolve({
      getBindGroupLayout: vi.fn(() => ({})),
    } as unknown as GPUComputePipeline);

    await expect(init).resolves.toBe(false);
    expect(tracking.createdBuffers).toHaveLength(0);
    expect(fallbackDevice.destroy).toHaveBeenCalledTimes(1);
  });

  it('keeps a transient pipeline-compilation failure retryable', async () => {
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid(), {
      onWarning: vi.fn(),
    });
    const device = makeMockDevice(tracking);
    vi.mocked(device.createComputePipelineAsync)
      .mockRejectedValueOnce(new Error('transient compile failure'));
    const renderer = { backend: { device, isWebGPUBackend: true as const } };

    await expect(pass.init(renderer)).resolves.toBe(false);
    expect((pass as unknown as { _initAttempted: boolean })._initAttempted).toBe(false);
    expect(tracking.createdBuffers).toHaveLength(0);

    await expect(pass.init(renderer)).resolves.toBe(true);
    // One failed attempt, then all five production DDGI pipelines.
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(6);
    expect(tracking.createdBuffers.length).toBeGreaterThan(0);
    pass.dispose();
  });
});

describe('ProbeUpdatePass — _initAttempted guard prevents repeated init() on WebGPU failure', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  /**
   * When navigator.gpu.requestAdapter returns null (no adapter — hard WebGPU
   * failure), init() must be called at most once regardless of how many times
   * runFrame() is invoked.  Previously, every runFrame call would re-issue
   * navigator.gpu.requestAdapter because the null-device path left _gpu null
   * without setting any guard flag.
   */
  it('runFrame() does not re-call init() after a hard WebGPU-init failure', async () => {
    // Make detectGpu return a non-SwiftShader result so the SwiftShader early
    // exit is not hit; the failure comes from navigator.gpu returning null below.
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: false,
      adapterKind: 'unknown',
      adapterVendor: '',
      adapterArchitecture: '',
    });

    // No backend device, no navigator.gpu — simulates a hard failure environment.
    const rendererNoGpu = {};

    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);

    // Spy on init to count calls.
    const initSpy = vi.spyOn(pass, 'init');

    // runFrame frame 1 — triggers init(); init fails because no device is
    // available; _gpu stays null; _initAttempted must be set to true.
    await pass.runFrame(rendererNoGpu, 0, 1);
    expect(initSpy).toHaveBeenCalledTimes(1);

    // runFrame frames 2–5 — must NOT re-call init() because _initAttempted is
    // already true.
    for (let i = 1; i < 5; i++) {
      await pass.runFrame(rendererNoGpu, 0, 1);
    }
    expect(initSpy).toHaveBeenCalledTimes(1);
  });

  it('routes hard WebGPU-unavailable init failure through structured warnings', async () => {
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: false,
      adapterKind: 'unknown',
      adapterVendor: '',
      adapterArchitecture: '',
    });
    vi.stubGlobal('navigator', {});
    const warnings: EngineWarning[] = [];

    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid, {
      onWarning: (warning) => warnings.push(warning),
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    await pass.runFrame({}, 0, 1);

    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toContainEqual(expect.objectContaining({
      code: 'walkaround-hybrid.ddgi-webgpu-unavailable',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'ProbeUpdatePass.init',
      details: { fallback: 'disable-ddgi-probe-update' },
    }));
  });

  it('accepts a conformant backend without adapter-provenance gates or debug globals', async () => {
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: true,
      adapterKind: 'swiftshader',
      adapterVendor: 'Google',
      adapterArchitecture: 'SwiftShader',
    });
    const tracking: MockDeviceTracking = {
      createdBuffers: [],
      destroyedBuffers: [],
      createdTextures: [],
      destroyedTextures: [],
    };
    const device = makeMockDevice(tracking);
    const pass = new ProbeUpdatePass(new SceneBvh(), new ProbeGrid());
    const sentinel = { untouched: true };
    const debugWindow = { __WG__: sentinel };
    vi.stubGlobal('window', debugWindow);

    await expect(pass.init({
      backend: { device, isWebGPUBackend: true },
    })).resolves.toBe(true);

    expect(detectGpu).not.toHaveBeenCalled();
    expect(debugWindow.__WG__).toBe(sentinel);
    pass.dispose();
  });

  it('routes DDGI shader compilation failures through structured warnings', async () => {
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: true,
      adapterKind: 'hardware',
      adapterVendor: 'test',
      adapterArchitecture: 'test',
    });
    const compileError = new Error('compile boom');
    const tracking: MockDeviceTracking = {
      createdBuffers: [],
      destroyedBuffers: [],
      createdTextures: [],
      destroyedTextures: [],
    };
    const device = makeMockDevice(tracking);
    vi.mocked(device.createComputePipelineAsync).mockRejectedValueOnce(compileError);
    const warnings: EngineWarning[] = [];

    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid, {
      onWarning: (warning) => warnings.push(warning),
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);

    await pass.runFrame({ backend: { device, isWebGPUBackend: true } }, 0, 1);

    expect(warnSpy).not.toHaveBeenCalled();
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnings).toContainEqual(expect.objectContaining({
      code: 'walkaround-hybrid.ddgi-shader-compilation-failed',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'ProbeUpdatePass.init',
      details: { fallback: 'disable-ddgi-probe-update' },
      raw: compileError,
    }));
  });

  it('routes navigator.gpu requestAdapter failures through structured warnings', async () => {
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: false,
      adapterKind: 'unknown',
      adapterVendor: '',
      adapterArchitecture: '',
    });
    const thrown = new Error('adapter boom');
    vi.stubGlobal('navigator', {
      gpu: {
        requestAdapter: vi.fn(async () => {
          throw thrown;
        }),
      },
    });
    const warnings: EngineWarning[] = [];

    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid, {
      onWarning: (warning) => warnings.push(warning),
    });
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    await pass.runFrame({}, 0, 1);

    expect(warnSpy).not.toHaveBeenCalled();
    expect(warnings).toContainEqual(expect.objectContaining({
      code: 'walkaround-hybrid.ddgi-request-adapter-failed',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'ProbeUpdatePass.init',
      details: { source: 'navigator.gpu', fallback: 'disable-ddgi-probe-update' },
      raw: thrown,
    }));
    expect(warnings).toContainEqual(expect.objectContaining({
      code: 'walkaround-hybrid.ddgi-webgpu-unavailable',
    }));
  });

  it('init() is attempted again after an explicit dispose() resets state', async () => {
    // All init calls fail (no GPU).
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: false,
      adapterKind: 'unknown',
      adapterVendor: '',
      adapterArchitecture: '',
    });

    const rendererNoGpu = {};

    const bvh = new SceneBvh();
    const grid = new ProbeGrid();
    const pass = new ProbeUpdatePass(bvh, grid);
    const initSpy = vi.spyOn(pass, 'init');

    // First attempt — init fails.
    await pass.runFrame(rendererNoGpu, 0, 1);
    expect(initSpy).toHaveBeenCalledTimes(1);

    // Subsequent calls are blocked by _initAttempted.
    await pass.runFrame(rendererNoGpu, 0, 1);
    expect(initSpy).toHaveBeenCalledTimes(1);

    // dispose() clears _gpu (already null) and resets the atlasCache. The
    // _initAttempted flag persists on the SAME instance (by design — this test
    // confirms the flag is NOT reset on dispose, so a disposed pass that somehow
    // gets runFrame called does not re-init).
    pass.dispose();
    await pass.runFrame(rendererNoGpu, 0, 1);
    // _initAttempted is NOT reset by dispose(), so init is still NOT retried.
    expect(initSpy).toHaveBeenCalledTimes(1);
  });
});
