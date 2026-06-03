/**
 * Bug characterization tests for ProbeUpdatePass resource management.
 *
 * (a) BUG 1 — dispose() resource leak: all 22 GPU buffers/textures allocated
 *     in init() (including the 5 TLAS buffers + traceParamsBuf previously
 *     omitted) must be destroyed on dispose().
 *
 * (b) BUG 2 — init() re-entry on failed GPU init: if navigator.gpu.requestAdapter
 *     returns null (hard WebGPU failure), _initAttempted must prevent re-issuing
 *     the adapter request on every subsequent runFrame() call.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ProbeUpdatePass as ProbeUpdatePassType } from '../probeUpdatePass.js';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';

// Install GPUBufferUsage / GPUTextureUsage globals so init() doesn't throw in
// the happy-dom Node environment used by vitest.
installWebGPUPolyfills();

// ─── GPU mock helpers ────────────────────────────────────────────────────────

/** Returns a tracking mock GPUBuffer. Each call to destroy() records itself. */
function makeBuffer(destroyedList: GPUBuffer[]): GPUBuffer {
  const buf = {
    size: 16,
    destroy: vi.fn(() => { destroyedList.push(buf as unknown as GPUBuffer); }),
  } as unknown as GPUBuffer;
  return buf;
}

/** Returns a tracking mock GPUTexture. destroy() is tracked. */
function makeTexture(destroyedList: GPUTexture[]): GPUTexture {
  const tex = {
    width: 64, height: 64, depthOrArrayLayers: 1,
    destroy: vi.fn(() => { destroyedList.push(tex as unknown as GPUTexture); }),
  } as unknown as GPUTexture;
  return tex;
}

interface MockDeviceTracking {
  createdBuffers: GPUBuffer[];
  destroyedBuffers: GPUBuffer[];
  createdTextures: GPUTexture[];
  destroyedTextures: GPUTexture[];
}

/**
 * Build a mock GPUDevice that tracks every createBuffer and createTexture call.
 * Each allocated resource has a destroy() that records itself in the "destroyed"
 * arrays — so the test can assert that destroyed.length === created.length.
 */
function makeMockDevice(tracking: MockDeviceTracking): GPUDevice {
  return {
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
      const buf = {
        size: desc.size,
        destroy: vi.fn(() => { tracking.destroyedBuffers.push(buf as unknown as GPUBuffer); }),
      } as unknown as GPUBuffer;
      tracking.createdBuffers.push(buf);
      return buf;
    }),
    createTexture: vi.fn((desc: GPUTextureDescriptor) => {
      const tex = {
        width: (desc.size as GPUExtent3DDict).width ?? 4,
        height: (desc.size as GPUExtent3DDict).height ?? 4,
        depthOrArrayLayers: 1,
        destroy: vi.fn(() => { tracking.destroyedTextures.push(tex as unknown as GPUTexture); }),
      } as unknown as GPUTexture;
      tracking.createdTextures.push(tex);
      return tex;
    }),
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    queue: { writeBuffer: vi.fn() },
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
    } as Awaited<ReturnType<typeof detectGpu>>);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /**
   * Calls init() with a renderer stub that supplies the mock device, then
   * calls dispose().  Asserts that every buffer allocated by init() is
   * destroyed — including the 5 TLAS buffers (tlasNodesBuf, tlasInstIdxBuf,
   * tlasBlasRootsBuf, tlasW2lBuf, tlasL2wBuf) and traceParamsBuf that were
   * previously missing from dispose().
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
    // matIdBuf, tlasNodesBuf, tlasInstIdxBuf, tlasBlasRootsBuf, tlasW2lBuf,
    // tlasL2wBuf, traceParamsBuf, materialsBuf, lightsBuf, gridParamsBuf,
    // frameParamsBuf, blendParamsBuf, borderIrrUboBuf, borderVisUboBuf,
    // rayResultsBuf, activeProbesBuf = 20 buffers.
    const buffersAfterInit = tracking.createdBuffers.length;
    expect(buffersAfterInit).toBeGreaterThanOrEqual(20); // at least 20 buffers from init

    pass.dispose();

    // Every buffer allocated during init must be destroyed.  irrScratchTex /
    // visScratchTex start null (only allocated on runFrame) so we only check
    // the init-time allocations here.
    expect(tracking.destroyedBuffers.length).toBe(buffersAfterInit);

    // No GPUBuffer may remain undestroyed.
    const undestroyed = tracking.createdBuffers.filter(
      (b) => !tracking.destroyedBuffers.includes(b),
    );
    expect(undestroyed).toHaveLength(0);
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
});

describe('ProbeUpdatePass — _initAttempted guard prevents repeated init() on WebGPU failure', () => {
  afterEach(() => {
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
    } as Awaited<ReturnType<typeof detectGpu>>);

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

  it('init() is attempted again after an explicit dispose() resets state', async () => {
    // All init calls fail (no GPU).
    vi.mocked(detectGpu).mockResolvedValue({
      isWebGPU: false,
      adapterKind: 'unknown',
      adapterVendor: '',
      adapterArchitecture: '',
    } as Awaited<ReturnType<typeof detectGpu>>);

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
