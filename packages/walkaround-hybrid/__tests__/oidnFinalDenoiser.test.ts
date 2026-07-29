/**
 * oidnFinalDenoiser.test.ts — W11 acceptance tests.
 *
 * Verifies that the OIDNFinalDenoiser wire is real:
 *   1. The disabled flag respects the modelUrl construction-time choice.
 *   2. `initialize` acquires an owned lease on the shared OIDN session.
 *   3. `dispatch` kicks off the background readback + inference + upload
 *      chain, calls `denoiseFinal` with color + albedo + normal aux inputs,
 *      and returns a GPU texture (the owned denoised output once an
 *      inference completes, the raw HDR target before that).
 *   4. `dispose` releases GPU resources and its owned OIDN session lease.
 *   5. After `registerBuiltinDenoisers` with `oidn.modelUrl`, the registry
 *      successfully looks up `'oidn-final'` (no "registered but disabled"
 *      error).
 *
 * Uses vi.mock to replace the `@vitrum/shared-denoisers` OIDN entry
 * points so no real ONNX runtime or model file is required.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

// ── Module mocks ────────────────────────────────────────────────────────────
// The mocked `@vitrum/shared-denoisers` re-exports the OIDN entry points as
// vi.fn spies the test can interrogate. denoiseFinal returns a deterministic
// flat Float32 array (HxWx3) so the upload path has real bytes to write.
//
// vi.hoisted is required because vi.mock factories are hoisted to the top of
// the file (above the imports) and so cannot reference top-level lexical
// bindings — vi.hoisted lets us define them in a hoisted scope that the
// factory can read.

const oidnMocks = vi.hoisted(() => {
  const leaseRelease = vi.fn(() => undefined);
  return {
    leaseRelease,
    acquireOIDNSession: vi.fn(async (_opts: unknown) => ({ release: leaseRelease })),
    denoiseFinal: vi.fn(async (inputs: { color: Float32Array; width: number; height: number }) => {
      // Echo input back as if OIDN denoised it (identity output is fine for the test).
      return new Float32Array(inputs.color);
    }),
    clearOIDNCache: vi.fn(() => undefined),
  };
});
const { acquireOIDNSession, denoiseFinal, leaseRelease } = oidnMocks;

vi.mock('@vitrum/shared-denoisers', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/shared-denoisers')>();
  return {
    ...actual,
    acquireOIDNSession: oidnMocks.acquireOIDNSession,
    denoiseFinal: oidnMocks.denoiseFinal,
    clearOIDNCache: oidnMocks.clearOIDNCache,
  };
});

// Import AFTER vi.mock so the mocked module is wired in.
import { OIDNFinalDenoiser } from '../src/pipeline/denoisers/oidnFinal.js';
import { DenoiserRegistry } from '../src/pipeline/denoisers/index.js';
import { registerBuiltinDenoisers } from '../src/pipeline/denoisers/registerBuiltinDenoisers.js';
import type {
  DenoiserDispatchContext,
  DenoiserInitContext,
} from '../src/pipeline/denoisers/index.js';
import type { EngineWarning } from '@vitrum/core';

// ── Stub GPU helpers ────────────────────────────────────────────────────────

/** Build a fake GPUTexture that records destroy() calls. */
function fakeTexture(): GPUTexture {
  return {
    label: '',
    destroy: vi.fn(),
    createView: vi.fn(() => ({} as GPUTextureView)),
    width: 64,
    height: 64,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba16float',
    usage: 0,
  } as unknown as GPUTexture;
}

/** Build a fake mapped readback buffer that resolves mapAsync immediately
 *  and returns a Float32-zeros-equivalent rgba16f payload on getMappedRange. */
function fakeReadbackBuffer(byteSize: number): GPUBuffer {
  const backing = new ArrayBuffer(byteSize);
  return {
    label: '',
    size: byteSize,
    usage: 0,
    mapAsync: vi.fn(async () => undefined),
    getMappedRange: vi.fn(() => backing),
    unmap: vi.fn(),
    destroy: vi.fn(),
  } as unknown as GPUBuffer;
}

function fakeDevice(): GPUDevice {
  return {
    createTexture: vi.fn(() => fakeTexture()),
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => fakeReadbackBuffer(desc.size)),
    createCommandEncoder: vi.fn(() => fakeEncoder()),
    queue: {
      writeTexture: vi.fn(),
      writeBuffer: vi.fn(),
      submit: vi.fn(),
    },
  } as unknown as GPUDevice;
}

function fakeEncoder(): GPUCommandEncoder {
  return {
    label: '',
    copyTextureToBuffer: vi.fn(),
    beginComputePass: vi.fn(),
    finish: vi.fn(() => ({} as GPUCommandBuffer)),
  } as unknown as GPUCommandEncoder;
}

/** Minimal DenoiserInitContext for the OIDN denoiser (which only uses
 *  device / width / height — bglCache + frameResources are untouched). */
function fakeInitCtx(device: GPUDevice): DenoiserInitContext {
  return {
    device,
    width: 64,
    height: 32,
    bglCache: {} as DenoiserInitContext['bglCache'],
    frameResources: {} as DenoiserInitContext['frameResources'],
  };
}

/** Minimal DenoiserDispatchContext exposing the textures OIDN reads
 *  (hdrColor, albedo, gNormalDepth) plus the encoder. */
function fakeDispatchCtx(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  hdr: GPUTexture,
  albedo: GPUTexture,
  normal: GPUTexture,
  frameIndex = 0,
  isMoving = false,
): DenoiserDispatchContext {
  return {
    device,
    encoder,
    width: 64,
    height: 32,
    frameIndex,
    resources: {
      common: {
        hdrColorTexture: hdr,
        albedoTexture: albedo,
        gNormalDepthTexture: normal,
      },
    } as DenoiserDispatchContext['resources'],
    sharedAtrousPipeline: {} as GPUComputePipeline,
    bglCache: {} as DenoiserDispatchContext['bglCache'],
    gNormalDepthView: {} as GPUTextureView,
    atrousDirectSigmas: [128.0, 5.0, 0.05] as const,
    readAccum: {} as GPUTexture,
    isMoving,
    wgX16: 4,
    wgY16: 2,
    computeDesc: (label) => ({ label }),
  };
}

// ── Tests ──────────────────────────────────────────────────────────────────

beforeEach(() => {
  acquireOIDNSession.mockClear();
  denoiseFinal.mockClear();
  leaseRelease.mockClear();
  acquireOIDNSession.mockImplementation(async () => ({ release: leaseRelease }));
});

describe('OIDNFinalDenoiser — disabled flag (W11)', () => {
  it('defaults to disabled=true when no modelUrl is supplied (back-compat placeholder)', () => {
    const d = new OIDNFinalDenoiser();
    expect(d.disabled).toBe(true);
  });

  it('flips to disabled=false when a modelUrl is supplied', () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/models/oidn.onnx' });
    expect(d.disabled).toBe(false);
  });

  it('treats empty-string modelUrl as "no modelUrl" → disabled=true', () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '' });
    expect(d.disabled).toBe(true);
  });
});

describe('OIDNFinalDenoiser.initialize', () => {
  it('acquires an OIDN session lease with the supplied URL', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/models/test-model.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    expect(acquireOIDNSession).toHaveBeenCalledTimes(1);
    const callArg = acquireOIDNSession.mock.calls[0]?.[0] as { modelUrl: string };
    expect(callArg.modelUrl).toBe('/models/test-model.onnx');
  });

  it('forwards executionProviders override when supplied', async () => {
    const d = new OIDNFinalDenoiser({
      modelUrl: '/models/m.onnx',
      executionProviders: ['wasm'],
    });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const callArg = acquireOIDNSession.mock.calls[0]?.[0] as {
      modelUrl: string;
      executionProviders?: string[];
    };
    expect(callArg.executionProviders).toEqual(['wasm']);
  });

  it('creates the owned denoised-output texture on the device', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    expect(device.createTexture).toHaveBeenCalled();
    const createCall = (device.createTexture as ReturnType<typeof vi.fn>).mock.calls[0]?.[0];
    expect(createCall.label).toBe('oidn-final-denoised-output');
    expect(createCall.format).toBe('rgba16float');
    expect(createCall.size).toEqual([64, 32]);
  });

  it('throws when initialize is called on a disabled (no-modelUrl) placeholder', async () => {
    const d = new OIDNFinalDenoiser();
    const device = fakeDevice();
    await expect(d.initialize(fakeInitCtx(device))).rejects.toThrow(/cannot initialize a placeholder/i);
  });

  it('rolls back a failed candidate and can retry without publishing poisoned state', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/retry-model.onnx' });
    const device = fakeDevice();
    acquireOIDNSession.mockRejectedValueOnce(new Error('mock preload failure'));

    await expect(d.initialize(fakeInitCtx(device))).rejects.toThrow('mock preload failure');
    const failedTexture = (device.createTexture as ReturnType<typeof vi.fn>)
      .mock.results[0]?.value as GPUTexture;
    expect(failedTexture.destroy).toHaveBeenCalledTimes(1);
    expect(leaseRelease).not.toHaveBeenCalled();
    expect(d.state()).toEqual({
      status: 'failed',
      reason: 'OIDN preload failed: mock preload failure',
      retryable: true,
    });

    await d.initialize(fakeInitCtx(device));
    const liveTexture = (device.createTexture as ReturnType<typeof vi.fn>)
      .mock.results[1]?.value as GPUTexture;
    expect(liveTexture.destroy).not.toHaveBeenCalled();
    expect(d.state()).toEqual({
      status: 'fallback',
      reason: 'waiting for first OIDN output',
    });
  });

  it('releases a late session and destroys the candidate when dispose wins initialization', async () => {
    let resolveAcquire!: (
      lease: Awaited<ReturnType<typeof acquireOIDNSession>>,
    ) => void;
    acquireOIDNSession.mockImplementationOnce(
      () => new Promise((resolve) => { resolveAcquire = resolve; }),
    );
    const d = new OIDNFinalDenoiser({ modelUrl: '/late-model.onnx' });
    const device = fakeDevice();

    const initializing = d.initialize(fakeInitCtx(device));
    await Promise.resolve();
    const candidateTexture = (device.createTexture as ReturnType<typeof vi.fn>)
      .mock.results[0]?.value as GPUTexture;
    d.dispose();
    d.dispose();
    expect(leaseRelease).not.toHaveBeenCalled();
    await expect(d.initialize(fakeInitCtx(device))).rejects.toThrow(/already in progress/i);

    resolveAcquire({ release: leaseRelease });
    await expect(initializing).rejects.toThrow(/cancelled by dispose/i);
    expect(candidateTexture.destroy).toHaveBeenCalledTimes(1);
    expect(leaseRelease).toHaveBeenCalledTimes(1);
    expect(d.state()).toEqual({
      status: 'fallback',
      reason: 'OIDN denoiser has been disposed',
    });
  });
});

describe('OIDNFinalDenoiser.dispatch', () => {
  it('on first dispatch returns the raw HDR texture (no inference completed yet)', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const encoder = fakeEncoder();
    const hdr = fakeTexture();
    const albedo = fakeTexture();
    const normal = fakeTexture();
    const ctx = fakeDispatchCtx(device, encoder, hdr, albedo, normal);

    const out = d.dispatch(ctx);
    // First dispatch hasn't awaited the background chain yet → returns raw HDR.
    expect(out).toBe(hdr);
  });

  it('issues 3 copyTextureToBuffer calls (color + albedo + normal) after frame submit', () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    // Cannot await init because the test wants synchronous dispatch behaviour;
    // we initialise first then dispatch.
    return d.initialize(fakeInitCtx(device)).then(() => {
      const encoder = fakeEncoder();
      const ctx = fakeDispatchCtx(device, encoder, fakeTexture(), fakeTexture(), fakeTexture());
      d.dispatch(ctx);
      d.afterFrameSubmit();

      // Three readback copies (one per aux input).
      const readbackEncoder = (device.createCommandEncoder as ReturnType<typeof vi.fn>)
        .mock.results[0]?.value as GPUCommandEncoder;
      expect(readbackEncoder.copyTextureToBuffer).toHaveBeenCalledTimes(3);
    });
  });

  it('calls denoiseFinal with color + albedo + normal aux inputs (W11 expected aux set)', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const encoder = fakeEncoder();
    const ctx = fakeDispatchCtx(device, encoder, fakeTexture(), fakeTexture(), fakeTexture());
    d.dispatch(ctx);
    d.afterFrameSubmit();

    // The background chain is awaited via microtasks. Let pending promises flush.
    // Two await boundaries (mapAsync × 3 then denoiseFinal); a single
    // round-trip through the microtask queue is enough.
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    expect(denoiseFinal).toHaveBeenCalledTimes(1);
    const callArg = denoiseFinal.mock.calls[0]?.[0] as {
      color: Float32Array;
      albedo?: Float32Array;
      normal?: Float32Array;
      width: number;
      height: number;
    };
    expect(callArg.color).toBeInstanceOf(Float32Array);
    expect(callArg.albedo).toBeInstanceOf(Float32Array);
    expect(callArg.normal).toBeInstanceOf(Float32Array);
    expect(callArg.width).toBe(64);
    expect(callArg.height).toBe(32);
    // 64×32 pixels × 3 channels.
    expect(callArg.color.length).toBe(64 * 32 * 3);
  });

  it('returns the owned denoised texture after a successful inference round-trip', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const encoder = fakeEncoder();
    const hdr = fakeTexture();
    const ctx = fakeDispatchCtx(device, encoder, hdr, fakeTexture(), fakeTexture());

    // Kick the cycle and wait for the background chain to complete.
    const firstOut = d.dispatch(ctx);
    expect(firstOut).toBe(hdr); // first call: no inference completed yet
    d.afterFrameSubmit();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
    // queue.writeTexture should have been called (upload of denoised result).
    expect(device.queue.writeTexture).toHaveBeenCalled();

    // Second dispatch — _haveDenoisedOutput is now true → returns owned texture.
    const secondOut = d.dispatch(
      fakeDispatchCtx(device, encoder, hdr, fakeTexture(), fakeTexture(), 1),
    );
    expect(secondOut).not.toBeNull();
    expect(secondOut).not.toBe(hdr); // it's now the owned denoised texture
  });

  it('drops an in-flight pre-reset result instead of publishing stale pixels', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    let releaseInference!: () => void;
    denoiseFinal.mockImplementationOnce(async (inputs: { color: Float32Array }) => {
      await new Promise<void>((resolve) => { releaseInference = resolve; });
      return new Float32Array(inputs.color);
    });

    const encoder = fakeEncoder();
    const hdr = fakeTexture();
    const stableCtx = fakeDispatchCtx(
      device,
      encoder,
      hdr,
      fakeTexture(),
      fakeTexture(),
      8,
    );
    expect(d.dispatch(stableCtx)).toBe(hdr);
    d.afterFrameSubmit();

    for (let i = 0; i < 8 && denoiseFinal.mock.calls.length === 0; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    const resetCtx = fakeDispatchCtx(
      device,
      encoder,
      hdr,
      fakeTexture(),
      fakeTexture(),
      0,
    );
    expect(d.dispatch(resetCtx)).toBe(hdr);
    releaseInference();
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(device.queue.writeTexture).not.toHaveBeenCalled();
    expect(d.state()).toEqual({
      status: 'fallback',
      reason: 'waiting for first OIDN output',
    });
  });

  it('skips stale result upload when resize happens during an in-flight inference cycle', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    let releaseInference!: () => void;
    denoiseFinal.mockImplementationOnce(async (inputs: { color: Float32Array }) => {
      await new Promise<void>((resolve) => { releaseInference = resolve; });
      return new Float32Array(inputs.color);
    });

    const encoder = fakeEncoder();
    const hdr = fakeTexture();
    const ctx = fakeDispatchCtx(device, encoder, hdr, fakeTexture(), fakeTexture());
    expect(d.dispatch(ctx)).toBe(hdr);
    d.afterFrameSubmit();

    for (let i = 0; i < 8 && denoiseFinal.mock.calls.length === 0; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    d.resize(128, 64);
    releaseInference();
    await new Promise((resolve) => setTimeout(resolve, 0));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(device.queue.writeTexture).not.toHaveBeenCalled();
    expect(d.state()).toEqual({
      status: 'fallback',
      reason: 'waiting for first OIDN output',
    });
  });

  it('falls back to raw HDR, reports a retryable failed state, and retries after dispatch-time OIDN failure', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const encoder = fakeEncoder();
    const hdr = fakeTexture();
    const ctx = fakeDispatchCtx(device, encoder, hdr, fakeTexture(), fakeTexture());
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      denoiseFinal
        .mockRejectedValueOnce(new Error('mock OIDN dispatch failure'))
        .mockImplementationOnce(async (inputs: { color: Float32Array }) => new Float32Array(inputs.color));

      const firstOut = d.dispatch(ctx);
      expect(firstOut).toBe(hdr);
      d.afterFrameSubmit();
      await new Promise((r) => setTimeout(r, 0));
      await new Promise((r) => setTimeout(r, 0));

      expect(d.state()).toEqual({
        status: 'failed',
        reason: 'OIDN inference cycle failed: mock OIDN dispatch failure',
        retryable: true,
      });
      expect(device.queue.writeTexture).not.toHaveBeenCalled();

      const retryOut = d.dispatch(ctx);
      expect(retryOut).toBe(hdr);
      d.afterFrameSubmit();
      await new Promise((r) => setTimeout(r, 0));
      await new Promise((r) => setTimeout(r, 0));

      expect(denoiseFinal).toHaveBeenCalledTimes(2);
      expect(d.state()).toEqual({ status: 'ready' });
      expect(device.queue.writeTexture).toHaveBeenCalledTimes(1);
    } finally {
      errorSpy.mockRestore();
    }
  });

  it('emits a structured host warning instead of console-only reporting after dispatch-time OIDN failure', async () => {
    const warnings: EngineWarning[] = [];
    const d = new OIDNFinalDenoiser({
      modelUrl: '/m.onnx',
      onWarning: (warning) => warnings.push(warning),
    });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const encoder = fakeEncoder();
    const hdr = fakeTexture();
    const ctx = fakeDispatchCtx(device, encoder, hdr, fakeTexture(), fakeTexture());
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      denoiseFinal.mockRejectedValueOnce(new Error('mock OIDN host-visible failure'));

      const out = d.dispatch(ctx);
      expect(out).toBe(hdr);
      d.afterFrameSubmit();
      await new Promise((r) => setTimeout(r, 0));
      await new Promise((r) => setTimeout(r, 0));

      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toMatchObject({
        code: 'walkaround-hybrid.oidn-final-inference-failed',
        backend: 'walkaround-hybrid',
        phase: 'renderFrame',
        method: 'renderFrame',
        details: {
          reason: 'OIDN inference cycle failed: mock OIDN host-visible failure',
          modelUrl: '/m.onnx',
          width: 64,
          height: 32,
          fallback: 'hdrColorTexture',
          retryable: true,
        },
      });
      expect(errorSpy).not.toHaveBeenCalled();
      expect(d.state()).toEqual({
        status: 'failed',
        reason: 'OIDN inference cycle failed: mock OIDN host-visible failure',
        retryable: true,
      });
    } finally {
      errorSpy.mockRestore();
    }
  });

  it('latches after three persistent failures instead of readback/inference every frame forever', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/bad.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));
    denoiseFinal.mockRejectedValue(new Error('persistent model failure'));
    const hdr = fakeTexture();
    const ctx = fakeDispatchCtx(device, fakeEncoder(), hdr, fakeTexture(), fakeTexture());
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      for (let attempt = 0; attempt < 4; attempt += 1) {
        expect(d.dispatch(ctx)).toBe(hdr);
        d.afterFrameSubmit();
        await new Promise((r) => setTimeout(r, 0));
        await new Promise((r) => setTimeout(r, 0));
      }
      expect(denoiseFinal).toHaveBeenCalledTimes(3);
      expect(d.state()).toEqual({
        status: 'failed',
        reason: 'OIDN inference cycle failed: persistent model failure',
        retryable: false,
      });
    } finally {
      errorSpy.mockRestore();
    }
  });
});

describe('OIDNFinalDenoiser.dispose', () => {
  it('releases its OIDN session lease exactly once', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    expect(leaseRelease).not.toHaveBeenCalled();
    d.dispose();
    d.dispose();
    expect(leaseRelease).toHaveBeenCalledTimes(1);
  });

  it('destroys the owned denoised-output texture', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    // The last-created texture is the denoised output.
    const createCalls = (device.createTexture as ReturnType<typeof vi.fn>).mock.results;
    const ownedTex = createCalls[createCalls.length - 1]?.value as GPUTexture;
    expect(ownedTex).toBeDefined();

    d.dispose();
    expect(ownedTex.destroy).toHaveBeenCalled();
  });

  it('is safe to call before initialize (no-op on undisposed state)', () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    expect(() => d.dispose()).not.toThrow();
    expect(leaseRelease).not.toHaveBeenCalled();
  });

  it('defers exactly-once lease release until an active inference settles', async () => {
    let resolveInference!: () => void;
    denoiseFinal.mockImplementationOnce(
      (_inputs: { color: Float32Array }) => new Promise((resolve) => {
        resolveInference = () => resolve(new Float32Array(64 * 32 * 3));
      }),
    );
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));
    const hdr = fakeTexture();
    d.dispatch(fakeDispatchCtx(device, fakeEncoder(), hdr, fakeTexture(), fakeTexture()));
    d.afterFrameSubmit();

    for (let i = 0; i < 8 && denoiseFinal.mock.calls.length === 0; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    expect(denoiseFinal).toHaveBeenCalledTimes(1);

    d.dispose();
    d.dispose();
    expect(leaseRelease).not.toHaveBeenCalled();
    resolveInference();
    for (let i = 0; i < 8 && leaseRelease.mock.calls.length === 0; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    expect(device.queue.writeTexture).not.toHaveBeenCalled();
    expect(leaseRelease).toHaveBeenCalledTimes(1);
  });
});

describe('OIDNFinalDenoiser — registry integration', () => {
  it('DenoiserRegistry.lookup("oidn-final") succeeds when oidn config is threaded', () => {
    const reg = new DenoiserRegistry();
    registerBuiltinDenoisers(reg, { oidn: { modelUrl: '/m.onnx' } });
    const d = reg.lookup('oidn-final');
    expect(d.id).toBe('oidn-final');
    expect(d.disabled).toBe(false);
  });

  it('forwards the host warning sink into the registered OIDN denoiser', () => {
    const onWarning = vi.fn();
    const reg = new DenoiserRegistry();
    registerBuiltinDenoisers(reg, {
      oidn: { modelUrl: '/m.onnx' },
      onWarning,
    });

    const oidn = reg.lookup('oidn-final') as unknown as {
      _onWarning: ((warning: EngineWarning) => void) | null;
    };
    expect(oidn._onWarning).toBe(onWarning);
  });

  it('DenoiserRegistry.lookup("oidn-final") throws "registered but disabled" without oidn config', () => {
    const reg = new DenoiserRegistry();
    registerBuiltinDenoisers(reg);
    expect(() => reg.lookup('oidn-final')).toThrow(/registered but disabled/);
  });
});

describe('HybridEngine — W11 OIDN model URL validation', () => {
  function mockDevice(): GPUDevice {
    return {
      createCommandEncoder: () => ({}),
      createBuffer: () => ({}),
      createShaderModule: () => ({}),
      createComputePipeline: () => ({}),
      createBindGroupLayout: () => ({}),
      createBindGroup: () => ({}),
      createPipelineLayout: () => ({}),
      queue: { writeBuffer: () => {}, submit: () => {} },
      addEventListener: () => {},
      removeEventListener: () => {},
      lost: new Promise<never>(() => {}),
    } as unknown as GPUDevice;
  }

  it('throws when denoiser="oidn-final" but extensions["walkaround-hybrid"].oidnModelUrl is missing', async () => {
    const { HybridEngine } = await import('../src/HybridEngine.js');
    expect(() => new HybridEngine({
      device: mockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [0.2, 0.4, 0.8],
      skyIrradiance: 0.5,
      denoiser: 'oidn-final',
      // extensions intentionally omitted
    })).toThrow(/oidn-final.*oidnModelUrl/i);
  });

  it('accepts denoiser="oidn-final" when extensions["walkaround-hybrid"].oidnModelUrl is supplied', async () => {
    const { HybridEngine } = await import('../src/HybridEngine.js');
    expect(() => new HybridEngine({
      device: mockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [0.2, 0.4, 0.8],
      skyIrradiance: 0.5,
      denoiser: 'oidn-final',
      extensions: {
        'walkaround-hybrid': { oidnModelUrl: '/models/oidn.onnx' },
      },
    })).not.toThrow();
  });
});
