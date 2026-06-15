/**
 * oidnFinalDenoiser.test.ts — W11 acceptance tests.
 *
 * Verifies that the OIDNFinalDenoiser wire is real:
 *   1. The disabled flag respects the modelUrl construction-time choice.
 *   2. `initialize` pre-warms the ONNX runtime via `preloadOIDNModel`.
 *   3. `dispatch` kicks off the background readback + inference + upload
 *      chain, calls `denoiseFinal` with color + albedo + normal aux inputs,
 *      and returns a GPU texture (the owned denoised output once an
 *      inference completes, the raw HDR target before that).
 *   4. `dispose` releases GPU resources AND clears the OIDN session cache.
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

const oidnMocks = vi.hoisted(() => ({
  preloadOIDNModel: vi.fn(async (_opts: unknown) => undefined),
  denoiseFinal: vi.fn(async (inputs: { color: Float32Array; width: number; height: number }) => {
    // Echo input back as if OIDN denoised it (identity output is fine for the test).
    return new Float32Array(inputs.color);
  }),
  releaseOIDNCacheEntry: vi.fn(() => undefined),
  clearOIDNCache: vi.fn(() => undefined),
}));
const { preloadOIDNModel, denoiseFinal, releaseOIDNCacheEntry } = oidnMocks;

vi.mock('@vitrum/shared-denoisers', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/shared-denoisers')>();
  return {
    ...actual,
    preloadOIDNModel: oidnMocks.preloadOIDNModel,
    denoiseFinal: oidnMocks.denoiseFinal,
    releaseOIDNCacheEntry: oidnMocks.releaseOIDNCacheEntry,
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
): DenoiserDispatchContext {
  return {
    device,
    encoder,
    width: 64,
    height: 32,
    frameIndex: 0,
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
    isMoving: false,
    wgX16: 4,
    wgY16: 2,
    computeDesc: (label) => ({ label }),
  };
}

// ── Tests ──────────────────────────────────────────────────────────────────

beforeEach(() => {
  preloadOIDNModel.mockClear();
  denoiseFinal.mockClear();
  releaseOIDNCacheEntry.mockClear();
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
  it('preloads the ONNX model via preloadOIDNModel with the supplied URL', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/models/test-model.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    expect(preloadOIDNModel).toHaveBeenCalledTimes(1);
    const callArg = preloadOIDNModel.mock.calls[0]?.[0] as { modelUrl: string };
    expect(callArg.modelUrl).toBe('/models/test-model.onnx');
  });

  it('forwards executionProviders override when supplied', async () => {
    const d = new OIDNFinalDenoiser({
      modelUrl: '/models/m.onnx',
      executionProviders: ['wasm'],
    });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const callArg = preloadOIDNModel.mock.calls[0]?.[0] as {
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

  it('issues 3 copyTextureToBuffer calls (color + albedo + normal) on dispatch', () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    // Cannot await init because the test wants synchronous dispatch behaviour;
    // we initialise first then dispatch.
    return d.initialize(fakeInitCtx(device)).then(() => {
      const encoder = fakeEncoder();
      const ctx = fakeDispatchCtx(device, encoder, fakeTexture(), fakeTexture(), fakeTexture());
      d.dispatch(ctx);

      // Three readback copies (one per aux input).
      expect(encoder.copyTextureToBuffer).toHaveBeenCalledTimes(3);
    });
  });

  it('calls denoiseFinal with color + albedo + normal aux inputs (W11 expected aux set)', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    const encoder = fakeEncoder();
    const ctx = fakeDispatchCtx(device, encoder, fakeTexture(), fakeTexture(), fakeTexture());
    d.dispatch(ctx);

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
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
    // queue.writeTexture should have been called (upload of denoised result).
    expect(device.queue.writeTexture).toHaveBeenCalled();

    // Second dispatch — _haveDenoisedOutput is now true → returns owned texture.
    const secondOut = d.dispatch(ctx);
    expect(secondOut).not.toBeNull();
    expect(secondOut).not.toBe(hdr); // it's now the owned denoised texture
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
      await new Promise((r) => setTimeout(r, 0));
      await new Promise((r) => setTimeout(r, 0));

      expect(denoiseFinal).toHaveBeenCalledTimes(2);
      expect(d.state()).toEqual({ status: 'ready' });
      expect(device.queue.writeTexture).toHaveBeenCalledTimes(1);
    } finally {
      errorSpy.mockRestore();
    }
  });
});

describe('OIDNFinalDenoiser.dispose', () => {
  it('releases the OIDN session cache entry via releaseOIDNCacheEntry', async () => {
    const d = new OIDNFinalDenoiser({ modelUrl: '/m.onnx' });
    const device = fakeDevice();
    await d.initialize(fakeInitCtx(device));

    expect(releaseOIDNCacheEntry).not.toHaveBeenCalled();
    d.dispose();
    expect(releaseOIDNCacheEntry).toHaveBeenCalledTimes(1);
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
    expect(releaseOIDNCacheEntry).toHaveBeenCalled();
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
