import { describe, expect, it, afterEach } from 'vitest';
import { extractGpuLimits, probeWebGPU, resetGpuDetectionCache } from '../gpuDetection.js';

describe('extractGpuLimits — portable GPUSupportedLimits reader', () => {
  it('reads canonical limits from a plain enumerable object (browser shape)', () => {
    const limits = {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
      maxBindGroups: 4,
    };
    const out = extractGpuLimits(limits);
    expect(out['maxStorageBuffersPerShaderStage']).toBe(16);
    expect(out['maxStorageTexturesPerShaderStage']).toBe(8);
    expect(out['maxBindGroups']).toBe(4);
  });

  it('reads NON-ENUMERABLE prototype getters (native-wgpu shape — Deno/wgpu-py)', () => {
    // The exact failure mode the createProgressiveEngine dzn run hit:
    // Object.keys(limits) === [] but direct property access works.
    const proto = {};
    Object.defineProperty(proto, 'maxStorageBuffersPerShaderStage', {
      get: () => 1_000_000,
      enumerable: false,
    });
    Object.defineProperty(proto, 'maxStorageTexturesPerShaderStage', {
      get: () => 1_000_000,
      enumerable: false,
    });
    const limits: object = Object.create(proto) as object;
    // Sanity: the enumeration-only extractor would have produced {} here.
    expect(Object.keys(limits)).toHaveLength(0);

    const out = extractGpuLimits(limits);
    expect(out['maxStorageBuffersPerShaderStage']).toBe(1_000_000);
    expect(out['maxStorageTexturesPerShaderStage']).toBe(1_000_000);
  });

  it('merges extra enumerable own-keys not in the canonical set (future/vendor limits)', () => {
    const limits = {
      maxStorageBuffersPerShaderStage: 32,
      someVendorLimit: 12345,
    };
    const out = extractGpuLimits(limits);
    expect(out['maxStorageBuffersPerShaderStage']).toBe(32);
    expect(out['someVendorLimit']).toBe(12345);
  });

  it('drops non-numeric / non-finite values and handles null', () => {
    expect(extractGpuLimits(null)).toEqual({});
    expect(extractGpuLimits(undefined)).toEqual({});
    const out = extractGpuLimits({
      maxBindGroups: 4,
      maxBufferSize: Number.POSITIVE_INFINITY, // not finite → dropped
      bogus: 'not-a-number',
    });
    expect(out['maxBindGroups']).toBe(4);
    expect('maxBufferSize' in out).toBe(false);
    expect('bogus' in out).toBe(false);
  });
});

describe('Bug6 fix — probeWebGPU carries error reason on adapter exception', () => {
  const origNavigator = globalThis.navigator;
  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', { value: origNavigator, configurable: true });
    resetGpuDetectionCache();
  });

  it('returns supported:false with no reason when navigator.gpu is absent', async () => {
    Object.defineProperty(globalThis, 'navigator', { value: {}, configurable: true });
    const result = await probeWebGPU();
    expect(result.supported).toBe(false);
    expect(result.reason).toBeUndefined();
  });

  it('returns supported:false with no reason when requestAdapter returns null', async () => {
    Object.defineProperty(globalThis, 'navigator', {
      value: { gpu: { requestAdapter: async () => null } },
      configurable: true,
    });
    const result = await probeWebGPU();
    expect(result.supported).toBe(false);
    expect(result.reason).toBeUndefined();
  });

  it('returns supported:false WITH a reason string when requestAdapter throws', async () => {
    const boom = new Error('driver crash: GPU device lost');
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: async () => { throw boom; },
        },
      },
      configurable: true,
    });
    const result = await probeWebGPU();
    expect(result.supported).toBe(false);
    // The reason must be present and carry the error message so callers can
    // distinguish a transient failure from a no-WebGPU environment.
    expect(result.reason).toBeDefined();
    expect(result.reason).toMatch(/driver crash/);
  });

  it('reason is the stringified error when the thrown value is not an Error', async () => {
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: async () => { throw new Error('adapter exploded'); },
        },
      },
      configurable: true,
    });
    const result = await probeWebGPU();
    expect(result.supported).toBe(false);
    expect(result.reason).toMatch(/adapter exploded/);
  });
});
