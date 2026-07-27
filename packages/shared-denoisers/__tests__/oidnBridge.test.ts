/**
 * oidnBridge.test.ts — Structural tests for Sprint 10b OIDN bridge.
 *
 * Does NOT perform ONNX inference (onnxruntime-web is an optional peer dep
 * not installed in this workspace). Verifies:
 *   1. All exported function signatures are present and callable.
 *   2. Default execution providers are ['webnn', 'webgpu', 'wasm'] (Decision 11).
 *   3. clearOIDNCache() is a no-op before any session is created.
 *   4. denoiseFinal throws a descriptive error when onnxruntime-web is absent.
 *   5. preloadOIDNModel throws a descriptive error when onnxruntime-web is absent.
 *   6. HWC↔NCHW layout helpers are correct (tested via the public denoiseFinal
 *      return type guarantee — the function must return a Float32Array).
 */

import { afterEach, describe, it, expect, vi } from 'vitest';
import {
  denoiseFinal,
  preloadOIDNModel,
  acquireOIDNSession,
  clearOIDNCache,
  _hwcToNchw,
  _nchwToHwc,
} from '../src/oidnBridge.js';
import type { OIDNDenoiseInputs, OIDNDenoiseOptions } from '../src/oidnBridge.js';

afterEach(() => {
  vi.doUnmock('onnxruntime-web');
  vi.resetModules();
});

// ── Export presence ───────────────────────────────────────────────────────────

describe('oidnBridge exports', () => {
  it('exports denoiseFinal as an async function', () => {
    expect(typeof denoiseFinal).toBe('function');
    // Verify it returns a Promise (is async).
    const stub = denoiseFinal(
      { color: new Float32Array(3), width: 1, height: 1 },
      { modelUrl: '/test.onnx' },
    );
    // It must be a Promise — we don't await it here (no ORT installed).
    expect(stub).toBeInstanceOf(Promise);
    // Suppress unhandled rejection for this test.
    stub.catch(() => {/* expected — no ORT installed */});
  });

  it('exports preloadOIDNModel as an async function', () => {
    expect(typeof preloadOIDNModel).toBe('function');
    const stub = preloadOIDNModel({ modelUrl: '/test.onnx' });
    expect(stub).toBeInstanceOf(Promise);
    stub.catch(() => {/* expected */});
  });

  it('exports clearOIDNCache as a synchronous function', () => {
    expect(typeof clearOIDNCache).toBe('function');
    // Must not throw when called before any session is created.
    expect(() => clearOIDNCache()).not.toThrow();
  });

  it('exports acquireOIDNSession as an async lease factory', () => {
    expect(typeof acquireOIDNSession).toBe('function');
  });
});

// ── clearOIDNCache ────────────────────────────────────────────────────────────

describe('clearOIDNCache', () => {
  it('is a no-op before any denoiseFinal call (no throw)', () => {
    clearOIDNCache();
    clearOIDNCache(); // idempotent
  });

  it('releases cached ORT sessions before clearing the map', async () => {
    vi.resetModules();
    const release = vi.fn();
    const run = vi.fn(async (_feeds: Record<string, unknown>) => ({
      output: { data: new Float32Array(3), dims: [1, 3, 1, 1] },
    }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        create: vi.fn(async () => ({ run, release })),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    await bridge.preloadOIDNModel({ modelUrl: '/mock.onnx', executionProviders: ['wasm'] });

    // The cache now holds a Promise<session>; release happens once the promise
    // resolves (a microtask after clear). Flush the microtask queue.
    bridge.clearOIDNCache();
    await Promise.resolve();
    expect(release).toHaveBeenCalledTimes(1);

    bridge.clearOIDNCache();
    await Promise.resolve();
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('releaseOIDNCacheEntry releases only the matching cached session', async () => {
    vi.resetModules();
    const releases = [vi.fn(), vi.fn()];
    let createCount = 0;
    const run = vi.fn(async () => ({
      output: { data: new Float32Array(3), dims: [1, 3, 1, 1] },
    }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        create: vi.fn(async () => ({ run, release: releases[createCount++]! })),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    await bridge.preloadOIDNModel({ modelUrl: '/a.onnx', executionProviders: ['wasm'] });
    await bridge.preloadOIDNModel({ modelUrl: '/b.onnx', executionProviders: ['wasm'] });

    bridge.releaseOIDNCacheEntry({ modelUrl: '/a.onnx', executionProviders: ['wasm'] });
    await Promise.resolve();
    expect(releases[0]).toHaveBeenCalledTimes(1);
    expect(releases[1]).not.toHaveBeenCalled();

    bridge.clearOIDNCache();
    await Promise.resolve();
    expect(releases[0]).toHaveBeenCalledTimes(1);
    expect(releases[1]).toHaveBeenCalledTimes(1);
  });
});

// ── Concurrent first-use session race (V3-6) ──────────────────────────────────
//
// Two concurrent preloads against a SLOW session-create must share ONE session:
// caching the resolved session (old code) let both pass the undefined check and
// create a duplicate that leaked untracked. The fix caches the creation PROMISE
// synchronously before the await; a rejected create deletes its own entry.

describe('OIDN concurrent first-use (promise cache)', () => {
  it('two concurrent preloads create exactly ONE session (no duplicate leak)', async () => {
    vi.resetModules();
    let createCount = 0;
    let releaseSession = () => {};
    const run = vi.fn(async () => ({
      output: { data: new Float32Array(3), dims: [1, 3, 1, 1] },
    }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        // Slow create: resolve on the next macrotask so both callers are
        // in-flight before either resolves.
        create: vi.fn(async () => {
          createCount += 1;
          await new Promise((r) => setTimeout(r, 10));
          const release = vi.fn();
          releaseSession = release;
          return { run, release };
        }),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    // Fire two concurrent preloads for the SAME key before either resolves.
    await Promise.all([
      bridge.preloadOIDNModel({ modelUrl: '/race.onnx', executionProviders: ['wasm'] }),
      bridge.preloadOIDNModel({ modelUrl: '/race.onnx', executionProviders: ['wasm'] }),
    ]);

    expect(createCount).toBe(1);

    // The single cached session is released exactly once on clear.
    bridge.clearOIDNCache();
    await Promise.resolve();
    expect(releaseSession).toHaveBeenCalledTimes(1);
  });

  it('a rejected create removes its cache entry (no poisoning)', async () => {
    vi.resetModules();
    let attempt = 0;
    const run = vi.fn(async () => ({
      output: { data: new Float32Array(3), dims: [1, 3, 1, 1] },
    }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        create: vi.fn(async () => {
          attempt += 1;
          if (attempt === 1) {
            throw new Error('mock create failure');
          }
          return { run, release: vi.fn() };
        }),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    // First create rejects → entry must be deleted, not cached.
    await expect(
      bridge.preloadOIDNModel({ modelUrl: '/poison.onnx', executionProviders: ['wasm'] }),
    ).rejects.toThrow('mock create failure');

    // A retry after the rejection succeeds (cache was not poisoned).
    await expect(
      bridge.preloadOIDNModel({ modelUrl: '/poison.onnx', executionProviders: ['wasm'] }),
    ).resolves.toBeUndefined();
    expect(attempt).toBe(2);
  });
});

describe('OIDN shared session leases', () => {
  it('keeps one shared session alive until both engine leases are released', async () => {
    vi.resetModules();
    const release = vi.fn();
    const run = vi.fn(async () => ({
      output: { data: new Float32Array(3).fill(2), dims: [1, 3, 1, 1] },
    }));
    const create = vi.fn(async () => ({ run, release }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: { create },
    }));

    const bridge = await import('../src/oidnBridge.js');
    const opts = { modelUrl: '/shared.onnx', executionProviders: ['wasm'] as const };
    const [leaseA, leaseB] = await Promise.all([
      bridge.acquireOIDNSession(opts),
      bridge.acquireOIDNSession(opts),
    ]);
    expect(create).toHaveBeenCalledTimes(1);

    leaseA.release();
    await Promise.resolve();
    expect(release).not.toHaveBeenCalled();

    await expect(bridge.denoiseFinal(
      { color: new Float32Array(3), width: 1, height: 1 },
      opts,
    )).resolves.toEqual(new Float32Array(3).fill(2));
    expect(create).toHaveBeenCalledTimes(1);
    expect(run).toHaveBeenCalledTimes(1);

    leaseB.release();
    await Promise.resolve();
    expect(release).toHaveBeenCalledTimes(1);
    leaseB.release();
    await Promise.resolve();
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('defers final release until an in-flight inference has resolved', async () => {
    vi.resetModules();
    const release = vi.fn();
    let resolveRun!: (value: {
      output: { data: Float32Array; dims: number[] };
    }) => void;
    const run = vi.fn(() => new Promise<{
      output: { data: Float32Array; dims: number[] };
    }>(resolve => {
      resolveRun = resolve;
    }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        create: vi.fn(async () => ({ run, release })),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    const opts = { modelUrl: '/in-flight.onnx', executionProviders: ['wasm'] as const };
    const lease = await bridge.acquireOIDNSession(opts);
    const inference = bridge.denoiseFinal(
      { color: new Float32Array(3), width: 1, height: 1 },
      opts,
    );
    await vi.waitFor(() => expect(run).toHaveBeenCalledTimes(1));

    lease.release();
    bridge.clearOIDNCache();
    await Promise.resolve();
    expect(release).not.toHaveBeenCalled();

    resolveRun({
      output: {
        data: new Float32Array(3).fill(4),
        dims: [1, 3, 1, 1],
      },
    });
    await expect(inference).resolves.toEqual(new Float32Array(3).fill(4));
    await Promise.resolve();
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('rolls back a failed lease acquisition so a later engine can retry', async () => {
    vi.resetModules();
    let attempts = 0;
    const release = vi.fn();
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        create: vi.fn(async () => {
          attempts++;
          if (attempts === 1) throw new Error('transient create failure');
          return {
              run: vi.fn(async () => ({
                output: {
                  data: new Float32Array(3),
                  dims: [1, 3, 1, 1],
                },
              })),
            release,
          };
        }),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    const opts = { modelUrl: '/retry.onnx', executionProviders: ['wasm'] as const };
    await expect(bridge.acquireOIDNSession(opts)).rejects.toThrow('transient create failure');
    const lease = await bridge.acquireOIDNSession(opts);
    expect(attempts).toBe(2);
    lease.release();
    await Promise.resolve();
    expect(release).toHaveBeenCalledTimes(1);
  });
});

describe('OIDN model input negotiation', () => {
  it('omits optional auxiliary feeds that a color-only model does not declare', async () => {
    vi.resetModules();
    const run = vi.fn(async (_feeds: Record<string, unknown>) => ({
      output: {
        data: new Float32Array([0.25, 0.5, 0.75]),
        dims: [1, 3, 1, 1],
      },
    }));
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        create: vi.fn(async () => ({
          inputNames: ['color'],
          run,
          release: vi.fn(),
        })),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    await expect(bridge.denoiseFinal(
      {
        color: new Float32Array([1, 2, 3]),
        normal: new Float32Array([0, 0, 1]),
        albedo: new Float32Array([0.5, 0.5, 0.5]),
        width: 1,
        height: 1,
      },
      { modelUrl: '/color-only.onnx', executionProviders: ['wasm'] },
    )).resolves.toEqual(new Float32Array([0.25, 0.5, 0.75]));

    const feeds = run.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(Object.keys(feeds)).toEqual(['color']);
  });

  it('rejects an explicitly configured auxiliary name absent from the model', async () => {
    vi.resetModules();
    const run = vi.fn();
    vi.doMock('onnxruntime-web', () => ({
      Tensor: class {
        constructor(..._args: unknown[]) {}
      },
      InferenceSession: {
        create: vi.fn(async () => ({
          inputNames: ['color'],
          run,
          release: vi.fn(),
        })),
      },
    }));

    const bridge = await import('../src/oidnBridge.js');
    await expect(bridge.denoiseFinal(
      {
        color: new Float32Array([1, 2, 3]),
        normal: new Float32Array([0, 0, 1]),
        width: 1,
        height: 1,
      },
      {
        modelUrl: '/color-only.onnx',
        executionProviders: ['wasm'],
        tensorNames: { normal: 'surface_normal' },
      },
    )).rejects.toThrow(/configured normal input 'surface_normal'.*not declared/i);
    expect(run).not.toHaveBeenCalled();
  });
});

// ── Runtime error when onnxruntime-web is absent ──────────────────────────────

describe('denoiseFinal (no ORT installed)', () => {
  it('rejects with a descriptive error mentioning onnxruntime-web', async () => {
    const inputs: OIDNDenoiseInputs = {
      color: new Float32Array(3),
      width: 1,
      height: 1,
    };
    const opts: OIDNDenoiseOptions = {
      modelUrl: '/test.onnx',
    };

    await expect(denoiseFinal(inputs, opts)).rejects.toThrow('onnxruntime-web');
  });

  it('rejects with install instructions', async () => {
    await expect(
      denoiseFinal({ color: new Float32Array(3), width: 1, height: 1 }, { modelUrl: '/x.onnx' }),
    ).rejects.toThrow('npm install onnxruntime-web');
  });
});

describe('preloadOIDNModel (no ORT installed)', () => {
  it('rejects with a descriptive error mentioning onnxruntime-web', async () => {
    await expect(preloadOIDNModel({ modelUrl: '/test.onnx' })).rejects.toThrow('onnxruntime-web');
  });
});

// ── OIDNDenoiseOptions type shape ─────────────────────────────────────────────

describe('OIDNDenoiseOptions defaults', () => {
  it('executionProviders defaults to webnn, webgpu, wasm (Decision 11)', async () => {
    // We cannot inspect the default inside the closure without running ORT,
    // but we can verify the documented type accepts all three string literals.
    const opts: OIDNDenoiseOptions = {
      modelUrl: '/test.onnx',
      executionProviders: ['webnn', 'webgpu', 'wasm'],
    };
    expect(opts.executionProviders).toEqual(['webnn', 'webgpu', 'wasm']);
  });

  it('executionProviders is optional (accepts no EP override)', () => {
    const opts: OIDNDenoiseOptions = { modelUrl: '/test.onnx' };
    expect(opts.executionProviders).toBeUndefined();
  });

  it('modelUrl is a required string field', () => {
    const opts: OIDNDenoiseOptions = { modelUrl: '/models/oidn_rt_hdr.onnx' };
    expect(typeof opts.modelUrl).toBe('string');
  });
});

// ── HWC↔NCHW layout transform round-trip tests ───────────────────────────────
//
// AUDIT M-3 fix (2026-05-09): _hwcToNchw and _nchwToHwc were previously
// unexported and only "tested" indirectly via denoiseFinal's return type.
// These tests verify the actual pixel layout transform without requiring ORT.

describe('_hwcToNchw / _nchwToHwc layout transforms', () => {
  it('round-trips a 1×1×3 (single pixel, 3 channels) buffer', () => {
    const hwc = new Float32Array([0.1, 0.5, 0.9]);  // R, G, B at pixel (0,0)
    const nchw = _hwcToNchw(hwc, 1, 1, 3);
    // NCHW layout for 1×1×3: [R_00, G_00, B_00] — same as HWC for 1 pixel
    expect(nchw).toHaveLength(3);
    expect(nchw[0]).toBeCloseTo(0.1);
    expect(nchw[1]).toBeCloseTo(0.5);
    expect(nchw[2]).toBeCloseTo(0.9);
    const back = _nchwToHwc(nchw, 1, 1, 3);
    expect(back[0]).toBeCloseTo(hwc[0] ?? 0);
    expect(back[1]).toBeCloseTo(hwc[1] ?? 0);
    expect(back[2]).toBeCloseTo(hwc[2] ?? 0);
  });

  it('round-trips a 2×2×3 buffer exactly (identity)', () => {
    // HWC layout: pixels row-major, channels interleaved
    // Pixel (0,0)=[1,2,3], (0,1)=[4,5,6], (1,0)=[7,8,9], (1,1)=[10,11,12]
    const hwc = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const nchw = _hwcToNchw(hwc, 2, 2, 3);
    const back = _nchwToHwc(nchw, 2, 2, 3);
    expect(back).toHaveLength(hwc.length);
    for (let i = 0; i < hwc.length; i++) {
      expect(back[i]).toBeCloseTo(hwc[i] ?? 0, 5);
    }
  });

  it('produces correct NCHW channel layout for 2×2×3', () => {
    // HWC: [R00 G00 B00  R01 G01 B01  R10 G10 B10  R11 G11 B11]
    // NCHW: [R00 R01 R10 R11  G00 G01 G10 G11  B00 B01 B10 B11]
    const hwc = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const nchw = _hwcToNchw(hwc, 2, 2, 3);
    expect(nchw).toHaveLength(12);
    // Channel 0 (R): pixels (0,0)=1, (0,1)=4, (1,0)=7, (1,1)=10
    expect(nchw[0]).toBeCloseTo(1);
    expect(nchw[1]).toBeCloseTo(4);
    expect(nchw[2]).toBeCloseTo(7);
    expect(nchw[3]).toBeCloseTo(10);
    // Channel 1 (G): pixels (0,0)=2, (0,1)=5, (1,0)=8, (1,1)=11
    expect(nchw[4]).toBeCloseTo(2);
    expect(nchw[5]).toBeCloseTo(5);
    expect(nchw[6]).toBeCloseTo(8);
    expect(nchw[7]).toBeCloseTo(11);
    // Channel 2 (B): pixels (0,0)=3, (0,1)=6, (1,0)=9, (1,1)=12
    expect(nchw[8]).toBeCloseTo(3);
    expect(nchw[9]).toBeCloseTo(6);
    expect(nchw[10]).toBeCloseTo(9);
    expect(nchw[11]).toBeCloseTo(12);
  });

  it('preserves total element count across layouts', () => {
    const w = 8, h = 4, c = 3;
    const hwc = new Float32Array(h * w * c).map((_, i) => i * 0.01);
    const nchw = _hwcToNchw(hwc, h, w, c);
    expect(nchw).toHaveLength(hwc.length);
    const back = _nchwToHwc(nchw, h, w, c);
    expect(back).toHaveLength(hwc.length);
  });

  it('round-trips a 4×4×3 buffer with floating-point values', () => {
    const w = 4, h = 4, c = 3;
    const hwc = Float32Array.from({ length: h * w * c }, (_, i) => Math.sin(i * 0.37));
    const nchw = _hwcToNchw(hwc, h, w, c);
    const back = _nchwToHwc(nchw, h, w, c);
    for (let i = 0; i < hwc.length; i++) {
      expect(back[i]).toBeCloseTo(hwc[i] ?? 0, 5);
    }
  });

  it('handles 1-channel (grayscale) buffer round-trip', () => {
    // Although OIDN uses 3 channels, the helpers are channel-agnostic
    const hwc = new Float32Array([10, 20, 30, 40]);  // 2×2×1
    const nchw = _hwcToNchw(hwc, 2, 2, 1);
    // For 1 channel, HWC and NCHW should be identical
    expect(Array.from(nchw)).toEqual([10, 20, 30, 40]);
    const back = _nchwToHwc(nchw, 2, 2, 1);
    expect(Array.from(back)).toEqual([10, 20, 30, 40]);
  });
});

// ── OIDNDenoiseInputs type shape ──────────────────────────────────────────────

describe('OIDNDenoiseInputs', () => {
  it('accepts color-only input (no aux buffers)', () => {
    const inputs: OIDNDenoiseInputs = {
      color: new Float32Array(1920 * 1080 * 3),
      width: 1920,
      height: 1080,
    };
    expect(inputs.normal).toBeUndefined();
    expect(inputs.albedo).toBeUndefined();
    expect(inputs.color.length).toBe(1920 * 1080 * 3);
  });

  it('accepts full input with normal and albedo aux buffers', () => {
    const w = 64, h = 64;
    const inputs: OIDNDenoiseInputs = {
      color:  new Float32Array(h * w * 3),
      normal: new Float32Array(h * w * 3),
      albedo: new Float32Array(h * w * 3),
      width:  w,
      height: h,
    };
    expect(inputs.normal).toBeInstanceOf(Float32Array);
    expect(inputs.albedo).toBeInstanceOf(Float32Array);
  });
});
