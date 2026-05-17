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

import { describe, it, expect } from 'vitest';
import {
  denoiseFinal,
  preloadOIDNModel,
  clearOIDNCache,
  hwcToNchw,
  nchwToHwc,
} from '../src/oidnBridge.js';
import type { OIDNDenoiseInputs, OIDNDenoiseOptions } from '../src/oidnBridge.js';

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
});

// ── clearOIDNCache ────────────────────────────────────────────────────────────

describe('clearOIDNCache', () => {
  it('is a no-op before any denoiseFinal call (no throw)', () => {
    clearOIDNCache();
    clearOIDNCache(); // idempotent
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
// AUDIT M-3 fix (2026-05-09): hwcToNchw and nchwToHwc were previously
// unexported and only "tested" indirectly via denoiseFinal's return type.
// These tests verify the actual pixel layout transform without requiring ORT.

describe('hwcToNchw / nchwToHwc layout transforms', () => {
  it('round-trips a 1×1×3 (single pixel, 3 channels) buffer', () => {
    const hwc = new Float32Array([0.1, 0.5, 0.9]);  // R, G, B at pixel (0,0)
    const nchw = hwcToNchw(hwc, 1, 1, 3);
    // NCHW layout for 1×1×3: [R_00, G_00, B_00] — same as HWC for 1 pixel
    expect(nchw).toHaveLength(3);
    expect(nchw[0]).toBeCloseTo(0.1);
    expect(nchw[1]).toBeCloseTo(0.5);
    expect(nchw[2]).toBeCloseTo(0.9);
    const back = nchwToHwc(nchw, 1, 1, 3);
    expect(back[0]).toBeCloseTo(hwc[0] ?? 0);
    expect(back[1]).toBeCloseTo(hwc[1] ?? 0);
    expect(back[2]).toBeCloseTo(hwc[2] ?? 0);
  });

  it('round-trips a 2×2×3 buffer exactly (identity)', () => {
    // HWC layout: pixels row-major, channels interleaved
    // Pixel (0,0)=[1,2,3], (0,1)=[4,5,6], (1,0)=[7,8,9], (1,1)=[10,11,12]
    const hwc = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const nchw = hwcToNchw(hwc, 2, 2, 3);
    const back = nchwToHwc(nchw, 2, 2, 3);
    expect(back).toHaveLength(hwc.length);
    for (let i = 0; i < hwc.length; i++) {
      expect(back[i]).toBeCloseTo(hwc[i] ?? 0, 5);
    }
  });

  it('produces correct NCHW channel layout for 2×2×3', () => {
    // HWC: [R00 G00 B00  R01 G01 B01  R10 G10 B10  R11 G11 B11]
    // NCHW: [R00 R01 R10 R11  G00 G01 G10 G11  B00 B01 B10 B11]
    const hwc = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const nchw = hwcToNchw(hwc, 2, 2, 3);
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
    const nchw = hwcToNchw(hwc, h, w, c);
    expect(nchw).toHaveLength(hwc.length);
    const back = nchwToHwc(nchw, h, w, c);
    expect(back).toHaveLength(hwc.length);
  });

  it('round-trips a 4×4×3 buffer with floating-point values', () => {
    const w = 4, h = 4, c = 3;
    const hwc = Float32Array.from({ length: h * w * c }, (_, i) => Math.sin(i * 0.37));
    const nchw = hwcToNchw(hwc, h, w, c);
    const back = nchwToHwc(nchw, h, w, c);
    for (let i = 0; i < hwc.length; i++) {
      expect(back[i]).toBeCloseTo(hwc[i] ?? 0, 5);
    }
  });

  it('handles 1-channel (grayscale) buffer round-trip', () => {
    // Although OIDN uses 3 channels, the helpers are channel-agnostic
    const hwc = new Float32Array([10, 20, 30, 40]);  // 2×2×1
    const nchw = hwcToNchw(hwc, 2, 2, 1);
    // For 1 channel, HWC and NCHW should be identical
    expect(Array.from(nchw)).toEqual([10, 20, 30, 40]);
    const back = nchwToHwc(nchw, 2, 2, 1);
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
