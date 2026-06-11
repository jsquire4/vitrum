/**
 * A8 GRIS-variant compile-time gate pin.
 *
 * Verifies that the `HybridEngineOptions.restirPtReuse` flag is correctly
 * threaded through `deriveHybridEngineConfig` into `_cfg.restirPtReuse`
 * (the value that `WalkaroundGPUPipeline.initialize` passes to
 * `compilePipelines({ restirPtReuse: this._cfg.restirPtReuse === 1 })`),
 * which selects the GRIS vs legacy shader variant at compile time.
 *
 * These tests run entirely in the HybridEngine constructor (synchronous)
 * and never touch GPU resources — the test seam is the `_cfg.restirPtReuse`
 * field on the parsed engine config record.
 *
 * Architecture decision (plan/road-to-100.md A8, 2026-06-10):
 *   Default `false` = biased clamped-Jacobian reuse (Sprint-17), retained
 *   for the realtime frame budget. Bias sources: B1 Jacobian clamp [0.1,10]
 *   (jacobianShift.wgsl.ts), B2 no reconnection-visibility ray (OFF variants
 *   of spatialGi/temporalGi), B3 no full-GBH MIS (OFF combine weights), B4
 *   centroid p̂ (restirPHat.wgsl.ts, shared ON/OFF).
 *   `true` = unbiased GRIS (Phase-1 reconnection shift + Phase-2 full-GBH MIS),
 *   compile-time gated (adds @group(1) scene BVH group to GI passes).
 */

import { describe, expect, it } from 'vitest';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';

function fakeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    features: new Set<string>(),
    queue: { writeBuffer() {}, writeTexture() {}, submit() {} },
    createBuffer: () => ({ destroy() {} }),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    createQuerySet: () => ({}),
    addEventListener: () => {},
    removeEventListener: () => {},
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function baseOpts(overrides: Partial<HybridEngineOptions> = {}): HybridEngineOptions {
  return {
    device: fakeDevice(),
    width: 64,
    height: 64,
    primaryLightDir: [0.3, -0.7, 0.6],
    primaryLightIntensity: 1.0,
    skyTint: [0.5, 0.7, 1.0],
    skyIrradiance: 0.3,
    ...overrides,
  };
}

// Test seam: read _cfg.restirPtReuse (the 0/1 numeric gate stored by
// deriveHybridEngineConfig — the exact value compilePipelines receives).
const grisGate = (e: HybridEngine): number =>
  (e as unknown as { _cfg: { restirPtReuse: number } })._cfg.restirPtReuse;

describe('A8 GRIS variant gate — _cfg.restirPtReuse threading', () => {
  it('default (no restirPtReuse) stores 0 (OFF — biased clamped-Jacobian path)', () => {
    const engine = new HybridEngine(baseOpts());
    expect(grisGate(engine)).toBe(0);
  });

  it('restirPtReuse: false stores 0 (explicit OFF)', () => {
    const engine = new HybridEngine(baseOpts({ restirPtReuse: false }));
    expect(grisGate(engine)).toBe(0);
  });

  it('restirPtReuse: true stores 1 (ON — unbiased GRIS path selected at compile time)', () => {
    const engine = new HybridEngine(baseOpts({ restirPtReuse: true }));
    expect(grisGate(engine)).toBe(1);
  });

  it('restirPtReuse: true is compatible with the full tier (GRIS requires BVH group)', () => {
    expect(() =>
      new HybridEngine(baseOpts({ tier: 'full', restirPtReuse: true }))
    ).not.toThrow();
    const engine = new HybridEngine(baseOpts({ tier: 'full', restirPtReuse: true }));
    expect(grisGate(engine)).toBe(1);
  });

  it('restirPtReuse: true is compatible with the lite tier (structural gate is compile-time, not resource-gated)', () => {
    // GRIS is a compile-time shader/layout choice only; it does not increase
    // the bind-group resource budget beyond what the scene BGL already occupies.
    // The lite-tier resource guard forbids rcEnabled/ppgEnabled/nrcEnabled
    // (they add extra bind groups or large buffers). GRIS reuses the existing
    // scene BGL binding (@group(1) is already present for RIS/shade), so it is
    // NOT forbidden on lite.
    expect(() =>
      new HybridEngine(baseOpts({ tier: 'lite', restirPtReuse: true }))
    ).not.toThrow();
  });

  it('default OFF is stable across qualityTier values (preset does not toggle the GRIS gate)', () => {
    for (const qualityTier of ['ultra', 'high', 'medium', 'low'] as const) {
      const engine = new HybridEngine(baseOpts({ qualityTier }));
      expect(grisGate(engine)).toBe(0);
    }
  });

  it('ON survives a qualityTier override (preset cannot un-set the host opt-in)', () => {
    for (const qualityTier of ['ultra', 'high', 'medium', 'low'] as const) {
      const engine = new HybridEngine(baseOpts({ qualityTier, restirPtReuse: true }));
      expect(grisGate(engine)).toBe(1);
    }
  });
});
