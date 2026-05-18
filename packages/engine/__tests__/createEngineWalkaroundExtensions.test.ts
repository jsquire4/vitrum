/**
 * W3-D12 — createEngine() walkaround-hybrid options merge.
 *
 * Verifies that the @vitrum/engine facade's `buildWalkaroundHybridOptions`
 * threads walkaround-specific knobs through `EngineOptions.extensions
 * ['walkaround-hybrid']` (per W3-D12), not through legacy top-level fields.
 *
 * Coverage:
 *   1. Defaults: lighting, debug, and the four scale-derived knobs land in
 *      the walkaround extension bag.
 *   2. User overrides on `advanced.extensions['walkaround-hybrid']` win
 *      per-key over the facade defaults.
 *   3. Other extension buckets (foreign keys) survive the merge unchanged.
 *   4. Top-level fields on `advanced` (e.g. `denoiser`) survive on the
 *      merged top-level options.
 */

import { describe, it, expect } from 'vitest';
import { buildWalkaroundHybridOptions } from '../src/createEngine.js';
import {
  WALKAROUND_HYBRID_EXT_KEY,
  type HybridEngineOptions,
  type WalkaroundHybridExtensions,
} from '@vitrum/walkaround-hybrid';

const FAKE_DEVICE = {} as GPUDevice;

describe('W3-D12 — buildWalkaroundHybridOptions defaults', () => {
  it('places lighting + scale-derived knobs under extensions["walkaround-hybrid"]', () => {
    const merged = buildWalkaroundHybridOptions({
      device:            FAKE_DEVICE,
      canvasWidth:       1920,
      canvasHeight:      1080,
      diagonal:          2.0, // Cornell-scale
      threeSceneForCtor: undefined,
      debug:             false,
      advanced:          undefined,
    });

    expect(merged.device).toBe(FAKE_DEVICE);
    expect(merged.width).toBe(1920);
    expect(merged.height).toBe(1080);

    const ext = merged.extensions?.[WALKAROUND_HYBRID_EXT_KEY] as WalkaroundHybridExtensions;
    expect(ext).toBeDefined();
    expect(ext.primaryLightDir).toEqual([0.3, -0.7, 0.6]);
    expect(ext.primaryLightIntensity).toBe(1.0);
    expect(ext.skyTint).toEqual([0.5, 0.7, 1.0]);
    expect(ext.skyIrradiance).toBe(0.3);
    expect(ext.debug).toBe(false);
    // Scale-derived: D=2 → cameraMove = (2e-3)² = 4e-6, emitterDist2Floor = (2e-4)² = 4e-8.
    expect(ext.cameraMoveResetThresholdSq).toBeCloseTo(4e-6, 12);
    expect(ext.temporalAccumAlpha).toBe(0.01);
    expect(ext.emitterDist2Floor).toBeCloseTo(4e-8, 14);
    expect(ext.triIntersectEpsilon).toBeCloseTo(2e-6, 12);
  });

  it('does NOT mirror lighting at the top level (W3-D12 contract)', () => {
    const merged = buildWalkaroundHybridOptions({
      device:            FAKE_DEVICE,
      canvasWidth:       64,
      canvasHeight:      64,
      diagonal:          1,
      threeSceneForCtor: undefined,
      debug:             false,
      advanced:          undefined,
    });

    // The merged options must NOT carry walkaround lighting at the top
    // level (legacy back-compat surface). Hosts that read this object
    // expect the canonical shape to be extensions-only.
    const top = merged as unknown as Record<string, unknown>;
    expect(top['primaryLightDir']).toBeUndefined();
    expect(top['primaryLightIntensity']).toBeUndefined();
    expect(top['skyTint']).toBeUndefined();
    expect(top['skyIrradiance']).toBeUndefined();
    expect(top['cameraMoveResetThresholdSq']).toBeUndefined();
  });
});

describe('W3-D12 — buildWalkaroundHybridOptions user overrides', () => {
  it('per-key user override on extensions["walkaround-hybrid"] wins over facade defaults', () => {
    const merged = buildWalkaroundHybridOptions({
      device:            FAKE_DEVICE,
      canvasWidth:       64,
      canvasHeight:      64,
      diagonal:          1,
      threeSceneForCtor: undefined,
      debug:             false,
      advanced: {
        extensions: {
          [WALKAROUND_HYBRID_EXT_KEY]: {
            primaryLightIntensity: 5.0,
            gtao: { radiusPx: 16, intensity: 1.0 },
          } satisfies Partial<WalkaroundHybridExtensions>,
        },
      } as Partial<HybridEngineOptions>,
    });

    const ext = merged.extensions?.[WALKAROUND_HYBRID_EXT_KEY] as WalkaroundHybridExtensions;
    expect(ext.primaryLightIntensity).toBe(5.0);
    expect(ext.gtao).toEqual({ radiusPx: 16, intensity: 1.0 });
    // Unspecified keys still come from defaults.
    expect(ext.primaryLightDir).toEqual([0.3, -0.7, 0.6]);
    expect(ext.skyIrradiance).toBe(0.3);
  });

  it('preserves foreign extension buckets', () => {
    const merged = buildWalkaroundHybridOptions({
      device:            FAKE_DEVICE,
      canvasWidth:       64,
      canvasHeight:      64,
      diagonal:          1,
      threeSceneForCtor: undefined,
      debug:             false,
      advanced: {
        extensions: {
          'stained-glass-extensions': { dichroicLUT: 'spec://lut.bin' },
        },
      } as Partial<HybridEngineOptions>,
    });

    expect(merged.extensions?.['stained-glass-extensions']).toEqual({
      dichroicLUT: 'spec://lut.bin',
    });
    // Walkaround bucket is still present.
    expect(merged.extensions?.[WALKAROUND_HYBRID_EXT_KEY]).toBeDefined();
  });

  it('forwards non-extension top-level overrides (e.g. denoiser) onto the merged options', () => {
    const merged = buildWalkaroundHybridOptions({
      device:            FAKE_DEVICE,
      canvasWidth:       64,
      canvasHeight:      64,
      diagonal:          1,
      threeSceneForCtor: undefined,
      debug:             false,
      advanced: {
        denoiser:   'svgf-real',
        maxBounces: 8,
      } as Partial<HybridEngineOptions>,
    });

    expect(merged.denoiser).toBe('svgf-real');
    expect(merged.maxBounces).toBe(8);
  });
});
