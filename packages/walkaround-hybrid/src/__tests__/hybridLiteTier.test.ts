import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
} from '../pipeline/WalkaroundGPUPipeline.js';

// Minimal GPUDevice stub. The HybridEngine constructor's validation +
// option-parsing path (where the lite-tier throws + bvhMode forcing happen)
// runs BEFORE any GPU call — it only stores the device handle for later async
// init. So a structural stub is enough to exercise ctor-time behaviour without
// a real GPU. (We never call `engine.renderFrame` here.)
function fakeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 7,
      maxSampledTexturesPerShaderStage: 20,
    },
    features: new Set<string>(),
    queue: { writeBuffer() {}, writeTexture() {}, submit() {} },
    createBuffer: vi.fn(() => ({ destroy() {} })),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    createQuerySet: () => ({}),
    addEventListener: () => {},
    removeEventListener: () => {},
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function baseOpts(overrides: Partial<HybridEngineOptions>): HybridEngineOptions {
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

// These tests pin the constructor-resolved preset/tier knobs. The resolved
// construction-immutable config lives on the engine's `_cfg` record
// (`deriveHybridEngineConfig` output); `_resolutionFactor` is a separate mutable
// runtime field. Read both through narrow test seams rather than per-field
// forwarder getters on the production class.
interface CfgPeek {
  nrcEnabled: number;
  restirBvhModeOverride: string;
  diSpatialPasses: number;
  giSpatialPasses: number;
  gtaoMode: string;
  ddgiUpdateDivisor: number;
  denoiser: string;
  checkerboard: boolean;
  resolutionFactor: number;
}
const cfg = (e: HybridEngine): CfgPeek => (e as unknown as { _cfg: CfgPeek })._cfg;
const resolutionFactor = (e: HybridEngine): number =>
  (e as unknown as { _resolutionFactor: number })._resolutionFactor;

describe('walkaround device-limit constructor preflight', () => {
  it('uses the same structural layout floor for lite and full', () => {
    expect(HYBRID_LITE_LIMITS).toEqual(HYBRID_WEBGPU_REQUIRED_LIMITS);
  });

  it('rejects an under-limit real device surface before any GPU allocation', () => {
    const device = fakeDevice();
    (device.limits as unknown as Record<string, number>).maxSampledTexturesPerShaderStage = 16;
    expect(() => new HybridEngine(baseOpts({ device, tier: 'full' })))
      .toThrow(/maxSampledTexturesPerShaderStage=16 \(requires >= 17\)/);
    expect(device.createBuffer).not.toHaveBeenCalled();
  });
});

describe('HybridEngine tier:lite — constructor validation', () => {
  it('throws on tier:lite + rcEnabled', () => {
    expect(() => new HybridEngine(baseOpts({ tier: 'lite', rcEnabled: true })))
      .toThrow(/tier:'lite' forbids rcEnabled/);
  });

  it('throws on tier:lite + ppgEnabled', () => {
    expect(() => new HybridEngine(baseOpts({ tier: 'lite', ppgEnabled: true })))
      .toThrow(/tier:'lite' forbids ppgEnabled/);
  });

  it('throws on tier:lite + denoiser:neural', () => {
    expect(() => new HybridEngine(baseOpts({ tier: 'lite', denoiser: 'neural' })))
      .toThrow(/tier:'lite' forbids denoiser:'neural'/);
  });

  it('throws on tier:lite + denoiser:bmfr before allocation', () => {
    const device = fakeDevice();
    expect(() => new HybridEngine(baseOpts({ device, tier: 'lite', denoiser: 'bmfr' })))
      .toThrow(/tier:'lite' forbids denoiser:'bmfr'/);
    expect(device.createBuffer).not.toHaveBeenCalled();
  });

  it('keeps BMFR available as an explicit full-tier selection', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'full', denoiser: 'bmfr' }));
    expect(cfg(engine).denoiser).toBe('bmfr');
  });

  it('never auto-selects BMFR on either runtime tier', () => {
    const lite = new HybridEngine(baseOpts({ tier: 'lite', denoiser: 'auto' }));
    const full = new HybridEngine(baseOpts({ tier: 'full', denoiser: 'auto' }));
    expect(cfg(lite).denoiser).toBe('atrous-variance');
    expect(cfg(full).denoiser).toBe('atrous-variance');
  });

  it('throws on tier:lite + nrcEnabled', () => {
    expect(() => new HybridEngine(baseOpts({ tier: 'lite', nrcEnabled: true })))
      .toThrow(/tier:'lite' forbids nrcEnabled/);
  });

  it('tier:lite + rcEnabled:false is allowed (the default for rc)', () => {
    expect(() => new HybridEngine(baseOpts({ tier: 'lite', rcEnabled: false })))
      .not.toThrow();
  });

  it('tier:lite + nrcEnabled:false is allowed (the default for nrc)', () => {
    expect(() => new HybridEngine(baseOpts({ tier: 'lite', nrcEnabled: false })))
      .not.toThrow();
  });

  it('reports neural support details from the runtime provisioning state', () => {
    const lite = new HybridEngine(baseOpts({ tier: 'lite' }));
    const full = new HybridEngine(baseOpts({ tier: 'full' }));
    expect(lite.capabilities.supportDetails?.denoisers.neural).toBe('unsupported');
    expect(lite.capabilities.supportDetails?.denoisers.auto).toBe('native');
    expect(lite.capabilities.supportDetails?.denoisers['atrous-variance']).toBe('native');
    expect(full.capabilities.supportDetails?.denoisers.neural).toBe('unsupported');
  });
});

describe('HybridEngine nrcEnabled — gate storage (full tier)', () => {
  it('default (no nrcEnabled) stores the gate as 0 (OFF)', () => {
    const engine = new HybridEngine(baseOpts({}));
    expect(cfg(engine).nrcEnabled).toBe(0);
  });

  it('nrcEnabled:true on the full tier stores the gate as 1', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'full', nrcEnabled: true }));
    expect(cfg(engine).nrcEnabled).toBe(1);
  });
});

describe('HybridEngine tier:lite — forces merged BVH', () => {
  it('forces bvhMode merged even when the host requested tlas', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    const engine = new HybridEngine(baseOpts({
      tier: 'lite',
      extensions: { 'walkaround-hybrid': { bvhMode: 'tlas' } },
      onWarning: (warning) => structured.push(warning),
    }));
    // The lite path overrides any host bvhMode to 'merged' (drops TLAS buffers).
    expect(cfg(engine).restirBvhModeOverride).toBe('merged');
    expect(structured).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'walkaround-hybrid.lite-bvh-mode-overridden',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        details: expect.objectContaining({
          tier: 'lite',
          requestedBvhMode: 'tlas',
          effectiveBvhMode: 'merged',
          fallback: 'merged-bvh',
        }),
      }),
    ]));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("tier:'lite' overrides bvhMode:'tlas'"));
    warn.mockRestore();
  });

  it('forces bvhMode merged when no host bvhMode was set', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'lite' }));
    expect(cfg(engine).restirBvhModeOverride).toBe('merged');
  });

  it('full tier preserves a host tlas override (no lite forcing)', () => {
    const engine = new HybridEngine(baseOpts({
      tier: 'full',
      extensions: { 'walkaround-hybrid': { bvhMode: 'tlas' } },
    }));
    expect(cfg(engine).restirBvhModeOverride).toBe('tlas');
  });
});

describe('HybridEngine tier:lite — biases default qualityTier to medium', () => {
  it('lite without explicit qualityTier resolves the medium preset knobs', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'lite' }));
    const c = cfg(engine);
    // medium preset: 1 spatial pass, gtao on, /8 ddgi, 0.67 resolution.
    expect(c.diSpatialPasses).toBe(1);
    expect(c.gtaoMode).toBe('on');
    expect(c.ddgiUpdateDivisor).toBe(8);
    expect(resolutionFactor(engine)).toBeCloseTo(0.67);
  });

  it('explicit qualityTier overrides the lite medium bias', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'lite', qualityTier: 'ultra' }));
    expect(cfg(engine).diSpatialPasses).toBe(2);
    expect(resolutionFactor(engine)).toBe(1.0);
  });

  it('returns to the preset resolution when a frame omits its temporary override', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'lite' }));
    const runtime = engine as unknown as {
      _resolutionFactor: number;
      _applyResolutionFactor(
        factor: number | undefined,
        nowMs: number,
      ): { width: number; height: number };
    };

    expect(runtime._applyResolutionFactor(0.5, 1_000)).toEqual({ width: 32, height: 32 });
    expect(runtime._resolutionFactor).toBe(0.5);

    expect(runtime._applyResolutionFactor(undefined, 1_251)).toEqual({
      width: Math.round(64 * cfg(engine).resolutionFactor),
      height: Math.round(64 * cfg(engine).resolutionFactor),
    });
    expect(runtime._resolutionFactor).toBe(cfg(engine).resolutionFactor);
  });
});

describe('HybridEngine qualityTier — explicit per-knob override beats preset', () => {
  it('qualityTier:low + explicit gtaoMode:on keeps gtao on (overriding low off)', () => {
    const engine = new HybridEngine(baseOpts({ qualityTier: 'low', gtaoMode: 'on' }));
    const c = cfg(engine);
    expect(c.gtaoMode).toBe('on');         // explicit override
    expect(c.diSpatialPasses).toBe(1);     // other low-tier values applied
  });

  it('default (no qualityTier, full tier) is the ultra baseline', () => {
    const engine = new HybridEngine(baseOpts({}));
    const c = cfg(engine);
    expect(c.diSpatialPasses).toBe(2);
    expect(c.giSpatialPasses).toBe(2);
    expect(c.gtaoMode).toBe('on');
    // No qualityTier ⇒ ultra preset. After the 2→32 cadence decision, ultra's
    // DDGI divisor is 2 (the fast end), so the no-preset default cadence is now
    // stride 2 — 4× the old hardcoded stride-8. Deliberate; pending GPU A/B.
    expect(c.ddgiUpdateDivisor).toBe(2);
    expect(resolutionFactor(engine)).toBe(1.0);
    expect(c.denoiser).toBe('atrous-variance');
  });
});

describe('HybridEngine checkerboard — preset enablement + host override threading', () => {
  it('no preset (engine default ⇒ ultra) resolves checkerboard OFF (byte-identical default)', () => {
    expect(cfg(new HybridEngine(baseOpts({}))).checkerboard).toBe(false);
  });

  it('quality tiers (ultra/high) render full-rate (checkerboard OFF)', () => {
    expect(cfg(new HybridEngine(baseOpts({ qualityTier: 'ultra' }))).checkerboard).toBe(false);
    expect(cfg(new HybridEngine(baseOpts({ qualityTier: 'high' }))).checkerboard).toBe(false);
  });

  it('degradation tiers (medium/low) enable checkerboard from the preset', () => {
    expect(cfg(new HybridEngine(baseOpts({ qualityTier: 'medium' }))).checkerboard).toBe(true);
    expect(cfg(new HybridEngine(baseOpts({ qualityTier: 'low' }))).checkerboard).toBe(true);
  });

  it('explicit checkerboardRendering:false OVERRIDES medium/low back to OFF', () => {
    expect(cfg(new HybridEngine(baseOpts({
      qualityTier: 'medium', checkerboardRendering: false,
    }))).checkerboard).toBe(false);
    expect(cfg(new HybridEngine(baseOpts({
      qualityTier: 'low', checkerboardRendering: false,
    }))).checkerboard).toBe(false);
  });

  it('explicit checkerboardRendering:true OVERRIDES ultra/high to ON (host opt-in still wins)', () => {
    expect(cfg(new HybridEngine(baseOpts({
      qualityTier: 'ultra', checkerboardRendering: true,
    }))).checkerboard).toBe(true);
    expect(cfg(new HybridEngine(baseOpts({
      qualityTier: 'high', checkerboardRendering: true,
    }))).checkerboard).toBe(true);
  });

  it('tier:lite (default ⇒ medium preset) resolves checkerboard ON', () => {
    expect(cfg(new HybridEngine(baseOpts({ tier: 'lite' }))).checkerboard).toBe(true);
  });
});
