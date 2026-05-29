import { describe, expect, it } from 'vitest';
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
      maxStorageTexturesPerShaderStage: 8,
    },
    features: new Set<string>(),
    queue: { writeBuffer() {}, writeTexture() {}, submit() {} },
    createBuffer: () => ({ destroy() {} }),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    createQuerySet: () => ({}),
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
  } as HybridEngineOptions;
}

describe('HYBRID_LITE_LIMITS', () => {
  it('is strictly below HYBRID_WEBGPU_REQUIRED_LIMITS on both axes', () => {
    expect(HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']!)
      .toBeLessThan(HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!);
    expect(HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!)
      .toBeLessThan(HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage']!);
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

  it('tier:lite + rcEnabled:false is allowed (the default for rc)', () => {
    expect(() => new HybridEngine(baseOpts({ tier: 'lite', rcEnabled: false })))
      .not.toThrow();
  });
});

describe('HybridEngine tier:lite — forces merged BVH', () => {
  it('forces bvhMode merged even when the host requested tlas', () => {
    const engine = new HybridEngine(baseOpts({
      tier: 'lite',
      extensions: { 'walkaround-hybrid': { bvhMode: 'tlas' } },
    }));
    // The lite path overrides any host bvhMode to 'merged' (drops TLAS buffers).
    expect((engine as unknown as { _restirBvhModeOverride: string })._restirBvhModeOverride)
      .toBe('merged');
  });

  it('forces bvhMode merged when no host bvhMode was set', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'lite' }));
    expect((engine as unknown as { _restirBvhModeOverride: string })._restirBvhModeOverride)
      .toBe('merged');
  });

  it('full tier preserves a host tlas override (no lite forcing)', () => {
    const engine = new HybridEngine(baseOpts({
      tier: 'full',
      extensions: { 'walkaround-hybrid': { bvhMode: 'tlas' } },
    }));
    expect((engine as unknown as { _restirBvhModeOverride: string })._restirBvhModeOverride)
      .toBe('tlas');
  });
});

describe('HybridEngine tier:lite — biases default qualityTier to medium', () => {
  it('lite without explicit qualityTier resolves the medium preset knobs', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'lite' }));
    const self = engine as unknown as {
      _diSpatialPasses: number;
      _gtaoMode: string;
      _ddgiUpdateDivisor: number;
      _resolutionFactor: number;
    };
    // medium preset: 1 spatial pass, gtao on, /8 ddgi, 0.67 resolution.
    expect(self._diSpatialPasses).toBe(1);
    expect(self._gtaoMode).toBe('on');
    expect(self._ddgiUpdateDivisor).toBe(8);
    expect(self._resolutionFactor).toBeCloseTo(0.67);
  });

  it('explicit qualityTier overrides the lite medium bias', () => {
    const engine = new HybridEngine(baseOpts({ tier: 'lite', qualityTier: 'ultra' }));
    const self = engine as unknown as { _diSpatialPasses: number; _resolutionFactor: number };
    expect(self._diSpatialPasses).toBe(2);
    expect(self._resolutionFactor).toBe(1.0);
  });
});

describe('HybridEngine qualityTier — explicit per-knob override beats preset', () => {
  it('qualityTier:low + explicit gtaoMode:on keeps gtao on (overriding low off)', () => {
    const engine = new HybridEngine(baseOpts({ qualityTier: 'low', gtaoMode: 'on' }));
    const self = engine as unknown as { _gtaoMode: string; _diSpatialPasses: number };
    expect(self._gtaoMode).toBe('on');         // explicit override
    expect(self._diSpatialPasses).toBe(1);     // other low-tier values applied
  });

  it('default (no qualityTier, full tier) is the ultra baseline', () => {
    const engine = new HybridEngine(baseOpts({}));
    const self = engine as unknown as {
      _diSpatialPasses: number;
      _giSpatialPasses: number;
      _gtaoMode: string;
      _ddgiUpdateDivisor: number;
      _resolutionFactor: number;
      _denoiser: string;
    };
    expect(self._diSpatialPasses).toBe(2);
    expect(self._giSpatialPasses).toBe(2);
    expect(self._gtaoMode).toBe('on');
    // No qualityTier ⇒ ultra preset. After the 2→32 cadence decision, ultra's
    // DDGI divisor is 2 (the fast end), so the no-preset default cadence is now
    // stride 2 — 4× the old hardcoded stride-8. Deliberate; pending GPU A/B.
    expect(self._ddgiUpdateDivisor).toBe(2);
    expect(self._resolutionFactor).toBe(1.0);
    expect(self._denoiser).toBe('atrous-variance');
  });
});
