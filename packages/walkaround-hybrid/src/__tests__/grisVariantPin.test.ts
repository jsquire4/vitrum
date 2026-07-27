/**
 * Construction-time pin for the fixed-width GRIS DDGI-proxy variant.
 *
 * `grisReuse` selects a 30-u32 reservoir layout and matching GI shader graph.
 * The enabled estimator reuses a diffuse/geometric one-bounce DDGI proxy with
 * exact surface-shift Jacobians, environment identity shifts, visibility-aware
 * inverse support, and a bounded all-technique density matrix. It is not
 * ReSTIR PT and does not claim an unbiased path-tracing result.
 */
import { describe, expect, it, vi } from 'vitest';
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

const grisGate = (engine: HybridEngine): number =>
  (engine as unknown as { _cfg: { grisReuse: number } })._cfg.grisReuse;

describe('GRIS DDGI-proxy variant gate', () => {
  it('defaults to the compact legacy layout', () => {
    expect(grisGate(new HybridEngine(baseOpts()))).toBe(0);
  });

  it('preserves an explicit false canonical option', () => {
    expect(grisGate(new HybridEngine(baseOpts({ grisReuse: false })))).toBe(0);
  });

  it('selects the fixed-width DDGI-proxy variant at construction', () => {
    expect(grisGate(new HybridEngine(baseOpts({ grisReuse: true })))).toBe(1);
  });

  it('accepts the GRIS layout on both full and lite tiers', () => {
    for (const tier of ['full', 'lite'] as const) {
      const engine = new HybridEngine(baseOpts({ tier, grisReuse: true }));
      expect(grisGate(engine)).toBe(1);
    }
  });

  it('does not let a quality preset change the construction-time choice', () => {
    for (const qualityTier of ['ultra', 'high', 'medium', 'low'] as const) {
      expect(grisGate(new HybridEngine(baseOpts({ qualityTier })))).toBe(0);
      expect(grisGate(new HybridEngine(baseOpts({ qualityTier, grisReuse: true })))).toBe(1);
    }
  });

  it('requires engine reconstruction to move between compact and GRIS layouts', () => {
    const compact = new HybridEngine(baseOpts({ grisReuse: false }));
    const gris = new HybridEngine(baseOpts({ grisReuse: true }));
    expect(grisGate(compact)).toBe(0);
    expect(grisGate(gris)).toBe(1);
  });

  it('accepts the deprecated alias, emits one structured warning, and preserves its value', () => {
    const warnings: unknown[] = [];
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const engine = new HybridEngine(baseOpts({
        restirPtReuse: true,
        onWarning: (warning) => warnings.push(warning),
      }));
      expect(grisGate(engine)).toBe(1);
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.restir-pt-reuse-deprecated',
          backend: 'walkaround-hybrid',
          phase: 'construction',
          details: { replacement: 'grisReuse', effectiveValue: true },
          message: expect.stringMatching(/not ReSTIR PT/),
        }),
      ]);
      expect(warnings).not.toEqual([
        expect.objectContaining({ message: expect.stringMatching(/unbiased/i) }),
      ]);
      expect(consoleWarn).toHaveBeenCalledOnce();
    } finally {
      consoleWarn.mockRestore();
    }
  });

  it('does not emit the alias warning for the canonical option', () => {
    const warnings: unknown[] = [];
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const engine = new HybridEngine(baseOpts({
        grisReuse: true,
        onWarning: (warning) => warnings.push(warning),
      }));
      expect(grisGate(engine)).toBe(1);
      expect(warnings).toEqual([]);
      expect(consoleWarn).not.toHaveBeenCalled();
    } finally {
      consoleWarn.mockRestore();
    }
  });

  it('accepts matching canonical and deprecated values but still warns about the alias', () => {
    const warnings: unknown[] = [];
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      const engine = new HybridEngine(baseOpts({
        grisReuse: true,
        restirPtReuse: true,
        onWarning: (warning) => warnings.push(warning),
      }));
      expect(grisGate(engine)).toBe(1);
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.restir-pt-reuse-deprecated',
          details: { replacement: 'grisReuse', effectiveValue: true },
        }),
      ]);
    } finally {
      consoleWarn.mockRestore();
    }
  });

  it('rejects conflicting canonical and deprecated values', () => {
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => new HybridEngine(baseOpts({
        grisReuse: true,
        restirPtReuse: false,
      }))).toThrow(/grisReuse and deprecated restirPtReuse disagree/);
    } finally {
      consoleWarn.mockRestore();
    }
  });
});
