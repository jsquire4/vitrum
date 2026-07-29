/**
 * Construction-time pin for the sole generalized-reuse GI path.
 *
 * The deprecated `grisReuse` spelling no longer selects a variant: omission or
 * `true` uses the fixed 28-u32 reservoir ABI, and `false` fails closed instead
 * of reviving the retired compact path. The estimator reuses a
 * diffuse/geometric one-bounce DDGI proxy with exact surface-shift Jacobians,
 * environment identity shifts, visibility-aware inverse support, and a bounded
 * all-technique density matrix. It is not ReSTIR PT and does not claim an
 * unbiased path-tracing result.
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

describe('generalized DDGI-proxy compatibility options', () => {
  it('requires no option for the generalized live layout', () => {
    expect(() => new HybridEngine(baseOpts())).not.toThrow();
  });

  it('fails closed when the retired compact path is requested', () => {
    expect(() =>
      new HybridEngine(baseOpts({ grisReuse: false })),
    ).toThrow(/retired compact ReSTIR-GI reuse path/);
  });

  it('accepts explicit true only as a deprecated compatibility spelling', () => {
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => new HybridEngine(baseOpts({ grisReuse: true }))).not.toThrow();
      expect(consoleWarn).toHaveBeenCalledOnce();
    } finally {
      consoleWarn.mockRestore();
    }
  });

  it('uses the generalized layout on both full and lite tiers', () => {
    for (const tier of ['full', 'lite'] as const) {
      expect(() => new HybridEngine(baseOpts({ tier }))).not.toThrow();
    }
  });

  it('does not let a quality preset change the sole live layout', () => {
    for (const qualityTier of ['ultra', 'high', 'medium', 'low'] as const) {
      expect(() => new HybridEngine(baseOpts({ qualityTier }))).not.toThrow();
    }
  });

  it('has no construction-time route back to the compact layout', () => {
    expect(() =>
      new HybridEngine(baseOpts({ restirPtReuse: false })),
    ).toThrow(/retired compact ReSTIR-GI reuse path/);
  });

  it('accepts the deprecated alias, emits one structured warning, and preserves its value', () => {
    const warnings: unknown[] = [];
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      new HybridEngine(baseOpts({
        restirPtReuse: true,
        onWarning: (warning) => warnings.push(warning),
      }));
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.gi-reuse-option-deprecated',
          backend: 'walkaround-hybrid',
          phase: 'construction',
          details: expect.objectContaining({
            replacement: 'omit-option',
            suppliedRestirPtReuse: true,
            effectiveValue: true,
          }),
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

  it('also deprecates the now-redundant canonical option', () => {
    const warnings: unknown[] = [];
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      new HybridEngine(baseOpts({
        grisReuse: true,
        onWarning: (warning) => warnings.push(warning),
      }));
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.gi-reuse-option-deprecated',
          details: expect.objectContaining({
            suppliedGrisReuse: true,
            replacement: 'omit-option',
          }),
        }),
      ]);
      expect(consoleWarn).toHaveBeenCalledOnce();
    } finally {
      consoleWarn.mockRestore();
    }
  });

  it('accepts matching canonical and deprecated values but still warns about the alias', () => {
    const warnings: unknown[] = [];
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      new HybridEngine(baseOpts({
        grisReuse: true,
        restirPtReuse: true,
        onWarning: (warning) => warnings.push(warning),
      }));
      expect(warnings).toEqual([
        expect.objectContaining({
          code: 'walkaround-hybrid.gi-reuse-option-deprecated',
          details: expect.objectContaining({
            replacement: 'omit-option',
            suppliedGrisReuse: true,
            suppliedRestirPtReuse: true,
            effectiveValue: true,
          }),
        }),
      ]);
    } finally {
      consoleWarn.mockRestore();
    }
  });

  it('rejects either compatibility spelling when false', () => {
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(() => new HybridEngine(baseOpts({
        grisReuse: true,
        restirPtReuse: false,
      }))).toThrow(/retired compact ReSTIR-GI reuse path/);
    } finally {
      consoleWarn.mockRestore();
    }
  });
});
