import { describe, expect, it } from 'vitest';
import type { HybridEngineOptions } from '../HybridEngineOptions.js';
import { validateHybridEngineOptions } from '../HybridEngineConfig.js';

function opts(patch: Record<string, unknown> = {}): HybridEngineOptions {
  return {
    device: {} as GPUDevice,
    width: 16,
    height: 16,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
    ...patch,
  };
}

describe('walkaround caustic strategy validation', () => {
  it.each(['none', 'refractive-trace', 'manifold-nee'] as const)(
    'accepts implemented strategy token %s',
    (causticStrategy) => {
      expect(() => validateHybridEngineOptions(opts({ causticStrategy }))).not.toThrow();
    },
  );

  it('rejects the unsupported photon-map strategy instead of substituting none', () => {
    expect(() => validateHybridEngineOptions(opts({ causticStrategy: 'photon-map' })))
      .toThrowError(/does not support/);
  });

  it('rejects unknown runtime tokens before engine state exists', () => {
    expect(() => validateHybridEngineOptions(opts({ causticStrategy: 'mnee-ish' })))
      .toThrowError(/unsupported causticStrategy/);
  });

  it('rejects non-object causticOptions', () => {
    expect(() => validateHybridEngineOptions(opts({ causticOptions: 4 })))
      .toThrowError(/causticOptions must be a plain object/);
  });

  it('rejects manifold options unless the manifold strategy owns them', () => {
    expect(() => validateHybridEngineOptions(opts({ causticOptions: { mneeMaxIterations: 8 } })))
      .toThrowError(/require causticStrategy/);
  });

  it('accepts every bounded manifold option at both limits', () => {
    for (const causticOptions of [
      { mneeMaxIterations: 1, mneeMaxChainLength: 1, mneeMultiplicityTrials: 1 },
      { mneeMaxIterations: 32, mneeMaxChainLength: 8, mneeMultiplicityTrials: 32 },
    ]) {
      expect(() => validateHybridEngineOptions(opts({
        causticStrategy: 'manifold-nee',
        causticOptions,
      }))).not.toThrow();
    }
  });

  it.each([
    ['mneeMaxIterations', 0], ['mneeMaxIterations', 33], ['mneeMaxIterations', 1.5],
    ['mneeMaxChainLength', 0], ['mneeMaxChainLength', 9], ['mneeMaxChainLength', 1.5],
    ['mneeMultiplicityTrials', 0], ['mneeMultiplicityTrials', 33], ['mneeMultiplicityTrials', 1.5],
  ] as const)('rejects malformed manifold option %s=%s', (key, value) => {
    expect(() => validateHybridEngineOptions(opts({
      causticStrategy: 'manifold-nee',
      causticOptions: { [key]: value },
    }))).toThrowError(/safe integer/);
  });

  it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects malformed maxBounces=%s instead of clamping it',
    (maxBounces) => {
      expect(() => validateHybridEngineOptions(opts({ maxBounces })))
        .toThrowError(/positive safe integer/);
    },
  );
});
