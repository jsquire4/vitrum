import { describe, expect, it } from 'vitest';
import { ptWebgpuCapabilities, type PtWebgpuCapabilitiesFlags } from '../capabilities.js';

function flags(traceTier: 'full' | 'lite'): PtWebgpuCapabilitiesFlags {
  return {
    traceTier,
    maxSamplesLimit: 64,
    maxBouncesLimit: 8,
    bdpt: false,
    restirPtReuse: false,
    restirPtBiasedWeightClamp: false,
    sampling: 'pcg',
    bvhTraversal: 'binary',
    causticStrategy: 'none',
    spectral: false,
    denoiser: 'none',
  };
}

describe('pt-webgpu inverse capability contract', () => {
  it('publishes the exact full-tier certified path-replay domain', () => {
    const inverse = ptWebgpuCapabilities(flags('full')).inverseRendering;

    expect(inverse?.methods).toEqual({
      'finite-difference': 'native',
      'path-replay': 'native',
    });
    const { materialFields, emitterFields, ...pathReplay } = inverse!.pathReplay!;
    expect(pathReplay).toEqual({
      failurePolicy: 'error',
      maxBounces: 1,
      supportsSpectral: false,
      supportsBdpt: false,
      supportsRestirPtReuse: false,
      supportsCausticStrategies: false,
    });
    expect([...materialFields]).toEqual(['emissive']);
    expect([...emitterFields]).toEqual([]);
    expect(Object.isFrozen(inverse)).toBe(true);
    expect(Object.isFrozen(inverse?.methods)).toBe(true);
    expect(Object.isFrozen(inverse?.pathReplay)).toBe(true);
    expect(Object.isFrozen(materialFields)).toBe(true);
    expect(Object.isFrozen(emitterFields)).toBe(true);
    expect(() => (materialFields as unknown as Set<string>).add('baseColor'))
      .toThrow(TypeError);
    expect(() => (emitterFields as unknown as Set<string>).add('intensity'))
      .toThrow(TypeError);
    expect([...materialFields]).toEqual(['emissive']);
    expect([...emitterFields]).toEqual([]);
  });

  it('does not advertise path replay on the uncertified lite tier', () => {
    expect(ptWebgpuCapabilities(flags('lite')).inverseRendering).toEqual({
      methods: {
        'finite-difference': 'native',
        'path-replay': 'unsupported',
      },
    });
  });
});
