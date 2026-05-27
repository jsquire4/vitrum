import { describe, expect, it } from 'vitest';
import { DenoiserAdapterPass } from '../src/pipeline/passes/DenoiserAdapterPass.js';
import type { Denoiser } from '../src/pipeline/denoisers/index.js';

function makeStubDenoiser(id: Denoiser['id']): Denoiser {
  return {
    id,
    passLabels: [],
    initialize: async () => {},
    dispatch: () => null,
    resize: () => {},
    dispose: () => {},
  };
}

describe('DenoiserAdapterPass runtime gate', () => {
  it('gates off when isPassEnabled returns false even for a real denoiser', () => {
    const p = new DenoiserAdapterPass(
      () => makeStubDenoiser('atrous-variance'),
      () => ({} as GPUComputePipeline),
      () => false,
    );
    expect(p.gates()).toBe(false);
  });

  it('gates on when isPassEnabled is true and denoiser is not none', () => {
    const p = new DenoiserAdapterPass(
      () => makeStubDenoiser('atrous-variance'),
      () => ({} as GPUComputePipeline),
      () => true,
    );
    expect(p.gates()).toBe(true);
  });
});
