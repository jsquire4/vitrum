import { describe, expect, it } from 'vitest';
import { DenoiserAdapterPass } from '../src/pipeline/passes/DenoiserAdapterPass.js';
import {
  DENOISER_READY_STATE,
  type Denoiser,
} from '../src/pipeline/denoisers/index.js';

function makeStubDenoiser(id: Denoiser['id']): Denoiser {
  return {
    id,
    passLabels: [],
    state: () => DENOISER_READY_STATE,
    initialize: async () => {},
    dispatch: () => null,
    prepareResize: () => ({
      commit: () => undefined,
      rollback: () => undefined,
      finalize: () => undefined,
    }),
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
