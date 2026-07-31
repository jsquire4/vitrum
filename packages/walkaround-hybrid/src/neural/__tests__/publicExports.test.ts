import { describe, expect, it } from 'vitest';

import {
  NEURAL_PREPROCESSING_CONTRACT,
  VITRUM_MODEL_LEGACY_VERSION,
  VITRUM_MODEL_MAGIC,
  VITRUM_MODEL_VERSION,
  WALKAROUND_DENOISER_UNET_SPEC,
  WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
  assessNeuralCheckpointProductionReadiness,
  assertWalkaroundNeuralDenoiserShape,
  executeNeuralInferenceCpu,
  isNeuralCheckpointProductionReady,
  loadWeightsFromArrayBuffer,
  postprocessNeuralRadiance,
  preprocessNeuralRadiance,
  sanitizeNeuralAlbedo,
  sanitizeNeuralNormal,
  serializeWeightsToArrayBuffer,
  walkaroundNeuralDenoiserShapeError,
} from '../../index.js';

describe('public neural package surface', () => {
  it('exports checkpoint, preprocessing, CPU-oracle, and shape contracts from the package root', () => {
    expect(VITRUM_MODEL_MAGIC).toBe(0xDEAF1984);
    expect(VITRUM_MODEL_LEGACY_VERSION).toBe(1);
    expect(VITRUM_MODEL_VERSION).toBe(2);
    expect(NEURAL_PREPROCESSING_CONTRACT.version).toBe(1);
    expect(WALKAROUND_DENOISER_UNET_SPEC.layers).toHaveLength(25);
    expect(WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT.widthMultiple).toBe(1);
    for (const value of [
      assessNeuralCheckpointProductionReadiness,
      assertWalkaroundNeuralDenoiserShape,
      executeNeuralInferenceCpu,
      isNeuralCheckpointProductionReady,
      loadWeightsFromArrayBuffer,
      postprocessNeuralRadiance,
      preprocessNeuralRadiance,
      sanitizeNeuralAlbedo,
      sanitizeNeuralNormal,
      serializeWeightsToArrayBuffer,
      walkaroundNeuralDenoiserShapeError,
    ]) {
      expect(typeof value).toBe('function');
    }
  });

  it('normalizes a tiny finite normal instead of replacing its direction', () => {
    const normal = sanitizeNeuralNormal(1e-20, 0, 0);
    expect(normal[0]).toBeCloseTo(1, 7);
    expect(normal[1]).toBeCloseTo(0, 7);
    expect(normal[2]).toBeCloseTo(0, 7);
  });
});
