import { describe, expect, it } from 'vitest';

import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
  roundNeuralTensorScalar,
} from '../float16.js';
import {
  NEURAL_F16_TENSOR_STORAGE,
  NEURAL_F32_TENSOR_STORAGE,
  resolveNeuralTensorStorage,
} from '../tensorPrecision.js';
import { NEURAL_PREPROCESSING_CONTRACT } from '../preprocessing.js';
import type { ModelWeights } from '../weights.js';

function weights(tensorStorage?: 'f32' | 'f16-compatible'): ModelWeights {
  return {
    layers: [],
    formatVersion: 2,
    checkpoint: {
      id: 'precision-test',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'test',
      captureBackend: 'walkaround-hybrid',
      tonemap: 'linear',
      hardware: 'test',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      ...(tensorStorage !== undefined ? { tensorStorage } : {}),
      ...(tensorStorage === 'f16-compatible' ? {
        mixedPrecision: {
          status: 'pass' as const,
          checkpointSha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
          architecture: 'vitrum-unet-9x3-v1' as const,
          preprocessing: NEURAL_PREPROCESSING_CONTRACT,
          quantization: 'f16-storage-per-logical-layer-f32-weight-bias-accumulation' as const,
          metricDomain: 'postprocessed-linear-hdr' as const,
          validationCorpusSha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
          validationScenes: 8,
          maxAbsError: 0.01,
          meanAbsError: 0.001,
          psnrDb: 48,
          finiteOutputs: true,
          outputMin: 0,
          outputMax: 64,
          accumulation: 'f32' as const,
          weights: 'f32' as const,
        },
      } : {}),
      qualityReport: { status: 'pass', reportPath: 'report.json' },
    },
  };
}

function device(features: readonly GPUFeatureName[]): GPUDevice {
  return { features: new Set(features) } as unknown as GPUDevice;
}

describe('neural tensor precision', () => {
  it('implements finite IEEE binary16 rounding and saturates neural overflow', () => {
    expect(float16BitsToFloat32(float32ToFloat16Bits(1))).toBe(1);
    expect(float16BitsToFloat32(float32ToFloat16Bits(-2))).toBe(-2);
    expect(roundNeuralTensorScalar(1 / 3, 'f16')).toBeCloseTo(0.333251953125, 12);
    expect(roundNeuralTensorScalar(1e20, 'f16')).toBe(65_504);
    expect(roundNeuralTensorScalar(Number.NaN, 'f16')).toBe(0);
    expect(roundNeuralTensorScalar(Number.POSITIVE_INFINITY, 'f16')).toBe(0);
  });

  it('requires both certified metadata and an enabled shader-f16 device feature', () => {
    expect(resolveNeuralTensorStorage(device(['shader-f16']), weights('f16-compatible')))
      .toBe(NEURAL_F16_TENSOR_STORAGE);
    expect(resolveNeuralTensorStorage(device([]), weights('f16-compatible')))
      .toBe(NEURAL_F32_TENSOR_STORAGE);
    expect(resolveNeuralTensorStorage(device(['shader-f16']), weights()))
      .toBe(NEURAL_F32_TENSOR_STORAGE);
    expect(resolveNeuralTensorStorage(device(['shader-f16']), weights('f32')))
      .toBe(NEURAL_F32_TENSOR_STORAGE);
  });

  it('never upgrades an uncertified f16-compatible declaration', () => {
    const malformed = weights('f16-compatible');
    const checkpoint = malformed.checkpoint!;
    const { mixedPrecision: _mixedPrecision, ...uncertifiedCheckpoint } = checkpoint;
    const uncertified: ModelWeights = {
      ...malformed,
      checkpoint: uncertifiedCheckpoint,
    };
    expect(resolveNeuralTensorStorage(device(['shader-f16']), uncertified))
      .toBe(NEURAL_F32_TENSOR_STORAGE);
  });
});
