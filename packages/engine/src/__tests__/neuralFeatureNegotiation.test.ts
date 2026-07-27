import { describe, expect, it } from 'vitest';
import {
  NEURAL_PREPROCESSING_CONTRACT,
  type HybridEngineOptions,
  type ModelWeights,
} from '@vitrum/walkaround-hybrid';
import { neuralCheckpointPayloadSha256 } from '../../../walkaround-hybrid/src/neural/checkpointDigest.js';
import {
  NEURAL_ARCHITECTURE_ID,
  NEURAL_F16_METRIC_DOMAIN,
  NEURAL_F16_QUANTIZATION,
} from '../../../walkaround-hybrid/src/neural/weights.js';
import { requiredWalkaroundNeuralDeviceFeatures } from '../neuralFeatureNegotiation.js';

function adapter(...features: GPUFeatureName[]): Pick<GPUAdapter, 'features'> {
  return { features: new Set(features) };
}

function weights(certified: boolean): ModelWeights {
  const layers = [{
    name: 'projection',
    weights: new Float32Array([1, 2, 3]),
    biases: new Float32Array([0.25]),
  }];
  return {
    formatVersion: 2,
    layers,
    checkpoint: {
      id: 'engine-feature-negotiation',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'test',
      captureBackend: 'webgpu',
      tonemap: 'linear-hdr',
      hardware: 'test-adapter',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      qualityReport: { status: 'pass', reportPath: 'test-report.json' },
      ...(certified ? {
        tensorStorage: 'f16-compatible' as const,
        mixedPrecision: {
          checkpointSha256: neuralCheckpointPayloadSha256(layers),
          architecture: NEURAL_ARCHITECTURE_ID,
          preprocessing: NEURAL_PREPROCESSING_CONTRACT,
          quantization: NEURAL_F16_QUANTIZATION,
          metricDomain: NEURAL_F16_METRIC_DOMAIN,
          validationCorpusSha256: '1'.repeat(64),
          status: 'pass' as const,
          validationScenes: 4,
          maxAbsError: 0.01,
          meanAbsError: 0.001,
          psnrDb: 40,
          finiteOutputs: true,
          outputMin: 0,
          outputMax: 8,
          accumulation: 'f32' as const,
          weights: 'f32' as const,
        },
      } : {}),
    },
  };
}

function options(
  denoiser: Exclude<HybridEngineOptions['denoiser'], undefined>,
  model: ModelWeights,
  precision: 'auto' | 'f32' | 'f16' = 'auto',
): Partial<HybridEngineOptions> {
  return {
    denoiser,
    neuralWeights: model,
    extensions: { 'walkaround-hybrid': { neuralTensorStorage: precision } },
  };
}

describe('owned-device neural feature negotiation', () => {
  it('requests shader-f16 only for an eligible neural selection', () => {
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter('shader-f16'), options('neural', weights(true)),
    )).toEqual(['shader-f16']);
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter('shader-f16'), options('auto', weights(true)),
    )).toEqual(['shader-f16']);
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter('shader-f16'), options('neural', weights(true), 'f32'),
    )).toEqual([]);
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter(), options('neural', weights(true)),
    )).toEqual([]);
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter('shader-f16'), options('atrous', weights(true)),
    )).toEqual([]);
  });

  it('fails explicit neural f16 before requestDevice when certification or adapter support is absent', () => {
    expect(() => requiredWalkaroundNeuralDeviceFeatures(
      adapter('shader-f16'), options('neural', weights(false), 'f16'),
    )).toThrow(/digest-bound, passing mixed-precision/);
    expect(() => requiredWalkaroundNeuralDeviceFeatures(
      adapter(), options('neural', weights(true), 'f16'),
    )).toThrow(/adapter supporting 'shader-f16'/);
  });

  it('leaves denoiser:auto deterministic instead of throwing on f16 ineligibility', () => {
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter('shader-f16'), options('auto', weights(false), 'f16'),
    )).toEqual([]);
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter(), options('auto', weights(true), 'f16'),
    )).toEqual([]);
  });

  it('derives shader-f16 independently from the resolved NRC trainer precision', () => {
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter('shader-f16'),
      { nrcEnabled: true, nrcConfig: { useF16: true } },
    )).toEqual(['shader-f16']);
    expect(requiredWalkaroundNeuralDeviceFeatures(
      adapter(),
      { nrcEnabled: true, nrcConfig: { useF16: false } },
    )).toEqual([]);
    expect(() => requiredWalkaroundNeuralDeviceFeatures(
      adapter(),
      { nrcEnabled: true, nrcConfig: { useF16: true } },
    )).toThrow(/nrcConfig\.useF16=true.*shader-f16/);
  });
});
