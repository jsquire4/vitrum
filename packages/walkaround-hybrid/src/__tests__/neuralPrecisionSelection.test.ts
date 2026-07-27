import { describe, expect, it, vi } from 'vitest';

import { parseHybridEngineOptions } from '../HybridEngineConfig.js';
import type { HybridEngineOptions } from '../HybridEngineOptions.js';
import { neuralCheckpointPayloadSha256 } from '../neural/checkpointDigest.js';
import { NEURAL_PREPROCESSING_CONTRACT } from '../neural/preprocessing.js';
import { WALKAROUND_DENOISER_UNET_SPEC } from '../neural/unetArchitecture.js';
import type { LayerWeights, ModelWeights } from '../neural/weights.js';

function model(certified: boolean): ModelWeights {
  const layers: LayerWeights[] = [];
  for (const layer of WALKAROUND_DENOISER_UNET_SPEC.layers) {
    if (layer.kind !== 'conv2d' && layer.kind !== 'transposedConv2d') continue;
    const kH = layer.params.kH ?? 1;
    const kW = layer.params.kW ?? 1;
    layers.push({
      name: layer.name,
      weights: new Float32Array(layer.params.outC * layer.params.inC * kH * kW),
      biases: new Float32Array(layer.params.outC),
    });
  }
  return {
    layers,
    formatVersion: 2,
    checkpoint: {
      id: certified ? 'certified-f16' : 'f32-only',
      trainingSamples: 500,
      noisySpp: 1,
      cleanSpp: 4096,
      auxiliaryInputs: ['albedo', 'normal'],
      captureSource: 'unit',
      captureBackend: 'walkaround-hybrid',
      tonemap: 'linear-hdr',
      hardware: 'unit',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      ...(certified ? {
        tensorStorage: 'f16-compatible' as const,
        mixedPrecision: {
          checkpointSha256: neuralCheckpointPayloadSha256(layers),
          architecture: 'vitrum-unet-9x3-v1' as const,
          preprocessing: NEURAL_PREPROCESSING_CONTRACT,
          quantization: 'f16-storage-per-logical-layer-f32-weight-bias-accumulation' as const,
          metricDomain: 'postprocessed-linear-hdr' as const,
          validationCorpusSha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
          status: 'pass' as const,
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
      } : { tensorStorage: 'f32' as const }),
      qualityReport: { status: 'pass', reportPath: 'quality.json' },
    },
  };
}

function limitedDevice(shaderF16: boolean): GPUDevice {
  return {
    features: new Set<GPUFeatureName>(shaderF16 ? ['shader-f16'] : []),
    limits: {
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxUniformBufferBindingSize: 64 * 1024,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxTextureDimension2D: 8192,
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
    },
    createBuffer: vi.fn(() => {
      throw new Error('selection must not allocate');
    }),
  } as unknown as GPUDevice;
}

function options(
  weights: ModelWeights,
  device: GPUDevice,
  denoiser: 'auto' | 'neural',
  storage: 'auto' | 'f32' | 'f16' = 'auto',
): HybridEngineOptions {
  return {
    device,
    width: 1920,
    height: 1080,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 0.25,
    denoiser,
    neuralWeights: weights,
    extensions: { 'walkaround-hybrid': { neuralTensorStorage: storage } },
  };
}

describe('neural precision selection before allocation', () => {
  it('on one 128 MiB device rejects f32 but selects certified f16 at 1080p', () => {
    const device = limitedDevice(true);
    const weights = model(true);
    expect(parseHybridEngineOptions(options(weights, device, 'auto', 'f32')))
      .toMatchObject({
        denoiser: 'atrous-variance',
        denoiserAutoResolution: {
          reason: 'host-neural-weights-device-infeasible',
          neuralTensorPrecision: 'f32',
        },
      });
    expect(parseHybridEngineOptions(options(weights, device, 'auto', 'auto')))
      .toMatchObject({
        denoiser: 'neural',
        denoiserAutoResolution: {
          reason: 'host-neural-weights',
          neuralTensorPrecision: 'f16',
        },
      });
    expect(device.createBuffer).not.toHaveBeenCalled();
  });

  it('falls back deterministically when auto cannot use certified f16', () => {
    const noFeature = parseHybridEngineOptions(
      options(model(true), limitedDevice(false), 'auto'),
    );
    expect(noFeature.denoiserAutoResolution).toMatchObject({
      resolved: 'atrous-variance',
      reason: 'host-neural-weights-device-infeasible',
      neuralTensorPrecision: 'f32',
      neuralDeviceFailure: expect.stringContaining('maxStorageBufferBindingSize=134217728'),
    });
    const noCertificate = parseHybridEngineOptions(
      options(model(false), limitedDevice(true), 'auto'),
    );
    expect(noCertificate.denoiserAutoResolution).toMatchObject({
      resolved: 'atrous-variance',
      reason: 'host-neural-weights-device-infeasible',
      neuralTensorPrecision: 'f32',
    });
  });

  it("never changes an explicit denoiser:'neural' into a fallback algorithm", () => {
    const forcedF32 = limitedDevice(true);
    expect(() => parseHybridEngineOptions(
      options(model(true), forcedF32, 'neural', 'f32'),
    )).toThrow(/denoiser:'neural' is infeasible before allocation.*maxStorageBufferBindingSize/);
    expect(forcedF32.createBuffer).not.toHaveBeenCalled();

    const noEligibleF16 = limitedDevice(false);
    expect(() => parseHybridEngineOptions(
      options(model(true), noEligibleF16, 'neural', 'auto'),
    )).toThrow(/denoiser:'neural' is infeasible before allocation.*maxStorageBufferBindingSize/);
    expect(noEligibleF16.createBuffer).not.toHaveBeenCalled();
  });

  it('makes explicit f16 fail for missing feature or certificate before allocation', () => {
    expect(() => parseHybridEngineOptions(
      options(model(true), limitedDevice(false), 'neural', 'f16'),
    )).toThrow(/requires 'shader-f16' to be enabled/);
    expect(() => parseHybridEngineOptions(
      options(model(false), limitedDevice(true), 'neural', 'f16'),
    )).toThrow(/requires a v2 checkpoint with passing mixed-precision certification/);
  });
});
