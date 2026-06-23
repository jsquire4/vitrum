import { describe, expect, it } from 'vitest';

import { parseHybridEngineOptions } from '../HybridEngineConfig.js';
import type { HybridEngineOptions } from '../HybridEngineOptions.js';
import { WALKAROUND_DENOISER_UNET_SPEC } from '../neural/unetArchitecture.js';
import type { LayerWeights, ModelWeights, NeuralCheckpointMetadata } from '../neural/weights.js';

const PRODUCTION_CHECKPOINT: NeuralCheckpointMetadata = {
  id: 'prod-fixture',
  trainingSamples: 512,
  noisySpp: 1,
  cleanSpp: 4096,
  auxiliaryInputs: ['albedo', 'normal'],
  captureSource: 'gpu-reference-capture',
  captureBackend: 'pt-webgpu-full',
  tonemap: 'linear-hdr',
  hardware: 'real-adapter-fixture',
  qualityReport: { status: 'pass', reportPath: 'tools/neural-denoiser-training/reports/prod-fixture.json' },
};

function fakeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    features: new Set<string>(),
    queue: { writeBuffer() {}, writeTexture() {}, submit() {} },
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

function hostWeights(): ModelWeights {
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
  return { layers };
}

function productionHostWeights(): ModelWeights {
  return { ...hostWeights(), checkpoint: PRODUCTION_CHECKPOINT };
}

function incompleteProductionHostWeights(): ModelWeights {
  return {
    ...hostWeights(),
    checkpoint: {
      ...PRODUCTION_CHECKPOINT,
      trainingSamples: 32,
      cleanSpp: 256,
      auxiliaryInputs: ['albedo'],
      qualityReport: { status: 'pass' },
    },
  };
}

function malformedHostWeights(): ModelWeights {
  return { layers: [] };
}

describe('learned-system option parsing', () => {
  it("keeps denoiser:'auto' on the non-learned default when no host model assets exist", () => {
    const cfg = parseHybridEngineOptions(baseOpts({ denoiser: 'auto' }));

    expect(cfg.denoiser).toBe('atrous-variance');
    expect(cfg.denoiserAutoResolution).toEqual({
      requested: 'auto',
      resolved: 'atrous-variance',
      reason: 'no-host-model-assets',
      packageProvidesProductionWeights: false,
      defaultEnabled: false,
      neuralCheckpointProductionReady: false,
      neuralCheckpointMissing: ['checkpoint metadata'],
    });
    expect(cfg.neuralWeights).toBeUndefined();
    expect(cfg.nrcEnabled).toBe(0);
    expect(cfg.ppgEnabled).toBe(0);
  });

  it("resolves denoiser:'auto' to neural only when full-tier host production weights exist", () => {
    const weights = productionHostWeights();
    const cfg = parseHybridEngineOptions(baseOpts({
      denoiser: 'auto',
      neuralWeights: weights,
    }));

    expect(cfg.denoiser).toBe('neural');
    expect(cfg.neuralWeights).toBe(weights);
    expect(cfg.denoiserAutoResolution).toMatchObject({
      resolved: 'neural',
      reason: 'host-neural-weights',
      packageProvidesProductionWeights: false,
      defaultEnabled: false,
      neuralCheckpointProductionReady: true,
      neuralCheckpointMissing: [],
    });
  });

  it("does not auto-select neural for shape-valid non-production weights", () => {
    const weights = hostWeights();
    const cfg = parseHybridEngineOptions(baseOpts({
      denoiser: 'auto',
      neuralWeights: weights,
    }));

    expect(cfg.denoiser).toBe('atrous-variance');
    expect(cfg.neuralWeights).toBe(weights);
    expect(cfg.neuralCheckpointAssessment.productionReady).toBe(false);
    expect(cfg.denoiserAutoResolution).toMatchObject({
      resolved: 'atrous-variance',
      reason: 'host-neural-weights-not-production-ready',
      neuralCheckpointProductionReady: false,
      neuralCheckpointMissing: ['checkpoint metadata'],
    });
  });

  it('requires production checkpoint thresholds before auto-selecting neural', () => {
    const cfg = parseHybridEngineOptions(baseOpts({
      denoiser: 'auto',
      neuralWeights: incompleteProductionHostWeights(),
    }));

    expect(cfg.denoiser).toBe('atrous-variance');
    expect(cfg.neuralCheckpointAssessment.productionReady).toBe(false);
    expect(cfg.neuralCheckpointAssessment.missing).toEqual(expect.arrayContaining([
      'trainingSamples>=500',
      'cleanSpp>=4096',
      'auxiliaryInputs.normal',
      'qualityReport.reportPath',
    ]));
    expect(cfg.denoiserAutoResolution).toMatchObject({
      resolved: 'atrous-variance',
      reason: 'host-neural-weights-not-production-ready',
      neuralCheckpointProductionReady: false,
    });
  });

  it("rejects denoiser:'auto' host weights that do not match the U-Net checkpoint contract", () => {
    expect(() => parseHybridEngineOptions(baseOpts({
      denoiser: 'auto',
      neuralWeights: malformedHostWeights(),
    }))).toThrow(/neuralWeights must match.*missing weights for layer 'enc1_conv'/);
  });

  it("rejects denoiser:'neural' host weights that do not match the U-Net checkpoint contract", () => {
    expect(() => parseHybridEngineOptions(baseOpts({
      denoiser: 'neural',
      neuralWeights: malformedHostWeights(),
    }))).toThrow(/neuralWeights must match.*missing weights for layer 'enc1_conv'/);
  });

  it("does not auto-select neural on tier:'lite', even when host weights exist", () => {
    const cfg = parseHybridEngineOptions(baseOpts({
      tier: 'lite',
      denoiser: 'auto',
      neuralWeights: hostWeights(),
    }));

    expect(cfg.denoiser).toBe('atrous-variance');
    expect(cfg.denoiserAutoResolution).toMatchObject({
      resolved: 'atrous-variance',
      reason: 'lite-neural-unavailable',
      packageProvidesProductionWeights: false,
      defaultEnabled: false,
      neuralCheckpointProductionReady: false,
      neuralCheckpointMissing: ['checkpoint metadata'],
    });
  });

  it("resolves denoiser:'auto' to OIDN when a host model URL is supplied", () => {
    const cfg = parseHybridEngineOptions(baseOpts({
      denoiser: 'auto',
      extensions: {
        'walkaround-hybrid': {
          oidnModelUrl: '/models/oidn_rt_hdr_alb_nrm.onnx',
        },
      },
    }));

    expect(cfg.denoiser).toBe('oidn-final');
    expect(cfg.oidnModelUrl).toBe('/models/oidn_rt_hdr_alb_nrm.onnx');
    expect(cfg.denoiserAutoResolution).toMatchObject({
      resolved: 'oidn-final',
      reason: 'host-oidn-model-url',
      packageProvidesProductionWeights: false,
      defaultEnabled: false,
    });
  });

  it('clamps learned-system cadence and mixture knobs into the effective config', () => {
    const cfg = parseHybridEngineOptions(baseOpts({
      ppgEnabled: true,
      nrcEnabled: true,
      ppgDispatchInterval: 0.25,
      ppgMixAlpha: 5,
      nrcWarmupSteps: -3.5,
      nrcSpreadC: -0.25,
    }));

    expect(cfg.ppgEnabled).toBe(1);
    expect(cfg.nrcEnabled).toBe(1);
    expect(cfg.ppgDispatchInterval).toBe(1);
    expect(cfg.ppgMixAlpha).toBe(1);
    expect(cfg.nrcWarmupSteps).toBe(0);
    expect(cfg.nrcSpreadC).toBe(0);
  });
});
