import { describe, expect, it } from 'vitest';

import { parseHybridEngineOptions } from '../HybridEngineConfig.js';
import type { HybridEngineOptions } from '../HybridEngineOptions.js';
import { WALKAROUND_DENOISER_UNET_SPEC } from '../neural/unetArchitecture.js';
import type { LayerWeights, ModelWeights, NeuralCheckpointMetadata } from '../neural/weights.js';
import { NEURAL_PREPROCESSING_CONTRACT } from '../neural/preprocessing.js';

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
  preprocessing: NEURAL_PREPROCESSING_CONTRACT,
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
  return { ...hostWeights(), formatVersion: 2, checkpoint: PRODUCTION_CHECKPOINT };
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
  it('rejects unknown constructor and nested option keys while leaving extensions open', () => {
    expect(() => parseHybridEngineOptions(baseOpts({ widht: 64 } as never)))
      .toThrow(/options: unknown key "widht"/);
    expect(() => parseHybridEngineOptions(baseOpts({
      tuning: { directFirelyClamp: 1 } as never,
    }))).toThrow(/options\.tuning: unknown key "directFirelyClamp"/);
    expect(() => parseHybridEngineOptions(baseOpts({
      gtao: { radiusPx: 12, intensitty: 2 } as never,
    }))).toThrow(/options\.gtao: unknown key "intensitty"/);
    expect(() => parseHybridEngineOptions(baseOpts({
      extensions: { 'future-host': { arbitrary: true } },
    }))).not.toThrow();
  });

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
      neuralCheckpointMissing: ['formatVersion=2', 'checkpoint metadata'],
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

  it("does not auto-select neural for shape-valid uncertified weights", () => {
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
      neuralCheckpointMissing: ['formatVersion=2', 'checkpoint metadata'],
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

  it("rejects explicit denoiser:'neural' for a shape-valid uncertified v1 checkpoint", () => {
    expect(() => parseHybridEngineOptions(baseOpts({
      denoiser: 'neural',
      neuralWeights: { ...hostWeights(), formatVersion: 1 },
    }))).toThrow(/requires a v2 production checkpoint.*formatVersion=2.*checkpoint metadata/);
  });

  it("rejects explicit denoiser:'neural' for mismatched v2 preprocessing metadata", () => {
    const weights: ModelWeights = {
      ...productionHostWeights(),
      checkpoint: {
        ...PRODUCTION_CHECKPOINT,
        preprocessing: {
          ...NEURAL_PREPROCESSING_CONTRACT,
          radianceScale: NEURAL_PREPROCESSING_CONTRACT.radianceScale / 2,
        },
      },
    };
    expect(() => parseHybridEngineOptions(baseOpts({
      denoiser: 'neural',
      neuralWeights: weights,
    }))).toThrow(/requires a v2 production checkpoint.*preprocessing=runtime-contract/);
  });
  it("accepts explicit denoiser:'neural' only for matching v2 production metadata", () => {
    const cfg = parseHybridEngineOptions(baseOpts({
      denoiser: 'neural',
      neuralWeights: productionHostWeights(),
    }));
    expect(cfg.denoiser).toBe('neural');
    expect(cfg.neuralCheckpointAssessment.productionReady).toBe(true);
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
      neuralCheckpointMissing: ['formatVersion=2', 'checkpoint metadata'],
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

  it('preserves valid learned-system cadence knobs and PPG mixture exactly', () => {
    const cfg = parseHybridEngineOptions(baseOpts({
      ppgEnabled: true,
      nrcEnabled: true,
      ppgDispatchInterval: 2,
      ppgMixAlpha: 0.35,
      nrcWarmupSteps: 3,
      nrcSpreadC: 0.25,
      nrcMaxResidentBytes: 24_000_000,
    }));

    expect(cfg.ppgEnabled).toBe(1);
    expect(cfg.nrcEnabled).toBe(1);
    expect(cfg.ppgDispatchInterval).toBe(2);
    expect(cfg.ppgMixAlpha).toBe(0.35);
    expect(cfg.nrcConfig.warmupSteps).toBe(3);
    expect(cfg.nrcConfig.spreadC).toBe(0.25);
    expect(cfg.nrcConfig.maxNrcResidentBytes).toBe(24_000_000);
  });

  it('resolves every public NRC field into one executable config contract', () => {
    const requested = {
      levels: 3,
      featuresPerEntry: 2,
      tableSize: 128,
      nMin: 8,
      growth: 1.5,
      oneBlobBins: 4,
      width: 32,
      hidden: 2,
      spreadC: 0.2,
      recordCap: 64,
      learningRate: 0.005,
      tableLearningRate: 0.05,
      useF16: true,
      tileB: 8,
      warmupSteps: 5,
      maxNrcResidentBytes: 50_000_000,
    } as const;
    const cfg = parseHybridEngineOptions(baseOpts({
      nrcEnabled: true,
      nrcConfig: requested,
    }));

    expect(cfg.nrcConfig).toEqual(requested);
  });

  it('accepts agreeing NRC aliases and rejects silent alias overrides', () => {
    expect(parseHybridEngineOptions(baseOpts({
      nrcConfig: {
        warmupSteps: 3,
        spreadC: 0.25,
        maxNrcResidentBytes: 24_000_000,
      },
      nrcWarmupSteps: 3,
      nrcSpreadC: 0.25,
      nrcMaxResidentBytes: 24_000_000,
    })).nrcConfig).toMatchObject({
      warmupSteps: 3,
      spreadC: 0.25,
      maxNrcResidentBytes: 24_000_000,
    });

    expect(() => parseHybridEngineOptions(baseOpts({
      nrcConfig: { warmupSteps: 4 },
      nrcWarmupSteps: 3,
    }))).toThrow(/nrcWarmupSteps.*disagrees with nrcConfig\.warmupSteps/);
    expect(() => parseHybridEngineOptions(baseOpts({
      nrcConfig: { spreadC: 0.5 },
      nrcSpreadC: 0.25,
    }))).toThrow(/nrcSpreadC.*disagrees with nrcConfig\.spreadC/);
    expect(() => parseHybridEngineOptions(baseOpts({
      nrcConfig: { maxNrcResidentBytes: 25_000_000 },
      nrcMaxResidentBytes: 24_000_000,
    }))).toThrow(/nrcMaxResidentBytes.*disagrees with nrcConfig\.maxNrcResidentBytes/);
  });

  it.each([
    [{ ppgDispatchInterval: 0.25 }, /ppgDispatchInterval.*safe integer/i],
    [{ nrcWarmupSteps: -3.5 }, /nrcWarmupSteps.*safe integer/i],
    [{ nrcSpreadC: -0.25 }, /nrcSpreadC.*>= 0/i],
  ] as const)('rejects malformed learned-system cadence %# instead of repairing it', (value, message) => {
    expect(() => parseHybridEngineOptions(baseOpts(value))).toThrow(message);
  });

  it.each([0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid NRC resident-byte budget %s before pipeline allocation',
    (nrcMaxResidentBytes) => {
      expect(() => parseHybridEngineOptions(baseOpts({ nrcMaxResidentBytes })))
        .toThrow(/maxNrcResidentBytes must be a positive safe integer/i);
    },
  );

  it.each([0, 1, -0.1, 1.1, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects invalid PPG mixture alpha %s instead of clamping it',
    (ppgMixAlpha) => {
      expect(() => parseHybridEngineOptions(baseOpts({ ppgMixAlpha })))
        .toThrow(/ppgMixAlpha must be finite and strictly between 0 and 1/);
    },
  );

  it('rejects PPG and GRIS together because GRIS bypasses the guide proposal', () => {
    expect(() => parseHybridEngineOptions(baseOpts({
      ppgEnabled: true,
      grisReuse: true,
    }))).toThrow(/ppgEnabled and grisReuse cannot be enabled together/);
  });

  it('accepts the canonical minimum neural internal shape', () => {
    const cfg = parseHybridEngineOptions(baseOpts({
      width: 8,
      height: 8,
      denoiser: 'neural',
      neuralWeights: productionHostWeights(),
    }));

    expect(cfg.denoiser).toBe('neural');
    expect(cfg.resolutionFactor).toBe(1);
  });

  it('accepts odd initial neural dimensions during pure option parsing', () => {
    expect(() => parseHybridEngineOptions(baseOpts({
      width: 9,
      height: 8,
      denoiser: 'neural',
      neuralWeights: productionHostWeights(),
    }))).not.toThrow();
  });

  it('validates the quality-scaled internal shape rather than physical dimensions', () => {
    expect(() => parseHybridEngineOptions(baseOpts({
      width: 16,
      height: 16,
      qualityTier: 'low',
      denoiser: 'neural',
      neuralWeights: productionHostWeights(),
    }))).not.toThrow();

    expect(() => parseHybridEngineOptions(baseOpts({
      width: 17,
      height: 16,
      qualityTier: 'low',
      denoiser: 'neural',
      neuralWeights: productionHostWeights(),
    }))).not.toThrow();
  });

  it('also accepts an odd shape when auto resolves to neural', () => {
    expect(() => parseHybridEngineOptions(baseOpts({
      width: 9,
      height: 8,
      denoiser: 'auto',
      neuralWeights: productionHostWeights(),
    }))).not.toThrow();
  });
});
