import { describe, expect, it } from 'vitest';

import { parseHybridEngineOptions } from '../HybridEngineConfig.js';
import type { HybridEngineOptions } from '../HybridEngineOptions.js';
import type { ModelWeights } from '../neural/weights.js';

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
    });
    expect(cfg.neuralWeights).toBeUndefined();
    expect(cfg.nrcEnabled).toBe(0);
    expect(cfg.ppgEnabled).toBe(0);
  });

  it("resolves denoiser:'auto' to neural only when full-tier host weights exist", () => {
    const weights = hostWeights();
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
    });
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
    }));

    expect(cfg.ppgEnabled).toBe(1);
    expect(cfg.nrcEnabled).toBe(1);
    expect(cfg.ppgDispatchInterval).toBe(1);
    expect(cfg.ppgMixAlpha).toBe(1);
    expect(cfg.nrcWarmupSteps).toBe(0);
  });
});
