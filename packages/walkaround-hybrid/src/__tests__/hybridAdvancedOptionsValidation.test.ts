import { describe, expect, it, vi } from 'vitest';

import { validateHybridEngineAdvancedOptions } from '../index.js';

describe('validateHybridEngineAdvancedOptions', () => {
  it('accepts a representative data-only advanced bag without invoking callbacks', () => {
    const isSceneReady = vi.fn(() => true);
    const getPipelineRebuildKey = vi.fn(() => 'next');
    const onWarning = vi.fn();
    const resolveEnvironmentMap = vi.fn();

    expect(() => validateHybridEngineAdvancedOptions({
      gpuSkinning: true,
      isSceneReady,
      pipelineRebuildKey: 'initial',
      getPipelineRebuildKey,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 3,
      skyTint: [0.2, 0.3, 0.4],
      skyIrradiance: 0.5,
      qualityTier: 'high',
      tier: 'full',
      gtaoMode: 'quarter',
      diSpatialPasses: 2,
      giSpatialPasses: 1,
      ddgiUpdateDivisor: 4,
      targetFrameIntervalMs: null,
      temporalAccumAlpha: 0.1,
      tuning: { triIntersectEpsilon: 1e-6 },
      caustic: { boost: 1, visClamp: 0.75 },
      stainedGlass: { sunCaustic: true, skyAperture: false },
      adaptiveSamplingThresholds: [0.01, 0.1],
      gtao: { radiusPx: 16, intensity: 2, bilateralDepthSigma: 0.25 },
      indirectFireflyClamp: [1, 2, 3],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      ppgEnabled: true,
      ppgDispatchInterval: 2,
      regir: { enabled: true, cellsPerAxis: 8, candidatesPerCell: 16, survivorsPerCell: 4 },
      rcEnabled: true,
      rcTransmittedInterfaceBudget: 6,
      rcWeight: 0.5,
      nrcEnabled: true,
      nrcConfig: {
        levels: 4,
        featuresPerEntry: 2,
        tableSize: 1024,
        width: 48,
        useF16: false,
        warmupSteps: 8,
        spreadC: 0.01,
      },
      nrcWarmupSteps: 8,
      nrcSpreadC: 0.01,
      onWarning,
      extensions: {
        foreign: { preserved: true },
        'walkaround-hybrid': {
          oidnModelUrl: '/models/oidn.onnx',
          oidnExecutionProviders: ['webgpu', 'wasm'],
          neuralTensorStorage: 'f32',
          bvhMode: 'tlas',
          resolveEnvironmentMap,
        },
      },
    })).not.toThrow();

    expect(isSceneReady).not.toHaveBeenCalled();
    expect(getPipelineRebuildKey).not.toHaveBeenCalled();
    expect(onWarning).not.toHaveBeenCalled();
    expect(resolveEnvironmentMap).not.toHaveBeenCalled();
  });

  it.each(['device', 'width', 'height'] as const)(
    'rejects facade-owned %s instead of ignoring it',
    (key) => {
      expect(() => validateHybridEngineAdvancedOptions({ [key]: 1 }))
        .toThrow(/unknown key/i);
    },
  );

  it('rejects prototype, symbol, accessor, non-enumerable, and nested typo surfaces', () => {
    expect(() => validateHybridEngineAdvancedOptions(new (class Bag {})()))
      .toThrow(/prototype/i);

    const symbolBag = { [Symbol('hidden')]: true };
    expect(() => validateHybridEngineAdvancedOptions(symbolBag)).toThrow(/symbol/i);

    const accessorBag: Record<string, unknown> = {};
    Object.defineProperty(accessorBag, 'debug', { enumerable: true, get: () => true });
    expect(() => validateHybridEngineAdvancedOptions(accessorBag)).toThrow(/data property/i);

    const hiddenBag: Record<string, unknown> = {};
    Object.defineProperty(hiddenBag, 'debug', { enumerable: false, value: true });
    expect(() => validateHybridEngineAdvancedOptions(hiddenBag)).toThrow(/enumerable/i);

    expect(() => validateHybridEngineAdvancedOptions({
      extensions: { 'walkaround-hybrid': { bvhMod: 'tlas' } },
    })).toThrow(/unknown key.*bvhMod/i);
    expect(() => validateHybridEngineAdvancedOptions({
      nrcConfig: { tableRows: 1024 },
    })).toThrow(/nrcConfig.*unknown key.*tableRows/i);
  });

  it.each([
    [{ debug: 1 }, /debug.*boolean/i],
    [{ qualityTier: 'cinematic' }, /qualityTier.*one of/i],
    [{ ddgiUpdateDivisor: 1.5 }, /ddgiUpdateDivisor.*safe integer/i],
    [{ targetFrameIntervalMs: Number.NaN }, /targetFrameIntervalMs.*finite/i],
    [{ temporalAccumAlpha: 1.01 }, /temporalAccumAlpha.*<= 1/i],
    [{ adaptiveSamplingThresholds: [0.2, 0.1] }, /low must be <= high/i],
    [{ atrousDirectSigmas: [1, 0, 1] }, /atrousDirectSigmas\[1\].*> 0/i],
    [{ ppgDispatchInterval: 0 }, /ppgDispatchInterval.*safe integer/i],
    [{ rcWeight: -0.1 }, /rcWeight.*>= 0/i],
    [{ rcTransmittedInterfaceBudget: 0 }, /rcTransmittedInterfaceBudget.*\[1, 8\]/i],
    [{ rcTransmittedInterfaceBudget: 8.5 }, /rcTransmittedInterfaceBudget.*\[1, 8\]/i],
    [{ nrcWarmupSteps: 1.5 }, /nrcWarmupSteps.*safe integer/i],
    [{ nrcConfig: { useF16: 'yes' } }, /NRC useF16 must be a boolean/i],
    [{ nrcConfig: { recordCap: 0 } }, /NRC recordCap must be a positive safe integer/i],
    [{ extensions: { 'walkaround-hybrid': { bvhMode: 'auto' } } }, /bvhMode.*one of/i],
    [{ extensions: { 'walkaround-hybrid': { oidnExecutionProviders: [] } } }, /must not be empty/i],
  ] as const)('rejects malformed supplied value %#', (advanced, expected) => {
    expect(() => validateHybridEngineAdvancedOptions(advanced)).toThrow(expected);
  });
});
