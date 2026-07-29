import { describe, expect, it } from 'vitest';

import { validateHybridEngineAdvancedOptions } from '../HybridEngineConfig.js';

describe('HybridEngine construction-only feature option dependencies', () => {
  it.each([
    { ppgMaxSpatialCells: 64 },
    { ppgMaxDTreeNodesPerCell: 17 },
    { ppgMixAlpha: 0.4 },
    { ppgDispatchInterval: 2 },
  ])('rejects a PPG-only option while PPG is disabled: %o', (options) => {
    expect(() => validateHybridEngineAdvancedOptions(options))
      .toThrow(/require ppgEnabled:true.*otherwise have no effect/i);
  });

  it.each([
    { nrcConfig: { warmupSteps: 4 } },
    { nrcWarmupSteps: 4 },
    { nrcSpreadC: 0.02 },
    { nrcMaxResidentBytes: 24_000_000 },
  ])('rejects an NRC-only option while NRC is disabled: %o', (options) => {
    expect(() => validateHybridEngineAdvancedOptions(options))
      .toThrow(/require nrcEnabled:true.*otherwise have no effect/i);
  });

  it.each([
    { rcTransmittedInterfaceBudget: 4 },
    { rcWeight: 0.25 },
    {
      cascadeDims: [{
        probes: [1, 1, 1],
        rays: 1,
        intervalNear: 0,
        intervalFar: 1,
      }],
    },
  ])('rejects an RC-only option while RC is disabled: %o', (options) => {
    expect(() => validateHybridEngineAdvancedOptions(options))
      .toThrow(/require rcEnabled:true.*otherwise have no effect/i);
  });

  it('accepts subsystem options when their immutable construction gate is enabled', () => {
    expect(() => validateHybridEngineAdvancedOptions({
      ppgEnabled: true,
      ppgMaxSpatialCells: 64,
      ppgMaxDTreeNodesPerCell: 17,
      ppgMixAlpha: 0.4,
      ppgDispatchInterval: 2,
      nrcEnabled: true,
      nrcConfig: { warmupSteps: 4 },
      rcEnabled: true,
      rcTransmittedInterfaceBudget: 4,
      rcWeight: 0.25,
    })).not.toThrow();
  });

  it('reports invalid values before reporting a missing feature gate', () => {
    expect(() => validateHybridEngineAdvancedOptions({ ppgMixAlpha: 1 }))
      .toThrow(/strictly between 0 and 1/i);
    expect(() => validateHybridEngineAdvancedOptions({ nrcWarmupSteps: 1.5 }))
      .toThrow(/safe integer/i);
    expect(() => validateHybridEngineAdvancedOptions({ rcWeight: 2 }))
      .toThrow(/rcWeight.*<= 1/i);
  });
});
