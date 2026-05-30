import { describe, it, expect } from 'vitest';
import {
  defaultSchedulerOptions,
  parseCausticConfig,
  parseAccumulationConfig,
  parseIblBakerConfig,
  type PTEngineWebGL2Options,
} from '../ptEngineWebGL2.js';

/**
 * Theme H characterization pins (Task 3b).
 *
 * The 2026-05-29 migration graduated first-class features (spectral / bdpt /
 * oidn / qualityMode) to typed options, but ~12 experimental TUNING knobs are
 * still sourced from the stringly-typed `opts.extensions` bag. Task 3b
 * consolidated every `opts.extensions?.['vitrum.ptWebgl.*']` read OUT of the
 * constructor body and INTO four frozen-config parser functions
 * (defaultSchedulerOptions / parseCausticConfig / parseAccumulationConfig /
 * parseIblBakerConfig).
 *
 * These tests are the behavior pin: for representative options objects (bag
 * values set AND unset → hit defaults) the parsed config must be IDENTICAL to
 * the pre-refactor golden values captured from the original inline reads. They
 * assert same default, same key, same coercion (extensionNumber /
 * extensionBoolean clamp/floor/finite semantics) for every one of the ~12 knobs.
 */

// A dummy device satisfies the EngineOptions `device: unknown` requirement; the
// parsers never touch it.
function makeOpts(extensions?: Record<string, unknown>): PTEngineWebGL2Options {
  return { device: {}, extensions } as unknown as PTEngineWebGL2Options;
}

describe('Theme H — defaultSchedulerOptions (scheduler knobs)', () => {
  it('returns the frozen capture-mode defaults when the bag is empty', () => {
    expect(defaultSchedulerOptions(undefined, 'capture')).toEqual({
      qualityMode: 'capture',
      adaptive: false,
      targetBatchMs: 0,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 1,
      initialSamplesPerFrame: 1,
      initialTileSize: 3, // DEFAULT_TILE_SIZE
      maxTileSize: 3, // DEFAULT_TILE_SIZE
      renderTargetBudgetBytes: 512 * 1024 * 1024,
    });
  });

  it('returns the frozen interactive-mode defaults when the bag is empty', () => {
    expect(defaultSchedulerOptions(undefined, 'interactive')).toEqual({
      qualityMode: 'interactive',
      adaptive: true,
      targetBatchMs: 40,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 48,
      initialSamplesPerFrame: 6,
      initialTileSize: 1,
      maxTileSize: 4,
      renderTargetBudgetBytes: 1024 * 1024 * 1024,
    });
  });

  it('overrides every scheduler knob from the bag with the original coercion', () => {
    const ext = {
      'vitrum.ptWebgl.samplesPerFrame': 30,
      'vitrum.ptWebgl.tileSize': 2,
      'vitrum.ptWebgl.maxSamplesPerFrame': 200, // clamped to 128
      'vitrum.ptWebgl.adaptiveScheduler': false,
      'vitrum.ptWebgl.targetBatchMs': 77,
      'vitrum.ptWebgl.maxTileSize': 6,
      'vitrum.ptWebgl.renderTargetBudgetBytes': 32 * 1024 * 1024, // floored to 64MiB min
    };
    expect(defaultSchedulerOptions(ext, 'final')).toEqual({
      qualityMode: 'final',
      adaptive: false,
      targetBatchMs: 77,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 128, // clampInt(200, 1, 128)
      initialSamplesPerFrame: 30, // clampInt(30, 1, 128)
      initialTileSize: 2, // clampInt(2, 1, final base.maxTileSize=4)
      maxTileSize: 6, // clampInt(6, 1, 8)
      renderTargetBudgetBytes: 64 * 1024 * 1024, // max(64MiB, 32MiB)
    });
  });

  it('clamps initialTileSize to the mode base.maxTileSize and budget to its floor', () => {
    // capture base.maxTileSize === DEFAULT_TILE_SIZE (3) — a requested tileSize of
    // 9 must clamp to 3, and a negative budget floors to 64MiB.
    const ext = {
      'vitrum.ptWebgl.tileSize': 9,
      'vitrum.ptWebgl.renderTargetBudgetBytes': -5,
    };
    const out = defaultSchedulerOptions(ext, 'capture');
    expect(out.initialTileSize).toBe(3);
    expect(out.renderTargetBudgetBytes).toBe(64 * 1024 * 1024);
  });

  it('ignores non-finite / wrong-typed bag values (falls back to default)', () => {
    const ext = {
      'vitrum.ptWebgl.samplesPerFrame': Number.NaN,
      'vitrum.ptWebgl.tileSize': 'big',
      'vitrum.ptWebgl.adaptiveScheduler': 'yes',
      'vitrum.ptWebgl.targetBatchMs': Number.POSITIVE_INFINITY,
    };
    // interactive defaults survive every malformed override.
    expect(defaultSchedulerOptions(ext, 'interactive')).toEqual({
      qualityMode: 'interactive',
      adaptive: true,
      targetBatchMs: 40,
      minSamplesPerFrame: 1,
      maxSamplesPerFrame: 48,
      initialSamplesPerFrame: 6,
      initialTileSize: 1,
      maxTileSize: 4,
      renderTargetBudgetBytes: 1024 * 1024 * 1024,
    });
  });
});

describe('Theme H — parseCausticConfig (radianceClamp bag knob)', () => {
  it('defaults radianceClamp to 0 when the bag is empty', () => {
    expect(parseCausticConfig(makeOpts()).radianceClamp).toBe(0);
  });

  it('reads + non-negative-clamps radianceClamp from the bag', () => {
    expect(
      parseCausticConfig(makeOpts({ 'vitrum.ptWebgl.radianceClamp': 12.5 })).radianceClamp,
    ).toBe(12.5);
    expect(
      parseCausticConfig(makeOpts({ 'vitrum.ptWebgl.radianceClamp': -3 })).radianceClamp,
    ).toBe(0);
  });

  it('ignores non-finite / wrong-typed radianceClamp (→ 0)', () => {
    expect(
      parseCausticConfig(makeOpts({ 'vitrum.ptWebgl.radianceClamp': Number.NaN })).radianceClamp,
    ).toBe(0);
    expect(
      parseCausticConfig(makeOpts({ 'vitrum.ptWebgl.radianceClamp': 'hot' })).radianceClamp,
    ).toBe(0);
  });
});

describe('Theme H — parseAccumulationConfig (pixel-adaptive + additive knobs)', () => {
  it('returns the all-off / cadence-4 defaults when the bag is empty', () => {
    expect(parseAccumulationConfig(makeOpts())).toEqual({
      pixelAdaptiveSampling: false,
      additiveAccumulation: false,
      pixelAdaptiveCadence: 4,
    });
  });

  it('pixel-adaptive sampling implies additive accumulation (original OR semantics)', () => {
    expect(
      parseAccumulationConfig(
        makeOpts({ 'vitrum.ptWebgl.pixelAdaptiveSampling': true }),
      ),
    ).toEqual({
      pixelAdaptiveSampling: true,
      additiveAccumulation: true, // implied
      pixelAdaptiveCadence: 4,
    });
  });

  it('additiveAccumulation can be requested without pixel-adaptive sampling', () => {
    expect(
      parseAccumulationConfig(
        makeOpts({ 'vitrum.ptWebgl.additiveAccumulation': true }),
      ),
    ).toEqual({
      pixelAdaptiveSampling: false,
      additiveAccumulation: true,
      pixelAdaptiveCadence: 4,
    });
  });

  it('reads pixelAdaptiveCadence and floors it to >= 1', () => {
    expect(
      parseAccumulationConfig(makeOpts({ 'vitrum.ptWebgl.pixelAdaptiveCadence': 9 }))
        .pixelAdaptiveCadence,
    ).toBe(9);
    expect(
      parseAccumulationConfig(makeOpts({ 'vitrum.ptWebgl.pixelAdaptiveCadence': 0 }))
        .pixelAdaptiveCadence,
    ).toBe(1);
    expect(
      parseAccumulationConfig(makeOpts({ 'vitrum.ptWebgl.pixelAdaptiveCadence': -5 }))
        .pixelAdaptiveCadence,
    ).toBe(1);
  });

  it('strictly equality-gates the boolean knobs (truthy non-true → false)', () => {
    expect(
      parseAccumulationConfig(
        makeOpts({
          'vitrum.ptWebgl.pixelAdaptiveSampling': 1, // not === true
          'vitrum.ptWebgl.additiveAccumulation': 'yes', // not === true
        }),
      ),
    ).toEqual({
      pixelAdaptiveSampling: false,
      additiveAccumulation: false,
      pixelAdaptiveCadence: 4,
    });
  });
});

describe('Theme H — parseIblBakerConfig (iblBakerMaxEntries bag knob)', () => {
  it('returns undefined cacheOpts (default LRU sizing) when the bag is empty', () => {
    expect(parseIblBakerConfig(makeOpts()).cacheOpts).toBeUndefined();
  });

  it('reads + floors + min-1-clamps iblBakerMaxEntries into maxEntries', () => {
    expect(
      parseIblBakerConfig(makeOpts({ 'vitrum.ptWebgl.iblBakerMaxEntries': 12.9 })).cacheOpts,
    ).toEqual({ maxEntries: 12 });
    expect(
      parseIblBakerConfig(makeOpts({ 'vitrum.ptWebgl.iblBakerMaxEntries': 0 })).cacheOpts,
    ).toEqual({ maxEntries: 1 });
  });

  it('ignores non-finite / wrong-typed iblBakerMaxEntries (→ undefined)', () => {
    expect(
      parseIblBakerConfig(makeOpts({ 'vitrum.ptWebgl.iblBakerMaxEntries': Number.NaN })).cacheOpts,
    ).toBeUndefined();
    expect(
      parseIblBakerConfig(makeOpts({ 'vitrum.ptWebgl.iblBakerMaxEntries': 'lots' })).cacheOpts,
    ).toBeUndefined();
  });
});
