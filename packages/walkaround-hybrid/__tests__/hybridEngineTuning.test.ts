/**
 * Theme-H (2026-05-30) — characterization tests for the nested `tuning`
 * namespace + the `parseHybridEngineOptions` validate/derive split.
 *
 * The API-shape change: the ~13 redundant FLAT audit-knob fields on
 * `HybridEngineOptions` (emitterDist2Floor, directFireflyClamp, ...,
 * restirGiSpatialCoplanarTol) were DELETED; host overrides now go through the
 * nested `opts.tuning?.<key>` namespace (`Partial<Tunables>`).
 *
 * CORE RULE — behaviour-preserving for resolved values: for equivalent input,
 * `readTunables` must resolve every knob to the SAME value the old flat shape
 * produced. These tests:
 *   1. capture the OLD flat-shape resolved values as GOLDEN (by simulating the
 *      old `grouped > flat > default` precedence directly), then assert the new
 *      `tuning`-shape produces identical resolved values;
 *   2. assert defaults (empty tuning) are unchanged;
 *   3. pin the subsystem-grouped sub-objects (caustic / gtao /
 *      adaptiveSamplingThresholds) still win over `tuning`;
 *   4. pin the validate/derive split — same throws, same derived config.
 */

import { describe, it, expect } from 'vitest';
import {
  readTunables,
  TUNABLE_DEFINITIONS,
  type Tunables,
} from '../src/HybridEngineTuning.js';
import {
  validateHybridEngineOptions,
  deriveHybridEngineConfig,
} from '../src/HybridEngine.js';
import type { HybridEngineOptions } from '../src/HybridEngineOptions.js';
import { resolveQualityPreset } from '../src/HybridEngineQualityPreset.js';

// ── Fixtures ─────────────────────────────────────────────────────────────────

function fakeDevice(): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    features: new Set<string>(),
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
  } as HybridEngineOptions;
}

/** The eight knobs that also have a subsystem sub-object (grouped source). */
const GROUPED_KEYS = new Set<keyof Tunables>([
  'causticBoost', 'causticVisClamp',
  'gtaoRadiusPx', 'gtaoIntensity', 'gtaoDepthThreshold', 'gtaoBilateralDepthSigma',
  'adaptiveSamplingThresholdLow', 'adaptiveSamplingThresholdHigh',
]);

/**
 * GOLDEN ORACLE — reproduce the PRE-Theme-H resolution exactly:
 * `grouped > flat > default`. `grouped` reads caustic/gtao/adaptive sub-objects;
 * `flat` reads the (now-deleted) top-level fields off a Record-typed bag.
 */
function resolveOldFlatShape(
  grouped: Partial<Record<keyof Tunables, number>>,
  flat: Partial<Record<keyof Tunables, number>>,
): Tunables {
  const out: Partial<Tunables> = {};
  for (const def of TUNABLE_DEFINITIONS) {
    const k = def.key;
    const g = grouped[k];
    const f = flat[k];
    (out as Record<string, number>)[k] =
      g !== undefined ? g : f !== undefined ? f : def.default;
  }
  return out as Tunables;
}

// A representative non-default override value per knob (each distinct from the
// Cornell default so a silent fallback to default would fail the assertion).
const ALL_KNOBS_OVERRIDE: Tunables = (() => {
  const out: Partial<Tunables> = {};
  for (const def of TUNABLE_DEFINITIONS) {
    (out as Record<string, number>)[def.key] = def.default * 2 + 1;
  }
  return out as Tunables;
})();

// ── Defaults ───────────────────────────────────────────────────────────────

describe('readTunables — defaults (empty tuning)', () => {
  it('every knob resolves to its TUNABLE_DEFINITIONS default when tuning is omitted', () => {
    const resolved = readTunables(baseOpts());
    const golden = resolveOldFlatShape({}, {}); // no grouped, no flat → all defaults
    expect(resolved).toEqual(golden);
    for (const def of TUNABLE_DEFINITIONS) {
      expect(resolved[def.key]).toBe(def.default);
    }
  });

  it('an empty tuning object is identical to omitting it', () => {
    expect(readTunables(baseOpts({ tuning: {} }))).toEqual(readTunables(baseOpts()));
  });
});

// ── Characterization: new tuning shape == old flat shape ─────────────────────

describe('readTunables — new tuning shape reproduces old flat-shape resolved values', () => {
  it('full knob matrix: tuning:{...all} == old flat:{...all}', () => {
    // OLD shape: every knob passed as a top-level flat field (no grouped).
    const golden = resolveOldFlatShape({}, { ...ALL_KNOBS_OVERRIDE });
    // NEW shape: the same values under the nested `tuning` namespace.
    const resolved = readTunables(baseOpts({ tuning: { ...ALL_KNOBS_OVERRIDE } }));
    expect(resolved).toEqual(golden);
    // And every knob must equal the override (not the default) — proves the
    // tuning path is actually consulted for all 21 knobs.
    for (const def of TUNABLE_DEFINITIONS) {
      expect(resolved[def.key]).toBe(ALL_KNOBS_OVERRIDE[def.key]);
    }
  });

  it('per-knob: a single tuning override resolves to that value, rest stay default', () => {
    for (const def of TUNABLE_DEFINITIONS) {
      const override = def.default * 3 + 0.5;
      const resolved = readTunables(baseOpts({ tuning: { [def.key]: override } }));
      const golden = resolveOldFlatShape({}, { [def.key]: override });
      expect(resolved).toEqual(golden);
      expect(resolved[def.key]).toBe(override);
    }
  });
});

// ── Subsystem sub-objects still win over tuning (grouped > tuning) ───────────

describe('readTunables — subsystem grouped sub-objects override tuning (grouped > tuning > default)', () => {
  it('caustic / gtao / adaptiveSamplingThresholds win over tuning for their knobs', () => {
    const opts = baseOpts({
      // tuning sets EVERY knob to a sentinel...
      tuning: { ...ALL_KNOBS_OVERRIDE },
      // ...but the grouped sub-objects set distinct values for the 8 grouped knobs.
      caustic: { boost: 100, visClamp: 101 },
      gtao: { radiusPx: 200, intensity: 201, depthThresholdWorldUnits: 202, bilateralDepthSigma: 203 },
      adaptiveSamplingThresholds: [300, 301],
    });
    const resolved = readTunables(opts);

    const grouped: Partial<Record<keyof Tunables, number>> = {
      causticBoost: 100, causticVisClamp: 101,
      gtaoRadiusPx: 200, gtaoIntensity: 201, gtaoDepthThreshold: 202, gtaoBilateralDepthSigma: 203,
      adaptiveSamplingThresholdLow: 300, adaptiveSamplingThresholdHigh: 301,
    };
    const golden = resolveOldFlatShape(grouped, { ...ALL_KNOBS_OVERRIDE });
    expect(resolved).toEqual(golden);

    // The 8 grouped knobs take the grouped value; the rest take the tuning value.
    for (const def of TUNABLE_DEFINITIONS) {
      if (GROUPED_KEYS.has(def.key)) {
        expect(resolved[def.key]).toBe(grouped[def.key]);
      } else {
        expect(resolved[def.key]).toBe(ALL_KNOBS_OVERRIDE[def.key]);
      }
    }
  });

  it('tuning fills in a grouped knob the sub-object omits (tuning > default)', () => {
    // gtao sub-object sets only radiusPx; tuning sets gtaoIntensity; the rest default.
    const resolved = readTunables(baseOpts({
      gtao: { radiusPx: 77 },
      tuning: { gtaoIntensity: 9 },
    }));
    expect(resolved.gtaoRadiusPx).toBe(77);   // grouped
    expect(resolved.gtaoIntensity).toBe(9);   // tuning (sub-object omitted it)
    expect(resolved.gtaoDepthThreshold).toBe(2.0); // default
  });
});

// ── parse split: validate ────────────────────────────────────────────────────

describe('validateHybridEngineOptions — throws (parse split, pure)', () => {
  it('does not throw on a valid default options object', () => {
    expect(() => validateHybridEngineOptions(baseOpts())).not.toThrow();
  });

  it('throws on tier:lite + rcEnabled / ppgEnabled / denoiser:neural / nrcEnabled', () => {
    expect(() => validateHybridEngineOptions(baseOpts({ tier: 'lite', rcEnabled: true }))).toThrow(/rcEnabled/);
    expect(() => validateHybridEngineOptions(baseOpts({ tier: 'lite', ppgEnabled: true }))).toThrow(/ppgEnabled/);
    expect(() => validateHybridEngineOptions(baseOpts({ tier: 'lite', denoiser: 'neural' }))).toThrow(/neural/);
    expect(() => validateHybridEngineOptions(baseOpts({ tier: 'lite', nrcEnabled: true }))).toThrow(/nrcEnabled/);
  });

  it('does NOT throw on tier:lite with the forbidden flags explicitly false', () => {
    expect(() => validateHybridEngineOptions(baseOpts({ tier: 'lite', rcEnabled: false }))).not.toThrow();
    expect(() => validateHybridEngineOptions(baseOpts({ tier: 'lite', nrcEnabled: false }))).not.toThrow();
  });

  it('accepts denoiser:none as an explicit pass-through mode', () => {
    expect(() => validateHybridEngineOptions(baseOpts({ denoiser: 'none' }))).not.toThrow();
  });

  it('throws on an unsupported denoiser enum', () => {
    const bad = { ...baseOpts(), denoiser: 'bogus-denoiser' } as unknown as HybridEngineOptions;
    expect(() => validateHybridEngineOptions(bad)).toThrow(/unsupported denoiser/);
  });

  it('throws on denoiser:neural without neuralWeights (tier:full)', () => {
    expect(() => validateHybridEngineOptions(baseOpts({ denoiser: 'neural' }))).toThrow(/neuralWeights/);
  });

  it('throws on denoiser:oidn-final without oidnModelUrl', () => {
    expect(() => validateHybridEngineOptions(baseOpts({ denoiser: 'oidn-final' }))).toThrow(/oidnModelUrl/);
    expect(() => validateHybridEngineOptions(baseOpts({
      denoiser: 'oidn-final',
      extensions: { 'walkaround-hybrid': { oidnModelUrl: '/m.onnx' } },
    }))).not.toThrow();
  });

  it('rejects ReGIR values and cross-field products outside the WGSL u32 domain', () => {
    expect(() => validateHybridEngineOptions(baseOpts({
      regir: { enabled: true, candidatesPerCell: 0x1_0000_0000 },
    }))).toThrow(/candidatesPerCell.*4294967295/);
    expect(() => validateHybridEngineOptions(baseOpts({
      regir: { enabled: true, cellsPerAxis: 1_024, survivorsPerCell: 4 },
    }))).toThrow(/invocation domain/);
  });
});

// ── parse split: derive ──────────────────────────────────────────────────────

describe('deriveHybridEngineConfig — defaulting record (parse split)', () => {
  function derive(opts: HybridEngineOptions) {
    const tier = opts.qualityTier ?? (opts.tier === 'lite' ? 'medium' : 'ultra');
    return deriveHybridEngineConfig(opts, resolveQualityPreset(tier));
  }

  it('threads the resolved tunables record (tuning-shape) into config.tunables', () => {
    const cfg = derive(baseOpts({ tuning: { directFireflyClamp: 8, triIntersectEpsilon: 1e-7 } }));
    expect(cfg.tunables.directFireflyClamp).toBe(8);
    expect(cfg.tunables.triIntersectEpsilon).toBe(1e-7);
    // Untouched knobs stay at default.
    expect(cfg.tunables.emitterDist2Floor).toBe(0.01);
  });

  it('default config (ultra) enables the canonical generalized-reuse path', () => {
    const cfg = derive(baseOpts());
    expect(cfg.denoiser).toBe('atrous-variance');
    expect(cfg.maxBounces).toBe(4);
    expect(cfg.indirectFireflyClamp).toEqual([1.0, 1.0, 1.0]);
    expect(cfg.nrcEnabled).toBe(0);
    // Resolved tunables all at default.
    for (const def of TUNABLE_DEFINITIONS) {
      expect(cfg.tunables[def.key]).toBe(def.default);
    }
  });

  it.each([0, -2, Number.NaN])(
    'rejects invalid maxBounces %s instead of normalizing it',
    (maxBounces) => {
      expect(() => validateHybridEngineOptions(baseOpts({ maxBounces })))
        .toThrow(/maxBounces.*positive safe integer/i);
    },
  );

  it.each([
    [{ targetFrameIntervalMs: Number.NaN }, /targetFrameIntervalMs.*finite/i],
    [{ indirectFireflyClamp: [Number.NaN, 2, 3] }, /indirectFireflyClamp\[0\].*finite/i],
    [{ atrousDirectSigmas: [1, Number.POSITIVE_INFINITY, 3] }, /atrousDirectSigmas\[1\].*finite/i],
    [{ cameraMoveResetThresholdSq: Number.NaN }, /cameraMoveResetThresholdSq.*finite/i],
    [{ temporalAccumAlpha: Number.POSITIVE_INFINITY }, /temporalAccumAlpha.*finite/i],
    [{ checkerboardMotionThresholdSq: Number.NEGATIVE_INFINITY }, /checkerboardMotionThresholdSq.*finite/i],
    [{ ddgiUpdateDivisor: Number.NaN }, /ddgiUpdateDivisor.*safe integer/i],
    [{ ppgDispatchInterval: Number.POSITIVE_INFINITY }, /ppgDispatchInterval.*safe integer/i],
    [{ ppgMaxSpatialCells: Number.NaN }, /ppgMaxSpatialCells.*integer/i],
    [{ ppgMaxDTreeNodesPerCell: Number.NEGATIVE_INFINITY }, /ppgMaxDTreeNodesPerCell.*integer/i],
    [{ tuning: { directFireflyClamp: Number.NaN } }, /directFireflyClamp.*finite/i],
  ] as const)('rejects malformed public tuning value %# before GPU state', (value, message) => {
    expect(() => validateHybridEngineOptions(baseOpts(value))).toThrow(message);
  });

  it('produces an identical config record over an option matrix', () => {
    const matrix: Partial<HybridEngineOptions>[] = [
      {},
      { denoiser: 'atrous' },
      { denoiser: 'bmfr', maxBounces: 8 },
      { qualityTier: 'low' },
      { tier: 'lite' },
      { rcEnabled: true, rcWeight: 0.25 },
      { tuning: { restirGiWCap: 32, glassMixScale: 0.9 } },
      { gtao: { radiusPx: 48 }, tuning: { gtaoIntensity: 3 } },
    ];
    for (const ov of matrix) {
      const a = derive(baseOpts(ov));
      const b = derive(baseOpts(ov));
      // Strip function-valued fields (getPipelineRebuildKey) before structural compare.
      const strip = (c: typeof a) => ({ ...c, getPipelineRebuildKey: undefined });
      expect(strip(a)).toEqual(strip(b));
    }
  });
});
