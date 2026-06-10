/**
 * Task 4.2 (Theme A) — characterization test for the HybridEngine `_cfg`
 * consolidation.
 *
 * The refactor: the ~25 construction-IMMUTABLE tunable-cluster fields that the
 * constructor used to splat one-by-one onto `this._x` private fields are
 * collapsed into a single `private readonly _cfg: ParsedHybridEngineConfig`.
 * Every value that used to be read via a `this._x` field is now read via
 * `this._cfg.x`.
 *
 * CORE RULE — behaviour-preserving. Every resolved config value must resolve
 * IDENTICALLY after the refactor as it did before. This test pins that through
 * the STABLE observable surfaces that survive the field deletion:
 *
 *   1. `_cfg` itself equals `deriveHybridEngineConfig(opts, preset)` — i.e. the
 *      engine holds the parsed config verbatim.
 *   2. `_initStaticConfig()` carries the same per-field values the old splatted
 *      `_x` fields produced (golden = derived from the same options).
 *   3. `_denoiserFilterDeps()` carries the same tuple-cluster values.
 *   4. `_buildFrameDeps()` carries the same per-frame config values.
 *   5. `capabilities.maxBounces` (read from the migrated `_maxBounces`) is
 *      unchanged.
 *
 * The golden values are computed straight from `deriveHybridEngineConfig` (the
 * pure derive half the constructor consumes), so they are independent of any
 * `this._x` field. That makes the test valid both BEFORE the refactor (when the
 * deps builders read `this._x`) and AFTER (when they read `this._cfg.x`).
 */

import { describe, it, expect, vi } from 'vitest';
import { HybridEngine, deriveHybridEngineConfig } from '../src/HybridEngine.js';
import type { HybridEngineOptions } from '../src/HybridEngineOptions.js';
import { resolveQualityPreset } from '../src/HybridEngineQualityPreset.js';

// ── Minimal mock GPUDevice — ctor only stores it + builds DDGI (no GPU work). ─
function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
    limits: {
      maxStorageBuffersPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 8,
    },
    features: new Set<string>(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

/** A rich, NON-default option set so every migrated field carries a value
 *  distinct from its default — a silent fallback would change the value and
 *  trip an assertion. */
function richOpts(overrides: Partial<HybridEngineOptions> = {}): HybridEngineOptions {
  return {
    device: makeMockDevice(),
    width: 128,
    height: 96,
    primaryLightDir: [0.3, -0.7, 0.6],
    primaryLightIntensity: 2.5,
    skyTint: [0.5, 0.7, 1.0],
    skyIrradiance: 0.3,
    maxBounces: 7,
    verbose: true,
    debug: true,
    denoiser: 'bmfr',
    indirectFireflyClamp: [2.0, 3.0, 4.0],
    atrousDirectSigmas: [100, 6, 0.06],
    atrousIndirectSigmas: [30, 18, 0.4],
    restirPtReuse: true,
    nrcEnabled: false,
    gtaoMode: 'quarter',
    diSpatialPasses: 2,
    giSpatialPasses: 1,
    ddgiUpdateDivisor: 3,
    ppgDispatchInterval: 5,
    targetFrameIntervalMs: 12,
    tuning: { directFireflyClamp: 8, triIntersectEpsilon: 1e-7 },
    pipelineRebuildKey: 'rk-42',
    extensions: { 'walkaround-hybrid': { bvhMode: 'tlas' } },
    ...overrides,
  } as HybridEngineOptions;
}

/** Reproduce the engine's parse path (validate is a no-op for valid opts). */
function deriveFor(opts: HybridEngineOptions) {
  const tier = opts.qualityTier ?? (opts.tier === 'lite' ? 'medium' : 'ultra');
  return deriveHybridEngineConfig(opts, resolveQualityPreset(tier));
}

type AnyRec = Record<string, unknown>;

/** Invoke a private nullary method on the engine via its name (test seam). */
function callPrivate(engine: HybridEngine, name: string): AnyRec {
  const fn = (engine as unknown as Record<string, unknown>)[name] as (this: HybridEngine) => AnyRec;
  return fn.call(engine);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1 — _cfg holds the parsed config verbatim
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine._cfg — holds the derived config verbatim', () => {
  it('_cfg structurally equals deriveHybridEngineConfig(opts, preset)', () => {
    const opts = richOpts();
    const engine = new HybridEngine(opts);
    const golden = deriveFor(opts);
    const cfg = (engine as unknown as AnyRec)['_cfg'] as Record<string, unknown>;

    expect(cfg).toBeDefined();
    // Strip the function-valued field before structural compare (functions are
    // not stably comparable; pin it separately below).
    const strip = (c: Record<string, unknown>) => ({ ...c, getPipelineRebuildKey: undefined });
    expect(strip(cfg)).toEqual(strip(golden as unknown as Record<string, unknown>));
    expect(cfg['getPipelineRebuildKey']).toBe(golden.getPipelineRebuildKey);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 2 — _initStaticConfig() carries the migrated values unchanged
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine._initStaticConfig — migrated config values unchanged', () => {
  it('every init-host static field resolves to the derived-config value', () => {
    const opts = richOpts();
    const engine = new HybridEngine(opts);
    const golden = deriveFor(opts);
    const cfg = callPrivate(engine, '_initStaticConfig');

    expect(cfg['restirBvhModeOverride']).toBe(golden.restirBvhModeOverride);
    expect(cfg['denoiser']).toBe(golden.denoiser);
    expect(cfg['neuralWeights']).toBe(golden.neuralWeights);
    expect(cfg['oidnModelUrl']).toBe(golden.oidnModelUrl);
    expect(cfg['oidnExecutionProviders']).toBe(golden.oidnExecutionProviders);
    expect(cfg['verbose']).toBe(golden.verbose);
    expect(cfg['debug']).toBe(golden.debug);
    expect(cfg['cameraMoveResetThresholdSq']).toBe(golden.initTunables.cameraMoveResetThresholdSq);
    expect(cfg['temporalAccumAlpha']).toBe(golden.initTunables.temporalAccumAlpha);
    expect(cfg['gtaoMode']).toBe(golden.gtaoMode);
    expect(cfg['diSpatialPasses']).toBe(golden.diSpatialPasses);
    expect(cfg['giSpatialPasses']).toBe(golden.giSpatialPasses);
    // restirPtReuse / nrcEnabled forwarded as booleans (=== 1).
    expect(cfg['restirPtReuse']).toBe(golden.restirPtReuse === 1);
    expect(cfg['nrcEnabled']).toBe(golden.nrcEnabled === 1);
    expect(cfg['ppgDispatchInterval']).toBe(golden.ppgDispatchInterval);
    expect(cfg['regirConfig']).toBe(golden.regirConfig);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3 — _denoiserFilterDeps() carries the tuple cluster unchanged
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine._denoiserFilterDeps — tuple cluster unchanged', () => {
  it('firefly clamp + atrous sigmas + flags resolve to the derived-config values', () => {
    const opts = richOpts();
    const engine = new HybridEngine(opts);
    const golden = deriveFor(opts);
    const deps = callPrivate(engine, '_denoiserFilterDeps');

    expect(deps['indirectFireflyClamp']).toEqual(golden.indirectFireflyClamp);
    expect(deps['atrousDirectSigmas']).toEqual(golden.atrousDirectSigmas);
    expect(deps['atrousIndirectSigmas']).toEqual(golden.atrousIndirectSigmas);
    expect(deps['stainedGlassFlags']).toBe(golden.stainedGlassFlags);
    expect(deps['restirPtReuse']).toBe(golden.restirPtReuse);
    expect(deps['nrcEnabled']).toBe(golden.nrcEnabled);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4 — _buildFrameDeps() carries the per-frame config values unchanged
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine._buildFrameDeps — per-frame config values unchanged', () => {
  it('tunables / targetFrameIntervalMs / verbose / debug + denoiser-filter cluster', () => {
    const opts = richOpts();
    const engine = new HybridEngine(opts);
    const golden = deriveFor(opts);
    const deps = callPrivate(engine, '_buildFrameDeps');

    // After struct grouping: flags.tunables, control.targetFrameIntervalMs,
    // telemetry.verbose/debug, filter.* for the denoiser-filter cluster.
    // Cast each sub-object through AnyRec so string indexing works.
    const flags = deps['flags'] as AnyRec;
    const control = deps['control'] as AnyRec;
    const telemetry = deps['telemetry'] as AnyRec;
    const filter = deps['filter'] as AnyRec;
    expect(flags['tunables']).toEqual(golden.tunables);
    expect(control['targetFrameIntervalMs']).toBe(golden.targetFrameIntervalMs);
    expect(telemetry['verbose']).toBe(golden.verbose);
    expect(flags['debug']).toBe(golden.debug);
    // Grouped denoiser-filter cluster (filter sub-object).
    expect(filter['indirectFireflyClamp']).toEqual(golden.indirectFireflyClamp);
    expect(filter['atrousDirectSigmas']).toEqual(golden.atrousDirectSigmas);
    expect(filter['atrousIndirectSigmas']).toEqual(golden.atrousIndirectSigmas);
    expect(filter['stainedGlassFlags']).toBe(golden.stainedGlassFlags);
    expect(filter['restirPtReuse']).toBe(golden.restirPtReuse);
    expect(filter['nrcEnabled']).toBe(golden.nrcEnabled);
  });

  it('consumeRebuildKeyChange uses the static rebuild key / getter from _cfg', () => {
    // No getPipelineRebuildKey getter → the static key drives the fingerprint.
    // The first call after construction must be a no-op (fingerprint already
    // seen at ctor), proving the static key resolves identically post-migration.
    const opts = richOpts({ pipelineRebuildKey: 'stable-key' });
    const engine = new HybridEngine(opts);
    const deps = callPrivate(engine, '_buildFrameDeps');
    const control = deps['control'] as AnyRec;
    const consume = control['consumeRebuildKeyChange'] as () => boolean;
    expect(consume()).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5 — capabilities.maxBounces reads the migrated value
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.capabilities — maxBounces from migrated config', () => {
  it('capabilities.maxBounces equals the derived maxBounces', () => {
    const opts = richOpts({ maxBounces: 9 });
    const engine = new HybridEngine(opts);
    const golden = deriveFor(opts);
    expect(engine.capabilities.maxBounces).toBe(golden.maxBounces);
    expect(engine.capabilities.maxBounces).toBe(9);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 6 — default options path is also identity-preserving
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine._cfg — default options identity', () => {
  it('a default engine carries the ultra-preset derived config', () => {
    const opts = {
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1.0,
      skyTint: [0.5, 0.6, 1.0],
      skyIrradiance: 0.8,
    } as HybridEngineOptions;
    const engine = new HybridEngine(opts);
    const golden = deriveFor(opts);
    const cfg = (engine as unknown as AnyRec)['_cfg'] as Record<string, unknown>;

    expect(cfg['denoiser']).toBe('atrous-variance');
    expect(cfg['maxBounces']).toBe(4);
    expect(cfg['indirectFireflyClamp']).toEqual([1.0, 1.0, 1.0]);
    expect(cfg['restirPtReuse']).toBe(0);
    expect(cfg['nrcEnabled']).toBe(0);
    expect(cfg['tunables']).toEqual(golden.tunables);
  });
});
