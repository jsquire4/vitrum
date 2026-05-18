/**
 * HybridEngineTuning — table-driven audit-tunable forwarding for HybridEngine.
 *
 * Extracted from `HybridEngine.ts` (refactor sweep 2026-05-18).
 *
 * Background:
 *   The walkaround engine threads ~16 audit-driven tuning knobs (per-frame
 *   plus init-time) from `HybridEngineOptions` into the underlying
 *   `WalkaroundGPUPipeline`. Pre-refactor each knob was an independent
 *   `private readonly _xxx: number` field, hand-defaulted in the constructor
 *   and hand-splatted in `renderFrame`. Adding a new tunable was a 3-step
 *   shotgun: declare the field, default it, splatter it.
 *
 * Design here:
 *   {@link TUNABLE_DEFINITIONS} is a single readonly table with one entry
 *   per tunable: `key` (the runtime field name and pipeline-input key) plus
 *   `default` (Cornell-baseline value the constructor falls back to) and
 *   `audit` (the audit citation that originally introduced the knob, for
 *   provenance). {@link readTunables} consumes the table + the constructor
 *   options object and produces a frozen {@link Tunables} record. The
 *   engine stores that one record and passes `...this._tunables` into
 *   `pipeline.renderFrame`.
 *
 *   Adding a new tunable becomes:
 *     1. Append one row to {@link TUNABLE_DEFINITIONS}.
 *     2. Add the matching field to {@link HybridEngineOptions} (in
 *        `HybridEngine.ts`).
 *     3. Add the matching field to `PipelineFrameInputs` (in
 *        `WalkaroundGPUPipeline.ts`).
 *
 *   No constructor edits, no `renderFrame` edits.
 */

import type { HybridEngineOptions } from './HybridEngine.js';

// ─── Per-frame tunables (splatted into pipeline.renderFrame each frame) ─────

/**
 * Frozen per-frame tunable record. Engine stores one of these post-ctor;
 * `renderFrame` spreads it into the pipeline call. All keys match the
 * corresponding `PipelineFrameInputs` field names so the spread lands
 * directly.
 */
export interface Tunables {
  /** Audit M12 — emitter geometry-term distance² floor. Default 0.01. */
  readonly emitterDist2Floor: number;
  /** Audit B4 — per-channel HDR clamp on the direct radiance channel. Default 4.0. */
  readonly directFireflyClamp: number;
  /** Audit B1 — stained-glass caustic boost. Default 1.0 (no boost). */
  readonly causticBoost: number;
  /** Audit B1 — tinted-visibility clamp before caustic boost. Default 1.0. */
  readonly causticVisClamp: number;
  /** Audit M6 — ReSTIR-DI temporal M-clamp. Default 20 frames. */
  readonly temporalMClampDI: number;
  /** Audit M7 — ReSTIR-DI spatial reuse radius in pixels. Default 30. */
  readonly spatialReuseRadiusPx: number;
  /** Audit M8 — ReSTIR-DI spatial depth-tolerance world-units floor. Default 0.05. */
  readonly spatialDepthTolFloor: number;
  /** Audit M1 — GTAO sampling radius in pixels. Default 32. */
  readonly gtaoRadiusPx: number;
  /** Audit M1 — GTAO intensity exponent. Default 2.0. */
  readonly gtaoIntensity: number;
  /** Audit M1 — GTAO depth threshold in world units. Default 2.0. */
  readonly gtaoDepthThreshold: number;
  /** Audit B3 — GTAO upsample bilateral depth sigma (world units). Default 0.25. */
  readonly gtaoBilateralDepthSigma: number;
  /** Audit M2 — adaptive-sampling low-variance threshold. Default 0.01. */
  readonly adaptiveSamplingThresholdLow: number;
  /** Audit M2 — adaptive-sampling high-variance threshold. Default 0.10. */
  readonly adaptiveSamplingThresholdHigh: number;
  /** D12 — Möller-Trumbore coplanarity epsilon. Default 1e-5. */
  readonly triIntersectEpsilon: number;
  /** 2026-05-18 sweep — probe-side glass-transmission perceptual mix scale. Default 0.7. */
  readonly glassMixScale: number;
  /** 2026-05-18 sweep — ReSTIR-GI per-pixel unbiased weight cap. Default 16.0. */
  readonly restirGiWCap: number;
  /** 2026-05-18 sweep — DDGI irradiance clamp at the ReSTIR-GI reconnection vertex. Default 5.0. */
  readonly restirGiIrrClamp: number;
  /** 2026-05-18 sweep — ReSTIR-GI temporal previous-frame M clamp. Default 50. */
  readonly restirGiMClamp: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse disc radius in half-res px. Default 12.0. */
  readonly restirGiSpatialRadiusPx: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse normal-alignment min cosine. Default 0.906. */
  readonly restirGiSpatialNormalDotMin: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse coplanarity tolerance (world units). Default 0.05. */
  readonly restirGiSpatialCoplanarTol: number;
}

/**
 * Frozen init-time tunable record. Engine passes these into
 * `WalkaroundGPUPipeline.initialize()` once and never per-frame.
 */
export interface InitTunables {
  /** Audit B8 — squared world-space camera move that resets the temporal
   *  accumulator. Default 1.0 (Cornell-scale; 1 m² jump). */
  readonly cameraMoveResetThresholdSq: number;
  /** Audit M3 — per-frame temporal-accumulator EMA weight. Default 0.01
   *  (~60 FPS Cornell convergence). */
  readonly temporalAccumAlpha: number;
}

/** Internal definition row — declares one tunable's wiring metadata. */
interface TunableDefinition<K extends keyof Tunables = keyof Tunables> {
  /** Runtime field name (matches both `HybridEngineOptions[key]` and
   *  `PipelineFrameInputs[key]`). */
  readonly key: K;
  /** Audit citation that introduced the knob — kept here for provenance so
   *  a future grep for an audit ID lands on the table row. */
  readonly audit: string;
  /** Cornell-baseline default applied when {@link HybridEngineOptions}
   *  omits the field. */
  readonly default: number;
}

/**
 * The single source of truth for per-frame audit tunables. Each row maps
 * an audit knob to (a) the HybridEngineOptions key the constructor reads
 * from, (b) the PipelineFrameInputs key the per-frame splat lands on, and
 * (c) the Cornell-baseline default. Adding a row + the matching fields on
 * the two interfaces is the only edit needed to ship a new tunable.
 */
export const TUNABLE_DEFINITIONS: readonly TunableDefinition[] = Object.freeze([
  { key: 'emitterDist2Floor',          audit: 'M12', default: 0.01 },
  { key: 'directFireflyClamp',         audit: 'B4',  default: 4.0  },
  { key: 'causticBoost',               audit: 'B1',  default: 1.0  },
  { key: 'causticVisClamp',            audit: 'B1',  default: 1.0  },
  { key: 'temporalMClampDI',           audit: 'M6',  default: 20   },
  { key: 'spatialReuseRadiusPx',       audit: 'M7',  default: 30.0 },
  { key: 'spatialDepthTolFloor',       audit: 'M8',  default: 0.05 },
  { key: 'gtaoRadiusPx',               audit: 'M1',  default: 32.0 },
  { key: 'gtaoIntensity',              audit: 'M1',  default: 2.0  },
  { key: 'gtaoDepthThreshold',         audit: 'M1',  default: 2.0  },
  { key: 'gtaoBilateralDepthSigma',    audit: 'B3',  default: 0.25 },
  { key: 'adaptiveSamplingThresholdLow',  audit: 'M2', default: 0.01 },
  { key: 'adaptiveSamplingThresholdHigh', audit: 'M2', default: 0.10 },
  { key: 'triIntersectEpsilon',        audit: 'D12', default: 1e-5 },
  // 2026-05-18 sweep — seven more Cornell-tuned magic constants migrated from
  // hardcoded WGSL consts to host-overridable UBO fields. `indirectFireflyClamp`
  // (tuple-typed) lives on HybridEngine directly; the seven below are number-
  // typed and flow through this table.
  { key: 'glassMixScale',              audit: 'sweep-20260518', default: 0.7   },
  { key: 'restirGiWCap',               audit: 'sweep-20260518', default: 16.0  },
  { key: 'restirGiIrrClamp',           audit: 'sweep-20260518', default: 5.0   },
  { key: 'restirGiMClamp',             audit: 'sweep-20260518', default: 50    },
  { key: 'restirGiSpatialRadiusPx',    audit: 'sweep-20260518', default: 12.0  },
  { key: 'restirGiSpatialNormalDotMin', audit: 'sweep-20260518', default: 0.906 },
  { key: 'restirGiSpatialCoplanarTol', audit: 'sweep-20260518', default: 0.05  },
] as const);

/**
 * Build the frozen per-frame Tunables record from constructor options. The
 * three knobs that live inside grouped option sub-objects (`caustic`, `gtao`,
 * `adaptiveSamplingThresholds`) are pulled out explicitly here — the
 * table-driven default-or-override loop handles the rest.
 */
export function readTunables(opts: HybridEngineOptions): Tunables {
  // Source map per tunable. Most knobs sit at the top level of opts;
  // a handful sit inside grouped sub-objects. The cast at the source
  // expression is narrow — only enough to retrieve a number-or-undefined.
  const grouped: Partial<Record<keyof Tunables, number | undefined>> = {
    causticBoost:                   opts.caustic?.boost,
    causticVisClamp:                opts.caustic?.visClamp,
    gtaoRadiusPx:                   opts.gtao?.radiusPx,
    gtaoIntensity:                  opts.gtao?.intensity,
    gtaoDepthThreshold:             opts.gtao?.depthThresholdWorldUnits,
    gtaoBilateralDepthSigma:        opts.gtao?.bilateralDepthSigma,
    adaptiveSamplingThresholdLow:   opts.adaptiveSamplingThresholds?.[0],
    adaptiveSamplingThresholdHigh:  opts.adaptiveSamplingThresholds?.[1],
  };

  const flat = opts as unknown as Partial<Record<keyof Tunables, number | undefined>>;

  const out: Partial<Tunables> = {};
  for (const def of TUNABLE_DEFINITIONS) {
    const k = def.key;
    const grouped_v = grouped[k];
    const flat_v = flat[k];
    const v = grouped_v !== undefined ? grouped_v
            : flat_v    !== undefined ? flat_v
            : def.default;
    (out as Record<string, number>)[k] = v;
  }
  return Object.freeze(out) as Tunables;
}

/**
 * Build the frozen init-time tunable record. Pulled out as a separate
 * record because init-tunables flow into `pipeline.initialize()` once,
 * not into `pipeline.renderFrame()` per frame.
 */
export function readInitTunables(opts: HybridEngineOptions): InitTunables {
  return Object.freeze({
    cameraMoveResetThresholdSq: opts.cameraMoveResetThresholdSq ?? 1.0,
    temporalAccumAlpha:         opts.temporalAccumAlpha ?? 0.01,
  });
}
