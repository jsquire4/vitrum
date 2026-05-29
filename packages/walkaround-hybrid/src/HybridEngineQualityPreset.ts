/**
 * HybridEngineQualityPreset — Phase-0 productization (roadmap §4.3 / §5.1).
 *
 * Maps a coarse {@link QualityTier} preset (`ultra`/`high`/`medium`/`low`) to
 * the concrete tunable / UBO / pass-gate knob values the engine threads into
 * the pipeline. Pure + table-driven (mirrors `HybridEngineTuning.ts`'s style).
 *
 * Design contract (matches the maintainer "fidelity + flexibility, no
 * hardcoded single-path" preference):
 *   - **`ultra` matches today's Cornell-baseline defaults EXCEPT for the DDGI
 *     probe-update cadence.** Every ultra value is the existing default or
 *     `undefined` (= "leave the engine default untouched"), with ONE deliberate
 *     exception: `ddgiUpdateDivisor: 2`. Before the H1 fix this knob was DEAD (it
 *     wrote an unread UBO field while the real cadence was a hardcoded stride 8);
 *     now that it is load-bearing, the maintainer chose a faster 2→32 cadence
 *     spread, so ultra updates probes at stride 2 (4× the old stride-8 rate) for
 *     a snappier GI response. This departs from the pre-H1 shipped cadence and is
 *     pending GPU A/B validation. The `resolveQualityPreset('ultra')` regression
 *     test pins the full value set (including the divisor=2). Presets are
 *     otherwise ADDITIVE — a host that never sets `qualityTier` gets ultra.
 *   - **Explicit per-knob options OVERRIDE the preset.** The engine applies the
 *     preset FIRST as a baseline, then `opts.X ?? preset.X` lets any explicit
 *     option win (`qualityTier:'low'` + `gtao:{radiusPx:99}` keeps radius 99).
 *   - Every §4.3 dimension resolves to a REAL field write or an explicitly
 *     documented "fixed at N, not yet tunable" note — presets never silently
 *     no-op a dimension.
 *
 * Dimensions intentionally LEFT FIXED for Phase 0 (documented in the plan):
 *   - DI/GI spatial NEIGHBOR count (`NEIGHBORS=5u` / `K_SPATIAL_GI=5u`) — a
 *     compile-time WGSL const; promoting it to a UBO field for a marginal lever
 *     is deferred. We degrade via spatial PASS count + resolution instead.
 *   - RC / PPG / neural — presets never force these ON (they need extra GPU
 *     resources / weights). `enableRcPpgNeuralByDefault` is informational for
 *     host UI; the engine default stays OFF regardless of preset.
 */

/** Coarse quality preset id. */
export type QualityTier = 'ultra' | 'high' | 'medium' | 'low';

/** GTAO dispatch mode resolved from a preset. `on` = half-res (today's
 *  behaviour), `quarter` = quarter-res dispatch, `off` = skip GTAO entirely. */
export type GtaoMode = 'on' | 'quarter' | 'off';

/**
 * Resolved preset knob values. `undefined` means "leave the engine's existing
 * default" (used by ultra to stay byte-identical). The engine reads these as
 * fallbacks under explicit `HybridEngineOptions` fields.
 */
export interface QualityPreset {
  /** Internal render-resolution scale (Deliverable 4 path). Always defined
   *  (ultra = 1.0). */
  readonly resolutionFactor: number;
  /** Adaptive-sampling tier classifier thresholds → ReSTIR-GI M_GI scale
   *  proxy. `undefined` ⇒ engine default `[0.01, 0.10]` (ultra/high). */
  readonly adaptiveSamplingThresholds: readonly [low: number, high: number] | undefined;
  /** GTAO dispatch mode (pass gate + half/quarter-res dispatch). */
  readonly gtaoMode: GtaoMode;
  /** Post-shade denoiser. `undefined` ⇒ engine default `'atrous-variance'`. */
  readonly denoiser: 'atrous' | 'atrous-variance' | undefined;
  /** Per-frame interval cap (ms). `undefined` ⇒ engine default (~60 FPS cap);
   *  a number sets an explicit cap; never `null` in any preset (null would
   *  DISABLE the cap, which is not a degradation lever). */
  readonly targetFrameIntervalMs: number | undefined;
  /** ReSTIR-DI spatial reuse ping-pong pass count (1 or 2). */
  readonly diSpatialPasses: 1 | 2;
  /** ReSTIR-GI spatial reuse ping-pong pass count (1 or 2). */
  readonly giSpatialPasses: 1 | 2;
  /** DDGI round-robin probe-update divisor (`probesPerFrame = ceil(total/N)`).
   *  Higher ⇒ fewer probes updated per frame ⇒ cheaper, slower GI response.
   *  Engine default is 4 (the historical hardcoded `/4`). */
  readonly ddgiUpdateDivisor: number;
  /** PPG (Müller 2017 path-guiding) train-pass dispatch cadence (roadmap §5.3).
   *  The guide + update compute passes run only on frames where
   *  `frameCount % ppgDispatchInterval === 0`; the learned sTree/dTree persists
   *  between updates, so a lower cadence is a pure cost lever — the gi-ris
   *  guided SAMPLING still consumes the most-recent tree EVERY frame (only the
   *  train passes skip). `1` = every frame (no behaviour change — ultra/high);
   *  `N > 1` runs the train passes every Nth frame (medium/low). Always ≥ 1
   *  (`0`/negative would skip forever); a host may override per-engine via
   *  `HybridEngineOptions.ppgDispatchInterval`. */
  readonly ppgDispatchInterval: number;
  /** Documentary only — whether the host UI should default-offer RC/PPG/neural
   *  for this tier. The engine NEVER forces these on from a preset. */
  readonly enableRcPpgNeuralByDefault: boolean;
}

/**
 * The preset → knob table. Grounded in the real field names + the §4.3
 * mapping. `ultra` is the Cornell baseline (every value either the existing
 * default or `undefined`).
 */
export const QUALITY_PRESETS: Readonly<Record<QualityTier, QualityPreset>> = Object.freeze({
  ultra: {
    resolutionFactor: 1.0,
    adaptiveSamplingThresholds: undefined, // ⇒ engine default [0.01, 0.10]
    gtaoMode: 'on',
    denoiser: undefined,                    // ⇒ engine default 'atrous-variance'
    targetFrameIntervalMs: undefined,       // ⇒ engine default (~60 FPS cap)
    diSpatialPasses: 2,
    giSpatialPasses: 2,
    ddgiUpdateDivisor: 2,                   // flagship: fastest GI cadence — stride 2 (4× the default-8 probe rate). H1 made the divisor load-bearing, so ultra is NO LONGER byte-identical to the old hardcoded stride-8 (intentional, per the 2→32 cadence decision).
    ppgDispatchInterval: 1,                  // every frame — no behaviour change when PPG is on.
    enableRcPpgNeuralByDefault: false,
  },
  high: {
    resolutionFactor: 0.85,
    adaptiveSamplingThresholds: undefined, // same GI budget as ultra
    gtaoMode: 'on',
    denoiser: undefined,
    targetFrameIntervalMs: undefined,
    diSpatialPasses: 2,
    giSpatialPasses: 2,
    ddgiUpdateDivisor: 4,                   // 2× the default-8 probe rate (stride 4)
    ppgDispatchInterval: 1,                  // every frame (high keeps full PPG cadence)
    enableRcPpgNeuralByDefault: false,
  },
  medium: {
    resolutionFactor: 0.67,
    // Raise thresholds so more pixels fall to tier 1 (M_GI→4) — the §4.3
    // "reduced" GI budget realized through the existing adaptive classifier.
    adaptiveSamplingThresholds: [0.04, 0.40],
    gtaoMode: 'on',
    denoiser: undefined,
    targetFrameIntervalMs: 20,
    diSpatialPasses: 1,
    giSpatialPasses: 1,
    ddgiUpdateDivisor: 8,                   // = the default probe cadence (stride 8)
    ppgDispatchInterval: 2,                  // train every 2nd frame — ~½ the PPG train cost; tree persists between updates so quality drift is negligible.
    enableRcPpgNeuralByDefault: false,
  },
  low: {
    resolutionFactor: 0.5,
    // §4.3 "minimal" GI budget — most pixels at tier 1.
    adaptiveSamplingThresholds: [0.20, 2.0],
    gtaoMode: 'off',
    denoiser: 'atrous',                     // cheaper legacy 3-pass atrous
    targetFrameIntervalMs: 33,
    diSpatialPasses: 1,
    giSpatialPasses: 1,
    ddgiUpdateDivisor: 32,                  // budget: slowest GI cadence — stride 32 (1/4 the default-8 probe rate)
    ppgDispatchInterval: 4,                  // budget: train every 4th frame — ~¼ the PPG train cost.
    enableRcPpgNeuralByDefault: false,
  },
});

/**
 * Resolve a quality tier to its concrete knob values. Defaults to `ultra`
 * (byte-identical to today's behaviour) when `tier` is undefined.
 */
export function resolveQualityPreset(tier: QualityTier | undefined): QualityPreset {
  return QUALITY_PRESETS[tier ?? 'ultra'];
}
