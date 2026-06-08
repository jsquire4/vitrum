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
 *   - Checkerboard — ENABLED on the degradation tiers (`medium` + `low`),
 *     OFF on the quality tiers (`ultra` + `high`). The promotion is backed by a
 *     measured whole-frame GPU-timestamp perf proof (dzn RTX-4090) PLUS a motion
 *     A/B (see {@link CHECKERBOARD_MEASURED_PERF_PROOF}): the feature now
 *     compacts the dispatch of FOUR passes — shade + the two ReSTIR-DI spatial
 *     passes + the ris initial-candidate pass — to half the pixels per frame and
 *     reconstructs the rest, saving ≈31% of whole-frame GPU time at 768px. A
 *     per-frame motion fallback (`checkerboardMotionThresholdSq`) forces
 *     full-rate under fast camera motion (bit-identical to OFF on those frames);
 *     the only quality cost is a sub-perceptible motion-ONSET transient that
 *     recovers within 2-3 frames. Quality/degradation tiers get it; the
 *     fidelity tiers (ultra/high) render full-rate. A host always overrides
 *     either way via `opts.checkerboardRendering`.
 */

/** Coarse quality preset id. */
export type QualityTier = 'ultra' | 'high' | 'medium' | 'low';

/** GTAO dispatch mode resolved from a preset. `on` = half-res (today's
 *  behaviour), `quarter` = quarter-res dispatch, `off` = skip GTAO entirely. */
type GtaoMode = 'on' | 'quarter' | 'off';

/**
 * Per-pass GPU-timestamp speedup ratios (ON vs full-rate OFF) for the four
 * passes checkerboard compacts. Each is `offMedian / onMedian` (>1 ⇒ faster on).
 * Recorded as verified medians — do not fabricate precision.
 */
export interface CheckerboardPerPassSpeedups {
  /** ReSTIR-DI spatial reuse pass 1 (`spatial-1`). */
  readonly diSpatial1: number;
  /** ReSTIR-DI spatial reuse pass 2 (`spatial-2`). */
  readonly diSpatial2: number;
  /** ReSTIR-DI ris initial-candidate pass. */
  readonly ris: number;
  /** The shade pass. */
  readonly shade: number;
}

/**
 * Motion / quality summary from the Cornell motion A/B (ON vs full-rate OFF).
 * Records the verified dB medians + the motion-fallback behaviour, so the
 * promotion's quality trade is honestly captured alongside the perf win.
 */
export interface CheckerboardQualitySummary {
  /** PSNR (dB) on a static / converged frame — identical to full-rate. */
  readonly staticDb: number;
  /** Worst-frame PSNR (dB) during the motion-ONSET transient. */
  readonly motionWorstDb: number;
  /** Human-readable note on the motion fallback + transient recovery. */
  readonly note: string;
}

/**
 * Checkerboard performance evidence. The `measured` variant now records a
 * WHOLE-FRAME speedup (the feature is multi-pass — shade + the two DI spatial
 * passes + ris — not the shade-only era it was first designed for), the
 * per-pass ratios, and the motion-quality summary. The `pending` variant is
 * retained for completeness (e.g. a future adapter/scene with no capture yet).
 */
export type CheckerboardPerfProof =
  | {
      readonly status: 'pending';
      readonly requiredMetric: 'whole-frame-gpu-timestamp-ab';
      readonly reason: string;
    }
  | {
      readonly status: 'measured';
      readonly requiredMetric: 'whole-frame-gpu-timestamp-ab';
      readonly benchmarkId: string;
      /** Whole-frame GPU-time speedup (`offMedian / onMedian`, >1 ⇒ faster). */
      readonly wholeFrameSpeedupRatio: number;
      /** Per-pass GPU-timestamp speedups for the four compacted passes. */
      readonly perPassSpeedups: CheckerboardPerPassSpeedups;
      /** Motion / quality A/B summary (static dB, motion-worst dB, fallback). */
      readonly quality: CheckerboardQualitySummary;
      /** Capture scene (e.g. `'cornell'`). */
      readonly scene: string;
      /** Render resolution the perf medians were captured at (px, square). */
      readonly resolutionPx: number;
      /** GPU adapter the capture ran on (e.g. `'dzn-rtx-4090'`). */
      readonly adapter: string;
      readonly capturedAt: string;
    };

export const CHECKERBOARD_PENDING_PERF_PROOF: CheckerboardPerfProof = Object.freeze({
  status: 'pending',
  requiredMetric: 'whole-frame-gpu-timestamp-ab',
  reason:
    'Checkerboard needs a whole-frame GPU-timestamp A/B with a measurable speedup and acceptable motion quality on this adapter/scene before a preset enables it.',
});

/**
 * Frozen measured perf proof — dzn RTX-4090, GPU timestamp + motion A/B.
 *
 * PERF (768px Cornell, interleaved-paired, both swap orders agree): the win is
 * concentrated in the ReSTIR-DI BVH re-cast passes and grows with resolution.
 * Per-pass medians: spatial-1 1.87×, spatial-2 1.88×, ris 1.90×, shade 1.66×;
 * WHOLE FRAME 1.46× (≈31% GPU-time saved). (P1-only — shade+spatial without ris
 * — was 1.26× / ~20%.)
 *
 * QUALITY (384px Cornell motion A/B, ON vs full-rate OFF): static/converged
 * identical (64.34 dB); sustained motion ≈ full-rate; the motion-ONSET transient
 * worst-frame is 43.6 dB (sub-perceptible gap error 0.00101 luma, comb 1.65),
 * recovering within 2-3 frames — well above the 35 dB bar. Fast motion forces
 * full-rate (bit-identical). OFF stays byte-identical (T1 golden unchanged).
 *
 * Harnesses (wsl-gpu): checkerboard-ris-perf-ab.ts,
 * checkerboard-spatial-perf-ab.ts, checkerboard-motion-ab.ts,
 * checkerboard-ris-isolate-ab.ts.
 */
export const CHECKERBOARD_MEASURED_PERF_PROOF: CheckerboardPerfProof = Object.freeze({
  status: 'measured',
  requiredMetric: 'whole-frame-gpu-timestamp-ab',
  benchmarkId: 'checkerboard-whole-frame-ab/dzn-rtx-4090',
  wholeFrameSpeedupRatio: 1.46,
  perPassSpeedups: Object.freeze({
    diSpatial1: 1.87,
    diSpatial2: 1.88,
    ris: 1.90,
    shade: 1.66,
  }),
  quality: Object.freeze({
    staticDb: 64.34,
    motionWorstDb: 43.6,
    note:
      'Static/converged identical to full-rate; sustained motion ≈ full-rate; ' +
      'motion-ONSET transient worst-frame 43.6 dB (sub-perceptible, 0.00101 luma ' +
      'gap, comb 1.65) recovers in 2-3 frames; fast motion forces full-rate ' +
      '(bit-identical via checkerboardMotionThresholdSq).',
  }),
  scene: 'cornell',
  resolutionPx: 768,
  adapter: 'dzn-rtx-4090',
  capturedAt: '2026-06-06',
});

/**
 * Capability summary for the host. Checkerboard is now VALIDATED + ENABLED on
 * the degradation tiers (medium + low) and OFF on the quality tiers (ultra +
 * high). `defaultEnabled` stays false — the bare engine default (no preset =
 * ultra) renders full-rate and is byte-identical.
 */
export const CHECKERBOARD_SUPPORT_DETAILS = Object.freeze({
  feature: 'checkerboardRendering',
  /** The bare engine default (no preset ⇒ ultra) renders full-rate. */
  defaultEnabled: false,
  /** Enabled on the degradation tiers, off on the quality tiers. */
  presetEnabled: Object.freeze({ ultra: false, high: false, medium: true, low: true }),
  perfProof: CHECKERBOARD_MEASURED_PERF_PROOF,
});

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
   *  The update compute pass runs only on frames where
   *  `frameCount % ppgDispatchInterval === 0`; the learned sTree/dTree persists
   *  between updates, so a lower cadence is a pure cost lever — the gi-ris
   *  guided SAMPLING still consumes the most-recent tree EVERY frame (only the
   *  update pass skips). `1` = every frame (no behaviour change — ultra/high);
   *  `N > 1` runs the train pass every Nth frame (medium/low). Always ≥ 1
   *  (`0`/negative would skip forever); a host may override per-engine via
   *  `HybridEngineOptions.ppgDispatchInterval`. */
  readonly ppgDispatchInterval: number;
  /** Documentary only — whether the host UI should default-offer RC/PPG/neural
   *  for this tier. The engine NEVER forces these on from a preset. */
  readonly enableRcPpgNeuralByDefault: boolean;
  /** Checkerboard half-res shading (HybridEngineOptions.checkerboardRendering).
   *  TRUE on the degradation tiers (medium + low), FALSE on the quality tiers
   *  (ultra + high) — backed by a measured whole-frame perf proof + motion A/B
   *  (see {@link CHECKERBOARD_MEASURED_PERF_PROOF}). A host always overrides
   *  either way via `opts.checkerboardRendering`. */
  readonly checkerboard: boolean;
  readonly checkerboardPerfProof: CheckerboardPerfProof;
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
    checkerboard: false,                     // QUALITY tier — full-rate (perf proof exists but the fidelity tiers don't trade quality for it).
    checkerboardPerfProof: CHECKERBOARD_MEASURED_PERF_PROOF,
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
    checkerboard: false,                     // QUALITY tier — full-rate (perf proof exists but high doesn't trade quality for it).
    checkerboardPerfProof: CHECKERBOARD_MEASURED_PERF_PROOF,
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
    checkerboard: true,                      // DEGRADATION tier — ON: ≈31% whole-frame GPU-time saved; motion fallback forces full-rate under fast motion (measured, see CHECKERBOARD_MEASURED_PERF_PROOF).
    checkerboardPerfProof: CHECKERBOARD_MEASURED_PERF_PROOF,
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
    checkerboard: true,                      // DEGRADATION tier — ON: ≈31% whole-frame GPU-time saved; motion fallback forces full-rate under fast motion (measured, see CHECKERBOARD_MEASURED_PERF_PROOF).
    checkerboardPerfProof: CHECKERBOARD_MEASURED_PERF_PROOF,
  },
});

/**
 * Resolve a quality tier to its concrete knob values. Defaults to `ultra`
 * (byte-identical to today's behaviour) when `tier` is undefined.
 */
export function resolveQualityPreset(tier: QualityTier | undefined): QualityPreset {
  return QUALITY_PRESETS[tier ?? 'ultra'];
}
