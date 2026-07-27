import { describe, expect, it } from 'vitest';
import {
  CHECKERBOARD_MEASURED_PERF_PROOF,
  CHECKERBOARD_SUPPORT_DETAILS,
  QUALITY_PRESETS,
  resolveQualityPreset,
  type QualityTier,
} from '../HybridEngineQualityPreset.js';
import {
  composePassLabels,
  diSpatialPassLabels,
  giSpatialPassLabels,
} from '../pipeline/passes/passOrder.js';

const TIERS: QualityTier[] = ['ultra', 'high', 'medium', 'low'];

describe('resolveQualityPreset — preset → knob table', () => {
  it('defaults to ultra when tier is undefined', () => {
    expect(resolveQualityPreset(undefined)).toBe(QUALITY_PRESETS.ultra);
  });

  it('ultra pins the Cornell baseline values (divisor=2 is the one deliberate cadence departure)', () => {
    const ultra = resolveQualityPreset('ultra');
    // resolutionFactor full; GI budget + denoiser + frame-cap LEFT at engine
    // defaults (undefined ⇒ "do not override"); 2 spatial passes. DDGI cadence
    // is the ONE intentional non-default: stride 2 (the fast end of the 2→32
    // spread), faster than the old hardcoded stride-8 — see the preset doc.
    expect(ultra.resolutionFactor).toBe(1.0);
    expect(ultra.adaptiveSamplingThresholds).toBeUndefined();
    expect(ultra.denoiser).toBeUndefined();
    expect(ultra.targetFrameIntervalMs).toBeUndefined();
    expect(ultra.diSpatialPasses).toBe(2);
    expect(ultra.giSpatialPasses).toBe(2);
    expect(ultra.gtaoMode).toBe('on');
    expect(ultra.ddgiUpdateDivisor).toBe(2);
    // PPG trains every frame on ultra — no cadence departure (interval 1).
    expect(ultra.ppgDispatchInterval).toBe(1);
  });

  it('each preset maps to the exact §4.3 table values', () => {
    expect(resolveQualityPreset('high')).toMatchObject({
      resolutionFactor: 0.85,
      adaptiveSamplingThresholds: undefined,
      gtaoMode: 'on',
      diSpatialPasses: 2,
      giSpatialPasses: 2,
      ddgiUpdateDivisor: 4,
      ppgDispatchInterval: 1,
      targetFrameIntervalMs: undefined,
    });
    expect(resolveQualityPreset('medium')).toMatchObject({
      resolutionFactor: 0.67,
      adaptiveSamplingThresholds: [0.04, 0.40],
      gtaoMode: 'on',
      diSpatialPasses: 1,
      giSpatialPasses: 1,
      ddgiUpdateDivisor: 8,
      ppgDispatchInterval: 2,
      targetFrameIntervalMs: 20,
    });
    expect(resolveQualityPreset('low')).toMatchObject({
      resolutionFactor: 0.5,
      adaptiveSamplingThresholds: [0.20, 2.0],
      gtaoMode: 'off',
      denoiser: 'atrous',
      diSpatialPasses: 1,
      giSpatialPasses: 1,
      ddgiUpdateDivisor: 32,
      ppgDispatchInterval: 4,
      targetFrameIntervalMs: 33,
    });
  });

  it('PPG train cadence is non-decreasing ultra → low and always ≥ 1 (never skips forever)', () => {
    const intervals = TIERS.map((t) => resolveQualityPreset(t).ppgDispatchInterval);
    // ultra/high = 1, medium = 2, low = 4 — cheaper tiers retrain less often.
    expect(intervals).toEqual([1, 1, 2, 4]);
    for (const n of intervals) expect(n).toBeGreaterThanOrEqual(1);
    for (let i = 1; i < intervals.length; i++) {
      expect(intervals[i]!).toBeGreaterThanOrEqual(intervals[i - 1]!);
    }
  });

  it('presets never carry null targetFrameIntervalMs (null would DISABLE the cap)', () => {
    for (const t of TIERS) {
      expect(resolveQualityPreset(t).targetFrameIntervalMs).not.toBeNull();
    }
  });

  it('presets never force RC/PPG/neural on (documentary flag only)', () => {
    for (const t of TIERS) {
      expect(resolveQualityPreset(t).enableRcPpgNeuralByDefault).toBe(false);
    }
  });

  it('degradation tiers (medium/low) enable checkerboard; quality tiers (ultra/high) keep it off', () => {
    expect(resolveQualityPreset('ultra').checkerboard).toBe(false);
    expect(resolveQualityPreset('high').checkerboard).toBe(false);
    expect(resolveQualityPreset('medium').checkerboard).toBe(true);
    expect(resolveQualityPreset('low').checkerboard).toBe(true);
  });

  it('every preset carries the MEASURED whole-frame perf proof (the evidence exists for all tiers)', () => {
    for (const t of TIERS) {
      const proof = resolveQualityPreset(t).checkerboardPerfProof;
      expect(proof).toBe(CHECKERBOARD_MEASURED_PERF_PROOF);
      expect(proof.status).toBe('measured');
      if (proof.status !== 'measured') {
        throw new Error('checkerboard preset unexpectedly has pending perf proof');
      }
      expect(proof.requiredMetric).toBe('whole-frame-gpu-timestamp-ab');
      // Whole-frame win recorded honestly (the verified median, ≈31% saved).
      expect(proof.wholeFrameSpeedupRatio).toBeCloseTo(1.46, 5);
      // The ris re-cast pass is the biggest per-pass win.
      expect(proof.perPassSpeedups.ris).toBeGreaterThan(proof.perPassSpeedups.shade);
      // Quality summary above the 35 dB bar at the motion-onset worst frame.
      expect(proof.quality.motionWorstDb).toBeGreaterThan(35);
      expect(proof.quality.staticDb).toBeGreaterThan(60);
      expect(proof.adapter).toBe('dzn-rtx-4090');
    }
  });

  it('checkerboard support details describe the measured proof + per-tier preset enablement', () => {
    expect(CHECKERBOARD_SUPPORT_DETAILS).toMatchObject({
      feature: 'checkerboardRendering',
      // Bare engine default (no preset ⇒ ultra) renders full-rate.
      defaultEnabled: false,
      // Enabled on the degradation tiers, off on the quality tiers.
      presetEnabled: { ultra: false, high: false, medium: true, low: true },
      perfProof: {
        status: 'measured',
        requiredMetric: 'whole-frame-gpu-timestamp-ab',
      },
    });
  });

  it('resolutionFactor monotonically decreases ultra → low', () => {
    const rf = TIERS.map((t) => resolveQualityPreset(t).resolutionFactor);
    expect(rf).toEqual([1.0, 0.85, 0.67, 0.5]);
    for (let i = 1; i < rf.length; i++) expect(rf[i]!).toBeLessThan(rf[i - 1]!);
  });
});

describe('pass-flag characterization — buildPassLayout / composePassLabels (GPU-free)', () => {
  const DENOISER_NONE: never[] = []; // composePassLabels takes denoiser labels.

  it('ultra config (2 spatial passes, GTAO on) lists both spatial labels + gtao', () => {
    const labels = composePassLabels(DENOISER_NONE, {
      diSpatialPasses: 2,
      giSpatialPasses: 2,
      gtaoEnabled: true,
    });
    expect(labels).toContain('spatial-1');
    expect(labels).toContain('spatial-2');
    expect(labels).toContain('gi-spatial-1');
    expect(labels).toContain('gi-spatial-2');
    expect(labels).toContain('gtao');
    expect(labels).toContain('gtao-upsample');
  });

  it('low config (1 spatial pass, GTAO off) omits gtao + the -1 spatial labels', () => {
    const labels = composePassLabels(DENOISER_NONE, {
      diSpatialPasses: 1,
      giSpatialPasses: 1,
      gtaoEnabled: false,
    });
    // GTAO gated off ⇒ neither label present.
    expect(labels).not.toContain('gtao');
    expect(labels).not.toContain('gtao-upsample');
    // 1-pass spatial keeps only the terminal label.
    expect(labels).not.toContain('spatial-1');
    expect(labels).toContain('spatial-2');
    expect(labels).not.toContain('gi-spatial-1');
    expect(labels).toContain('gi-spatial-2');
  });

  it('omitting config defaults to the full layout (additive / regression-safe)', () => {
    const withDefaults = composePassLabels(DENOISER_NONE);
    const explicitFull = composePassLabels(DENOISER_NONE, {
      diSpatialPasses: 2,
      giSpatialPasses: 2,
      gtaoEnabled: true,
    });
    expect(withDefaults).toEqual(explicitFull);
  });

  it('spatial label slicing helpers match the pass count', () => {
    expect(diSpatialPassLabels(2)).toEqual(['spatial-1', 'spatial-2']);
    expect(diSpatialPassLabels(1)).toEqual(['spatial-2']);
    expect(giSpatialPassLabels(2)).toEqual(['gi-spatial-1', 'gi-spatial-2']);
    expect(giSpatialPassLabels(1)).toEqual(['gi-spatial-2']);
  });

  it('terminal labels (spatial-2 / gi-spatial-2) survive in BOTH pass counts (shade dependency)', () => {
    expect(diSpatialPassLabels(1)).toContain('spatial-2');
    expect(diSpatialPassLabels(2)).toContain('spatial-2');
    expect(giSpatialPassLabels(1)).toContain('gi-spatial-2');
    expect(giSpatialPassLabels(2)).toContain('gi-spatial-2');
  });
});
