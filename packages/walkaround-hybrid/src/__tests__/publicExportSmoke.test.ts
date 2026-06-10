/**
 * Smoke tests: verify that QualityTier / QUALITY_PRESETS / resolveQualityPreset
 * are importable from the @vitrum/walkaround-hybrid package root.
 *
 * Previously these were only accessible via the internal
 * `HybridEngineQualityPreset` path, so host code that needed to enumerate tiers
 * or build quality-picker UIs had to import an internal module.
 */

import { describe, expect, it } from 'vitest';
import {
  QUALITY_PRESETS,
  resolveQualityPreset,
  CHECKERBOARD_SUPPORT_DETAILS,
  type QualityTier,
  type QualityPreset,
} from '../index.js';

const TIERS: QualityTier[] = ['ultra', 'high', 'medium', 'low'];

describe('QualityTier / QUALITY_PRESETS / resolveQualityPreset public export', () => {
  it('QUALITY_PRESETS has an entry for every tier', () => {
    for (const tier of TIERS) {
      expect(QUALITY_PRESETS).toHaveProperty(tier);
    }
  });

  it('resolveQualityPreset(undefined) returns the ultra preset', () => {
    const preset: QualityPreset = resolveQualityPreset(undefined);
    expect(preset).toBe(QUALITY_PRESETS.ultra);
    expect(preset.resolutionFactor).toBe(1.0);
  });

  it('resolveQualityPreset returns the correct preset for each tier', () => {
    for (const tier of TIERS) {
      expect(resolveQualityPreset(tier)).toBe(QUALITY_PRESETS[tier]);
    }
  });

  it('each preset carries the required shape fields', () => {
    for (const tier of TIERS) {
      const p = QUALITY_PRESETS[tier];
      expect(typeof p.resolutionFactor).toBe('number');
      expect(typeof p.diSpatialPasses).toBe('number');
      expect(typeof p.giSpatialPasses).toBe('number');
      expect(typeof p.ddgiUpdateDivisor).toBe('number');
      expect(typeof p.checkerboard).toBe('boolean');
    }
  });

  it('CHECKERBOARD_SUPPORT_DETAILS is exported and has a measured perf proof', () => {
    expect(CHECKERBOARD_SUPPORT_DETAILS.perfProof.status).toBe('measured');
    expect(CHECKERBOARD_SUPPORT_DETAILS.presetEnabled.ultra).toBe(false);
    expect(CHECKERBOARD_SUPPORT_DETAILS.presetEnabled.medium).toBe(true);
  });
});
