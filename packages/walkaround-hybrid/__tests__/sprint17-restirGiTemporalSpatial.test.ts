/**
 * Sprint 17 — ReSTIR-GI temporal + spatial reuse structural tests.
 *
 * Validates WGSL contents (entry points, bindings, reuse formulae),
 * pass-layout placement (PPG training → gi-temporal → gi-spatial-1 →
 * gi-spatial-2 after the initial gi-ris and before shade), and that
 * MAX_PASS_COUNT was bumped to fit the new slots.
 */

import { describe, expect, it } from 'vitest';
import { TEMPORAL_GI_WGSL } from '../src/shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_WGSL } from '../src/shaders/spatialGi.wgsl.js';
import { SPATIAL_GI_COMMON_WGSL } from '../src/shaders/spatialGiCommon.wgsl.js';
import {
  MAX_PASS_COUNT,
  buildPassLayout,
} from '../src/pipeline/timestampQueries.js';

describe('Sprint 17 — temporal-GI WGSL', () => {
  it('declares temporalGiMain entry point with 8x8 workgroup', () => {
    expect(TEMPORAL_GI_WGSL).toContain('fn temporalGiMain');
    expect(TEMPORAL_GI_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('binds reservoirGiCurrent (rw), reservoirGiPrevious (read), WalkaroundUBO at group(0)', () => {
    expect(TEMPORAL_GI_WGSL).toContain(
      '@group(0) @binding(0) var<storage, read_write> tgi_resCurrent',
    );
    expect(TEMPORAL_GI_WGSL).toContain(
      '@group(0) @binding(1) var<storage, read>       tgi_resPrev',
    );
    expect(TEMPORAL_GI_WGSL).toContain('@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO');
  });

  it('clamps M to a bounded history via the UBO (Cornell default 50; the WGSL const was migrated to ubo.restirGiMClamp in the 2026-05-18 sweep so library consumers can override per scene)', () => {
    expect(TEMPORAL_GI_WGSL).toContain('ubo.restirGiMClamp');
  });

  it('reprojects via the previous-frame view-projection matrix', () => {
    expect(TEMPORAL_GI_WGSL).toContain('ubo.prevViewProjMatrix');
    expect(TEMPORAL_GI_WGSL).toContain('projectToPrevHalfPx');
  });

  it('applies geometric-consistency rejection on depth and normal', () => {
    expect(TEMPORAL_GI_WGSL).toContain('DEPTH_REL_TOL');
    expect(TEMPORAL_GI_WGSL).toContain('NORMAL_DOT_MIN');
  });

  it('evaluates both native receiver techniques in the canonical transformed domain', () => {
    expect(TEMPORAL_GI_WGSL).toContain('grisLogDomainToCanonicalJacobian');
    expect(TEMPORAL_GI_WGSL).toContain('grisLogWeightedTransformedDensity');
    expect(TEMPORAL_GI_WGSL).toContain('var techniqueLogMass: array<f32, 2>');
    expect(TEMPORAL_GI_WGSL).not.toContain('jacobianReconnectionShift');
  });

  it('admits a canonical local estimator only as an identity row and marks shifted tint for recast', () => {
    expect(TEMPORAL_GI_WGSL).toContain(
      'i == 0u && reservoirGiHasLocalEstimator(candidate)',
    );
    expect(TEMPORAL_GI_WGSL).toContain(
      'candidateLogWeight[i] = candidate.H;',
    );
    expect(TEMPORAL_GI_WGSL).toContain(
      'GI_SAMPLE_FLAG_RECAST_TINT,\n          candidate.sampleFlags,\n          identityOnly,',
    );
    expect(TEMPORAL_GI_WGSL).not.toContain(
      '|| reservoirGiHasLocalEstimator(current)',
    );
  });
});

describe('Sprint 17 — spatial-GI WGSL', () => {
  it('declares spatialGiMain entry point with 8x8 workgroup', () => {
    expect(SPATIAL_GI_WGSL).toContain('fn spatialGiMain');
    expect(SPATIAL_GI_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('binds input + output reservoirs and WalkaroundUBO at group(0)', () => {
    expect(SPATIAL_GI_WGSL).toContain('@group(0) @binding(0) var<storage, read>       sgi_resIn');
    expect(SPATIAL_GI_WGSL).toContain('@group(0) @binding(1) var<storage, read_write> sgi_resOut');
    expect(SPATIAL_GI_WGSL).toContain('@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO');
  });

  it('pulls K_SPATIAL_GI = 5 random neighbours from a UBO-driven disc radius (the radius const was migrated to ubo.restirGiSpatialRadiusPx in the 2026-05-18 sweep; Cornell default 12.0 px lives on HybridEngineOptions)', () => {
    // Task3 — K_SPATIAL_GI / sampleDiscPx hoisted into spatialGiCommon; both
    // the WGSL body (SPATIAL_GI_WGSL) and the shared module carry the symbol.
    expect(SPATIAL_GI_COMMON_WGSL).toContain('K_SPATIAL_GI: u32 = 5u');
    expect(SPATIAL_GI_COMMON_WGSL).toContain('ubo.restirGiSpatialRadiusPx');
    // The body still USES sampleDiscPx and K_SPATIAL_GI (calls + loop var).
    expect(SPATIAL_GI_WGSL).toContain('sampleDiscPx');
    expect(SPATIAL_GI_WGSL).toContain('K_SPATIAL_GI');
  });

  it('clamps M at 500 (spatial-reuse history bound)', () => {
    // Task3 — M_CLAMP_SPATIAL declaration moved to spatialGiCommon.
    expect(SPATIAL_GI_COMMON_WGSL).toContain('M_CLAMP_SPATIAL: u32 = 500u');
    // The body still USES M_CLAMP_SPATIAL.
    expect(SPATIAL_GI_WGSL).toContain('M_CLAMP_SPATIAL');
  });

  it('applies geometric-consistency rejection and complete transformed-density evaluation', () => {
    expect(SPATIAL_GI_WGSL).toContain('ubo.restirGiSpatialNormalDotMin');
    expect(SPATIAL_GI_WGSL).toContain('ubo.restirGiSpatialCoplanarTol');
    expect(SPATIAL_GI_WGSL).toContain('grisLogDomainToCanonicalJacobian');
    expect(SPATIAL_GI_WGSL).toContain('grisLogWeightedTransformedDensity');
    expect(SPATIAL_GI_WGSL).toContain('var techniqueLogMass: array<f32, 6>');
    expect(SPATIAL_GI_WGSL).not.toContain('jacobianReconnectionShift');
  });

  it('gathers every compatible technique before evaluating fixed-sample densities', () => {
    const gatherStart = SPATIAL_GI_WGSL.indexOf(
      'for (var gather: u32 = 0u; gather < K_SPATIAL_GI;',
    );
    const gatherEnd = SPATIAL_GI_WGSL.indexOf(
      'domains[domainCount] = q;',
      gatherStart,
    );
    expect(gatherStart).toBeGreaterThanOrEqual(0);
    expect(gatherEnd).toBeGreaterThan(gatherStart);
    const gather = SPATIAL_GI_WGSL.slice(gatherStart, gatherEnd);
    expect(gather).not.toContain('grisLogDomainToCanonicalJacobian(');
    expect(gather).not.toContain('grisLogMaterialPHatAt(');

    const matrixStart = SPATIAL_GI_WGSL.indexOf(
      'for (var i: u32 = 0u; i < domainCount;',
      gatherEnd,
    );
    const matrixEnd = SPATIAL_GI_WGSL.indexOf(
      'var out = emptyReservoirGI();',
      matrixStart,
    );
    expect(matrixStart).toBeGreaterThan(gatherEnd);
    expect(matrixEnd).toBeGreaterThan(matrixStart);
    const matrix = SPATIAL_GI_WGSL.slice(matrixStart, matrixEnd);
    expect(matrix).toContain('grisLogDomainToCanonicalJacobian(');
    expect(matrix).toContain('grisLogMaterialPHatAt(');
  });

  it('uses Eq. 7 attempt mass in log m_i and derives each source logW from H', () => {
    expect(SPATIAL_GI_WGSL).toContain('grisLogWeightedTransformedDensity(');
    expect(SPATIAL_GI_WGSL).toContain('grisLogCanonicalResamplingWeight(');
    expect(SPATIAL_GI_WGSL).toContain('candidate.H,');
    expect(SPATIAL_GI_WGSL).toContain('candidate.nativeLogPHat,');
    expect(SPATIAL_GI_WGSL).toContain(
      'let logMisWeight = techniqueLogMass[i] - logDenominator;',
    );
    expect(SPATIAL_GI_WGSL).not.toMatch(/candidate\.W\b/);
    expect(SPATIAL_GI_WGSL).not.toContain('sourceScaledMass');
  });

  it('keeps neighbour local rows denominator-only and marks shifted tint for recast', () => {
    expect(SPATIAL_GI_WGSL).toContain(
      'i == 0u && reservoirGiHasLocalEstimator(candidate)',
    );
    expect(SPATIAL_GI_WGSL).toContain(
      'candidateLogWeight[i] = candidate.H;',
    );
    expect(SPATIAL_GI_WGSL).toContain(
      'GI_SAMPLE_FLAG_RECAST_TINT,\n          candidate.sampleFlags,\n          identityOnly,',
    );
    expect(SPATIAL_GI_WGSL).not.toContain(
      '|| reservoirGiHasLocalEstimator(rCenter)',
    );
  });
});

describe('Sprint 17 — pass-layout placement', () => {
  it('places PPG training before the contiguous GI-reuse block in every layout variant', () => {
    for (const denoiserMode of ['atrous-variance', 'atrous'] as const) {
      const layout = buildPassLayout({ denoiserMode });
      const labels = layout.labels;
      const giRis = labels.indexOf('gi-ris');
      const ppgUpdate = labels.indexOf('ppg-update');
      const giTemporal = labels.indexOf('gi-temporal');
      const giS1 = labels.indexOf('gi-spatial-1');
      const giS2 = labels.indexOf('gi-spatial-2');
      const shade = labels.indexOf('shade');
      expect(giRis).toBeGreaterThanOrEqual(0);
      expect(ppgUpdate).toBe(giRis + 1);
      expect(giTemporal).toBe(ppgUpdate + 1);
      expect(giS1).toBe(giTemporal + 1);
      expect(giS2).toBe(giS1 + 1);
      expect(shade).toBe(giS2 + 1);
    }
  });

  it('MAX_PASS_COUNT accommodates the GI block (≥ 20 — sample-budget + 4 DI + 4 GI + shade + 2 GTAO + denoiser + indirect-combine + 3 tail)', () => {
    expect(MAX_PASS_COUNT).toBeGreaterThanOrEqual(20);
  });
});
