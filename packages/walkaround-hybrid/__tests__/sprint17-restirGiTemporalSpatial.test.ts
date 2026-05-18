/**
 * Sprint 17 — ReSTIR-GI temporal + spatial reuse structural tests.
 *
 * Validates WGSL contents (entry points, bindings, reuse formulae),
 * pass-layout placement (gi-temporal → gi-spatial-1 → gi-spatial-2 in
 * a contiguous block right after gi-ris and before shade), and that
 * MAX_PASS_COUNT was bumped to fit the new slots.
 */

import { describe, expect, it } from 'vitest';
import { TEMPORAL_GI_WGSL } from '../src/shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_WGSL } from '../src/shaders/spatialGi.wgsl.js';
import { MAX_PASS_COUNT, buildPassLayout } from '../src/pipeline/timestampQueries.js';

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

  it('clamps M to a bounded history (post-Sprint-18-followup default = 50; see comment)', () => {
    expect(TEMPORAL_GI_WGSL).toMatch(/M_CLAMP_GI:\s*u32\s*=\s*\d+u/);
  });

  it('reprojects via the previous-frame view matrix', () => {
    expect(TEMPORAL_GI_WGSL).toContain('ubo.prevViewMatrix');
    expect(TEMPORAL_GI_WGSL).toContain('projectToPrevHalfPx');
  });

  it('applies geometric-consistency rejection on depth and normal', () => {
    expect(TEMPORAL_GI_WGSL).toContain('DEPTH_REL_TOL');
    expect(TEMPORAL_GI_WGSL).toContain('NORMAL_DOT_MIN');
  });

  it('applies reconnection-shift Jacobian for prev sample', () => {
    expect(TEMPORAL_GI_WGSL).toContain('jacobianReconnectionShift');
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

  it('pulls K_SPATIAL_GI = 5 random neighbours from a 12-pixel disc', () => {
    expect(SPATIAL_GI_WGSL).toContain('K_SPATIAL_GI: u32 = 5u');
    expect(SPATIAL_GI_WGSL).toContain('SPATIAL_RADIUS_GI: f32 = 12.0');
  });

  it('clamps M at 500 (spatial-reuse history bound)', () => {
    expect(SPATIAL_GI_WGSL).toContain('M_CLAMP_SPATIAL: u32 = 500u');
  });

  it('applies geometric-consistency reject + Jacobian shift', () => {
    expect(SPATIAL_GI_WGSL).toContain('NORMAL_DOT_MIN_S');
    expect(SPATIAL_GI_WGSL).toContain('DEPTH_REL_TOL_S');
    expect(SPATIAL_GI_WGSL).toContain('jacobianReconnectionShift');
  });

  it('uses w_q = p̂(z_q) · W_q · M_q · J (standard RIS combine)', () => {
    expect(SPATIAL_GI_WGSL).toMatch(/let w_q\s*=\s*pHatZ\s*\*\s*rQ\.W\s*\*\s*f32\(Mq\)\s*\*\s*J/);
  });
});

describe('Sprint 17 — pass-layout placement', () => {
  it('places gi-temporal + gi-spatial-1 + gi-spatial-2 contiguously after gi-ris in every layout variant', () => {
    for (const denoiserMode of ['atrous-variance', 'atrous'] as const) {
      const layout = buildPassLayout({ denoiserMode });
      const labels = layout.labels;
      const giRis = labels.indexOf('gi-ris');
      const giTemporal = labels.indexOf('gi-temporal');
      const giS1 = labels.indexOf('gi-spatial-1');
      const giS2 = labels.indexOf('gi-spatial-2');
      const shade = labels.indexOf('shade');
      expect(giRis).toBeGreaterThanOrEqual(0);
      expect(giTemporal).toBe(giRis + 1);
      expect(giS1).toBe(giRis + 2);
      expect(giS2).toBe(giRis + 3);
      expect(shade).toBe(giS2 + 1);
    }
  });

  it('MAX_PASS_COUNT accommodates the GI block (≥ 20 — sample-budget + 4 DI + 4 GI + shade + 2 GTAO + denoiser + indirect-combine + 3 tail)', () => {
    expect(MAX_PASS_COUNT).toBeGreaterThanOrEqual(20);
  });
});
