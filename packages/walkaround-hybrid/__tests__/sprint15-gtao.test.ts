/**
 * Sprint 15 — GTAO (Ground-Truth Ambient Occlusion) structural tests.
 *
 * Verifies the WGSL strings contain expected entry points + key constants,
 * the pass layout includes `gtao` and `gtao-upsample` in the right order,
 * and MAX_PASS_COUNT was bumped to accommodate the new slots.
 */

import { describe, expect, it } from 'vitest';
import { GTAO_WGSL } from '../src/shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_WGSL } from '../src/shaders/gtaoUpsample.wgsl.js';
import {
  MAX_PASS_COUNT,
  buildPassLayout,
} from '../src/pipeline/timestampQueries.js';

describe('Sprint 15 — GTAO WGSL', () => {
  it('GTAO_WGSL contains gtaoMain entry point', () => {
    expect(GTAO_WGSL).toContain('fn gtaoMain');
    expect(GTAO_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('GTAO_WGSL declares its 3 expected bindings', () => {
    expect(GTAO_WGSL).toContain('@group(0) @binding(0) var gtao_normalDepth');
    expect(GTAO_WGSL).toContain('@group(0) @binding(1) var gtao_aoOut');
    expect(GTAO_WGSL).toContain('@group(0) @binding(2) var<uniform> gtao_ubo');
  });

  it('GTAO_WGSL has NUM_DIRECTIONS=4 and NUM_STEPS=6 (horizon-based AO defaults)', () => {
    expect(GTAO_WGSL).toContain('NUM_DIRECTIONS: u32 = 4u');
    expect(GTAO_WGSL).toContain('NUM_STEPS:      u32 = 6u');
  });

  it('GTAO_WGSL writes 1.0 (unoccluded) for sky-miss pixels (depth = 0)', () => {
    // The sky-miss early-out should set the output to fully lit.
    expect(GTAO_WGSL).toMatch(/centerDepth\s*<\s*1e-4/);
    expect(GTAO_WGSL).toContain('textureStore(gtao_aoOut, gid.xy, vec4f(1.0))');
  });
});

describe('Sprint 15 — GTAO upsample WGSL', () => {
  it('contains gtaoUpsampleMain entry point', () => {
    expect(GTAO_UPSAMPLE_WGSL).toContain('fn gtaoUpsampleMain');
    expect(GTAO_UPSAMPLE_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('declares 3 bindings (aoHalf in, normalDepth in, aoFull out)', () => {
    expect(GTAO_UPSAMPLE_WGSL).toContain('@group(0) @binding(0) var up_aoHalf');
    expect(GTAO_UPSAMPLE_WGSL).toContain('@group(0) @binding(1) var up_normalDepth');
    expect(GTAO_UPSAMPLE_WGSL).toContain('@group(0) @binding(2) var up_aoFullOut');
  });

  it('uses joint bilateral with depth + normal edge stops', () => {
    expect(GTAO_UPSAMPLE_WGSL).toContain('similarityWeight');
    expect(GTAO_UPSAMPLE_WGSL).toMatch(/depthDelta|depthW/);
    expect(GTAO_UPSAMPLE_WGSL).toMatch(/dot\(centerNormal/);
  });

  it('passes through sky-miss as 1.0', () => {
    expect(GTAO_UPSAMPLE_WGSL).toMatch(/centerDepth\s*<\s*1e-4/);
    expect(GTAO_UPSAMPLE_WGSL).toContain('textureStore(up_aoFullOut, gid.xy, vec4f(1.0))');
  });
});

describe('Sprint 15 — pass layout integration', () => {
  it('MAX_PASS_COUNT is at least 19 (room for gtao + gtao-upsample)', () => {
    expect(MAX_PASS_COUNT).toBeGreaterThanOrEqual(19);
  });

  it('SVGF + PPG-off layout includes gtao and gtao-upsample between shade and welford-temporal', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'svgf' });
    const labels = layout.labels;
    const shadeIdx = labels.indexOf('shade');
    const gtaoIdx = labels.indexOf('gtao');
    const gtaoUpsampleIdx = labels.indexOf('gtao-upsample');
    const welfordIdx = labels.indexOf('welford-temporal');
    expect(shadeIdx).toBeGreaterThanOrEqual(0);
    expect(gtaoIdx).toBeGreaterThan(shadeIdx);
    expect(gtaoUpsampleIdx).toBe(gtaoIdx + 1);
    expect(welfordIdx).toBeGreaterThan(gtaoUpsampleIdx);
  });

  it('atrous layout (denoiserMode=atrous) also includes gtao slots', () => {
    const layout = buildPassLayout({ ppgEnabled: false, denoiserMode: 'atrous' });
    expect(layout.labels).toContain('gtao');
    expect(layout.labels).toContain('gtao-upsample');
  });

  it('PPG-on layout still includes gtao slots after ppg-update', () => {
    const layout = buildPassLayout({ ppgEnabled: true, denoiserMode: 'svgf' });
    const labels = layout.labels;
    const ppgIdx = labels.indexOf('ppg-update');
    const gtaoIdx = labels.indexOf('gtao');
    // ppg-update sits before gtao in the order (ppg-update is added between
    // shade and gtao); both should be present.
    expect(ppgIdx).toBeGreaterThanOrEqual(0);
    expect(gtaoIdx).toBeGreaterThan(ppgIdx);
  });

  it('slotCount fits within MAX_PASS_COUNT', () => {
    const layoutSvgf = buildPassLayout({ ppgEnabled: false, denoiserMode: 'svgf' });
    const layoutPpg = buildPassLayout({ ppgEnabled: true, denoiserMode: 'svgf' });
    expect(layoutSvgf.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
    expect(layoutPpg.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
  });
});
