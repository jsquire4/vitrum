/**
 * Sprint 16 — ReSTIR-GI RIS pass structural tests.
 *
 * Verifies the RIS_GI_WGSL string contains the expected entry point, bindings,
 * candidate count, and reservoir helpers; the pass layout places `gi-ris`
 * between `spatial-2` and `shade`; and the static/common ReservoirGI export
 * still carries the widened 30-u32 GRIS layout. Runtime compilation selects the
 * compact 20-u32 default via `compilePipelines`; that gate is covered by
 * `giStructuralGate.test.ts`.
 */

import { describe, expect, it } from 'vitest';
import { RIS_GI_WGSL } from '../src/shaders/risGi.wgsl.js';
import { COMMON_WGSL } from '../src/shaders/common.wgsl.js';
import { DDGI_GRID_UBO_WGSL } from '../src/ddgi/ddgiSampleWgsl.js';
import {
  MAX_PASS_COUNT,
  buildPassLayout,
} from '../src/pipeline/timestampQueries.js';

describe('Sprint 16 — RIS_GI WGSL', () => {
  it('declares risGiMain entry point with 8x8 workgroup', () => {
    expect(RIS_GI_WGSL).toContain('fn risGiMain');
    expect(RIS_GI_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('binds gNormalDepth at (0,10) and reservoir buffer at (0,11)', () => {
    expect(RIS_GI_WGSL).toContain('@group(0) @binding(10) var gi_gNormalDepth');
    expect(RIS_GI_WGSL).toContain(
      '@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent',
    );
  });

  it('binds the BVH triple at group(1)', () => {
    expect(RIS_GI_WGSL).toContain('@group(1) @binding(0) var<storage, read> bvh:');
    expect(RIS_GI_WGSL).toContain('@group(1) @binding(1) var<storage, read> bvh_index:');
    expect(RIS_GI_WGSL).toContain('@group(1) @binding(2) var<storage, read> bvh_position:');
  });

  it('binds WalkaroundUBO + DDGI atlas at the canonical slots', () => {
    expect(RIS_GI_WGSL).toContain('@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO');
    expect(RIS_GI_WGSL).toContain('@group(3) @binding(0) var ddgiIrradiance');
    expect(RIS_GI_WGSL).toContain('@group(3) @binding(1) var ddgiVisibility');
    // D5.1: @group(3) @binding(3) ddgiGrid is now in the shared ddgiGridUbo module.
    expect(DDGI_GRID_UBO_WGSL).toContain('@group(3) @binding(3) var<uniform> ddgiGrid');
  });

  it('uses an adaptive M_GI scaled from the M_GI_BASE=8 candidate count (Majercik 2021 §4.2 + Sprint 9 tier)', () => {
    expect(RIS_GI_WGSL).toContain('M_GI_BASE: u32 = 8u');
    expect(RIS_GI_WGSL).toMatch(/let M_GI\s*=\s*M_GI_BASE\s*\*\s*tier/);
  });

  it('writes an empty reservoir on primary-ray miss, glass, or metal', () => {
    expect(RIS_GI_WGSL).toContain('emptyReservoirGI()');
    // No-hit early-out should call store with empty reservoir.
    expect(RIS_GI_WGSL).toMatch(/storeReservoirGI_rw\([^)]*emptyReservoirGI\(\)/);
  });

  it('uses cosine-weighted hemisphere candidates (defined in common.wgsl)', () => {
    expect(COMMON_WGSL).toContain('fn sampleCosineHemisphere');
    expect(RIS_GI_WGSL).toContain('sampleCosineHemisphere(normal, &rng)');
  });

  it('uses DDGI-atlas reconnection-vertex radiance estimation through the material payload helper', () => {
    expect(RIS_GI_WGSL).toContain('sampleDDGIAtPoint');
    expect(RIS_GI_WGSL).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    expect(RIS_GI_WGSL).toContain('Lo = xsPayload.Lo');
  });

  it('runs the final visibility test on the chosen sample and attenuates W by tinted alpha transmittance', () => {
    expect(RIS_GI_WGSL).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(RIS_GI_WGSL).toContain('traceSceneAlphaTintTransmittanceTextured(');
    expect(RIS_GI_WGSL).toContain('let shadowT = clamp(luminance(shadowTint), 0.0, 1.0);');
    expect(RIS_GI_WGSL).not.toContain('traceSceneAlphaTransmittanceTextured(');
    expect(RIS_GI_WGSL).toMatch(/if \(shadowT <= 0\.001\)/);
    expect(RIS_GI_WGSL).toContain('r.w_sum = r.w_sum * shadowT;');
    expect(RIS_GI_WGSL).toMatch(/r\.W\s*=\s*0\.0/);
  });

  it('computes W = w_sum / (M · p̂(z)) per the RIS estimator', () => {
    expect(RIS_GI_WGSL).toMatch(/r\.w_sum\s*\/\s*\(f32\(r\.M\)\s*\*\s*pHatZ\)/);
  });

  it('only dispatches over half-resolution pixels (W/2 × H/2)', () => {
    expect(RIS_GI_WGSL).toContain('halfDims = fullDims / 2u');
    expect(RIS_GI_WGSL).toContain('if (any(gid.xy >= halfDims)) { return; }');
  });
});

describe('Sprint 16 — ReservoirGI byte-pack helpers (static common.wgsl export)', () => {
  it('declares the 30 × u32 stride constant on the static GRIS-compatible export', () => {
    // GRIS Phase-0 appended the reconnection-shift cache at indices [20..29],
    // widening the per-pixel reservoir from 20 u32 (80 bytes) to 30 u32
    // (120 bytes). The [0..19] prefix stays byte-identical — see
    // reservoirPtLayout.test.ts for the bit-identity guard.
    expect(COMMON_WGSL).toContain('RESERVOIR_GI_STRIDE: u32 = 30u');
  });

  it('exposes _rw + _ro load helpers and a _rw store helper', () => {
    expect(COMMON_WGSL).toContain('fn loadReservoirGI_rw');
    expect(COMMON_WGSL).toContain('fn loadReservoirGI_ro');
    expect(COMMON_WGSL).toContain('fn storeReservoirGI_rw');
  });

  it('updateReservoirGI weights by w_sum (canonical reservoir update)', () => {
    expect(COMMON_WGSL).toContain('fn updateReservoirGI');
    expect(COMMON_WGSL).toMatch(/rand_f32\(rng\)\s*\*\s*\(\*r\)\.w_sum\s*<\s*w/);
  });
});

describe('Sprint 16 — pass-layout placement', () => {
  it('gi-ris sits directly after spatial-2 and before shade in every layout variant', () => {
    for (const denoiserMode of ['atrous-variance', 'atrous'] as const) {
      const layout = buildPassLayout({ denoiserMode });
      const spatial2 = layout.labels.indexOf('spatial-2');
      const giRis = layout.labels.indexOf('gi-ris');
      const shade = layout.labels.indexOf('shade');
      expect(spatial2).toBeGreaterThanOrEqual(0);
      expect(giRis).toBe(spatial2 + 1);
      // Sprint 17 inserts gi-temporal + gi-spatial-1 + gi-spatial-2 between
      // gi-ris and shade; shade still trails the full GI block.
      expect(shade).toBeGreaterThan(giRis);
    }
  });

  it('MAX_PASS_COUNT accommodates the gi-ris slot (≥ 20)', () => {
    expect(MAX_PASS_COUNT).toBeGreaterThanOrEqual(20);
  });
});
