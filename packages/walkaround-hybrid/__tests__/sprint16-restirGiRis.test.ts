/**
 * Sprint 16 — ReSTIR-GI RIS pass structural tests.
 *
 * Verifies the RIS_GI_WGSL string contains the expected entry point, bindings,
 * candidate count, and reservoir helpers; the pass layout places `gi-ris`
 * between `spatial-2` and `shade`; and the static/common ReservoirGI export
 * carries the sole live 28-u32 generalized-reuse layout.
 */

import { describe, expect, it } from 'vitest';
import { RIS_GI_MODULE, RIS_GI_WGSL } from '../src/shaders/risGi.wgsl.js';
import { COMMON_WGSL } from '../src/shaders/common.wgsl.js';
import { reservoirGiAccessorsWgsl } from '../src/shaders/reservoirGi.wgsl.js';
import { DDGI_GRID_UBO_WGSL } from '../src/ddgi/ddgiSampleWgsl.js';
import { WGSL_MODULES } from '../src/pipeline/wgslModules.js';
import { composeWgsl } from '../src/pipeline/wgslComposer.js';
import {
  MAX_PASS_COUNT,
  buildPassLayout,
} from '../src/pipeline/timestampQueries.js';

describe('Sprint 16 — RIS_GI WGSL', () => {
  it('declares risGiMain entry point with 8x8 workgroup', () => {
    expect(RIS_GI_WGSL).toContain('fn risGiMain');
    expect(RIS_GI_WGSL).toContain('@workgroup_size(8, 8, 1)');
  });

  it('binds only the reservoir buffer from the frame group', () => {
    expect(RIS_GI_WGSL).not.toContain('gi_gNormalDepth');
    expect(RIS_GI_WGSL).toContain(
      '@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent',
    );
  });

  it('binds the three versioned scene arenas at group(1)', () => {
    const composed = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    expect(composed).toContain('@group(1) @binding(0) var<storage, read> sceneGeometryArena:');
    expect(composed).toContain('@group(1) @binding(1) var<storage, read> sceneTlasArena:');
    expect(composed).toContain('@group(1) @binding(2) var<storage, read> sceneLightingArena:');
    expect(composed).toContain('fn bvhLoadNode(');
    expect(composed).toContain('fn bvhLoadIndex(');
    expect(composed).toContain('fn bvhLoadPosition(');
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

  it('writes an empty reservoir on a primary-ray miss', () => {
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
    expect(RIS_GI_WGSL).toContain('Lo = xsPayload.Lo;');
  });

  it('evaluates tinted alpha visibility for every candidate before reservoir selection', () => {
    expect(RIS_GI_WGSL).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(RIS_GI_WGSL).toContain('traceSceneAlphaTintTransmittanceTextured(');
    expect(RIS_GI_WGSL).toContain('candidateVisibility = clamp(luminance(shadowTint), 0.0, 1.0);');
    expect(RIS_GI_WGSL).toContain('pHat = restir_gi_receiver_phat_from_payload(');
    expect(RIS_GI_WGSL).toContain(') * candidateVisibility;');
    expect(RIS_GI_WGSL).not.toContain('traceSceneAlphaTransmittanceTextured(');
    expect(RIS_GI_WGSL).not.toContain('r.w_sum = r.w_sum * shadowT;');
  });

  it('computes W = w_sum / (M · p̂(z)) per the RIS estimator', () => {
    expect(RIS_GI_WGSL).toContain('finaliseGIReservoirWFromPHat(&r, ubo.restirGiWCap, r.nativePHat);');
    expect(COMMON_WGSL).toContain('let denominator = f32((*r).M) * pHatF;');
    expect(COMMON_WGSL).toContain('let W_raw = (*r).w_sum / denominator;');
  });

  it('only dispatches over half-resolution pixels (W/2 × H/2)', () => {
    expect(RIS_GI_WGSL).toContain('halfDims = fullDims / 2u');
    expect(RIS_GI_WGSL).toContain('if (any(gid.xy >= halfDims)) { return; }');
  });
});

describe('Sprint 16 — ReservoirGI canonical byte pack and pass-local accessors', () => {
  it('declares the 28-u32 stride constant on the canonical export', () => {
    // GRIS Phase-0 appended the reconnection-shift cache at indices [20..27],
    // widening the per-pixel reservoir from 20 u32 (80 bytes) to 28 u32
    // (112 bytes). The [0..19] prefix stays byte-identical — see
    // reservoirPtLayout.test.ts for the bit-identity guard.
    expect(COMMON_WGSL).toContain('RESERVOIR_GI_STRIDE: u32 = 28u');
  });

  it('exposes canonical pack/unpack and generates exact-binding accessors', () => {
    expect(COMMON_WGSL).toContain('fn unpackReservoirGI');
    expect(COMMON_WGSL).toContain('fn packReservoirGI');
    const accessors = reservoirGiAccessorsWgsl({
      loadReadWriteBinding: 'currentGi',
      loadReadBinding: 'previousGi',
      storeReadWriteBinding: 'currentGi',
    });
    expect(accessors).toContain('fn loadReservoirGI_rw');
    expect(accessors).toContain('fn loadReservoirGI_ro');
    expect(accessors).toContain('fn storeReservoirGI_rw');
    expect(accessors).toContain('return unpackReservoirGI(words);');
    expect(accessors).toContain('let words = packReservoirGI(r);');
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
