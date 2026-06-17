/**
 * A1 — ReSTIR-PT SPATIAL reuse pass + COMPOSITE megakernel structure tests.
 *
 * These pin the estimator-split ALGEBRA + the GRIS spatial structure at the WGSL
 * string level (the GPU radiometric A/Bs — equal-spp variance vs the megakernel,
 * and the BDPT caustic scene — are V28 queue entries; see road-to-100 A1). They
 * are the cheap structural guard the byte-identity goldens + the naga compile gate
 * complement: a symbol-scope / binding / MIS-form regression that does not change
 * the digest is still caught here.
 */
import { describe, expect, it } from 'vitest';

import {
  composeRestirPtSpatialWgsl,
  composeRestirPtResolveWgsl,
  composePtWebgpuReuseWgsl,
} from '../wgsl/pathTrace/restirPtCompose.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';
import {
  composePtWebgpuCompositeTraceWgsl,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('A1 — ReSTIR-PT spatial pass (GRIS full GBH)', () => {
  it('declares exactly one restirPtSpatial @compute entry point', () => {
    const wgsl = composeRestirPtSpatialWgsl();
    expect((wgsl.match(/fn restirPtSpatial\(/g) ?? []).length).toBe(1);
    // It must NOT pull in the other passes' entry points (per-pass module split).
    expect(wgsl).not.toContain('fn restirPtProduce(');
    expect(wgsl).not.toContain('fn restirPtTemporal(');
    expect(wgsl).not.toContain('fn restirPtResolve(');
  });

  it('uses the hybrid shift Jacobian + the GRIS finalize (W=w_sum/p̂, NO /M)', () => {
    // The shift Jacobian re-roots a neighbour's reconnection edge onto the centre
    // pixel and multiplies the half-G ratio by the source/target BSDF replay-pdf ratio.
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'restirPtHybridShiftJacobianForPair(rQ, rCenter, woQ, woCenter)',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain('let qReplayPdfAtR = restirPtVisibleReplayPdfForDomain(rCenter, woCenter, qR[i].xs);');
    // GRIS finalize (the m_i already sum to 1 — no /M).
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      'finaliseReservoirPTWGris(&rOut, rptParams.wCap, params.cameraPos.xyz);',
    );
  });

  it('the reused-sample weight is m·p̂·W·J with NO /p_src (the V19 grison guard)', () => {
    // w_q = m_q · p̂_r(T z_q) · W_q · J  — the reservoir W already bakes the source
    // pdf; an extra /p_src would diverge the feedback loop. Pin the exact form.
    expect(RESTIR_PT_SPATIAL_WGSL).toContain('let w_q = m_q * pHatQ_atR * qW[i] * qJ[i];');
    // And it must NOT divide any reused weight by a source pdf.
    expect(RESTIR_PT_SPATIAL_WGSL).not.toMatch(/w_q\s*=\s*[^;]*\/\s*\w*[pP]dfSrc/);
  });

  it('gates each neighbour on reconnection VISIBILITY (unbiasedness)', () => {
    expect(RESTIR_PT_SPATIAL_WGSL).toContain('rptSpatialReconVisible(rCenter.xv, rCenter.nv, rQ.xs)');
  });

  it('reads the temporal output (b21) + writes the spatial output (b25) — hazard-free neighbour reads', () => {
    // The pass samples rpt_resSpatialIn (the temporal "current" slot) and writes a
    // SEPARATE rpt_resSpatialOut, so it never writes the slot it samples.
    expect(RESTIR_PT_SPATIAL_WGSL).toContain('@group(4) @binding(1) var<storage, read>       rpt_resSpatialIn:');
    expect(RESTIR_PT_SPATIAL_WGSL).toContain('@group(4) @binding(5) var<storage, read_write> rpt_resSpatialOut:');
    // After relocation the spatial output sits at the relocated group-0 binding 25.
    const composed = composeRestirPtSpatialWgsl();
    expect(composed).toContain('@group(0) @binding(25)');
    // Resolve reads the SAME relocated slot (25) the spatial pass wrote.
    expect(composeRestirPtResolveWgsl()).toContain('@group(0) @binding(25)');
  });

  it('the combined reuse unit declares all four entry points exactly once each', () => {
    const composed = composePtWebgpuReuseWgsl();
    for (const fn of ['restirPtProduce', 'restirPtTemporal', 'restirPtSpatial', 'restirPtResolve']) {
      expect((composed.match(new RegExp(`fn ${fn}\\(`, 'g')) ?? []).length).toBe(1);
    }
  });
});

describe('A1 — ReSTIR-PT composite megakernel (estimator split)', () => {
  it('the composite megakernel reads the resolve indirect at the relocated binding 23', () => {
    const composite = composePtWebgpuCompositeTraceWgsl(false);
    expect(composite).toContain('@group(0) @binding(23) var<storage, read_write> rpt_result_in: array<vec4f>;');
    // For a pixel the producer contributed to it composites the indirect + breaks (E0-only).
    expect(composite).toContain('let rptCompositeContributed = rptComposite.a > 0.5;');
    expect(composite).toContain('if (rptCompositeContributed) {');
    expect(composite).toContain('radiance = radiance + rptComposite.rgb;');
  });

  it('the composite runs the BSDF→light/env area-MIS at E0 for ALL pixels (analytic lights not in TLAS — no double-count)', () => {
    const composite = composePtWebgpuCompositeTraceWgsl(false);
    const dflt = composePtWebgpuTraceWgsl(false);
    // Both the default and composite megakernels run the BSDF-area connection on
    // sampleAllowsAreaMis only — no additional gate in composite mode.
    // Analytic lights (rect-area, disc, env, sky, directional) are NOT in the TLAS,
    // so the producer's xs cannot be an analytic light, and rptComposite.rgb cannot
    // double-count bsdfAreaLightConnectionContribution or
    // bsdfEnvironmentConnectionContribution. Dropping these (the previous
    // !rptCompositeContributed gate) caused a ~46% energy under-bias (2026-06-10 A/B).
    expect(dflt).toContain('if (sampleAllowsAreaMis) {');
    expect(dflt).not.toContain('!rptCompositeContributed');
    expect(composite).toContain('if (sampleAllowsAreaMis) {');
    expect(composite).not.toContain('if (sampleAllowsAreaMis && !rptCompositeContributed) {');
    // The connection BODY is present and runs for composited pixels.
    expect(composite).toContain('radiance = radiance + bsdfAreaLightConnectionContribution(');
  });

  it('the DEFAULT (non-composite) megakernel is unchanged — no rpt_result_in, full path', () => {
    const dflt = composePtWebgpuTraceWgsl(false);
    expect(dflt).not.toContain('rpt_result_in');
    expect(dflt).not.toContain('@group(0) @binding(23) var<storage, read_write>');
    // The default keeps the full indirect bounce loop (no E0-only break-after-direct).
    expect(dflt).not.toContain('A1 composite');
  });
});
