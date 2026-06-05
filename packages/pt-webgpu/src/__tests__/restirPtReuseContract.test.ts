/**
 * restirPtReuseContract.test.ts — string-contract pins for the ReSTIR-PT reuse
 * unit (temporal-only, reconnection-shift-only, prefix-length-1). These guard the
 * load-bearing invariants that make the temporal feedback loop UNBIASED + stable,
 * mirroring the discipline of walkaround-hybrid's temporalGi GRIS notes.
 *
 * NO GPU here: the EXECUTED correctness check (the reconnection-shift Jacobian FD,
 * a converged A/B, etc.) is a separate hardware step (GPU reserved). This file
 * pins (a) that the unit composes as one string with every consumed symbol
 * defined, (b) that the temporal pass calls the FD-validated shift + the GRIS
 * finalize, and (c) the SINGLE most important radiometric invariant: the reused
 * reservoir's weight is m·p̂·W·J with NO division by a source pdf (an extra
 * /p_src diverges the feedback loop — the V19 grison lesson).
 */

import { describe, it, expect } from 'vitest';
import { composePtWebgpuReuseWgsl } from '../wgsl/pathTrace/restirPtCompose.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from '../wgsl/pathTrace/restirPtTemporal.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { RESTIR_PT_RESOLVE_WGSL } from '../wgsl/pathTrace/restirPtResolve.wgsl.js';
import { RESERVOIR_PT_HERO_WGSL } from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';

const composed = composePtWebgpuReuseWgsl();

/** Strip `//` line comments so a string-contract match counts only real code. */
function codeOnly(src: string): string {
  return src
    .split('\n')
    .filter((l) => !l.trim().startsWith('//'))
    .join('\n');
}

describe('ReSTIR-PT reuse — composes as a single WGSL unit', () => {
  it('declares all three @compute entry points exactly once each', () => {
    expect((composed.match(/@compute @workgroup_size\(8, 8, 1\)\s*\nfn restirPtProduce\(/g) ?? []).length).toBe(1);
    expect((composed.match(/@compute @workgroup_size\(8, 8, 1\)\s*\nfn restirPtTemporal\(/g) ?? []).length).toBe(1);
    expect((composed.match(/@compute @workgroup_size\(8, 8, 1\)\s*\nfn restirPtResolve\(/g) ?? []).length).toBe(1);
  });

  it('includes the reconnection-shift Jacobian + reservoir ADT exactly once', () => {
    // restirPtShiftJacobian (FD-validated) must be DEFINED once (not double-
    // included by both the shared modules and the reservoir composite).
    const code = codeOnly(composed);
    expect((code.match(/fn restirPtShiftJacobian\(/g) ?? []).length).toBe(1);
    expect((code.match(/fn restirPtReconnectionGeometryTerm\(/g) ?? []).length).toBe(1);
    expect((code.match(/struct ReservoirPTHero \{/g) ?? []).length).toBe(1);
    expect((code.match(/struct RestirPtParams \{/g) ?? []).length).toBe(1);
  });

  it('defines every shared symbol the passes reference (no dangling identifier)', () => {
    // Spot-check the load-bearing shared symbols the reuse passes call — each
    // must be DEFINED somewhere in the composed unit.
    for (const def of [
      'fn traceClosest(',
      'fn traceAny(',
      'fn evaluateBrdf(',
      'fn brdfDirectionalPdf(',
      'fn cosineHemisphereSample(',
      'fn glossyReflectionSample(',
      'fn decodeMaterial(',
      'fn hitMaterialId(',
      'fn sampleEnvironmentColor(',
      'fn sampleEnvironmentImportance(',
      'fn powerHeuristic(',
      'fn pcgInit(',
      'fn rand_f32(',
      'fn luminance(',
      'fn buildOnb(',
      'fn fresnelSchlick(',
      'struct FrameParams {',
    ]) {
      expect(composed.includes(def), `composed unit defines ${def}`).toBe(true);
    }
  });

  it('declares the ReSTIR-PT resources in @group(4) (separate from inherited 0..3)', () => {
    expect(composed).toContain('@group(4) @binding(0) var<storage, read_write> rpt_reservoirOut: array<u32>;');
    expect(composed).toContain('@group(4) @binding(1) var<storage, read_write> rpt_resCurrent: array<u32>;');
    expect(composed).toContain('@group(4) @binding(2) var<storage, read>       rpt_resPrev:    array<u32>;');
    expect(composed).toContain('@group(4) @binding(3) var<storage, read_write> rpt_result:      array<vec4f>;');
  });
});

describe('ReSTIR-PT temporal — calls the FD-validated shift + the GRIS finalize', () => {
  it('calls restirPtShiftJacobian with (rPrev.xv, rCur.xv, rPrev.xs, rPrev.ns) — prefix-1, xPre==xv', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'J = restirPtShiftJacobian(rPrev.xv, rCur.xv, rPrev.xs, rPrev.ns);',
    );
  });

  it('finalises with the GRIS form finaliseReservoirPTWGris (W = w_sum/p̂, NO /M)', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('finaliseReservoirPTWGris(&rGris, rptParams.wCap, params.cameraPos.xyz);');
    // The GRIS finalize must NOT divide by M (the MIS weights already sum to 1).
    const finalizeBody = RESERVOIR_PT_HERO_WGSL.slice(
      RESERVOIR_PT_HERO_WGSL.indexOf('fn finaliseReservoirPTWGris('),
    ).split('\n').slice(0, 8).join('\n');
    expect(finalizeBody).toContain('(*r).w_sum / pHatF');
    expect(finalizeBody).not.toMatch(/f32\(\(\*r\)\.M\)/); // no ·M normalisation
  });

  it('uses the INTEGRAND-MATCHING target (evaluateBrdf·cos·Lo), not the diffuse-cosine proxy (B3)', () => {
    // The hero target p̂ matches the integrand (the real visible-vertex BRDF) so the
    // temporal MIS weights glossy candidates correctly. Unbiased by construction
    // (W = w_sum/p̂ cancels p̂ — see the producer 1-sample note), variance-reducing
    // for a glossy visible vertex whose BRDF the old cosine proxy mis-weighted.
    const targetBody = RESERVOIR_PT_HERO_WGSL.slice(
      RESERVOIR_PT_HERO_WGSL.indexOf('fn restirPtTargetAt('),
    ).split('\n').slice(0, 12).join('\n');
    expect(targetBody).toContain('evaluateBrdf(albV, roughnessV, metalV, nv, wo, wi)');
    expect(targetBody).toContain('luminance(f * cosTheta * Lo)');
    expect(targetBody).not.toContain('INV_PI'); // the old diffuse-cosine proxy is gone
    // wo is threaded from the camera at the call sites (producer + temporal + finalize).
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('restirpt_safe_normalize(params.cameraPos.xyz - rCur.xv)');
  });

  it('reprojects via the previous-frame camera matrix (params.prevViewProj)', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('params.prevViewProj * vec4f(worldPos, 1.0)');
  });

  it('gates reuse on a reconnection-visibility ray (traceAny along xv → xs)', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('fn rptReconnectionVisible(');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('rptReconnectionVisible(rCur.xv, rCur.nv, rPrev.xs)');
  });

  it('M-clamps the previous history before contributing', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('let prevM = min(rPrev.M, rptParams.mClamp);');
  });
});

describe('ReSTIR-PT temporal — the w_prev weight is m·p̂·W·J with NO /p_src (V19 lesson)', () => {
  // THE load-bearing invariant. temporalGi.wgsl.ts:358-367: a reused RESERVOIR's
  // weight is m_prev·p̂_cur(T z_prev)·W_prev·J — NO division by a source pdf (the
  // reservoir's W already bakes its source pdf in). An extra /p_src multiplies the
  // carried weight by ≈π…∞ each frame and diverges the temporal feedback loop.
  it('the w_prev expression multiplies by the Jacobian J', () => {
    // Locate the w_prev assignment and assert it ends in `* J`.
    const m = RESTIR_PT_TEMPORAL_WGSL.match(/let w_prev = ([^;]+);/);
    expect(m, 'w_prev assignment present').not.toBeNull();
    const rhs = (m![1] ?? '').trim();
    // Must include the Jacobian factor.
    expect(/\*\s*J\b/.test(rhs), `w_prev RHS multiplies by J: "${rhs}"`).toBe(true);
    // Must be the canonical m·p̂·W·J shape.
    expect(rhs).toContain('m_prev');
    expect(rhs).toContain('pHatPrev_atCur');
    expect(rhs).toContain('rPrev.W');
  });

  it('the w_prev expression does NOT divide by any source pdf', () => {
    const m = RESTIR_PT_TEMPORAL_WGSL.match(/let w_prev = ([^;]+);/);
    const rhs = (m![1] ?? '').trim();
    // No division at all in the canonical reuse weight, and specifically no
    // /pdfSrc / /p_src / /pHat-source.
    expect(rhs.includes('/'), `w_prev RHS has NO division: "${rhs}"`).toBe(false);
    expect(rhs.toLowerCase().includes('pdfsrc')).toBe(false);
    expect(rhs.toLowerCase().includes('p_src')).toBe(false);
  });

  it('the canonical (current) sample weight is also division-free (m·p̂·W)', () => {
    const m = RESTIR_PT_TEMPORAL_WGSL.match(/let w_cur = ([^;]+);/);
    expect(m, 'w_cur assignment present').not.toBeNull();
    const rhs = (m![1] ?? '').trim();
    expect(rhs).toBe('m_cur * pHatCur_native * rCur.W');
    expect(rhs.includes('/')).toBe(false);
  });
});

describe('ReSTIR-PT producer — unbiased candidate weight + specular gate', () => {
  it('stores the REAL source BSDF pdf (pdfSrc) for unbiased glossy reconstruction', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'let pdfSrc = brdfDirectionalPdf(baseColorV, roughnessV, metallicV, 0.0, vMat.ior, nv, woV, wiRecon);',
    );
  });

  it('the candidate weight is p̂ / p_src (RIS), and finalises with the GRIS W', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let wCandidate = select(0.0, pHat / pdfSrc, pdfSrc > 1e-8);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('finaliseReservoirPTWGris(&r, rptParams.wCap, params.cameraPos.xyz);');
  });

  it('gates specular / transmissive visible vertices to an EMPTY reservoir (no reuse)', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptIsReusableVisibleVertex(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (transmission > 0.01) { return false; }');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'if (!rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV)) {',
    );
  });
});

describe('ReSTIR-PT resolve — reconstructs with the FULL BRDF (not the proxy target)', () => {
  it('evaluates the full visible-vertex BRDF and forms f·cos·Lo·W', () => {
    expect(RESTIR_PT_RESOLVE_WGSL).toContain(
      'let fBsdf = evaluateBrdf(r.albV, r.roughnessV, r.metalV, r.nv, wo, wiRecon);',
    );
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('let indirect = fBsdf * cosTheta * r.Lo * r.W;');
  });

  it('does NOT use the diffuse-cosine proxy (restirPtTargetAt) in reconstruction', () => {
    expect(RESTIR_PT_RESOLVE_WGSL).not.toContain('restirPtTargetAt');
  });
});
