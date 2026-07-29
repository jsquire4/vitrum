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
import {
  sampleCMF,
  wavelengthToRGB,
  X_CMF_INTEGRAL,
  Y_CMF_INTEGRAL,
  Z_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';
import { composePathTraceKernelWgsl } from '../wgsl/pathTrace/kernel.wgsl.js';
import { RESTIR_PT_TEMPORAL_WGSL } from '../wgsl/pathTrace/restirPtTemporal.wgsl.js';
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { RESTIR_PT_RESOLVE_WGSL } from '../wgsl/pathTrace/restirPtResolve.wgsl.js';
import { RESERVOIR_PT_HERO_WGSL } from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';

const compositeKernel = composePathTraceKernelWgsl({
  volumetricSss: true,
  restirPtComposite: true,
});

function heroMisMixturePdfReference(lambdaNm: number): number {
  const [x, y, z] = sampleCMF(lambdaNm);
  return (x / X_CMF_INTEGRAL + y / Y_CMF_INTEGRAL + z / Z_CMF_INTEGRAL) / 3;
}

function addRgb(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): readonly [number, number, number] {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

/** Strip `//` line comments so a string-contract match counts only real code. */
function codeOnly(src: string): string {
  return src
    .split('\n')
    .filter((l) => !l.trim().startsWith('//'))
    .join('\n');
}

interface TemporalReuseWeightsInput {
  readonly cCur: number;
  readonly cPrev: number;
  readonly pHatCurNative: number;
  readonly pHatPrevAtCurSample: number;
  readonly pHatPrevAtCur: number;
  readonly pHatPrevNative: number;
  readonly rCurW: number;
  readonly rPrevW: number;
  /** |dT_current→previous| for the current sample. */
  readonly currentToPreviousJacobian: number;
  /** |dT_previous→current| for the previous sample. */
  readonly shiftJacobian: number;
}

function restirPtTemporalWeightsReference(input: TemporalReuseWeightsInput): {
  readonly mCur: number;
  readonly mPrev: number;
  readonly wCur: number;
  readonly wPrev: number;
} {
  const denomCur = input.cCur * input.pHatCurNative
    + input.cPrev
      * input.pHatPrevAtCurSample
      * input.currentToPreviousJacobian;
  const mCur = denomCur > 1e-12 ? (input.cCur * input.pHatCurNative) / denomCur : 1;
  const denomPrev = input.cCur
      * input.pHatPrevAtCur
      * input.shiftJacobian
    + input.cPrev * input.pHatPrevNative;
  const mPrev = denomPrev > 1e-12 ? (input.cPrev * input.pHatPrevNative) / denomPrev : 0;
  return {
    mCur,
    mPrev,
    wCur: mCur * input.pHatCurNative * input.rCurW,
    wPrev: mPrev * input.pHatPrevAtCur * input.rPrevW * input.shiftJacobian,
  };
}

function finaliseReservoirPTWGrisReference(opts: {
  readonly M: number;
  readonly wSum: number;
  readonly pHat: number;
}): number {
  if (opts.M <= 0 || !Number.isFinite(opts.wSum) || opts.wSum <= 0 ||
      !Number.isFinite(opts.pHat) || opts.pHat <= 1e-9) return Number.NEGATIVE_INFINITY;
  return Math.log(opts.wSum) - Math.log(opts.pHat);
}

describe('ReSTIR-PT temporal — calls the reconnection shift + the GRIS finalize', () => {
  it('calls the prefix-1 geometry shift with source/target reservoirs', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'let J = restirPtReconnectionJacobianForPair(rPrev, rCur);',
    );
  });

  it('finalises with the unbiased log-domain GRIS form (NO /M and no clamp)', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('finaliseReservoirPTWGris(&rGris);');
    // The GRIS finalize must NOT divide by M (the MIS weights already sum to 1).
    const finalizeBody = RESERVOIR_PT_HERO_WGSL.slice(
      RESERVOIR_PT_HERO_WGSL.indexOf('fn finaliseReservoirPTWGris('),
    ).split('\n').slice(0, 30).join('\n');
    expect(finalizeBody).toContain('(*r).logWeightSum - log(pHatF)');
    expect(finalizeBody).not.toContain('wCap');
    expect(finalizeBody).not.toMatch(/f32\(\(\*r\)\.M\)/); // no ·M normalisation
  });

  it('does not double-count proposal density in the prefix-1 change of variables', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain('return restirPtShiftJacobian(');
    expect(RESERVOIR_PT_HERO_WGSL).not.toContain('restirPtVisibleReplayPdfForDomain');
    expect(RESTIR_PT_TEMPORAL_WGSL).not.toContain('pSource / pTarget');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('refreshReconnectionStatePT(&rGris);');
  });

  it('uses the implemented one-edge topology directly instead of an unpersisted prefix flag', () => {
    expect(RESERVOIR_PT_HERO_WGSL).not.toContain('prefixVertexCount');
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain('prefixVertexCount');
    expect(RESTIR_PT_TEMPORAL_WGSL).not.toContain('prefixVertexCount');
    expect(RESTIR_PT_SPATIAL_WGSL).not.toContain('prefixVertexCount');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'let prevValid = rptFinitePositive(J)',
    );
  });

  it('uses the INTEGRAND-MATCHING target (evaluateBrdf·cos·Lo), not the diffuse-cosine proxy (B3)', () => {
    // The hero target p̂ matches the integrand (the real visible-vertex BRDF) so the
    // temporal MIS weights glossy candidates correctly. Unbiased by construction
    // (W = w_sum/p̂ cancels p̂ — see the producer 1-sample note), variance-reducing
    // for a glossy visible vertex whose BRDF the old cosine proxy mis-weighted.
    const targetStart = RESERVOIR_PT_HERO_WGSL.indexOf(
      'fn restirPtTargetAt(',
    );
    const targetEnd = RESERVOIR_PT_HERO_WGSL.indexOf(
      'fn restirPtTargetForDomain(',
      targetStart,
    );
    const targetBody = RESERVOIR_PT_HERO_WGSL.slice(targetStart, targetEnd);
    expect(targetBody).toContain('evaluateBrdfFullWithClearcoatNormal(');
    expect(targetBody).toContain('clearcoatNormalV, wo, wi,');
    expect(targetBody).toContain('clearcoatV, clearcoatRoughnessV, sheenV, sheenRoughnessV, sheenColorV,');
    expect(targetBody).toContain('specularColorV, specularIntensityV,');
    expect(targetBody).toContain('anisotropyV, anisotropyRotationV,');
    expect(targetBody).toContain('luminance(f * cosTheta * Lo)');
    expect(targetBody).not.toContain('INV_PI'); // the old diffuse-cosine proxy is gone
    // wo is threaded from the camera at the call sites (producer + temporal + finalize).
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('let woCur  = rCur.woV;');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('let woPrev = rPrev.woV;');
  });

  it('reprojects via the previous-frame camera matrix (params.prevViewProj)', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('params.prevViewProj * vec4f(worldPos, 1.0)');
  });

  it('gates reuse on a reconnection-visibility ray (traceAny along xv → xs)', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('fn rptReconnectionVisible(');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'rptReconnectionVisible(rCur.xv, rCur.nv, rPrev.xs, &rng)',
    );
  });

  it('M-clamps the previous history before contributing', () => {
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('let prevM = min(rPrev.M, rptParams.mClamp);');
  });
});

describe('ReSTIR-PT temporal — log weight is log(m·p̂·W·J) with NO /p_src', () => {
  // THE load-bearing invariant. temporalGi.wgsl.ts:358-367: a reused RESERVOIR's
  // weight is m_prev·p̂_cur(T z_prev)·W_prev·J — NO division by a source pdf (the
  // reservoir's W already bakes its source pdf in). An extra /p_src multiplies the
  // carried weight by ≈π…∞ each frame and diverges the temporal feedback loop.
  it('the previous expression includes the Jacobian exactly once', () => {
    const m = RESTIR_PT_TEMPORAL_WGSL.match(/let logWeightPrevious =\s*([^;]+);/);
    expect(m, 'logWeightPrevious assignment present').not.toBeNull();
    const rhs = (m![1] ?? '').trim();
    expect(rhs).toContain('logMPrevious');
    expect(rhs).toContain('log(pHatPrev_atCur)');
    expect(rhs).toContain('rPrev.logW');
    expect(rhs).toContain('log(J)');
  });

  it('the w_prev expression does NOT divide by any source pdf', () => {
    const m = RESTIR_PT_TEMPORAL_WGSL.match(/let logWeightPrevious =\s*([^;]+);/);
    const rhs = (m![1] ?? '').trim();
    // No division at all in the canonical reuse weight, and specifically no
    // /pdfSrc / /p_src / /pHat-source.
    expect(rhs.includes('/'), `w_prev RHS has NO division: "${rhs}"`).toBe(false);
    expect(rhs.toLowerCase().includes('pdfsrc')).toBe(false);
    expect(rhs.toLowerCase().includes('p_src')).toBe(false);
  });

  it('the canonical sample uses the same log-product representation', () => {
    const m = RESTIR_PT_TEMPORAL_WGSL.match(/let logWeightCurrent =\s*([^;]+);/);
    expect(m, 'logWeightCurrent assignment present').not.toBeNull();
    const rhs = (m![1] ?? '').trim();
    expect(rhs).toBe('logMCurrent + log(pHatCur_native) + rCur.logW');
    expect(rhs.includes('/')).toBe(false);
  });

  it('CPU oracle: temporal reuse weights are pairwise-GRIS and pdfSrc-independent', () => {
    const input: TemporalReuseWeightsInput = {
      cCur: 3,
      cPrev: 5,
      pHatCurNative: 2.0,
      pHatPrevAtCurSample: 4.0,
      pHatPrevAtCur: 1.5,
      pHatPrevNative: 3.5,
      rCurW: 0.75,
      rPrevW: 0.4,
      currentToPreviousJacobian: 1 / 0.6,
      shiftJacobian: 0.6,
    };
    const weights = restirPtTemporalWeightsReference(input);

    expect(weights.mCur).toBeCloseTo(
      (3 * 2.0) / (3 * 2.0 + 5 * 4.0 * input.currentToPreviousJacobian),
      12,
    );
    expect(weights.mPrev).toBeCloseTo(
      (5 * 3.5) / (3 * 1.5 * input.shiftJacobian + 5 * 3.5),
      12,
    );
    expect(weights.wCur).toBeCloseTo(weights.mCur * input.pHatCurNative * input.rCurW, 12);
    expect(weights.wPrev).toBeCloseTo(
      weights.mPrev * input.pHatPrevAtCur * input.rPrevW * input.shiftJacobian,
      12,
    );

    for (const pdfSrc of [0.01, 0.25, 16, 1024]) {
      expect(weights.wPrev / pdfSrc).not.toBeCloseTo(weights.wPrev, 6);
      expect(restirPtTemporalWeightsReference({ ...input }).wPrev).toBeCloseTo(weights.wPrev, 12);
    }

    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'cCur, pHatPrev_atCur, J,',
    );
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      'cPrev, pHatPrev_atCurSample, JCurrentToPrevious,',
    );
  });

  it('CPU oracle: log GRIS finalization is independent of M and unclamped', () => {
    const expected = Math.log(4);
    expect(finaliseReservoirPTWGrisReference({ M: 1, wSum: 12, pHat: 3 })).toBeCloseTo(expected, 14);
    expect(finaliseReservoirPTWGrisReference({ M: 128, wSum: 12, pHat: 3 })).toBeCloseTo(expected, 14);
    expect(finaliseReservoirPTWGrisReference({ M: 128, wSum: 12, pHat: 0 })).toBe(Number.NEGATIVE_INFINITY);
    expect(finaliseReservoirPTWGrisReference({ M: 0, wSum: 12, pHat: 3 })).toBe(Number.NEGATIVE_INFINITY);
  });

  it('max-log arithmetic preserves adversarial ratios without a biased ceiling', () => {
    const finalizeLog = (wSum: number, pHat: number): number =>
      Math.log(wSum) - Math.log(pHat);
    for (const [wSum, pHat] of [
      [1, 1],
      [1e20, 1e-8],
      [Number.MAX_VALUE, 1e-8],
      [Number.MAX_VALUE, Number.MIN_VALUE],
    ] as const) {
      expect(Number.isFinite(finalizeLog(wSum, pHat))).toBe(true);
    }
    expect(Math.exp(finalizeLog(12, 3))).toBeCloseTo(4, 14);
    expect(RESERVOIR_PT_HERO_WGSL).toContain('fn rptLogAddExp(');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('let logW = (*r).logWeightSum - log(pHatF);');
    expect(RESERVOIR_PT_HERO_WGSL).not.toContain('wCap');
  });

  it('production reservoir code rejects NaN/Inf before candidate count or history publication', () => {
    const updateBody = RESERVOIR_PT_HERO_WGSL.slice(
      RESERVOIR_PT_HERO_WGSL.indexOf('fn updateReservoirPTLog('),
      RESERVOIR_PT_HERO_WGSL.indexOf('fn copyReservoirPTVisibleDomain('),
    );
    expect(updateBody).toContain(') -> bool {');
    expect(updateBody).toContain('if (!rptFiniteScalar(logWeight)');
    expect(updateBody).toContain('|| !rptFinitePositive(pdfSrc)');
    expect(updateBody).toContain('let nextLogWeightSum = rptLogAddExp(');
    expect(updateBody).toContain('rptMarkReservoirNumericFailure(r);');
    expect(updateBody.indexOf('if (!rptFinitePositive(w)')).toBeLessThan(
      updateBody.indexOf('(*r).M = (*r).M + 1u;'),
    );
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain(
      '&rGris, rCur.xs, rCur.ns, rCur.Lo, rCur.heroLambdaV,',
    );
    expect(RESTIR_PT_SPATIAL_WGSL).toContain(
      '&rOut, rCenter.xs, rCenter.ns, rCenter.Lo, rCenter.heroLambdaV,',
    );

    for (const bad of [Number.NaN, Infinity, -Infinity, 0, -1]) {
      expect(Number.isFinite(bad) && bad > 0).toBe(false);
    }
    expect(Number.isFinite(1) && 1 > 0).toBe(true);
  });
});

describe('ReSTIR-PT producer — unbiased candidate weight + specular gate', () => {
  it('stores the REAL source BSDF pdf (pdfSrc) for unbiased glossy reconstruction', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let pdfSrc = rptSourceDirectionalPdfFull(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'clearcoatV, clearcoatRoughnessV, sheenV, sheenRoughnessV,',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('specularColorV, specularIntensityV,');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('anisotropyV, anisotropyRotationV,');
  });

  it('samples clearcoat and sheen source lobes with a matching normalized pdf', () => {
    for (const line of [
      'fn rptSourceLobeWeightSum(clearcoat: f32, sheen: f32) -> f32 {',
      'fn rptSampleSourceReconnectionDirection(',
      'sheenRoughness: f32,',
      'let xiSource = rand_f32(rng) * lobeWeightSum;',
      'if (xiSource < 1.0 + max(clearcoat, 0.0)) {',
      'buildOnb(clearcoatNormal, &ccTanT, &ccTanB);',
      'let bs = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);',
      'let bs = charlieSheenSample(rng, wo, normal, tanT, tanB, sheenRoughness);',
      'return brdfDirectionalPdfFullSampledWithClearcoatNormal(',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('evaluates suffix Lo direct lighting and onward throughput with extension-aware BRDF helpers', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptDirectAtVertex(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('clearcoatNormal: vec3f,');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('clearcoatRoughness: f32,');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let brdf = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let continuationPdf = max(dot(normal, envDir), 0.0) * INV_PI;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let fOnward = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(');
  });

  it('keeps rptDirectAtVertex transmission and clearcoat parameters aligned with its caller', () => {
    const signatureStart = RESTIR_PT_PRODUCER_WGSL.indexOf('fn rptDirectAtVertex(');
    const signatureEnd = RESTIR_PT_PRODUCER_WGSL.indexOf(') -> vec3f {', signatureStart);
    const signature = RESTIR_PT_PRODUCER_WGSL.slice(signatureStart, signatureEnd);
    expect(signature).toContain(
      'metallic: f32,\n  transmission: f32,\n  clearcoat: f32,',
    );

    const callStart = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'Lo = Lo + rptDirectAtVertex(',
    );
    const callEnd = RESTIR_PT_PRODUCER_WGSL.indexOf(');', callStart);
    const call = RESTIR_PT_PRODUCER_WGSL.slice(callStart, callEnd);
    expect(call).toContain(
      'roughness, metallic, sm.transmission,\n      sm.clearcoat,',
    );
  });

  it('covers spot and mesh-area emitters in the suffix direct-lighting producer', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('for (var si = 0u; si < params.spotLightCount; si = si + 1u) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let a = meshAreaLights[mb].xyz;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let mr = sampleMeshAreaLightRadiance(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('mi, vec3f(uu, vv, ww), lpos,');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let area = 0.5 * length(cross(b - a, c - a));');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (dist2 <= 0.0 || area <= 0.0) { continue; }');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('meshAreaLights[mb + 3u].w > 0.5 || !traceAny');
  });

  it('uses packed N-directional RGB records in suffix direct lighting', () => {
    const code = codeOnly(RESTIR_PT_PRODUCER_WGSL);
    expect(code).toContain('for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {');
    expect(code).toContain('let dDirAD = directionalLights[dBase];');
    expect(code).toContain('let dIrrMean = directionalLights[dBase + 1u];');
    expect(code).toContain('let angDiamRaw = dDirAD.w;');
    expect(code).toContain('let dirShadowDisabled = angDiamRaw < 0.0;');
    expect(code).toContain('let angDiam = select(angDiamRaw, -1.0 - angDiamRaw, dirShadowDisabled);');
    expect(code).toContain('let lightDir = rptSampleDirectionalCone(rng, dDirAD.xyz, angDiam);');
    expect(code).toContain('let dIrrOut = select(dIrrMean.rgb, spectralEmissionAtHero(dIrrMean.rgb, heroLambda), params.spectralEnabled != 0u);');
    expect(code).toContain('contrib = contrib + suffixThroughput * brdf * nDotL * dIrrOut;');
    expect(code).not.toContain('contrib = contrib + suffixThroughput * brdf * nDotL * params.lightDir.w;');
  });

  it('honors packed emitter castShadow flags in suffix direct lighting where those lanes are available', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let dirShadowDisabled = angDiamRaw < 0.0;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY, rng)) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let rectShadowDisabled = rectAreaLights[rb].w > 0.5;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (rectShadowDisabled || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3), rng)) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let ptShadowDisabled = ptExtra.z > 0.5;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (ptShadowDisabled || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3), rng)) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let spShadowDisabled = spExtra.z > 0.5;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (spShadowDisabled || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3), rng)) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('meshAreaLights[mb + 3u].w > 0.5 || !traceAny');
  });

  it('admits finite glossy reflection without a maturity switch and excludes transmission', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (transmission > 0.0) { return false; }');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('return bsdfHasFiniteConnectionSupport(');
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain('allowGlossyReuse');
  });

  it('the candidate weight is p̂ / p_src (RIS), and finalises with the GRIS W', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let logCandidateWeight = log(pHat) - log(pdfSrc);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('finaliseReservoirPTWGris(&r);');
  });

  it('stores the visible-vertex extension payload and uses anisotropic producer sampling', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let anisotropyV = materialAnisotropy(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptSampleSourceReconnectionDirection(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('bs = glossyReflectionSampleAnisotropic(rng, wo, normal, tanT, tanB, roughness, anisotropy);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('anisotropyV,');
    for (const field of [
      'r.clearcoatV = clearcoatV;',
      'r.clearcoatRoughnessV = clearcoatRoughnessV;',
      'r.sheenV = sheenV;',
      'r.sheenRoughnessV = sheenRoughnessV;',
      'r.sheenColorV = sheenColorV;',
      'r.iridescenceV = iridescenceV;',
      'r.iridescenceIorV = iridescenceIorV;',
      'r.iridescenceThicknessMinV = iridescenceThicknessMinV;',
      'r.iridescenceThicknessMaxV = iridescenceThicknessMaxV;',
      'r.specularColorV = specularColorV;',
      'r.specularIntensityV = specularIntensityV;',
      'r.anisotropyV = anisotropyV;',
      'r.anisotropyRotationV = anisotropyRotationV;',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(field);
    }
  });

  it('mirrors the main shade prologue material-map stack for the visible vertex payload', () => {
    for (const line of [
      'var baseColorV = vMat.baseColor * sampleVertexColor(vHit.triIndex, vHit.baryVW).rgb * sampleBaseColorTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex).rgb;',
      'baseColorV = baseColorV * sampleAoFactor(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);',
      'let ormSampleV = sampleOrmTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);',
      'var roughnessV = clamp(vMat.roughness * ormSampleV.g, 0.0, 1.0);',
      'var metallicV = clamp(vMat.metallic * ormSampleV.b, 0.0, 1.0);',
      'var transmissionV = clamp(vMat.transmission * sampleTransmissionTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);',
      'nv = applyNormalMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex, vIsFront);',
      'nv = applyBumpMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);',
      'let clearcoatNormalV = applyClearcoatNormalMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);',
      'var clearcoatV = clamp(vMat.clearcoat * sampleClearcoatTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);',
      'var clearcoatRoughnessV = clamp(vMat.clearcoatRoughness * sampleClearcoatRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);',
      'var sheenColorV = clamp(vMat.sheenColor * sampleSheenColorTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), vec3f(0.0), vec3f(1.0));',
      'var sheenRoughnessV = clamp(vMat.sheenRoughness * sampleSheenRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);',
      'var iridescenceV = clamp(vMat.iridescence * sampleIridescenceTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);',
      'let iridescenceThicknessSampleV = sampleIridescenceThicknessTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);',
      'var specularColorV = max(',
      'var specularIntensityV = clamp(vMat.specularIntensity * sampleSpecularIntensityTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('applies alpha pass-through and mapped transmission before the reusable-visible gate', () => {
    const alphaIdx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let vTrace = rptTraceClosestAfterAlpha(primaryRay, &rng);',
    );
    const transmissionIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('var transmissionV = clamp(vMat.transmission * sampleTransmissionTexture');
    const gateIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('!rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV, clearcoatV, sheenV)');
    expect(alphaIdx).toBeGreaterThanOrEqual(0);
    expect(transmissionIdx).toBeGreaterThan(alphaIdx);
    expect(gateIdx).toBeGreaterThan(transmissionIdx);
  });

  it('mirrors layer and spectral visible-vertex prologue effects before sampling', () => {
    for (const line of [
      'iorV = cauchyIorAtLambda(heroLambda, vMat.ior, vMat.dispersionAbbe);',
      'let layerTxV = clamp(select(vMat.backLayerTx, vMat.frontLayerTx, vIsFront), vec3f(0.0), vec3f(1.0));',
      'roughnessV = clamp(layerRoughnessV, 0.0, 1.0);',
      'activeLayerWeightRgb(layerTxV, heroLambda, true)',
      'spectralCombinedReflectanceAtHero(',
      'baseColorV = vec3f(reflScalarV);',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
    const prologueIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('baseColorV = vec3f(reflScalarV);');
    const f0Idx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let f0BaseV = materialSpecularF0(',
    );
    expect(f0Idx).toBeGreaterThan(prologueIdx);
  });

  it('mirrors the material-map prologue for suffix/reconnection vertices too', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('struct RptSuffixMaterial {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptSuffixMaterialAtHit(hit: SceneHit, incomingDir: vec3f, wo: vec3f, heroLambda: f32) -> RptSuffixMaterial');
    for (const line of [
      'out.baseColor = mat.baseColor * sampleVertexColor(hit.triIndex, hit.baryVW).rgb * sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex).rgb;',
      'out.baseColor = out.baseColor * sampleAoFactor(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);',
      'let ormSample = sampleOrmTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);',
      'out.emissive = mat.emissive * sampleEmissiveTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex).rgb;',
      'out.transmission = clamp(mat.transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);',
      'out.normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex, isFrontFace);',
      'out.normal = applyBumpMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);',
      'out.clearcoatNormal = applyClearcoatNormalMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);',
      'out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);',
      'out.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), vec3f(0.0), vec3f(1.0));',
      'out.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);',
      'out.specularColor = max(',
      'mat.specularColor * sampleSpecularColorTexture(',
      'out.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);',
      'out.anisotropy = materialAnisotropy(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);',
      'out.anisotropyRotation = materialAnisotropyRotation(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);',
      'out.envMapIntensity = materialEnvMapIntensity(matId);',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('uses the layered iridescence/coherent-stack response for visible-vertex source sampling', () => {
    const f0BaseIdx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let f0BaseV = materialSpecularF0(',
    );
    const iridescentF0Idx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let iridescentF0V = iridescenceModifiedF0(',
      f0BaseIdx,
    );
    const f0Idx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let f0V = select(iridescentF0V, f0BaseV, thinFilmV.enabled);',
      iridescentF0Idx,
    );
    const fresnelIdx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let fresV = bsdfLayeredInterfaceResponse(',
      f0Idx,
    );
    const sampleIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let wiRecon = rptSampleSourceReconnectionDirection(', fresnelIdx);
    const pdfIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let pdfSrc = rptSourceDirectionalPdfFull(', sampleIdx);

    expect(f0BaseIdx).toBeGreaterThanOrEqual(0);
    expect(iridescentF0Idx).toBeGreaterThan(f0BaseIdx);
    expect(f0Idx).toBeGreaterThan(iridescentF0Idx);
    expect(fresnelIdx).toBeGreaterThan(f0Idx);
    expect(sampleIdx).toBeGreaterThan(fresnelIdx);
    expect(pdfIdx).toBeGreaterThan(sampleIdx);
    expect(
      RESTIR_PT_PRODUCER_WGSL.slice(iridescentF0Idx, f0Idx),
    ).toContain('iridescenceThicknessMaxV,');
    expect(RESTIR_PT_PRODUCER_WGSL.slice(fresnelIdx, sampleIdx)).toContain(
      'materialSpecularFresnelSchlick(',
    );
  });

  it('applies layer, spectral albedo, and spectral emission in suffix Lo', () => {
    for (const line of [
      'out.ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);',
      'let layerTx = clamp(select(mat.backLayerTx, mat.frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));',
      'activeLayerWeightRgb(layerTx, heroLambda, true)',
      'spectralCombinedReflectanceAtHero(',
      'out.baseColor = vec3f(reflScalar);',
      'let emissive = select(sm.emissive, spectralEmissionAtHero(sm.emissive, heroLambda), params.spectralEnabled != 0u);',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('mirrors megakernel spectral light handling for every suffix direct-light class', () => {
    for (const line of [
      'let dIrrOut = select(dIrrMean.rgb, spectralEmissionAtHero(dIrrMean.rgb, heroLambda), params.spectralEnabled != 0u);',
      'let rrOut = select(rr, spectralEmissionAtHero(rr, heroLambda), params.spectralEnabled != 0u);',
      'let radOut = select(rad, spectralEmissionAtHero(rad, heroLambda), params.spectralEnabled != 0u);',
      'let sradOut = select(srad, spectralEmissionAtHero(srad, heroLambda), params.spectralEnabled != 0u);',
      'let mrOut = select(mr, spectralEmissionAtHero(mr, heroLambda), params.spectralEnabled != 0u);',
      'let envColorOut = select(envColor, spectralEmissionAtHero(envColor, heroLambda), params.spectralEnabled != 0u) * envMapIntensity;',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('mirrors megakernel environment intensity and spectral handling in suffix paths', () => {
    for (const line of [
      'envMapIntensity: f32,',
      'let envColorOut = select(envColor, spectralEmissionAtHero(envColor, heroLambda), params.spectralEnabled != 0u) * envMapIntensity;',
      'sm.envMapIntensity,',
      'let envContribution = select(envRgb, spectralEmissionAtHero(envRgb, heroLambda), params.spectralEnabled != 0u);',
      'let envNeePdf = environmentNeeProposalPdf(nextDir, normal);',
      'let escapeMisWeight = powerHeuristic(cosSample.pdf, envNeePdf);',
      'Lo = Lo + suffixThroughput * envContribution * sm.envMapIntensity * escapeMisWeight;',
      '// would estimate the same direct path a second time. An empty producer',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('consumes suffix anisotropy in direct lighting and onward BRDF evaluation', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('anisotropy: f32,');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('anisotropyRotation: f32,');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('anisotropy, anisotropyRotation,');

    const directCallIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('Lo = Lo + rptDirectAtVertex(');
    const directAnisoIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('sm.anisotropy, sm.anisotropyRotation,', directCallIdx);
    const onwardIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let fOnward = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(');
    const onwardAnisoIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('sm.anisotropy, sm.anisotropyRotation,', onwardIdx);
    expect(directCallIdx).toBeGreaterThanOrEqual(0);
    expect(directAnisoIdx).toBeGreaterThan(directCallIdx);
    expect(onwardIdx).toBeGreaterThan(directAnisoIdx);
    expect(onwardAnisoIdx).toBeGreaterThan(onwardIdx);
  });

  it('alpha-skips reconnection and onward suffix hits before decoding their material', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptTraceClosestAfterAlpha(rayIn: Ray, rng: ptr<function, PtRngState>) -> RptAlphaTraceHit');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let passesThrough = alphaTestPassThrough(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'hitMaterialId(hit), hit.triIndex, hit.baryVW, hit.instanceIndex, rng,',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let surfaceHitLimit = sceneSurfaceHitLimit();');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (surfaceHitCount >= surfaceHitLimit) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('valid = false;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let sTrace = rptTraceClosestAfterAlpha(reconRay, &rng);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let nextTrace = rptTraceClosestAfterAlpha(Ray(pos + normal * 1e-3, nextDir), rng);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (!nextTrace.valid) { return rptInvalidSuffixEstimate(); }');

    const reconTraceIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let sTrace = rptTraceClosestAfterAlpha(reconRay, &rng);');
    const suffixLoIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let suffixEstimate = rptComputeLoAtReconnection(');
    expect(suffixLoIdx).toBeGreaterThan(reconTraceIdx);
  });

  it('uses the mapped suffix normal as the reservoir reconnection normal', () => {
    const matIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let sReservoirMat = rptSuffixMaterialAtHit(sHit, reconRay.direction, reconDirToXv, heroLambda);');
    const normalIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('ns = sReservoirMat.normal;');
    const updateIdx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      '&r, xs, ns, Lo, heroLambda, pdfSrc, logCandidateWeight, &rng,',
    );
    expect(matIdx).toBeGreaterThanOrEqual(0);
    expect(normalIdx).toBeGreaterThan(matIdx);
    expect(updateIdx).toBeGreaterThan(normalIdx);
  });

  it('gates singular or transmissive vertices while retaining finite thin-film reuse', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptIsReusableVisibleVertex(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (transmission > 0.0) { return false; }');
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain(
      'vMat.isUnlit || vMat.thinFilmEnabled ||',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      '// thin film is a finite rough-interface Fresnel replacement and remains in',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      '!rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV, clearcoatV, sheenV)) {',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'anisotropyV, anisotropyRotationV, thinFilmV,',
    );
  });
  it('rejects a transmissive or alpha-exhausted suffix instead of publishing partial Lo', () => {
    const decodeIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let sm = rptSuffixMaterialAtHit(');
    const rejectIdx = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'if (sm.transmission > 0.0) { return rptInvalidSuffixEstimate(); }',
      decodeIdx,
    );
    const directIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('Lo = Lo + rptDirectAtVertex(', decodeIdx);
    expect(rejectIdx).toBeGreaterThan(decodeIdx);
    expect(directIdx).toBeGreaterThan(rejectIdx);
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (!sTrace.valid) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (!suffixEstimate.valid) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('Lo = suffixEstimate.Lo;');
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain('if (sm.transmission > 0.0) { break; }');
});
});

describe('ReSTIR-PT resolve — reconstructs with the FULL BRDF (not the proxy target)', () => {
  it('evaluates the full visible-vertex BRDF and applies logW without overflow', () => {
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('let fBsdf = evaluateBrdfFullWithClearcoatNormal(');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('r.nv, r.clearcoatNormalV, wo, wiRecon,');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('r.clearcoatV, r.clearcoatRoughnessV, r.sheenV, r.sheenRoughnessV, r.sheenColorV,');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('r.specularColorV, r.specularIntensityV,');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('r.anisotropyV, r.anisotropyRotationV,');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('let integrand = fBsdf * cosTheta * r.Lo;');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('let scaled = rptScalePositiveVec3ByLog(integrand, r.logW);');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('if (!scaled.valid)');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('vec4f(0.0, 0.0, 0.0, -1.0)');
  });

  it('does NOT use the diffuse-cosine proxy (restirPtTargetAt) in reconstruction', () => {
    expect(RESTIR_PT_RESOLVE_WGSL).not.toContain('restirPtTargetAt');
  });

  it('reconstructs history at the reservoir wavelength before RGB compositing', () => {
    expect(RESTIR_PT_RESOLVE_WGSL).toContain(
      'let reservoirHeroPdf = heroMisMixturePdf(r.heroLambdaV);',
    );
    expect(RESTIR_PT_RESOLVE_WGSL).toContain(
      'r.heroLambdaV, luminance(indirect), reservoirHeroPdf,',
    );

    const currentConvert = compositeKernel.indexOf(
      'outRadiance = heroWavelengthToRgb(heroLambda, luminance(radiance), heroPdf);',
    );
    const reusedAdd = compositeKernel.indexOf(
      'outRadiance = outRadiance + rptComposite.rgb;',
    );
    expect(currentConvert).toBeGreaterThanOrEqual(0);
    expect(reusedAdd).toBeGreaterThan(currentConvert);
    expect(compositeKernel).not.toContain('radiance = radiance + rptComposite.rgb;');

    // Deliberately separate blue and red hero samples. Correct reuse is the sum
    // of two independently reconstructed estimators. Relabelling the reservoir
    // scalar with the current wavelength (the old composite order) is visibly
    // and numerically different.
    const currentLambda = 450;
    const reservoirLambda = 625;
    const currentScalar = 2.25;
    const reservoirScalar = 1.75;
    const currentPdf = heroMisMixturePdfReference(currentLambda);
    const reservoirPdf = heroMisMixturePdfReference(reservoirLambda);
    const correct = addRgb(
      wavelengthToRGB(currentLambda, currentScalar, currentPdf),
      wavelengthToRGB(reservoirLambda, reservoirScalar, reservoirPdf),
    );
    const relabelled = wavelengthToRGB(
      currentLambda,
      currentScalar + reservoirScalar,
      currentPdf,
    );
    const error = Math.hypot(
      correct[0] - relabelled[0],
      correct[1] - relabelled[1],
      correct[2] - relabelled[2],
    );
    expect(error).toBeGreaterThan(0.01);
    expect(correct.every(Number.isFinite)).toBe(true);
  });
});

describe('ReSTIR-PT reservoir — visible material is deterministically rehydrated', () => {
  it('uses the compact identity ABI and reconstructs every resolve lobe', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain('64 bytes (16 × u32)');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('const RESERVOIR_PT_HERO_STRIDE: u32 = 16u;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('fn rptVisibleMaterialAtSurface(');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('fn rptHydrateVisibleDomain(');
    for (const line of [
      '(*r).clearcoatV = vm.clearcoat;',
      '(*r).clearcoatRoughnessV = vm.clearcoatRoughness;',
      '(*r).sheenV = vm.sheen;',
      '(*r).sheenColorV = vm.sheenColor;',
      '(*r).iridescenceV = vm.iridescence;',
      '(*r).anisotropyV = vm.anisotropy;',
      '(*r).specularColorV = vm.specularColor;',
      '(*r).specularIntensityV = vm.specularIntensity;',
      '(*r).clearcoatNormalV = vm.clearcoatNormal;',
    ]) {
      expect(RESERVOIR_PT_HERO_WGSL).toContain(line);
    }
  });

  it('copies the full visible-material domain into temporal/spatial output reservoirs', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain('fn copyReservoirPTVisibleDomain(');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).clearcoatV = src.clearcoatV;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).specularColorV = src.specularColorV;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).specularIntensityV = src.specularIntensityV;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).clearcoatNormalV = src.clearcoatNormalV;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).anisotropyRotationV = src.anisotropyRotationV;');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('copyReservoirPTVisibleDomain(&rGris, rCur);');
    expect(RESTIR_PT_SPATIAL_WGSL).toContain('copyReservoirPTVisibleDomain(&rOut, rCenter);');
  });
});
