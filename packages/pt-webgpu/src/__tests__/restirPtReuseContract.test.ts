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
import { RESTIR_PT_SPATIAL_WGSL } from '../wgsl/pathTrace/restirPtSpatial.wgsl.js';
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
  it('declares all four @compute entry points exactly once each', () => {
    expect((composed.match(/@compute @workgroup_size\(8, 8, 1\)\s*\nfn restirPtProduce\(/g) ?? []).length).toBe(1);
    expect((composed.match(/@compute @workgroup_size\(8, 8, 1\)\s*\nfn restirPtTemporal\(/g) ?? []).length).toBe(1);
    expect((composed.match(/@compute @workgroup_size\(8, 8, 1\)\s*\nfn restirPtSpatial\(/g) ?? []).length).toBe(1);
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
      'fn evaluateBrdfFull(',
      'fn brdfDirectionalPdf(',
      'fn brdfDirectionalPdfFull(',
      'fn brdfDirectionalPdfFullSampled(',
      'fn cosineHemisphereSample(',
      'fn glossyReflectionSample(',
      'fn glossyReflectionSampleAnisotropic(',
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
      'fn materialAnisotropy(',
      'fn materialAnisotropyRotation(',
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
    ).split('\n').slice(0, 42).join('\n');
    expect(targetBody).toContain('evaluateBrdfFull(');
    expect(targetBody).toContain('clearcoatV, clearcoatRoughnessV, sheenV, sheenRoughnessV, sheenColorV,');
    expect(targetBody).toContain('specularColorV, specularIntensityV,');
    expect(targetBody).toContain('anisotropyV, anisotropyRotationV,');
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
      'let xiSource = rand_f32(rng) * lobeWeightSum;',
      'if (xiSource < 1.0 + max(clearcoat, 0.0)) {',
      'let bs = glossyReflectionSample(rng, wo, normal, tanT, tanB, clearcoatRoughness);',
      'return brdfDirectionalPdfFullSampled(',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('evaluates suffix Lo direct lighting and onward throughput with extension-aware BRDF helpers', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptDirectAtVertex(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('clearcoatRoughness: f32,');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let brdf = evaluateBrdfFull(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let brdfPdf = brdfDirectionalPdfFullSampled(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let fOnward = evaluateBrdfFull(');
  });

  it('covers spot and mesh-area emitters in the suffix direct-lighting producer', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('for (var si = 0u; si < params.spotLightCount; si = si + 1u) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let a = meshAreaLights[mb].xyz;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let mr = meshAreaLights[mb + 3u].rgb;');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('meshAreaLights[mb + 3u].w > 0.5 || !traceAny');
  });

  it('the candidate weight is p̂ / p_src (RIS), and finalises with the GRIS W', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let wCandidate = select(0.0, pHat / pdfSrc, pdfSrc > 1e-8);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('finaliseReservoirPTWGris(&r, rptParams.wCap, params.cameraPos.xyz);');
  });

  it('stores the visible-vertex extension payload and uses anisotropic producer sampling', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let anisotropyV = materialAnisotropy(vMatId, vHit.triIndex, vHit.baryVW);');
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
      'baseColorV = vMat.baseColor * sampleBaseColorTexture(vMatId, vHit.triIndex, vHit.baryVW).rgb;',
      'baseColorV = baseColorV * sampleAoFactor(vMatId, vHit.triIndex, vHit.baryVW);',
      'let ormSampleV = sampleOrmTexture(vMatId, vHit.triIndex, vHit.baryVW);',
      'roughnessV = clamp(vMat.roughness * ormSampleV.g, 0.02, 1.0);',
      'metallicV = clamp(vMat.metallic * ormSampleV.b, 0.0, 1.0);',
      'transmissionV = clamp(vMat.transmission * sampleTransmissionTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);',
      'nv = applyNormalMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);',
      'nv = applyBumpMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);',
      'clearcoatV = clamp(vMat.clearcoat * sampleClearcoatTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);',
      'clearcoatRoughnessV = clamp(vMat.clearcoatRoughness * sampleClearcoatRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);',
      'sheenColorV = clamp(vMat.sheenColor * sampleSheenColorTexture(vMatId, vHit.triIndex, vHit.baryVW), vec3f(0.0), vec3f(1.0));',
      'sheenRoughnessV = clamp(vMat.sheenRoughness * sampleSheenRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);',
      'iridescenceV = clamp(vMat.iridescence * sampleIridescenceTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);',
      'let iridescenceThicknessSampleV = sampleIridescenceThicknessTexture(vMatId, vHit.triIndex, vHit.baryVW);',
      'specularColorV = clamp(vMat.specularColor * sampleSpecularColorTexture(vMatId, vHit.triIndex, vHit.baryVW), vec3f(0.0), vec3f(1.0));',
      'specularIntensityV = clamp(vMat.specularIntensity * sampleSpecularIntensityTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('applies alpha pass-through and mapped transmission before the reusable-visible gate', () => {
    const alphaIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('alphaTestPassThrough(hitMaterialId(vHit), vHit.triIndex, vHit.baryVW, &rng)');
    const transmissionIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('var transmissionV = clamp(vMat.transmission * sampleTransmissionTexture');
    const gateIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('!rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV)');
    expect(alphaIdx).toBeGreaterThanOrEqual(0);
    expect(transmissionIdx).toBeGreaterThan(alphaIdx);
    expect(gateIdx).toBeGreaterThan(transmissionIdx);
  });

  it('mirrors layer, thin-film, and spectral visible-vertex prologue effects before sampling', () => {
    for (const line of [
      'iorV = cauchyIorAtLambda(heroLambda, vMat.ior, vMat.dispersionAbbe);',
      'let layerTxV = clamp(select(vMat.backLayerTx, vMat.frontLayerTx, vIsFront), vec3f(0.0), vec3f(1.0));',
      'roughnessV = clamp(layerRoughnessV, 0.02, 1.0);',
      'activeLayerWeightRgb(layerTxV, heroLambda, true)',
      'let rt = thinFilmTmmRt(',
      'baseColorV = mix(baseColorV, baseColorV * thinFilmReflectTintV, filmStrengthV);',
      'evalJakobHanikaSpectrum(vMat.spectralReflCoeffs, heroLambda)',
      'baseColorV = vec3f(reflScalarV);',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
    const prologueIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('baseColorV = vec3f(reflScalarV);');
    const f0Idx = RESTIR_PT_PRODUCER_WGSL.indexOf('let f0V = materialSpecularF0(baseColorV, metallicV, specularColorV, specularIntensityV);');
    expect(f0Idx).toBeGreaterThan(prologueIdx);
  });

  it('mirrors the material-map prologue for suffix/reconnection vertices too', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('struct RptSuffixMaterial {');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptSuffixMaterialAtHit(hit: SceneHit, incomingDir: vec3f, wo: vec3f, heroLambda: f32) -> RptSuffixMaterial');
    for (const line of [
      'out.baseColor = mat.baseColor * sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW).rgb;',
      'out.baseColor = out.baseColor * sampleAoFactor(matId, hit.triIndex, hit.baryVW);',
      'let ormSample = sampleOrmTexture(matId, hit.triIndex, hit.baryVW);',
      'out.emissive = mat.emissive * sampleEmissiveTexture(matId, hit.triIndex, hit.baryVW).rgb;',
      'out.transmission = clamp(mat.transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);',
      'out.normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);',
      'out.normal = applyBumpMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);',
      'out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);',
      'out.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, hit.triIndex, hit.baryVW), vec3f(0.0), vec3f(1.0));',
      'out.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);',
      'out.specularColor = clamp(mat.specularColor * sampleSpecularColorTexture(matId, hit.triIndex, hit.baryVW), vec3f(0.0), vec3f(1.0));',
      'out.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);',
      'out.anisotropy = materialAnisotropy(matId, hit.triIndex, hit.baryVW);',
      'out.anisotropyRotation = materialAnisotropyRotation(matId, hit.triIndex, hit.baryVW);',
    ]) {
      expect(RESTIR_PT_PRODUCER_WGSL).toContain(line);
    }
  });

  it('applies layer, thin-film, spectral albedo, and spectral emission in suffix Lo', () => {
    for (const line of [
      'out.ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);',
      'let layerTx = clamp(select(mat.backLayerTx, mat.frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));',
      'activeLayerWeightRgb(layerTx, heroLambda, true)',
      'let rt = thinFilmTmmRt(',
      'out.baseColor = mix(out.baseColor, out.baseColor * thinFilmReflectTint, filmStrength);',
      'evalJakobHanikaSpectrum(mat.spectralReflCoeffs, heroLambda)',
      'out.baseColor = vec3f(reflScalar);',
      'let emissive = select(sm.emissive, spectralEmissionAtHero(sm.emissive, heroLambda), params.spectralEnabled != 0u);',
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
    const onwardIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let fOnward = evaluateBrdfFull(');
    const onwardAnisoIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('sm.anisotropy, sm.anisotropyRotation,', onwardIdx);
    expect(directCallIdx).toBeGreaterThanOrEqual(0);
    expect(directAnisoIdx).toBeGreaterThan(directCallIdx);
    expect(onwardIdx).toBeGreaterThan(directAnisoIdx);
    expect(onwardAnisoIdx).toBeGreaterThan(onwardIdx);
  });

  it('alpha-skips reconnection and onward suffix hits before decoding their material', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptTraceClosestAfterAlpha(rayIn: Ray, rng: ptr<function, u32>) -> RptAlphaTraceHit');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('alphaTestPassThrough(hitMaterialId(hit), hit.triIndex, hit.baryVW, rng)');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let sTrace = rptTraceClosestAfterAlpha(reconRay, &rng);');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('let nextTrace = rptTraceClosestAfterAlpha(Ray(pos + normal * 1e-3, nextDir), rng);');

    const reconTraceIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let sTrace = rptTraceClosestAfterAlpha(reconRay, &rng);');
    const suffixLoIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('Lo = rptComputeLoAtReconnection(&rng, xs, sHit, reconRay.direction, reconDirToXv, heroLambda, suffixBounces);');
    expect(suffixLoIdx).toBeGreaterThan(reconTraceIdx);
  });

  it('uses the mapped suffix normal as the reservoir reconnection normal', () => {
    const matIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('let sReservoirMat = rptSuffixMaterialAtHit(sHit, reconRay.direction, reconDirToXv, heroLambda);');
    const normalIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('ns = sReservoirMat.normal;');
    const updateIdx = RESTIR_PT_PRODUCER_WGSL.indexOf('updateReservoirPT(&r, xs, ns, Lo, pdfSrc, wCandidate, &rng);');
    expect(matIdx).toBeGreaterThanOrEqual(0);
    expect(normalIdx).toBeGreaterThan(matIdx);
    expect(updateIdx).toBeGreaterThan(normalIdx);
  });

  it('gates specular / transmissive visible vertices to an EMPTY reservoir (no reuse)', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('fn rptIsReusableVisibleVertex(');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain('if (transmission > 0.01) { return false; }');
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'if (vMat.isUnlit || !rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV)) {',
    );
  });
});

describe('ReSTIR-PT resolve — reconstructs with the FULL BRDF (not the proxy target)', () => {
  it('evaluates the full visible-vertex BRDF and forms f·cos·Lo·W', () => {
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('let fBsdf = evaluateBrdfFull(');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('r.clearcoatV, r.clearcoatRoughnessV, r.sheenV, r.sheenRoughnessV, r.sheenColorV,');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('r.specularColorV, r.specularIntensityV,');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('r.anisotropyV, r.anisotropyRotationV,');
    expect(RESTIR_PT_RESOLVE_WGSL).toContain('let indirect = fBsdf * cosTheta * r.Lo * r.W;');
  });

  it('does NOT use the diffuse-cosine proxy (restirPtTargetAt) in reconstruction', () => {
    expect(RESTIR_PT_RESOLVE_WGSL).not.toContain('restirPtTargetAt');
  });
});

describe('ReSTIR-PT reservoir — visible-material payload is serialized with the reservoir', () => {
  it('bumps the reservoir stride and stores scalar extension-lobe fields', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain('ReservoirPTHero, 208 bytes = 52 × u32');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('const RESERVOIR_PT_HERO_STRIDE: u32 = 52u;');
    for (const line of [
      'buf[b + 31u] = bitcast<u32>(r.clearcoatV);',
      'buf[b + 32u] = bitcast<u32>(r.clearcoatRoughnessV);',
      'buf[b + 33u] = bitcast<u32>(r.sheenV);',
      'buf[b + 34u] = bitcast<u32>(r.sheenRoughnessV);',
      'buf[b + 35u] = bitcast<u32>(r.sheenColorV.x);',
      'buf[b + 38u] = bitcast<u32>(r.iridescenceV);',
      'buf[b + 42u] = bitcast<u32>(r.anisotropyV);',
      'buf[b + 43u] = bitcast<u32>(r.anisotropyRotationV);',
      'buf[b + 44u] = bitcast<u32>(r.specularColorV.x);',
      'buf[b + 47u] = bitcast<u32>(r.specularIntensityV);',
      'buf[b + 51u] = r._padHybrid;',
    ]) {
      expect(RESERVOIR_PT_HERO_WGSL).toContain(line);
    }
  });

  it('copies the full visible-material domain into temporal/spatial output reservoirs', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain('fn copyReservoirPTVisibleDomain(');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).clearcoatV = src.clearcoatV;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).specularColorV = src.specularColorV;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).specularIntensityV = src.specularIntensityV;');
    expect(RESERVOIR_PT_HERO_WGSL).toContain('(*dst).anisotropyRotationV = src.anisotropyRotationV;');
    expect(RESTIR_PT_TEMPORAL_WGSL).toContain('copyReservoirPTVisibleDomain(&rGris, rCur);');
    expect(RESTIR_PT_SPATIAL_WGSL).toContain('copyReservoirPTVisibleDomain(&rOut, rCenter);');
  });
});
