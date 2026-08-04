import { describe, expect, it } from 'vitest';
import { RIS_WGSL } from '../ris.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from '../risGiGlassWalk.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../nrcIndependentSuffix.wgsl.js';
import { RESERVOIR_DI_WGSL } from '../reservoirDi.wgsl.js';
import { RESERVOIR_GI_WGSL } from '../reservoirGi.wgsl.js';
import { SAMPLE_CASCADE_C0_WGSL } from '../sampleCascadeC0.wgsl.js';
import { TEMPORAL_GI_WGSL } from '../temporalGi.wgsl.js';
import { SPATIAL_GI_WGSL } from '../spatialGi.wgsl.js';

describe('exact represented radiometric support', () => {
  it('retains tiny positive DI source PDFs in log-domain represented WRS', () => {
    expect(RIS_WGSL).toContain('reservoirDiInitialCandidateLogWeight(');
    expect(RIS_WGSL).toContain('emitterLogSelectionPmf,');
    expect(RIS_WGSL).toContain('ls.pdfArea,');
    expect(RIS_WGSL).not.toContain('max(1e-15, emitterSelPmf * ls.pdfArea)');
    expect(RESERVOIR_DI_WGSL).toContain('log2(targetDensity) -');
    expect(RESERVOIR_DI_WGSL).toContain('log2(withinEmitterPdf) +');
    expect(RESERVOIR_DI_WGSL).toContain(
      'if (representedWrsUpdate(wrs, logWeight, rng))',
    );

    const tinyPdf = 1e-20;
    const tinyTarget = 3e-20;
    const representedWeight = 2 ** (
      Math.log2(tinyTarget) - Math.log2(tinyPdf)
    );
    expect(Number.isFinite(representedWeight)).toBe(true);
    expect(representedWeight).toBeCloseTo(3, 12);
  });

  it('does not erase tiny positive GI targets, visibility, or mixture PDFs', () => {
    const glass = NATIVE_GLASS_GI_WGSL;
    expect(glass).toContain(
      'let logPHat = reservoirGiLogPositive(luminance(receiverContribution));',
    );
    expect(glass).toContain(
      'let logPSrc = reservoirGiLogPositive(cosTheta * INV_PI);',
    );
    expect(glass).toContain('let logWeight = logPHat - logPSrc;');
    expect(glass).toContain('var wrs = representedWrsInit();');
    expect(glass).toContain('&reservoir,');
    expect(glass).toContain('&wrs,');
    expect(glass).toContain('finaliseGIReservoirFromNativeWrs(');
    expect(glass).not.toContain('w_sum');
    expect(glass).not.toMatch(/pHat\s*\/\s*pSrc/);
    expect(glass).not.toMatch(/logPHat\s*[<=>]+\s*1e-/);
    expect(glass).not.toMatch(/logPSrc\s*[<=>]+\s*1e-/);

    const tinyLogWeight = Math.log2(3e-38) - Math.log2(1e-38);
    expect(Number.isFinite(tinyLogWeight)).toBe(true);
    expect(2 ** tinyLogWeight).toBeCloseTo(3, 12);

    for (const src of [TEMPORAL_GI_WGSL, SPATIAL_GI_WGSL]) {
      expect(src).not.toMatch(/pHat[^\n]*1e-9/);
      expect(src).not.toMatch(/denominator[^\n]*1e-12/);
      expect(src).toContain('candidate.H,');
      expect(src).not.toMatch(/candidate\.W\b/);
    }
  });

  it('preserves tiny positive cosine, shadow, RC, and NRC support', () => {
    expect(SHADING_TERMS_WGSL).toContain('let rcHasEnergy = max(Lo_rc.r, max(Lo_rc.g, Lo_rc.b)) > 0.0;');
    expect(SHADING_TERMS_WGSL).not.toMatch(/(?:nDotL|nDotSun|sunShadowT|rcHasEnergy)[^\n]*(?:1e-|0\.001)/);
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('if (cosTheta <= 0.0) { continue; }');
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('if (Wsum > 0.0)');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain('if (!(proposalPdf > 0.0)) { break; }');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).not.toMatch(/(?:selectionPmf|pdfArea|proposalPdf|cosLight)[^\n]*1e-/);
  });

  it('keeps reservoir normalization logarithmic through the final radiometric product', () => {
    expect(SHADING_TERMS_WGSL).toContain(
      'fn restirShadeLogProductChannel(',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'fn restirShadeNormaliseLogSum(logSum: vec3f, denominator: f32)',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'let responseLog = restirShadeDirectionalVolumeLog(',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'return restirShadeExp2Clamped3(responseLog);',
    );
    expect(SHADING_TERMS_WGSL).not.toMatch(/\b(?:r|g)\.W\b/);
    expect(SHADING_TERMS_WGSL).not.toMatch(/exp2\((?:r|g)\.logW\)/);
    expect(SHADING_TERMS_WGSL).toContain(
      'logDomainJacobian = grisLogDomainToCanonicalJacobian(',
    );
    expect(SHADING_TERMS_WGSL).not.toContain(
      'domainJacobian = grisDomainToCanonicalJacobian(',
    );

    const logProduct = (logW: number, ...factors: number[]): number =>
      logW + factors.reduce((sum, factor) => sum + Math.log2(factor), 0);

    // Both products are representable even though a premature exp2(logW)
    // would respectively underflow or overflow a binary32 endpoint.
    expect(2 ** logProduct(-200, 2 ** 100)).toBe(2 ** -100);
    expect(2 ** logProduct(200, 2 ** -100)).toBe(2 ** 100);

    // A single bilinear contributor's tiny blend weight cancels only after
    // normalization. Exponentiating the -200 intermediate would lose it.
    const blendWeight = 2 ** -100;
    const weightedLog = logProduct(-200, 2 ** 100, blendWeight);
    const normalizedLog = weightedLog - Math.log2(blendWeight);
    expect(normalizedLog).toBe(-100);
    expect(2 ** normalizedLog).toBe(2 ** -100);
  });

  it('converts log radiance to f32 only at the true rounding and overflow boundaries', () => {
    for (const source of [SHADING_TERMS_WGSL, RESERVOIR_GI_WGSL]) {
      expect(source).toContain('ROUND_TO_ZERO');
      expect(source).toContain('-150.0');
      expect(source).toContain('OVERFLOW');
      expect(source).toContain('128.0');
      expect(source).not.toContain('127.999');
      expect(source).not.toContain('LOG_MIN_NORMAL');
    }

    const maxF32 = Math.fround(3.402823466e38);
    const endpoint = (logValue: number): number => {
      if (!Number.isFinite(logValue) || logValue <= -150) return 0;
      if (logValue >= 128) return maxF32;
      return Math.min(Math.fround(2 ** logValue), maxF32);
    };
    expect(endpoint(-150)).toBe(0);
    expect(endpoint(-149.5)).toBe(2 ** -149);
    expect(endpoint(127.5)).toBe(Math.fround(2 ** 127.5));
    expect(endpoint(127.5)).toBeLessThan(maxF32);
    expect(endpoint(128)).toBe(maxF32);
  });
});
