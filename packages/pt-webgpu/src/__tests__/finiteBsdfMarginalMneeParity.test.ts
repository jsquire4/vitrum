import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';

function exactDielectricFresnel(cosTheta: number, etaTOverI: number): number {
  const cosI = Math.min(Math.max(Math.abs(cosTheta), 0), 1);
  const eta = Math.max(etaTOverI, 1e-4);
  const sinT2 = (1 - cosI * cosI) / (eta * eta);
  if (sinT2 >= 1) return 1;
  const cosT = Math.sqrt(Math.max(0, 1 - sinT2));
  const rParallel = (eta * cosI - cosT) / (eta * cosI + cosT);
  const rPerpendicular = (cosI - eta * cosT) / (cosI + eta * cosT);
  return 0.5 * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

function schlick(cosTheta: number, f0: number): number {
  return f0 + (1 - f0) * (1 - Math.abs(cosTheta)) ** 5;
}

function materialFresnelWithoutIridescence(
  cosTheta: number,
  etaTOverI: number,
  specularColor: readonly number[],
  specularIntensity: number,
): number[] {
  const exact = exactDielectricFresnel(cosTheta, etaTOverI);
  if (exact >= 1) return [1, 1, 1];
  return specularColor.map((channel) => {
    const authoredF0 =
      0.04 * Math.min(Math.max(channel, 0), 1) * Math.min(Math.max(specularIntensity, 0), 1);
    return Math.min(
      Math.max((exact * schlick(cosTheta, authoredF0)) / schlick(cosTheta, 0.04), 0),
      1,
    );
  });
}

describe('finite BSDF normalized-mixture closure', () => {
  it('uses the marginal proposal density for every overlapping source lobe', () => {
    const selectPmf = [0.35, 0.65];
    const proposalPdf = [
      [0.8, 0.2],
      [0.25, 0.75],
    ];
    const fullFiniteBsdf = [1.75, 0.6];
    const marginal = [0, 1].map((direction) =>
      selectPmf.reduce((sum, pmf, lobe) => sum + pmf * proposalPdf[lobe]![direction]!, 0),
    );

    let expectedIntegral = 0;
    for (let lobe = 0; lobe < selectPmf.length; lobe += 1) {
      for (let direction = 0; direction < 2; direction += 1) {
        const pathProbability = selectPmf[lobe]! * proposalPdf[lobe]![direction]!;
        expectedIntegral += (pathProbability * fullFiniteBsdf[direction]!) / marginal[direction]!;
      }
    }
    expect(expectedIntegral).toBeCloseTo(
      fullFiniteBsdf.reduce((sum, value) => sum + value, 0),
      14,
    );
    expect(marginal[0]).not.toBe(selectPmf[0]! * proposalPdf[0]![0]!);
    expect(marginal[0]).not.toBe(selectPmf[1]! * proposalPdf[1]![0]!);
  });

  it('finalizes both finite branch families with the full BSDF and marginal pdf', () => {
    const helper = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf('fn finalizeFiniteBounceSampleWithClearcoatNormal('),
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf('// D9.1 — shared anisotropy axis helper'),
    );
    expect(helper).toContain('brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(helper).toContain('evaluateFiniteBsdfFullWithClearcoatNormal(');
    expect(helper).toContain('finiteBsdf * cosine / marginalPdf');
    expect(helper).toContain('(*result).sampledEventPdf = marginalPdf;');
    expect(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.match(/finalizeFiniteBounceSampleWithClearcoatNormal\(/g),
    ).toHaveLength(3);
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).not.toContain(
      'fn sampledLobeDirectionalPdfWithClearcoatNormal(',
    );
  });
});

describe('MNEE authored-interface parity', () => {
  it('keeps coloured Fresnel R/T complementary after expected alpha coverage', () => {
    const coverage = 0.37;
    const baseColor = [0.8, 0.45, 0.2];
    const transmission = 0.73;
    const fresnel = materialFresnelWithoutIridescence(0.42, 1.52, [0.25, 0.8, 1], 0.65);
    for (let channel = 0; channel < 3; channel += 1) {
      const reflection = coverage * fresnel[channel]!;
      const transmissionFactor =
        coverage * baseColor[channel]! * transmission * (1 - fresnel[channel]!);
      const normalized =
        reflection / coverage +
        transmissionFactor / (coverage * baseColor[channel]! * transmission);
      expect(normalized).toBeCloseTo(1, 14);
    }
    expect(new Set(fresnel.map((value) => value.toFixed(12))).size).toBe(3);
  });

  it('rehydrates mapped optics and rejects absent transport surfaces', () => {
    for (const token of [
      'sampleIridescenceTexture(',
      'sampleIridescenceThicknessTexture(',
      'sampleSpecularColorTexture(',
      'sampleSpecularIntensityTexture(',
      'spectralRgbFactorAtHero(specularColor, heroLambda)',
      'out.isUnlit = mat.isUnlit;',
      'optics.isUnlit || optics.coverage <= 0.0',
      'fn mneeFacetCoverage(',
      'sampleBaseColorTexture(matId, triIndex, baryVW, instanceIndex).a',
      'sampleVertexColor(triIndex, baryVW).a',
      'sampleAlphaTexture(matId, triIndex, baryVW, instanceIndex)',
    ]) {
      expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(token);
    }
  });

  it('uses the shared vector dielectric Fresnel and coverage for film and non-film events', () => {
    const reflection = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('fn mneeFacetReflectionFactorWithEta('),
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('fn mneeFacetTransmissionFactorWithEta('),
    );
    const transmission = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('fn mneeFacetTransmissionFactorWithEta('),
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('fn mneeEmitterRadiance('),
    );
    expect(reflection).toContain('optics.coverage *');
    expect(reflection).toContain('materialDielectricLayeredInterface(');
    expect(reflection).toContain(').reflectance');
    expect(transmission).toContain('optics.coverage * optics.baseColor');
    expect(transmission).toContain('materialDielectricLayeredInterface(');
    expect(transmission).toContain('interfaceResponse.baseTransmittance');
    expect(reflection).not.toContain('vec3f(frDielectric(');
    expect(transmission).not.toContain('1.0 - frDielectric(');
  });
});
