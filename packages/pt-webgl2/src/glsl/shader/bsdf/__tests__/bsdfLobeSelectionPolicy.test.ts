import { describe, expect, it } from 'vitest';
import * as BsdfFns from '../bsdf_functions.glsl.js';

const bsdf_functions = (BsdfFns as unknown as Record<string, string>)['bsdf_functions']!;

function sourceBetween(startNeedle: string, endNeedle: string): string {
  const start = bsdf_functions.indexOf(startNeedle);
  expect(start, `${startNeedle} must exist`).toBeGreaterThanOrEqual(0);
  const end = bsdf_functions.indexOf(endNeedle, start + startNeedle.length);
  expect(end, `${endNeedle} must follow ${startNeedle}`).toBeGreaterThan(start);
  return bsdf_functions.slice(start, end);
}

function sourceFrom(startNeedle: string): string {
  const start = bsdf_functions.indexOf(startNeedle);
  expect(start, `${startNeedle} must exist`).toBeGreaterThanOrEqual(0);
  return bsdf_functions.slice(start);
}

describe('pt-webgl2 BSDF lobe-selection PDF policy', () => {
  it('uses one sampling-lobe policy helper for sampled and evaluated PDFs', () => {
    const helper = sourceBetween('void getSamplingLobeWeights', 'float bsdfEval');
    expect(helper).toContain(
      'getLobeWeights( wo, wo, vec3( 0, 0, 1 ), clearcoatWo, surf, heroWavelength',
    );

    const result = sourceBetween('float bsdfResult', '// Sprint 7: SSS');
    const sample = sourceFrom('ScatterRecord bsdfSample');
    expect(result).toContain('getSamplingLobeWeights( wo, clearcoatWo, surf, heroWavelength');
    expect(sample).toContain('getSamplingLobeWeights( wo, clearcoatWo, surf, heroWavelength');
  });

  it('does not recompute mixture probabilities from arbitrary light directions in bsdfResult', () => {
    const result = sourceBetween('float bsdfResult', '// Sprint 7: SSS');
    expect(result).not.toContain('getLobeWeights( wo, wi');
    expect(result).not.toContain('getHalfVector( wo, wi');
  });

  it('keeps spectral dispersion transmission on the same anisotropic GGX sampler as ordinary transmission', () => {
    const ordinaryTransmission = sourceBetween(
      'vec3 transmissionDirection',
      '// ── Sprint 8: Chromatic dispersion',
    );
    const dispersionTransmission = sourceBetween(
      'vec3 dispersionTransmissionDirection',
      '// clearcoat',
    );

    expect(ordinaryTransmission).toContain('ggxDirectionForSurface( wo, surf, rand2( 13 ) )');
    expect(dispersionTransmission).toContain('ggxDirectionForSurface( wo, surf, rand2( 13 ) )');
    expect(dispersionTransmission).not.toMatch(/vec3\s+halfVector\s*=\s*ggxDirection\s*\(/);
  });

  it('does not apply an unmatched second GGX perturbation after rough refraction', () => {
    const ordinaryTransmission = sourceBetween(
      'vec3 transmissionDirection',
      '// ── Sprint 8: Chromatic dispersion',
    );
    const dispersionTransmission = sourceBetween(
      'vec3 dispersionTransmissionDirection',
      '// clearcoat',
    );

    expect(bsdf_functions).not.toContain('perturbDirectionByGGX');
    expect(ordinaryTransmission).not.toContain('rand2( 47 )');
    expect(dispersionTransmission).not.toContain('rand2( 47 )');
  });

  it('uses hero-wavelength eta for dispersion sampling and transmission PDF evaluation', () => {
    const transmissionEval = sourceBetween('float transmissionEval', 'vec3 transmissionDirection');
    const bsdfEval = sourceBetween('float bsdfEval', 'float bsdfResult');
    const dispersionTransmission = sourceBetween(
      'vec3 dispersionTransmissionDirection',
      '// clearcoat',
    );
    const sample = sourceFrom('ScatterRecord bsdfSample');

    expect(bsdf_functions).toContain(
      'float transmissionEtaAtHero( const in SurfaceRecord surf, float heroWavelength )',
    );
    expect(transmissionEval).toContain(
      'float eta = transmissionEtaAtHero( surf, heroWavelength );',
    );
    expect(bsdfEval.replace(/\s+/g, ' ')).toContain(
      'getHalfVector( wi, wo, transmissionEtaAtHero( surf, heroWavelength ) )',
    );
    expect(dispersionTransmission).toContain(
      'float eta = transmissionEtaAtHero( surf, heroWavelength );',
    );
    expect(sample.replace(/\s+/g, ' ')).toContain(
      'wi = cauchyDispersionEnabled( surf ) ? dispersionTransmissionDirection( wo, surf, heroWavelength ) : transmissionDirection( wo, surf );',
    );
    expect(dispersionTransmission).not.toContain('iorDelta');
  });

  it('preserves every positive finite dispersion and medium-scattering coefficient', () => {
    const dispersion = sourceBetween(
      'float cauchyIORatLambda',
      'float transmissionEtaAtHero',
    ).replace(/\s+/g, ' ');
    const sss = sourceBetween('ScatterRecord sssSample', 'ScatterRecord bsdfSample');

    expect(dispersion).toContain('if ( C == 0.0 )');
    expect(dispersion).toContain('surf.dispersionStrength > 0.0');
    expect(dispersion).toContain('abs( iorCauchyB ) > 0.0');
    expect(dispersion).toContain('abs( iorCauchyC ) > 0.0');
    expect(dispersion).toContain('surf.dispersionStrength / dispersionBasis');
    expect(dispersion).not.toContain('dispersionStrength > 1e-5');
    expect(sss).toContain('sigmaT.x > 0.0 ? sigmaS.x / sigmaT.x : 0.0');
    expect(sss).not.toContain('sigmaT.x > 1e-6');

    const sigmaT = 1e-12;
    const sigmaS = 0.5e-12;
    expect(sigmaS / sigmaT).toBe(0.5);
  });

  it('represents exact-zero roughness as a discrete event without a PDF threshold', () => {
    const deltaPdf = sourceBetween('float bsdfDeltaPdfLocal', 'float bsdfPdfLocal');
    const continuousEval = sourceBetween('float bsdfEval', 'float bsdfResult');
    const result = sourceBetween('float bsdfResult', '// Sprint 7: SSS');
    const sample = sourceFrom('ScatterRecord bsdfSample');

    expect(bsdf_functions).toContain(
      'return surf.filteredRoughness <= 0.0;',
    );
    expect(bsdf_functions).not.toContain(
      'surf.filteredRoughness <= 0.0 && surf.anisotropy <= 0.0',
    );
    expect(deltaPdf).toContain('bsdfDeltaDirectionMatches( reflected, wi )');
    expect(deltaPdf).toContain('bsdfDeltaTransmissionDirection');
    expect(continuousEval).toContain('! bsdfBaseLobesAreDelta( surf )');
    expect(result).toContain('float deltaPdf = bsdfDeltaEvalLocal(');
    expect(sample).toContain('result.sampledDelta = sampledDelta;');
    expect(sample).not.toContain('result.sampledDelta = result.pdf >');
    expect(sample).not.toContain('sampledDelta = result.pdf >');
  });

  it('preserves sub-milliscale authored anisotropic roughness and tiny HG asymmetry', () => {
    const anisotropic = sourceBetween(
      'vec2 anisotropicRoughnessAxes',
      'float ggxDistributionAnisotropic',
    );
    const hg = sourceBetween('vec3 sampleHG_glsl', '// diffuse');
    const volumeHg = sourceBetween('vec3 sampleMediumPhase', '// Sprint 12');
    const volumeSample = sourceFrom('ScatterRecord bsdfSample');

    expect(anisotropic).toContain('clamp( surf.filteredRoughness, 0.0, 1.0 )');
    expect(anisotropic).toContain('if ( roughness == 0.0 ) roughness = 0.001;');
    expect(anisotropic).not.toContain('clamp( surf.filteredRoughness, 0.001');
    expect(anisotropic).not.toContain('vec2( 0.001 ), vec2( 1.0 )');
    expect(hg).toContain('if ( gg == 0.0 )');
    expect(hg).toContain('1.5 * gg * ( 1.0 - a2 )');
    expect(hg).not.toContain('if ( abs( gg ) < 1e-4 )');
    expect(volumeHg).toContain('else if ( abs( g ) < 1e-3 )');
    expect(volumeHg).toContain('1.5 * g * ( 1.0 - a2 )');
    expect(volumeHg).toContain('float xi = 1.0 - uv.x;');
    expect(volumeSample).toMatch(
      /surf\.volumeParticle[\s\S]*?sampleMediumPhase\(\s*worldWo, surf\.sssAnisotropyG/,
    );

    const authoredRoughness = 1e-7;
    expect(Math.max(0, Math.min(1, authoredRoughness))).toBe(authoredRoughness);
    const a = 0.25;
    const tinyG = 1e-8;
    const sampledCos = a + 1.5 * tinyG * (1 - a * a)
      + 2 * tinyG * tinyG * (a * a * a - a);
    expect(sampledCos).not.toBe(a);
  });

  it('rejects entry and thin-exit TIR with finite zero-weight scatter records', () => {
    const ordinaryTransmission = sourceBetween(
      'vec3 transmissionDirection',
      '// ── Sprint 8: Chromatic dispersion',
    );
    const dispersionTransmission = sourceBetween(
      'vec3 dispersionTransmissionDirection',
      '// clearcoat',
    );
    const deltaTransmission = sourceBetween(
      'vec3 bsdfDeltaTransmissionDirection',
      'float bsdfDeltaPdfLocal',
    );
    const sample = sourceFrom('ScatterRecord bsdfSample');

    for (const transmission of [ordinaryTransmission, dispersionTransmission]) {
      expect(transmission).toContain('dot( lightDirection, lightDirection ) > 1e-16');
      expect(transmission).toContain('dot( exitDirection, exitDirection ) > 1e-16');
      expect(transmission).toContain('return vec3( 0.0 );');
    }
    expect(deltaTransmission).toContain('dot( direction, direction ) > 1e-16');
    expect(deltaTransmission).toContain('dot( exitDirection, exitDirection ) > 1e-16');
    expect(sample).toContain('bool rejectedTransmission = false;');
    expect(sample).toContain('result.pdf = 0.0;');
    expect(sample).toContain('result.specularPdf = 0.0;');
    expect(sample).toContain('result.throughput = vec3( 0.0 );');
    expect(sample).toContain('result.sampledDelta = sampledDelta;');
  });

  it('reports the full mixed-lobe marginal PDF for sampled and reversed paths', () => {
    const continuousEval = sourceBetween('float bsdfEval', 'float bsdfResult');
    const sample = sourceFrom('ScatterRecord bsdfSample');
    const weights = [0.2, 0.5, 0.3] as const;
    const lobePdfs = [0.3, 0.4, 0.1] as const;
    const marginal = weights.reduce(
      (sum, weight, index) => sum + weight * lobePdfs[index]!,
      0,
    );

    expect(marginal).toBeCloseTo(0.29, 14);
    expect(marginal).not.toBe(lobePdfs[1]);
    expect(continuousEval.replace(/\s+/g, ' ')).toContain(
      'dpdf * diffuseWeight + spdf * specularWeight + tpdf * transmissionWeight + cpdf * clearcoatWeight',
    );
    expect(sample).toContain('result.pdf = bsdfEval(');
  });
});
