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
    expect(helper).toContain('getLobeWeights( wo, wo, vec3( 0, 0, 1 ), clearcoatWo, surf, heroWavelength');

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
    const dispersionTransmission = sourceBetween('vec3 dispersionTransmissionDirection', '// clearcoat');

    expect(ordinaryTransmission).toContain('ggxDirectionForSurface( wo, surf, rand2( 13 ) )');
    expect(dispersionTransmission).toContain('ggxDirectionForSurface( wo, surf, rand2( 13 ) )');
    expect(dispersionTransmission).not.toMatch(/vec3\s+halfVector\s*=\s*ggxDirection\s*\(/);
  });

  it('does not apply an unmatched second GGX perturbation after rough refraction', () => {
    const ordinaryTransmission = sourceBetween(
      'vec3 transmissionDirection',
      '// ── Sprint 8: Chromatic dispersion',
    );
    const dispersionTransmission = sourceBetween('vec3 dispersionTransmissionDirection', '// clearcoat');

    expect(bsdf_functions).not.toContain('perturbDirectionByGGX');
    expect(ordinaryTransmission).not.toContain('rand2( 47 )');
    expect(dispersionTransmission).not.toContain('rand2( 47 )');
  });

  it('uses hero-wavelength eta for dispersion sampling and transmission PDF evaluation', () => {
    const transmissionEval = sourceBetween(
      'float transmissionEval',
      'vec3 transmissionDirection',
    );
    const bsdfEval = sourceBetween('float bsdfEval', 'float bsdfResult');
    const dispersionTransmission = sourceBetween('vec3 dispersionTransmissionDirection', '// clearcoat');
    const sample = sourceFrom('ScatterRecord bsdfSample');

    expect(bsdf_functions).toContain('float transmissionEtaAtHero( SurfaceRecord surf, float heroWavelength )');
    expect(transmissionEval).toContain('float eta = transmissionEtaAtHero( surf, heroWavelength );');
    expect(bsdfEval).toContain('vec3 transmissionHalfVector = getHalfVector( wi, wo, transmissionEtaAtHero( surf, heroWavelength ) );');
    expect(dispersionTransmission).toContain('float eta = transmissionEtaAtHero( surf, heroWavelength );');
    expect(sample).toContain('if ( cauchyDispersionEnabled( surf ) )');
    expect(dispersionTransmission).not.toContain('iorDelta');
  });
});
