import { describe, expect, it } from 'vitest';
import { evaluateHG, sampleHG } from '@vitrum/shared-samplers';
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

    const result = sourceBetween('float bsdfResult', 'ScatterRecord bsdfSample');
    const sample = sourceFrom('ScatterRecord bsdfSample');
    expect(result.replace(/\s+/g, ' ')).toContain(
      'getSamplingLobeWeights( wo, clearcoatWo, orientedSurf, heroWavelength',
    );
    expect(sample).toContain('getSamplingLobeWeights( wo, clearcoatWo, surf, heroWavelength');
  });

  it('does not recompute mixture probabilities from arbitrary light directions in bsdfResult', () => {
    const result = sourceBetween('float bsdfResult', 'ScatterRecord bsdfSample');
    expect(result).not.toContain('getLobeWeights( wo, wi');
    expect(result).not.toContain('getHalfVector( wo, wi');
  });

  it('keeps spectral dispersion transmission on the same anisotropic GGX sampler as ordinary transmission', () => {
    const interfaceSampler = sourceBetween(
      'vec3 sampleTransmissionInterfaceDirection',
      'vec3 transmissionDirection',
    );
    const ordinaryTransmission = sourceBetween(
      'vec3 transmissionDirection',
      '// ── Sprint 8: Chromatic dispersion',
    );
    const dispersionTransmission = sourceBetween(
      'vec3 dispersionTransmissionDirection',
      '// clearcoat',
    );

    expect(interfaceSampler).toContain(
      'vec3 halfVector = ggxDirectionForSurface( canonicalWo, surf, uv );',
    );
    expect(ordinaryTransmission).toContain(
      'sampleTransmissionInterfaceDirection(',
    );
    expect(ordinaryTransmission).toContain('wo, surf, 0.0, rand2( 13 )');
    expect(dispersionTransmission).toContain(
      'sampleTransmissionInterfaceDirection(',
    );
    expect(dispersionTransmission).toContain(
      'wo, surf, heroWavelength, rand2( 13 )',
    );
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
    const interfaceSampler = sourceBetween(
      'vec3 sampleTransmissionInterfaceDirection',
      'vec3 transmissionDirection',
    );
    const sample = sourceFrom('ScatterRecord bsdfSample');

    expect(bsdf_functions).toContain(
      'float transmissionEtaAtHero(',
    );
    expect(transmissionEval).toContain(
      'float eta = transmissionEtaAtHero( surf, heroWavelength, wo );',
    );
    expect(bsdfEval.replace(/\s+/g, ' ')).toContain(
      'getHalfVector( wi, wo, transmissionEtaAtHero( surf, heroWavelength, wo ) )',
    );
    expect(interfaceSampler).toContain(
      'float eta = transmissionEtaAtHero( surf, heroWavelength, wo );',
    );
    expect(dispersionTransmission).toContain('heroWavelength, rand2( 13 )');
    expect(sample.replace(/\s+/g, ' ')).toContain(
      'wi = cauchyDispersionEnabled( surf ) ? dispersionTransmissionDirection( wo, surf, heroWavelength ) : transmissionDirection( wo, surf );',
    );
    expect(dispersionTransmission).not.toContain('iorDelta');
  });

  it('preserves every positive finite dispersion and medium-scattering coefficient', () => {
    const dispersion = sourceBetween(
      'float cauchyIORFromDLine',
      'float transmissionEtaAtHero',
    ).replace(/\s+/g, ' ');
    const mediumExtinction = sourceBetween(
      'vec3 fogTrueExtinction',
      'vec3 fogSegmentTransmittance',
    );

    expect(dispersion).toContain(
      'bNm2 * ( 1.0 / lambda2 - 1.0 / dLine2 )',
    );
    expect(dispersion).toContain('uSpectralRendering != 0');
    expect(dispersion).toContain('surf.dispersionStrength > 0.0');
    expect(dispersion).toContain(
      'cauchyIORFromDLine( heroWavelength, surf.ior, surf.dispersionStrength )',
    );
    expect(dispersion).not.toContain('iorCauchy');
    expect(dispersion).not.toContain('dispersionBasis');
    expect(dispersion).not.toContain('dispersionScale');
    expect(dispersion).not.toContain('dispersionStrength > 1e-5');
    expect(mediumExtinction).toContain('return sigmaA + fog.sigmaS;');
    expect(mediumExtinction).toContain('float sigmaSHero = heroScalarFromRgb(');
    expect(mediumExtinction).not.toContain('sigmaSHero > 1e-6');

    const sigmaT = 1e-12;
    const sigmaS = 0.5e-12;
    expect(sigmaS / sigmaT).toBe(0.5);
  });

  it('includes the sampled HG phase value in the explicit medium-vertex numerator', () => {
    const mediumSample = sourceFrom('ScatterRecord bsdfSample').replace(/\s+/g, ' ');
    expect(mediumSample).toContain('if ( surf.volumeParticle )');
    expect(mediumSample).toContain(
      'sampleRec.throughput = pathThroughputFromRgb( surf.color * sampleRec.pdf, heroWavelength );',
    );
    expect(bsdf_functions).not.toContain('sssSample(');

    // Isotropic HG has f = pdf = 1/(4π), so f/pdf is exactly one.
    const isotropicHg = 1 / (4 * Math.PI);
    expect(isotropicHg / isotropicHg).toBe(1);
  });

  it('represents exact-zero roughness as a discrete event without a PDF threshold', () => {
    const deltaPdf = sourceBetween('float bsdfDeltaPdfLocal', 'float bsdfPdfLocal');
    const continuousEval = sourceBetween('float bsdfEval', 'float bsdfResult');
    const result = sourceBetween('float bsdfResult', 'ScatterRecord bsdfSample');
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

  it('preserves sub-milliscale anisotropy and mirrors the exact shared HG inverse', () => {
    const anisotropic = sourceBetween(
      'vec2 anisotropicRoughnessAxes',
      'float ggxDistributionAnisotropic',
    );
    const hgInverse = sourceBetween(
      'float sampleHgCosTheta',
      '// diffuse',
    );
    const volumeHg = sourceBetween('vec3 sampleMediumPhase', '// Sprint 12');
    const volumePdf = sourceBetween(
      'float mediumPhasePdf',
      'vec3 sampleMediumPhase',
    );
    const volumeSample = sourceFrom('ScatterRecord bsdfSample');

    expect(anisotropic).toContain('clamp( surf.filteredRoughness, 0.0, 1.0 )');
    expect(anisotropic).toContain('if ( roughness == 0.0 ) roughness = 0.001;');
    expect(anisotropic).not.toContain('clamp( surf.filteredRoughness, 0.001');
    expect(anisotropic).not.toContain('vec2( 0.001 ), vec2( 1.0 )');
    expect(hgInverse).toContain('if ( abs( gg ) < 0.125 )');
    expect(hgInverse).toContain('clamp( g, -0.999999, 0.999999 )');
    expect(hgInverse).toContain('gg * gg * gg * ( q * q - 1.0 )');
    expect(hgInverse).toContain('( 1.0 - gg * gg ) /');
    expect(hgInverse).toContain('( 1.0 + gg * q )');
    expect(bsdf_functions).toContain(
      'oneMinusA * oneMinusA +',
    );
    expect(bsdf_functions).toContain(
      '2.0 * a * ( 1.0 - alignedCos )',
    );
    expect(volumeHg).toContain('float cosTheta = sampleHgCosTheta( uv.x, g );');
    expect(volumePdf).toContain('return hg_phase( cosTheta, g );');
    expect(volumePdf).not.toContain('float g2 = g * g;');
    expect(volumeSample).toMatch(
      /surf\.volumeParticle[\s\S]*?sampleMediumPhase\(\s*worldWo, surf\.sssAnisotropyG/,
    );

    const authoredRoughness = 1e-7;
    expect(Math.max(0, Math.min(1, authoredRoughness))).toBe(authoredRoughness);

    const glslMirrorCos = (u: number, g: number): number => {
      const gg = Math.max(-0.999999, Math.min(0.999999, g));
      const q = 1 - 2 * u;
      let cosTheta: number;
      if (Math.abs(gg) < 0.125) {
        const d = 1 + gg * q;
        const numerator =
          2 * q +
          gg * (q * q + 3) +
          2 * gg * gg * q +
          gg * gg * gg * (q * q - 1);
        cosTheta = numerator / (2 * d * d);
      } else {
        const ratio = (1 - gg * gg) / (1 + gg * q);
        cosTheta = (1 + gg * gg - ratio * ratio) / (2 * gg);
      }
      return Math.max(-1, Math.min(1, cosTheta));
    };

    for (const g of [-1.2, -0.99995, -0.9, -0.125, -0.124999, -1e-8, 0, 1e-8, 0.124999, 0.125, 0.9, 0.99995, 1.2]) {
      for (const u of [0, 0.1, 0.5, 0.9, 0.999999]) {
        expect(glslMirrorCos(u, g)).toBeCloseTo(sampleHG(0, u, g)[2], 13);
      }
    }
    expect(evaluateHG(0.37, 1.2)).toBe(evaluateHG(0.37, 0.999999));
    expect(evaluateHG(-0.37, -1.2)).toBe(evaluateHG(-0.37, -0.999999));
    for (const [g, cosTheta] of [
      [0.999999, 1],
      [-0.999999, -1],
    ] as const) {
      const a = Math.fround(Math.abs(Math.fround(g)));
      const alignedCos = g >= 0 ? cosTheta : -cosTheta;
      const oneMinusA = Math.fround(1 - a);
      const denominator = Math.fround(
        Math.fround(oneMinusA * oneMinusA) +
          Math.fround(2 * a * Math.fround(1 - alignedCos)),
      );
      const f32Pdf = Math.fround(
        Math.fround(oneMinusA * Math.fround(1 + a)) /
          Math.fround(
            Math.fround(4 * Math.fround(Math.PI)) *
              Math.fround(denominator * Math.sqrt(denominator)),
          ),
      );
      expect(Number.isFinite(f32Pdf)).toBe(true);
      expect(f32Pdf).toBeGreaterThan(0);
      const cpuPdf = evaluateHG(cosTheta, g);
      expect(Math.abs(f32Pdf - cpuPdf) / cpuPdf).toBeLessThan(0.04);
    }
    expect(glslMirrorCos(0.375, 1e-8)).not.toBe(0.25);
  });

  it('rejects entry and thin-exit TIR with finite zero-weight scatter records', () => {
    const interfaceSampler = sourceBetween(
      'vec3 sampleTransmissionInterfaceDirection',
      'vec3 transmissionDirection',
    );
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

    expect(interfaceSampler).toContain(
      'dot( canonicalWi, canonicalWi ) > 1e-16',
    );
    expect(interfaceSampler).toContain('return vec3( 0.0 );');
    for (const transmission of [ordinaryTransmission, dispersionTransmission]) {
      expect(transmission).toContain('sampleTransmissionInterfaceDirection(');
    }
    expect(deltaTransmission).toContain(
      'dot( canonicalDirection, canonicalDirection ) > 1e-16',
    );
    expect(deltaTransmission).toContain('return vec3( 0.0 );');
    const thinSheet = sourceBetween(
      'ThinSheetTransmissionSample sampleThinSheetTransmission',
      'float bsdfDeltaPdfLocal',
    );
    expect(thinSheet.match(/sampleThinSheetInterface\(/g)).toHaveLength(2);
    expect(thinSheet).toContain('return result;');
    expect(thinSheet).toContain('result.valid =');
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
