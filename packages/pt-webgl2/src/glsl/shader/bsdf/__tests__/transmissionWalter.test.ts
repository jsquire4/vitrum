import { describe, expect, it } from 'vitest';
import * as BsdfFns from '../bsdf_functions.glsl.js';

const bsdf_functions = (BsdfFns as unknown as Record<string, string>)['bsdf_functions']!;

describe('rough-transmission BTDF/PDF pairing', () => {
  it('pins the Walter half-vector Jacobian and radiance-mode BTDF terms', () => {
    expect(bsdf_functions).toContain(
      'float sqrtDenom = wiDotH + eta * woDotH;',
    );
    expect(bsdf_functions).toContain(
      'float pdfWi = pdfWh * abs( wiDotH ) / denom;',
    );
    expect(bsdf_functions).toContain(
      '( 1.0 - F ) * D * G * abs( wiDotH * woDotH ) * eta * eta',
    );
    expect(bsdf_functions).toContain(
      'float F = dielectricFresnel( abs( woDotH ), eta );',
    );
  });

  it('matches one physical diffuse attenuation to the diffuse-mixture PDF', () => {
    const diffuseStart = bsdf_functions.indexOf('float diffuseEval');
    const diffuseEnd = bsdf_functions.indexOf(
      'vec3 diffuseDirection',
      diffuseStart,
    );
    const diffuseEval = bsdf_functions.slice(diffuseStart, diffuseEnd);
    const evalStart = bsdf_functions.indexOf('float bsdfEval');
    const evalEnd = bsdf_functions.indexOf('float bsdfResult', evalStart);
    const evaluatedBsdf = bsdf_functions.slice(evalStart, evalEnd);
    const lobeStart = bsdf_functions.indexOf('void getLobeWeights');
    const lobeEnd = bsdf_functions.indexOf('void getSamplingLobeWeights', lobeStart);
    const lobePolicy = bsdf_functions.slice(lobeStart, lobeEnd);

    expect(diffuseEval).toContain(
      'float transFactor = ( 1.0 - surf.transmission );',
    );
    expect(evaluatedBsdf).not.toContain('1.0 - surf.transmission');
    expect(lobePolicy).toContain(
      'diffuseWeight = ( 1.0 - transmission ) * ( 1.0 - diffSpecularProb );',
    );

    // Use a smooth dielectric so the competing reflection/transmission lobes
    // are delta measures. Away from their discrete directions, the continuous
    // mixture density is exactly diffuseWeight * cosinePdf. The expected value
    // of the diffuse strategy must therefore retain one (1 - transmission)
    // factor. The former evaluator applied a second factor and converged to
    // (1 - transmission)^2 instead.
    const cosine = 0.63;
    const cosinePdf = cosine / Math.PI;
    const baseResponse = 0.72;
    const diffSpecularProbability = 0.5;
    const transmissionSpecularProbability = 0.31;

    for (const transmission of [0, 0.1, 0.35, 0.7, 0.95, 0.999]) {
      const diffuseRaw =
        (1 - transmission) * (1 - diffSpecularProbability);
      const specularRaw =
        transmission * transmissionSpecularProbability +
        (1 - transmission) * diffSpecularProbability;
      const transmissionRaw =
        transmission * (1 - transmissionSpecularProbability);
      const total = diffuseRaw + specularRaw + transmissionRaw;
      const diffuseWeight = diffuseRaw / total;
      const mixturePdf = diffuseWeight * cosinePdf;
      const physicalDiffuseCosine =
        (1 - transmission) * baseResponse * cosinePdf;
      const conditionalEstimate = physicalDiffuseCosine / mixturePdf;
      const expectedDiffuseEnergy =
        diffuseWeight * conditionalEstimate;

      expect(total).toBeCloseTo(1, 15);
      expect(mixturePdf).toBeGreaterThan(0);
      expect(Number.isFinite(conditionalEstimate)).toBe(true);
      expect(expectedDiffuseEnergy).toBeCloseTo(
        (1 - transmission) * baseResponse,
        14,
      );

      if (transmission > 0) {
        const formerlyDoubleAttenuatedEnergy =
          expectedDiffuseEnergy * (1 - transmission);
        expect(formerlyDoubleAttenuatedEnergy).not.toBeCloseTo(
          expectedDiffuseEnergy,
          10,
        );
      }
    }
  });

  it('has a finite, positive matched throughput for entering and exiting cases', () => {
    const cases = [
      { eta: 1 / 1.5, woDotH: 0.9, wiDotH: -0.8, woZ: 0.85 },
      { eta: 1.5, woDotH: 0.7, wiDotH: -0.9, woZ: 0.75 },
    ];

    for (const { eta, woDotH, wiDotH, woZ } of cases) {
      const D = 0.7;
      const G = 0.6;
      const G1 = 0.8;
      const F = 0.04;
      const transmission = 0.75;
      const sqrtDenom = wiDotH + eta * woDotH;
      const denom = sqrtDenom * sqrtDenom;

      // ggxPdfForSurface(wo, wh) for VNDF sampling.
      const pdfWh = D * G1 * Math.abs(woDotH) / Math.abs(woZ);
      const pdfWi = pdfWh * Math.abs(wiDotH) / denom;
      // transmissionEval stores f * |n.wi| in ScatterRecord.throughput.
      const fCos =
        transmission *
        (1 - F) *
        D *
        G *
        Math.abs(wiDotH * woDotH) *
        eta *
        eta /
        (Math.abs(woZ) * denom);
      const sampledWeight = fCos / pdfWi;

      // D, the half-vector geometry, and the Jacobian cancel exactly only when
      // the evaluator and sampler use the same Walter convention.
      const expected =
        transmission * (1 - F) * (G / G1) * eta * eta;
      expect(pdfWi).toBeGreaterThan(0);
      expect(fCos).toBeGreaterThan(0);
      expect(Number.isFinite(sampledWeight)).toBe(true);
      expect(sampledWeight).toBeCloseTo(expected, 12);
    }
  });

  it('rejects same-hemisphere directions before evaluating the BTDF', () => {
    expect(bsdf_functions).toContain(
      'if ( woDotH * wiDotH >= 0.0 || abs( wo.z ) <= EPSILON || abs( wi.z ) <= EPSILON )',
    );
  });
});
