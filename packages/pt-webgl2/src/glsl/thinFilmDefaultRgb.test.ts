import { describe, expect, it } from 'vitest';
import {
  thinFilmRtAtWavelength,
} from '../../../pt-webgpu/src/math/thinFilm.js';
import * as ThinFilmModule from './shader/bsdf/thin_film_tmm.glsl.js';
import * as BsdfModule from './shader/bsdf/bsdf_functions.glsl.js';

const thin_film_tmm = (
  ThinFilmModule as unknown as Record<string, string>
)['thin_film_tmm']!;
const bsdf_functions = (
  BsdfModule as unknown as Record<string, string>
)['bsdf_functions']!;

function bareFresnel(cosI: number, etaI: number, etaT: number): number {
  const sinT = etaI / etaT * Math.sqrt(Math.max(0, 1 - cosI * cosI));
  if (sinT >= 1) return 1;
  const cosT = Math.sqrt(Math.max(0, 1 - sinT * sinT));
  const rs = (etaI * cosI - etaT * cosT) /
    (etaI * cosI + etaT * cosT);
  const rp = (etaT * cosI - etaI * cosT) /
    (etaT * cosI + etaI * cosT);
  return 0.5 * (rs * rs + rp * rp);
}

function coatedInterfaceChannel(
  authoredF: number,
  stackR: number,
  stackT: number,
  bareR: number,
): readonly [number, number, number] {
  const baseF = Math.max(0, Math.min(1, authoredF));
  const surviving = Math.max(0, Math.min(1, stackR + stackT));
  const reflectedWeight = baseF * stackR / Math.max(bareR, 1e-6);
  const transmittedWeight =
    (1 - baseF) * stackT / Math.max(1 - bareR, 1e-6);
  const weightSum = reflectedWeight + transmittedWeight;
  let reflectedFraction = weightSum > 1e-20
    ? reflectedWeight / weightSum
    : baseF;
  if (stackT <= 1e-20 && stackR > 1e-20) reflectedFraction = 1;
  if (stackR <= 1e-20 && stackT > 1e-20) reflectedFraction = 0;
  const reflectance = surviving * Math.max(0, Math.min(1, reflectedFraction));
  const transmittance = surviving - reflectance;
  return [reflectance, transmittance, 1 - surviving];
}

describe('WebGL2 coherent thin-film transport', () => {
  it('contains the complete Snell-aware S/P network in both physical directions', () => {
    const compact = thin_film_tmm.replace(/\s+/g, ' ');
    for (const token of [
      'struct ThinFilmScatter',
      'ThinFilmScatter thinFilmCascade(',
      'vec2 thinFilmPhysicalCosine(',
      'vec2 thinFilmAdmittance(',
      'vec2 thinFilmPolarizedRt(',
      'bool frontFace, bool pPolarized',
      'pPolarized ? cDiv( cosine, n ) : cMul( n, cosine )',
      'frontFace ? mediumIndex - 1 : layerCount - mediumIndex',
      'microfacetCos, frontFace, false',
      'microfacetCos, frontFace, true',
    ]) {
      expect(compact).toContain(token);
    }
    expect(compact).toContain('return vec3( 0.0, 0.0, 1.0 );');
    expect(compact).not.toContain('vec2 totalM = vec2( 1.0, 0.0 )');
  });

  it('collapses zero phase to the bare oblique interface and conserves lossless energy', () => {
    for (const cosTheta of [1, 0.8, 0.35, 0.08]) {
      const result = thinFilmRtAtWavelength({
        layers: [{ ior: 2.1, thicknessNm: 0 }],
        incidentIor: 1,
        substrateIor: 1.52,
        wavelengthNm: 510,
        cosTheta,
        angleDependent: true,
      });
      expect(result.reflectance).toBeCloseTo(
        bareFresnel(cosTheta, 1, 1.52),
        9,
      );
      expect(result.reflectance + result.transmittance).toBeCloseTo(1, 11);
      expect(result.absorption).toBeCloseTo(0, 11);
    }
  });

  it('keeps lossy stacks passive and lossless stacks reciprocal under reversed layer order', () => {
    const layers = [
      { ior: 1.31, thicknessNm: 90 },
      { ior: 2.05, thicknessNm: 145 },
    ];
    const cosForward = 0.47;
    const sinForward = Math.sqrt(1 - cosForward * cosForward);
    const cosReverse = Math.sqrt(1 - (sinForward / 1.52) ** 2);
    const forward = thinFilmRtAtWavelength({
      layers,
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 610,
      cosTheta: cosForward,
      angleDependent: true,
    });
    const reverse = thinFilmRtAtWavelength({
      layers,
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 610,
      cosTheta: cosReverse,
      angleDependent: true,
      reverse: true,
    });
    for (const result of [forward, reverse]) {
      expect(result.reflectance).toBeGreaterThanOrEqual(0);
      expect(result.transmittance).toBeGreaterThanOrEqual(0);
      expect(result.absorption).toBeCloseTo(0, 11);
      expect(
        result.reflectance + result.transmittance + result.absorption,
      ).toBeCloseTo(1, 11);
    }
    expect(reverse.reflectance).toBeCloseTo(forward.reflectance, 9);
    expect(reverse.transmittance).toBeCloseTo(forward.transmittance, 9);

    const lossy = thinFilmRtAtWavelength({
      layers: [
        { ior: 1.31, thicknessNm: 90 },
        { ior: 2.05, extinctionCoefficient: 0.04, thicknessNm: 145 },
      ],
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 610,
      cosTheta: cosForward,
      angleDependent: true,
    });
    expect(lossy.absorption).toBeGreaterThan(0);
    expect(
      lossy.reflectance + lossy.transmittance + lossy.absorption,
    ).toBeCloseTo(1, 11);
  });

  it('defaults authored angle dependence on and uses hero or channel-local wavelengths', () => {
    const compact = thin_film_tmm.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'float cos0 = angleDependent ? clamp( microfacetCos, 0.0, 1.0 ) : 1.0;',
    );
    expect(compact).toContain('if ( uSpectralRendering != 0 )');
    expect(compact).toContain('thinFilmLayerCount, heroWavelengthNm,');
    expect(compact).toContain('thinFilmLayerCount, 650.0,');
    expect(compact).toContain('thinFilmLayerCount, 510.0,');
    expect(compact).toContain('thinFilmLayerCount, 475.0,');
    expect(compact).toContain(
      'result.reflectance = vec3( red.x, green.x, blue.x )',
    );
    expect(compact).toContain(
      'result.transmittance = vec3( red.y, green.y, blue.y )',
    );

    const stack = {
      layers: [{ ior: 1.82, thicknessNm: 237 }],
      incidentIor: 1,
      substrateIor: 1.52,
      cosTheta: 0.43,
      angleDependent: true,
    } as const;
    const rgb = [650, 510, 475].map((wavelengthNm) =>
      thinFilmRtAtWavelength({ ...stack, wavelengthNm }).reflectance
    );
    expect(new Set(rgb.map((value) => value.toFixed(8))).size).toBeGreaterThan(1);
  });

  it('replaces rather than double-counts the authored bare interface', () => {
    const input = {
      layers: [
        { ior: 1.31, thicknessNm: 90 },
        { ior: 2.05, extinctionCoefficient: 0.07, thicknessNm: 145 },
      ],
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 610,
      cosTheta: 0.47,
      angleDependent: true,
    } as const;
    const stack = thinFilmRtAtWavelength(input);
    const bareR = bareFresnel(input.cosTheta, 1, 1.52);
    const adjusted = coatedInterfaceChannel(
      bareR,
      stack.reflectance,
      stack.transmittance,
      bareR,
    );
    expect(adjusted[0]).toBeCloseTo(stack.reflectance, 12);
    expect(adjusted[1]).toBeCloseTo(stack.transmittance, 12);
    expect(adjusted[2]).toBeCloseTo(stack.absorption, 12);

    const compact = thin_film_tmm.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'baseF * filmRt.reflectance / max( bareR, 1e-6 )',
    );
    expect(compact).toContain(
      'baseT * filmRt.transmittance / max( bareT, 1e-6 )',
    );
    expect(compact).toContain(
      'response.reflectance = survivingEnergy * reflectedFraction;',
    );
    expect(compact).toContain(
      'response.baseTransmittance = survivingEnergy * ( vec3( 1.0 ) - reflectedFraction );',
    );
    expect(compact).not.toMatch(/baseF\s*\+\s*\([^)]*baseF[^)]*\)\s*\*\s*filmRt\.reflectance/);
    expect(compact).not.toMatch(/\(\s*vec3\( 1\.0 \)\s*-\s*baseF\s*\)\s*\*\s*filmRt\.transmittance/);
  });

  it('routes every finite evaluator and sampler through the same layered response', () => {
    expect(bsdf_functions).not.toContain('thinFilmTMMRgb(');
    expect(bsdf_functions).not.toContain('vec2 thinFilmRt = thinFilmTMM(');
    expect(
      bsdf_functions.match(/surfaceLayeredInterfaceResponse\(/g)?.length ?? 0,
    ).toBeGreaterThanOrEqual(7);
    for (const token of [
      'color = interfaceResponse.baseTransmittance *',
      'vec3 F = interfaceResponse.reflectance;',
      'surf.transmission * surf.color *',
      'interfaceCos = interfaceResponse.baseTransmittance *',
      ').reflectance;',
      'interfaceResponse.baseTransmittance *',
    ]) {
      expect(bsdf_functions).toContain(token);
    }
  });
});
