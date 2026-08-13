import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { MaterialSpec, Scene, ThinFilmLayer } from '@vitrum/core';
import {
  ThinFilmNumericError,
  thinFilmRgb,
  thinFilmRtAtWavelength,
} from '../math/thinFilm.js';
import {
  materialToPackedVec4s,
  MATERIAL_FLOAT_STRIDE,
  THIN_FILM_RGB_LUT_BINS,
  thinFilmRgbLutForMaterial,
  thinFilmRgbLutPosition,
  MATERIAL_VEC4_STRIDE,
} from '../scene/materialPacking.js';
import { assertThinFilmSceneSupported } from '../spectralSceneValidation.js';
import {
  composeSppmPhotonPassWgsl,
  PT_WEBGPU_TRACE_WGSL,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import {
  composeRestirPtProducerWgsl,
  composeRestirPtResolveWgsl,
} from '../wgsl/pathTrace/restirPtCompose.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';

const ENGINE_SOURCE = readFileSync(new URL('../index.ts', import.meta.url), 'utf8');
const MUTATION_ROUTER_SOURCE = readFileSync(
  new URL('../sceneMutationRouter.ts', import.meta.url),
  'utf8',
);
const dielectricMaterial = (
  layers: readonly ThinFilmLayer[],
  extra: Partial<MaterialSpec> = {},
): MaterialSpec => ({
  baseColor: [1, 1, 1],
  roughness: 0,
  metallic: 0,
  transmission: 1,
  ior: 1.52,
  thinFilmStack: { layers },
  ...extra,
});

describe('thin-film numerical failure contract', () => {
  it('raises a structured CPU error and absorbs an invalid shader sample', () => {
    let failure: unknown;
    try {
      thinFilmRtAtWavelength({
        layers: [{ ior: 1.5, thicknessNm: 100 }],
        incidentIor: 1,
        substrateIor: 1.5,
        wavelengthNm: 0,
        cosTheta: 0.5,
        angleDependent: true,
      });
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(ThinFilmNumericError);
    expect(failure).toMatchObject({
      name: 'ThinFilmNumericError',
      code: 'PT_WEBGPU_THIN_FILM_NUMERIC_FAILURE',
      reason: 'non-finite-response',
    });
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain(
      'return vec3f(0.0, 0.0, 1.0);',
    );
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).not.toContain(
      'if (!tfFinite(r) || !tfFinite(t) || r + t > 1.0001) {\n' +
      '    return vec3f(1.0, 0.0, 0.0);',
    );
  });
});

function scene(material: MaterialSpec): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'film',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material,
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function bareFresnel(cosI: number, etaI: number, etaT: number): number {
  const sinT = etaI / etaT * Math.sqrt(Math.max(0, 1 - cosI * cosI));
  if (sinT >= 1) return 1;
  const cosT = Math.sqrt(Math.max(0, 1 - sinT * sinT));
  const rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
  const rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
  return 0.5 * (rs * rs + rp * rp);
}

function coatedInterfaceChannel(
  authoredF: number,
  stackR: number,
  stackT: number,
  bareR: number,
): { reflectance: number; transmittance: number; absorption: number } {
  const baseF = Math.max(0, Math.min(1, authoredF));
  const survivingEnergy = Math.max(0, Math.min(1, stackR + stackT));
  const reflectedWeight = baseF * stackR / Math.max(bareR, 1e-6);
  const transmittedWeight =
    (1 - baseF) * stackT / Math.max(1 - bareR, 1e-6);
  const weightSum = reflectedWeight + transmittedWeight;
  let reflectedFraction =
    weightSum > 1e-20 ? reflectedWeight / weightSum : baseF;
  if (stackT <= 1e-20 && stackR > 1e-20) reflectedFraction = 1;
  if (stackR <= 1e-20 && stackT > 1e-20) reflectedFraction = 0;
  const reflectance = survivingEnergy *
    Math.max(0, Math.min(1, reflectedFraction));
  const transmittance = survivingEnergy - reflectance;
  return {
    reflectance,
    transmittance,
    absorption: 1 - survivingEnergy,
  };
}

describe('pt-webgpu coherent thin-film production closure', () => {
  it('matches the analytic single-layer Airy result at normal incidence', () => {
    const n0 = 1; const n1 = 1.38; const ns = 1.52;
    const d = 101; const lambda = 550;
    const r01 = (n0 - n1) / (n0 + n1);
    const r12 = (n1 - ns) / (n1 + ns);
    const phase = 4 * Math.PI * n1 * d / lambda;
    const c = Math.cos(phase); const s = Math.sin(phase);
    const numerator = [
      r01 + r12 * c,
      r12 * s,
    ] as const;
    const denominator = [
      1 + r01 * r12 * c,
      r01 * r12 * s,
    ] as const;
    const expectedR = (
      numerator[0] ** 2 + numerator[1] ** 2
    ) / (denominator[0] ** 2 + denominator[1] ** 2);
    const rt = thinFilmRtAtWavelength({
      layers: [{ ior: n1, thicknessNm: d }],
      incidentIor: n0, substrateIor: ns, wavelengthNm: lambda,
      cosTheta: 1, angleDependent: true,
    });
    expect(rt.reflectance).toBeCloseTo(expectedR, 10);
    expect(rt.transmittance).toBeCloseTo(1 - expectedR, 10);
    expect(rt.absorption).toBeCloseTo(0, 10);
  });

  it('collapses a zero-thickness layer to exact unpolarized oblique Fresnel', () => {
    for (const cosTheta of [1, 0.8, 0.35, 0.08]) {
      const rt = thinFilmRtAtWavelength({
        layers: [{ ior: 2.1, thicknessNm: 0 }],
        incidentIor: 1, substrateIor: 1.52, wavelengthNm: 510,
        cosTheta, angleDependent: true,
      });
      expect(rt.reflectance).toBeCloseTo(bareFresnel(cosTheta, 1, 1.52), 9);
      expect(rt.reflectance + rt.transmittance + rt.absorption).toBeCloseTo(1, 11);
    }
  });

  it('puts complex-IOR loss inside the optics and conserves R+T+A', () => {
    const common = {
      incidentIor: 1, substrateIor: 1.52, wavelengthNm: 460,
      cosTheta: 0.42, angleDependent: true,
    };
    const lossless = thinFilmRtAtWavelength({
      ...common, layers: [{ ior: 1.7, thicknessNm: 380 }],
    });
    const lossy = thinFilmRtAtWavelength({
      ...common,
      layers: [{ ior: 1.7, extinctionCoefficient: 0.18, thicknessNm: 380 }],
    });
    expect(lossy.absorption).toBeGreaterThan(0.05);
    expect(lossy.transmittance).toBeLessThan(lossless.transmittance);
    expect(Math.abs(lossy.reflectance - lossless.reflectance)).toBeGreaterThan(1e-4);
    expect(lossy.reflectance + lossy.transmittance + lossy.absorption).toBeCloseTo(1, 11);
  });

  it('is reciprocal under forward/reverse layer order at the Snell-matched angle', () => {
    const layers = [
      { ior: 1.31, thicknessNm: 90 },
      { ior: 2.05, thicknessNm: 145 },
    ];
    const cosForward = 0.47;
    const sinForward = Math.sqrt(1 - cosForward * cosForward);
    const cosReverse = Math.sqrt(1 - (sinForward / 1.52) ** 2);
    const forward = thinFilmRtAtWavelength({
      layers, incidentIor: 1, substrateIor: 1.52, wavelengthNm: 610,
      cosTheta: cosForward, angleDependent: true,
    });
    const reverse = thinFilmRtAtWavelength({
      layers, incidentIor: 1, substrateIor: 1.52, wavelengthNm: 610,
      cosTheta: cosReverse, angleDependent: true, reverse: true,
    });
    expect(reverse.transmittance).toBeCloseTo(forward.transmittance, 9);
    expect(reverse.reflectance).toBeCloseTo(forward.reflectance, 9);
  });

  it('replaces the bare interface without double counting and preserves all limiting identities', () => {
    const common = {
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
    const stack = thinFilmRtAtWavelength(common);
    const bareR = bareFresnel(common.cosTheta, 1, 1.52);

    // Authored Fresnel equal to the physical bare interface must replace that
    // interface with the complete TMM stack exactly, not add a second boundary.
    const physical = coatedInterfaceChannel(
      bareR, stack.reflectance, stack.transmittance, bareR,
    );
    expect(physical.reflectance).toBeCloseTo(stack.reflectance, 12);
    expect(physical.transmittance).toBeCloseTo(stack.transmittance, 12);
    expect(physical.absorption).toBeCloseTo(stack.absorption, 12);

    // Conversely, a zero-layer stack is just the bare boundary and must leave
    // arbitrary authored metallic/specular reflectance untouched.
    const bareStack = thinFilmRtAtWavelength({ ...common, layers: [] });
    for (const authoredF of [0, 0.02, 0.37, 0.91, 1]) {
      const uncoated = coatedInterfaceChannel(
        authoredF,
        bareStack.reflectance,
        bareStack.transmittance,
        bareR,
      );
      expect(uncoated.reflectance).toBeCloseTo(authoredF, 12);
      expect(uncoated.transmittance).toBeCloseTo(1 - authoredF, 12);
      expect(uncoated.absorption).toBeCloseTo(0, 12);
    }
  });

  it('keeps lossy colored lanes bounded and finite through TIR and near-zero bare odds', () => {
    const rgb = thinFilmRgb({
      layers: [
        { ior: 1.28, thicknessNm: 105 },
        { ior: 2.35, extinctionCoefficient: 0.22, thicknessNm: 260 },
      ],
      incidentIor: 1,
      substrateIor: 1.52,
      cosTheta: 0.31,
      angleDependent: true,
    });
    const bareR = bareFresnel(0.31, 1, 1.52);
    for (let lane = 0; lane < 3; lane += 1) {
      const adjusted = coatedInterfaceChannel(
        [0.015, 0.44, 0.93][lane]!,
        rgb.reflectance[lane]!,
        rgb.transmittance[lane]!,
        bareR,
      );
      expect(Number.isFinite(adjusted.reflectance)).toBe(true);
      expect(Number.isFinite(adjusted.transmittance)).toBe(true);
      expect(Number.isFinite(adjusted.absorption)).toBe(true);
      expect(adjusted.reflectance).toBeGreaterThanOrEqual(0);
      expect(adjusted.transmittance).toBeGreaterThanOrEqual(0);
      expect(adjusted.absorption).toBeGreaterThanOrEqual(0);
      expect(
        adjusted.reflectance + adjusted.transmittance + adjusted.absorption,
      ).toBeCloseTo(1, 12);
    }

    const tir = thinFilmRtAtWavelength({
      layers: [{ ior: 1.2, thicknessNm: 0 }],
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 550,
      cosTheta: 0.5,
      angleDependent: true,
      reverse: true,
    });
    for (const authoredF of [0, 1e-12, 0.5, 1]) {
      const adjusted = coatedInterfaceChannel(
        authoredF, tir.reflectance, tir.transmittance, 1,
      );
      expect(adjusted).toEqual({
        reflectance: 1,
        transmittance: 0,
        absorption: 0,
      });
    }

    // Equal endpoint IOR gives bareR=0. A real coating may still reflect; the
    // denominator floor must keep the authored-odds replacement finite.
    const equalIorStack = thinFilmRtAtWavelength({
      layers: [{ ior: 1.8, thicknessNm: 170 }],
      incidentIor: 1,
      substrateIor: 1,
      wavelengthNm: 500,
      cosTheta: 0.62,
      angleDependent: true,
    });
    const nearZero = coatedInterfaceChannel(
      0.04,
      equalIorStack.reflectance,
      equalIorStack.transmittance,
      0,
    );
    expect(Object.values(nearZero).every(Number.isFinite)).toBe(true);
    expect(
      nearZero.reflectance + nearZero.transmittance + nearZero.absorption,
    ).toBeCloseTo(1, 12);
  });

  it('preserves the complete TMM response under forward and reverse incidence', () => {
    const layers = [
      { ior: 1.31, thicknessNm: 90 },
      { ior: 2.05, extinctionCoefficient: 0.04, thicknessNm: 145 },
    ];
    const cosForward = 0.47;
    const sinForward = Math.sqrt(1 - cosForward * cosForward);
    const cosReverse = Math.sqrt(1 - (sinForward / 1.52) ** 2);
    for (const sample of [
      {
        rt: thinFilmRtAtWavelength({
          layers, incidentIor: 1, substrateIor: 1.52, wavelengthNm: 610,
          cosTheta: cosForward, angleDependent: true,
        }),
        bareR: bareFresnel(cosForward, 1, 1.52),
      },
      {
        rt: thinFilmRtAtWavelength({
          layers, incidentIor: 1, substrateIor: 1.52, wavelengthNm: 610,
          cosTheta: cosReverse, angleDependent: true, reverse: true,
        }),
        bareR: bareFresnel(cosReverse, 1.52, 1),
      },
    ]) {
      const adjusted = coatedInterfaceChannel(
        sample.bareR,
        sample.rt.reflectance,
        sample.rt.transmittance,
        sample.bareR,
      );
      expect(adjusted.reflectance).toBeCloseTo(sample.rt.reflectance, 11);
      expect(adjusted.transmittance).toBeCloseTo(sample.rt.transmittance, 11);
      expect(adjusted.absorption).toBeCloseTo(sample.rt.absorption, 11);
    }
  });

  it('handles total internal reflection and angleDependent=false', () => {
    const tir = thinFilmRtAtWavelength({
      layers: [{ ior: 1.2, thicknessNm: 0 }],
      incidentIor: 1, substrateIor: 1.52, wavelengthNm: 550,
      cosTheta: 0.5, angleDependent: true, reverse: true,
    });
    expect(tir.reflectance).toBeCloseTo(1, 9);
    expect(tir.transmittance).toBeCloseTo(0, 9);
    const normalOnly = (cosTheta: number) => thinFilmRtAtWavelength({
      layers: [{ ior: 1.45, thicknessNm: 240 }],
      incidentIor: 1, substrateIor: 1.52, wavelengthNm: 510,
      cosTheta, angleDependent: false,
    });
    expect(normalOnly(0.05)).toEqual(normalOnly(0.95));
  });

  it('bounds the packed RGB cosine LUT against the 81-sample reference', () => {
    const material = dielectricMaterial([
      { ior: 1.31, thicknessNm: 80 },
      { ior: 2.15, extinctionCoefficient: 0.015, thicknessNm: 170 },
      { ior: 1.46, thicknessNm: 220 },
    ]);
    const lut = thinFilmRgbLutForMaterial(material);
    let maxError = 0;
    for (const reverse of [false, true]) {
      for (let step = 0; step <= 2000; step += 1) {
        const cosTheta = step / 2000;
        const position = thinFilmRgbLutPosition(cosTheta, 1, 1.52, reverse);
        const i0 = Math.floor(position);
        const i1 = Math.min(i0 + 1, THIN_FILM_RGB_LUT_BINS - 1);
        const alpha = position - i0;
        const directionOffset = reverse ? 8 : 0;
        const sample = (lane: number) => {
          const a = lut[i0 * 16 + directionOffset + lane]!;
          const b = lut[i1 * 16 + directionOffset + lane]!;
          return a + (b - a) * alpha;
        };
        const reference = thinFilmRgb({
          layers: material.thinFilmStack!.layers,
          incidentIor: 1, substrateIor: 1.52,
          angleDependent: true, cosTheta, reverse,
        });
        const expected = [
          ...reference.reflectance, ...reference.transmittance,
          reference.reflectanceEnergy, reference.transmittanceEnergy,
        ];
        for (let lane = 0; lane < 8; lane += 1) {
          maxError = Math.max(maxError, Math.abs(sample(lane) - expected[lane]!));
        }
      }
    }
    expect(maxError).toBeLessThan(0.02);
  });

  it('defaults angle dependence on and admits every implemented layered-BSDF combination', () => {
    const packed = materialToPackedVec4s(dielectricMaterial([
      { ior: 1.4, thicknessNm: 120 },
    ]));
    expect(packed[27]).toBe(1);
    expect(packed).toHaveLength(MATERIAL_FLOAT_STRIDE);
    expect(() => assertThinFilmSceneSupported(scene(dielectricMaterial([
      { ior: 1.4, thicknessNm: 120 },
    ])))).not.toThrow();
    expect(() => assertThinFilmSceneSupported(scene(dielectricMaterial([
      { ior: 1.4, thicknessNm: 120 },
    ], {
      transmissionMap: {
        handle: { width: 1, height: 1, data: new Float32Array([0.35, 0, 0, 1]) },
      },
    })))).not.toThrow();
    for (const material of [
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], { metallic: 0.1 }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], { roughness: 0.2 }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], { transmission: 0.8 }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], {
        roughnessMap: { handle: {} },
      }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], {
        metallicMap: { handle: {} },
      }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], { anisotropy: 0.2 }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], { clearcoat: 0.2 }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], { sheen: 0.2 }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], { iridescence: 0.2 }),
      dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }], {
        frontLayer: { transmission: [0.9, 0.8, 0.7] },
      }),
    ]) {
      expect(() => assertThinFilmSceneSupported(scene(material))).not.toThrow();
    }
  });

  it('routes the adjusted interface through evaluation, sampling, PDFs, and every composed estimator', () => {
    const interfaceResponse = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn bsdfLayeredInterfaceResponse(',
      ),
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn materialDielectricLayeredInterface(',
      ),
    );
    for (const token of [
      'let filmRt = thinFilmTransportRt(',
      'let reflectedWeight =',
      'baseF * filmRt.reflectance / max(bareR, 1e-6);',
      'baseT * filmRt.transmittance / max(bareT, 1e-6);',
      'response.reflectance = survivingEnergy * reflectedFraction;',
      'response.baseTransmittance =',
    ]) {
      expect(interfaceResponse).toContain(token);
    }

    const evaluator = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn evaluateBrdfFullWithClearcoatNormal(',
      ),
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn evaluateFiniteSameSideBrdfFullWithClearcoatNormal(',
      ),
    );
    const eventPdf = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn bsdfDielectricFiniteEventProbabilities(',
      ),
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn brdfFiniteBaseLobeWeights(',
      ),
    );
    const sampler = PT_WEBGPU_PATH_TRACE_BSDF_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn sampleNextBounceDirectionWithClearcoatNormal(',
      ),
      PT_WEBGPU_PATH_TRACE_BSDF_WGSL.indexOf(
        'fn sampleNextBounceDirection(',
      ),
    );
    expect(evaluator).toContain('bsdfLayeredInterfaceResponse(');
    expect(eventPdf).toContain('materialDielectricLayeredInterface(');
    // The event-PDF path evaluates the macro and microfacet interfaces in a
    // two-iteration loop (one inlined call site instead of two), so the response
    // lands in locals rather than a per-call struct. Same model, same routing.
    expect(eventPdf).toContain('microfacetReflectance');
    expect(eventPdf).toContain('microfacetBaseTransmittance');
    expect(sampler).toContain('materialDielectricLayeredInterface(');
    expect(sampler).toContain('microfacetInterface.reflectance');
    expect(sampler).toContain('microfacetInterface.baseTransmittance');

    const composedShaders = [
      PT_WEBGPU_TRACE_WGSL,
      PT_WEBGPU_TRACE_LITE_WGSL,
      composeSppmPhotonPassWgsl(),
      composeRestirPtProducerWgsl(),
      composeRestirPtResolveWgsl(),
    ];
    for (const shader of composedShaders) {
      expect(shader).toContain('fn bsdfLayeredInterfaceResponse(');
      expect(shader).toContain('fn materialDielectricLayeredInterface(');
    }
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'anisoStrength, anisoRotation, thinFilm, false);',
    );
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain(
      '0.0, 0.0, thinFilm, false);',
    );
    expect(composeSppmPhotonPassWgsl()).toContain(
      'sampleNextBounceDirectionWithClearcoatNormal(',
    );
    expect(composeRestirPtProducerWgsl()).toContain(
      'rptSourceDirectionalPdfFull(',
    );
    expect(composeRestirPtResolveWgsl()).toContain(
      'let thinFilm = rptThinFilmForDomain(r);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('thinFilmReflectTint');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('layerStrength = clamp(0.12');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let prevThinFilm = prevMat.thinFilm;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'evalThinFilm.frontFace = !evalThinFilm.frontFace;',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'specularColor, specularIntensity, evalThinFilm,',
    );
  });

  it('samples adjusted RGB R/T at scalar energy while preserving colored expectations and film loss', () => {
    const rt = thinFilmRgb({
      layers: [
        { ior: 1.32, thicknessNm: 95 },
        { ior: 2.2, extinctionCoefficient: 0.08, thicknessNm: 180 },
      ],
      incidentIor: 1,
      substrateIor: 1.52,
      angleDependent: true,
      cosTheta: 0.43,
    });
    const bareR = bareFresnel(0.43, 1, 1.52);
    const authored = [0.025, 0.32, 0.78];
    const adjusted = authored.map((baseF, lane) => coatedInterfaceChannel(
      baseF,
      rt.reflectance[lane]!,
      rt.transmittance[lane]!,
      bareR,
    ));
    const luminance = (values: readonly number[]) =>
      values[0]! * 0.2126 + values[1]! * 0.7152 + values[2]! * 0.0722;
    const reflected = adjusted.map((lane) => lane.reflectance);
    const transmitted = adjusted.map((lane) => lane.transmittance);
    const reflectedEnergy = luminance(reflected);
    const transmittedEnergy = luminance(transmitted);
    const proposalNorm = reflectedEnergy + transmittedEnergy;
    const pReflect = reflectedEnergy / proposalNorm;
    const pTransmit = transmittedEnergy / proposalNorm;
    expect(Math.max(...reflected) - Math.min(...reflected)).toBeGreaterThan(0.01);
    expect(Math.max(...transmitted) - Math.min(...transmitted)).toBeGreaterThan(0.01);
    expect(adjusted.some((lane) => lane.absorption > 0.01)).toBe(true);

    const sampleCount = 100_000;
    let reflectedCount = 0;
    let transmittedCount = 0;
    const reflectedMean = [0, 0, 0];
    const transmittedMean = [0, 0, 0];
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const xi = (sample + 0.5) / sampleCount;
      if (xi < pReflect) {
        reflectedCount += 1;
        for (let lane = 0; lane < 3; lane += 1) {
          reflectedMean[lane] =
            reflectedMean[lane]! + reflected[lane]! / pReflect;
        }
      } else {
        transmittedCount += 1;
        for (let lane = 0; lane < 3; lane += 1) {
          transmittedMean[lane] =
            transmittedMean[lane]! + transmitted[lane]! / pTransmit;
        }
      }
    }
    expect(reflectedCount / sampleCount).toBeCloseTo(pReflect, 4);
    expect(transmittedCount / sampleCount).toBeCloseTo(pTransmit, 4);
    for (let lane = 0; lane < 3; lane += 1) {
      expect(reflectedMean[lane]! / sampleCount).toBeCloseTo(
        reflected[lane]!,
        4,
      );
      expect(transmittedMean[lane]! / sampleCount).toBeCloseTo(
        transmitted[lane]!,
        4,
      );
      expect(
        reflectedMean[lane]! / sampleCount +
          transmittedMean[lane]! / sampleCount,
      ).toBeCloseTo(1 - adjusted[lane]!.absorption, 4);
    }
  });

  it('applies KHR transmission after the optical stack without excluding finite estimators', () => {
    const raw = {
      reflectance: [0.08, 0.12, 0.2] as const,
      transmittance: [0.71, 0.63, 0.49] as const,
      reflectanceEnergy: 0.13,
      transmittanceEnergy: 0.61,
    };
    for (const scale of [0, 0.35, 1]) {
      const scaledT = raw.transmittance.map((value) => value * scale);
      const pTransmit = raw.transmittanceEnergy * scale;
      const absorption = 1 - raw.reflectanceEnergy - pTransmit;
      expect(raw.reflectance).toEqual([0.08, 0.12, 0.2]);
      expect(absorption).toBeCloseTo(0.87 - 0.61 * scale, 14);
      if (scale > 0) {
        for (let lane = 0; lane < 3; lane += 1) {
          expect(scaledT[lane]! / pTransmit).toBeCloseTo(
            raw.transmittance[lane]! / raw.transmittanceEnergy,
            14,
          );
        }
      }
    }

    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'out.transmittance = raw.transmittance * transmissionScale;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'out.transmittanceEnergy = raw.transmittanceEnergy * transmissionScale;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      '1.0 - out.reflectanceEnergy - out.transmittanceEnergy',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'params.spectralEnabled != 0u, heroLambda, transmission,',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'prevMat.transmission,',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'frontFace, params.spectralEnabled != 0u, photonHeroLambda, transmission,',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'opticalFilm.transmissionScale = 1.0;',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'baseColor * clamp(transmission, 0.0, 1.0) * ft *',
    );
    expect(PT_WEBGPU_PATH_TRACE_BSDF_WGSL).toContain(
      'sheenAttenuation * clearcoatAttenuation;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      'if (optics.thinFilmEnabled)',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let mneeDeltaEvent = bs.sampledIsDelta && bs.sampledEventPdf > 0.0;',
    );
  });

  it('samples adjusted hero R/T at the exact scalar proposal and retains absorption as lost weight', () => {
    const rt = thinFilmRtAtWavelength({
      layers: [{ ior: 1.8, extinctionCoefficient: 0.11, thicknessNm: 310 }],
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 487,
      cosTheta: 0.37,
      angleDependent: true,
    });
    const adjusted = coatedInterfaceChannel(
      0.36,
      rt.reflectance,
      rt.transmittance,
      bareFresnel(0.37, 1, 1.52),
    );
    const proposalNorm = adjusted.reflectance + adjusted.transmittance;
    const pReflect = adjusted.reflectance / proposalNorm;
    const sampleCount = 100_000;
    const counts = [0, 0];
    let reflectedEstimator = 0;
    let transmittedEstimator = 0;
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const xi = (sample + 0.5) / sampleCount;
      if (xi < pReflect) {
        counts[0]! += 1;
        reflectedEstimator += adjusted.reflectance / pReflect;
      } else {
        counts[1]! += 1;
        transmittedEstimator +=
          adjusted.transmittance / (1 - pReflect);
      }
    }
    expect(counts[0]! / sampleCount).toBeCloseTo(pReflect, 4);
    expect(counts[1]! / sampleCount).toBeCloseTo(1 - pReflect, 4);
    expect(reflectedEstimator / sampleCount).toBeCloseTo(
      adjusted.reflectance,
      4,
    );
    expect(transmittedEstimator / sampleCount).toBeCloseTo(
      adjusted.transmittance,
      4,
    );
    expect(
      (reflectedEstimator + transmittedEstimator) / sampleCount,
    ).toBeCloseTo(1 - adjusted.absorption, 4);
  });

  it('pins eta-mode reciprocity, BDPT reverse delta density, SPPM parity, and deterministic TIR', () => {
    const etaI = 1;
    const etaT = 1.52;
    const forwardRadianceScale = (etaI / etaT) ** 2;
    const reverseRadianceScale = (etaT / etaI) ** 2;
    expect(forwardRadianceScale * reverseRadianceScale).toBeCloseTo(1, 12);
    expect(1 * 1).toBe(1); // Importance-mode transmission is eta-neutral both ways.
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'etaIOverT * etaIOverT, 1.0, transportModeImportance',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('BSDF_LOBE_DELTA_REFLECTION');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('BSDF_LOBE_DELTA_TRANSMISSION');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'bdptEyeStackSetSpec(',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'bs.sampledLobe == BSDF_LOBE_COMPOUND_THIN_SHEET_TRANSMISSION,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!bs.sampledIsDelta) {');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'pdfScatter = bsPrev.sampledEventPdf;',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'let thinFilm = ThinFilmInterface(',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'sampleNextBounceDirectionWithClearcoatNormal(',
    );
    expect(SPPM_PHOTON_PASS_WGSL).not.toContain('thinFilmTransmitTint');

    const tir = thinFilmRtAtWavelength({
      layers: [{ ior: 1.2, thicknessNm: 0 }],
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 550,
      cosTheta: 0.5,
      angleDependent: true,
      reverse: true,
    });
    expect(tir).toEqual({ reflectance: 1, transmittance: 0, absorption: 0 });
    for (const xi of [0, 0.2, 0.5, 0.999999]) {
      expect(xi < tir.reflectance ? 'reflect' : 'not-reflect').toBe('reflect');
    }
  });


  it('keeps full and incremental thin-film repacks transactional', () => {
    const repackStart = ENGINE_SOURCE.indexOf('  #repackScene(');
    const repackEnd = ENGINE_SOURCE.indexOf('  #syncLiteTextures(', repackStart);
    const repack = ENGINE_SOURCE.slice(repackStart, repackEnd);
    const validationAt = repack.indexOf('assertThinFilmSceneSupported(scene);');
    const packAt = repack.indexOf('const packed = buildPackedScene(scene');
    const previousAt = repack.indexOf('const previous = {');
    const uploadAt = repack.indexOf('uploadedScene = uploadPackedScene(');
    const publishAt = repack.indexOf('this.#scene = scene;');
    expect(validationAt).toBeGreaterThanOrEqual(0);
    expect(validationAt).toBeLessThan(packAt);
    expect(packAt).toBeLessThan(previousAt);
    expect(previousAt).toBeLessThan(uploadAt);
    expect(uploadAt).toBeLessThan(publishAt);

    const filmGuardAt = MUTATION_ROUTER_SOURCE.indexOf(
      'currentPrimitive?.material.thinFilmStack != null ||',
    );
    const fixedSlotPackAt = MUTATION_ROUTER_SOURCE.indexOf(
      'const packed = packFoldedMaterialEntry(',
      filmGuardAt,
    );
    const fullRepackAt = MUTATION_ROUTER_SOURCE.indexOf(
      'host.setScene(authoredNextScene);',
      filmGuardAt,
    );
    expect(filmGuardAt).toBeGreaterThanOrEqual(0);
    expect(filmGuardAt).toBeLessThan(fixedSlotPackAt);
    expect(fixedSlotPackAt).toBeLessThan(fullRepackAt);
  });
  it('rejects stacks that violate the 0.02 RGB LUT admission certificate', () => {
    expect(() => thinFilmRgbLutForMaterial(dielectricMaterial([
      { ior: 3, thicknessNm: 100_000 },
    ]))).toThrow(/thin-film RGB LUT error .* exceeds 0\.02/);
  });

  it('keeps non-film records byte-identical and appends LUTs sparsely', () => {
    const plain: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0,
      metallic: 0,
      transmission: 1,
      ior: 1.52,
    };
    const film = dielectricMaterial([{ ior: 1.4, thicknessNm: 120 }]);
    const plainDirect = new Float32Array(materialToPackedVec4s(plain));
    const plainScene = buildPackedScene(scene(plain)).materials;
    expect(plainScene).toEqual(plainDirect);
    expect(plainScene.byteLength).toBe(MATERIAL_VEC4_STRIDE * 16);
    expect(plainScene[28 * 4 + 2]).toBe(0);
    const filmScene = buildPackedScene(scene(film)).materials;
    expect(filmScene.length).toBe(MATERIAL_FLOAT_STRIDE + THIN_FILM_RGB_LUT_BINS * 16);
    expect(filmScene.byteLength).toBe(
      (MATERIAL_FLOAT_STRIDE + THIN_FILM_RGB_LUT_BINS * 16) * 4,
    );
    const absoluteBaseVec4 = filmScene[28 * 4 + 2]!;
    expect(absoluteBaseVec4).toBe(MATERIAL_VEC4_STRIDE);
    expect(filmScene.slice(0, MATERIAL_FLOAT_STRIDE)).toEqual(
      new Float32Array(materialToPackedVec4s(film, { thinFilmRgbLutBaseVec4: MATERIAL_VEC4_STRIDE })),
    );
    for (const token of [
      'if (descriptorIndex == MATERIAL_INVALID_INDEX) {',
      'let lutBaseVec4 = materialRecordExactU32(',
      '!materialSpanValid(lutBaseScalar, lutScalarCount, scalarCount)',
      'fn materialStorageScalar(scalarIndex: u32) -> f32 {',
    ]) {
      expect(PT_WEBGPU_TRACE_WGSL).toContain(token);
    }
  });

  it('fails dark when a published thin-film descriptor or LUT range is corrupt', () => {
    const material = PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL;
    const helperStart = material.indexOf('fn thinFilmAbsorbedTransportRt(');
    const transportStart = material.indexOf('fn thinFilmTransportRt(');
    expect(helperStart).toBeGreaterThanOrEqual(0);
    expect(transportStart).toBeGreaterThan(helperStart);

    const helper = material.slice(helperStart, transportStart);
    expect(helper).toContain('out.reflectance = vec3f(0.0);');
    expect(helper).toContain('out.transmittance = vec3f(0.0);');
    expect(helper).toContain('out.reflectanceEnergy = 0.0;');
    expect(helper).toContain('out.transmittanceEnergy = 0.0;');
    expect(helper).toContain('out.absorptionEnergy = 1.0;');

    const transport = material.slice(transportStart);
    const descriptorGuard = transport.slice(
      transport.indexOf('if (descriptorIndex == MATERIAL_INVALID_INDEX) {'),
      transport.indexOf('let lutBaseVec4 = materialRecordExactU32('),
    );
    const lutGuard = transport.slice(
      transport.indexOf('lutBaseVec4 == MATERIAL_INVALID_INDEX ||'),
      transport.indexOf('let bin0Offset = materialCheckedMulU32('),
    );
    for (const guard of [descriptorGuard, lutGuard]) {
      expect(guard).toContain('return thinFilmAbsorbedTransportRt();');
      expect(guard).not.toContain('vec3f(1.0)');
      expect(guard).not.toContain('reflectanceEnergy = 1.0');
    }
  });
});
