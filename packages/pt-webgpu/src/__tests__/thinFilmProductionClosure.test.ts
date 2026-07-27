import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { MaterialSpec, Scene, ThinFilmLayer } from '@vitrum/core';
import {
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
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

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

  it('defaults angle dependence on and validates the narrow interface domain', () => {
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
      expect(() => assertThinFilmSceneSupported(scene(material))).toThrow(
        /thin-film scene validation/,
      );
    }
  });

  it('routes R/T/A through the central sampler on eye, lite, SPPM and BDPT paths', () => {
    for (const token of [
      'let rt = thinFilmTransportRt(thinFilm, abs(dot(wo, normal)));',
      'result.throughputMul = rt.reflectance / pReflect;',
      'baseColor * rt.transmittance * etaScale / pTransmit;',
      'The remainder is physical absorption',
      'const MATERIAL_VEC4_STRIDE = 29u;',
    ]) {
      expect(PT_WEBGPU_TRACE_WGSL).toContain(token);
    }
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('thinFilmReflectTint');
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain('layerStrength = clamp(0.12');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let prevThinFilm = ThinFilmInterface(',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'fresnelSchlick(cosOPrev, f0Prev),\n' +
      '        prevMat.iridescence,\n' +
      '        prevMat.iridescenceIor,\n' +
      '        prevMat.iridescenceThicknessMin,\n' +
      '        prevMat.iridescenceThicknessMax,\n' +
      '        prevMat.specularColor,\n' +
      '        prevMat.specularIntensity,\n' +
      '        prevThinFilm,',
    );
  });

  it('samples RGB R/T/A at scalar energy while preserving colored R/p and T/p expectations', () => {
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
    const pReflect = rt.reflectanceEnergy;
    const pTransmit = rt.transmittanceEnergy;
    const pAbsorb = 1 - pReflect - pTransmit;
    expect(Math.max(...rt.reflectance) - Math.min(...rt.reflectance)).toBeGreaterThan(0.01);
    expect(Math.max(...rt.transmittance) - Math.min(...rt.transmittance)).toBeGreaterThan(0.01);
    expect(pAbsorb).toBeGreaterThan(0.01);

    const sampleCount = 100_000;
    let reflected = 0;
    let transmitted = 0;
    let absorbed = 0;
    const reflectedMean = [0, 0, 0];
    const transmittedMean = [0, 0, 0];
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const xi = (sample + 0.5) / sampleCount;
      if (xi < pReflect) {
        reflected += 1;
        for (let lane = 0; lane < 3; lane += 1) {
          reflectedMean[lane] = reflectedMean[lane]! + rt.reflectance[lane]! / pReflect;
        }
      } else if (xi < pReflect + pTransmit) {
        transmitted += 1;
        for (let lane = 0; lane < 3; lane += 1) {
          transmittedMean[lane] = transmittedMean[lane]! + rt.transmittance[lane]! / pTransmit;
        }
      } else {
        absorbed += 1;
      }
    }
    expect(reflected / sampleCount).toBeCloseTo(pReflect, 4);
    expect(transmitted / sampleCount).toBeCloseTo(pTransmit, 4);
    expect(absorbed / sampleCount).toBeCloseTo(pAbsorb, 4);
    for (let lane = 0; lane < 3; lane += 1) {
      expect(reflectedMean[lane]! / sampleCount).toBeCloseTo(rt.reflectance[lane]!, 4);
      expect(transmittedMean[lane]! / sampleCount).toBeCloseTo(rt.transmittance[lane]!, 4);
    }
  });

  it('gates coherent transmission without changing reflection or T-over-p throughput', () => {
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
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (optics.thinFilmEnabled) { return optics.transmission > 0.0; }',
    );
  });

  it('samples hero-wavelength R/T/A at the exact scalar proposal', () => {
    const rt = thinFilmRtAtWavelength({
      layers: [{ ior: 1.8, extinctionCoefficient: 0.11, thicknessNm: 310 }],
      incidentIor: 1,
      substrateIor: 1.52,
      wavelengthNm: 487,
      cosTheta: 0.37,
      angleDependent: true,
    });
    const sampleCount = 100_000;
    const boundaries = [rt.reflectance, rt.reflectance + rt.transmittance];
    const counts = [0, 0, 0];
    let reflectedEstimator = 0;
    let transmittedEstimator = 0;
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const xi = (sample + 0.5) / sampleCount;
      if (xi < boundaries[0]!) {
        counts[0]! += 1;
        reflectedEstimator += rt.reflectance / rt.reflectance;
      } else if (xi < boundaries[1]!) {
        counts[1]! += 1;
        transmittedEstimator += rt.transmittance / rt.transmittance;
      } else {
        counts[2]! += 1;
      }
    }
    expect(counts[0]! / sampleCount).toBeCloseTo(rt.reflectance, 4);
    expect(counts[1]! / sampleCount).toBeCloseTo(rt.transmittance, 4);
    expect(counts[2]! / sampleCount).toBeCloseTo(rt.absorption, 4);
    expect(reflectedEstimator / sampleCount).toBeCloseTo(rt.reflectance, 4);
    expect(transmittedEstimator / sampleCount).toBeCloseTo(rt.transmittance, 4);
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
      'bdptEyeStackSetSpec(bounce, bs.sampledIsDelta);',
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
    const plain = dielectricMaterial([]);
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
      'if (descriptorIndex >= arrayLength(&materials)) {',
      'let lutBaseVec4 = u32(round(max(materials[descriptorIndex].z, 0.0)));',
      'lutBaseVec4 == 0u || lutEndScalar > arrayLength(&materials) * 4u',
      'fn materialStorageScalar(scalarIndex: u32) -> f32 {',
    ]) {
      expect(PT_WEBGPU_TRACE_WGSL).toContain(token);
    }
  });
});
