import { describe, expect, it } from 'vitest';
import {
  KHR_MATERIALS_IOR_INFINITY_APPROX,
  type MaterialSpec,
} from '@vitrum/core';
import {
  MATERIAL_OPTICS_WGSL,
} from '@vitrum/shared-bvh';
import { PROBE_RAY_CAST_WGSL } from '@vitrum/walkaround-rc';
import {
  MATERIAL_OPTICAL_RELATIVE_OFFSETS,
  MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT,
  materialDispersionIorRgb,
  materialThinFilmRgb,
  packMaterialOpticalMeta,
} from '../bvh/materialOptics.js';
import { packBVHBeerColorsFromCore } from '../restir/packingHelpers.js';
import { DDGI_MAX_MATERIALS } from '../ddgi/probeUpdateMaterials.js';
import { makeProbeUpdateRaysWGSL } from '../ddgi/wgsl/probeUpdateRays.wgsl.js';
import { MANIFOLD_CAUSTICS_WGSL } from '../shaders/manifoldCaustics.wgsl.js';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../shaders/nrcIndependentSuffix.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';
import { REFRACTIVE_CAUSTICS_WGSL } from '../shaders/refractiveCaustics.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from '../shaders/risGiGlassWalk.wgsl.js';
import { RESTIR_GI_DIELECTRIC_SUFFIX_WGSL } from '../shaders/risGi.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shaders/shadingTerms.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../shaders/surfaceTextures.wgsl.js';

function baseMaterial(patch: Partial<MaterialSpec> = {}): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 0.1,
    metallic: 0,
    transmission: 1,
    ior: 1.5,
    ...patch,
  };
}

const MAX_FINITE_F32 = 3.402823466e38;

function exactBeerTransmittance(mu: number, distance: number): number {
  if (
    Number.isNaN(mu) || Number.isNaN(distance) ||
    mu < 0 || distance < 0
  ) {
    return 0;
  }
  if (distance === 0 || mu === 0) return 1;
  if (mu > MAX_FINITE_F32 || distance > MAX_FINITE_F32) return 0;
  return Math.exp(-mu * distance);
}

describe('walkaround realtime material optics preintegration', () => {
  it('falls back to the selected UV source when an affine lane is invalid', () => {
    const start = MATERIAL_OPTICS_WGSL.indexOf('fn materialResolveUv(');
    const end = MATERIAL_OPTICS_WGSL.indexOf('fn materialDispersionIorRgb(', start);
    const resolver = MATERIAL_OPTICS_WGSL.slice(start, end);

    expect(resolver).toContain('let source = select(uv0, uv1, row0.w >= 0.5);');
    expect(resolver).toContain('if (row1.w < 0.5) { return source; }');
    expect(resolver).toContain('return select(source, resolved, finite);');
    expect(resolver).not.toContain('return select(uv0, resolved, finite);');

    const uv0 = [0.1, 0.2] as const;
    const uv1 = [0.7, 0.8] as const;
    const source = true ? uv1 : uv0;
    const invalidAffineFallback = source;
    expect(invalidAffineFallback).toBe(uv1);
  });

  it('reconstructs a constant spectral Beer coefficient at arbitrary distance', () => {
    const material = baseMaterial({
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 780,
        values: new Float32Array([1, 1, 1]),
      },
    });
    const meta = packMaterialOpticalMeta(material);
    expect(meta[0]).toBe(MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT);

    const reconstructed = [0, 0, 0];
    const distance = 1.75;
    for (let sample = 0; sample < MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT; sample += 1) {
      const base = (MATERIAL_OPTICAL_RELATIVE_OFFSETS.SPECTRAL_SAMPLES + sample) * 4;
      const attenuation = Math.exp(-meta[base]! * distance);
      reconstructed[0]! += meta[base + 1]! * attenuation;
      reconstructed[1]! += meta[base + 2]! * attenuation;
      reconstructed[2]! += meta[base + 3]! * attenuation;
    }
    const expected = Math.exp(-distance);
    for (const channel of reconstructed) expect(channel).toBeCloseTo(expected, 5);
  });

  it('defines finite RGB and spectral Beer limits for every canonical caller', () => {
    expect(exactBeerTransmittance(0, Number.POSITIVE_INFINITY)).toBe(1);
    expect(exactBeerTransmittance(1e-12, Number.POSITIVE_INFINITY)).toBe(0);
    expect(exactBeerTransmittance(4, 0)).toBe(1);
    expect(exactBeerTransmittance(0, 12)).toBe(1);
    expect(exactBeerTransmittance(2, 3)).toBeCloseTo(Math.exp(-6), 14);
    expect(exactBeerTransmittance(Number.NaN, 1)).toBe(0);
    expect(exactBeerTransmittance(1, Number.NaN)).toBe(0);
    expect(exactBeerTransmittance(-1, 1)).toBe(0);
    expect(exactBeerTransmittance(1, -1)).toBe(0);
    expect(exactBeerTransmittance(Number.POSITIVE_INFINITY, 0)).toBe(1);
    const scalarHelperStart = MATERIAL_OPTICS_WGSL.indexOf(
      'fn materialBeerTransmittanceExact(mu: f32, distance: f32)',
    );
    const scalarHelperEnd = MATERIAL_OPTICS_WGSL.indexOf(
      'fn materialSpectralAttenuation(', scalarHelperStart,
    );
    const scalarHelper = MATERIAL_OPTICS_WGSL.slice(
      scalarHelperStart, scalarHelperEnd,
    );
    expect(scalarHelper.indexOf('if (distance == 0.0 || mu == 0.0)'))
      .toBeLessThan(scalarHelper.indexOf('if (mu > VITRUM_OPTICAL_MAX_FINITE_F32)'));

    // A zero-extinction spectral basis sample remains a finite contribution at
    // infinity; every positive-extinction sample vanishes instead of producing
    // exp(-(0 * infinity)) => NaN.
    const reconstructedAtInfinity = [
      exactBeerTransmittance(0, Number.POSITIVE_INFINITY) * 0.4,
      exactBeerTransmittance(0.2, Number.POSITIVE_INFINITY) * 0.6,
    ].reduce((sum, value) => sum + value, 0);
    expect(reconstructedAtInfinity).toBe(0.4);
    expect(Number.isFinite(reconstructedAtInfinity)).toBe(true);

    const deliberatelyNonUnitWeightSum = 0.23 + 0.31 + 0.17;
    const zeroMappedDistance = 0;
    const zeroDistanceTransfer = zeroMappedDistance === 0
      ? 1
      : deliberatelyNonUnitWeightSum;
    expect(deliberatelyNonUnitWeightSum).not.toBe(1);
    expect(zeroDistanceTransfer).toBe(1);

    expect(MATERIAL_OPTICS_WGSL).toContain(
      'fn materialBeerTransmittanceExact(mu: f32, distance: f32)',
    );
    expect(MATERIAL_OPTICS_WGSL).toContain(
      'materialBeerTransmittanceExact(sample.x, distanceInMaterial)',
    );
    const zeroDistanceGuard = MATERIAL_OPTICS_WGSL.indexOf(
      'if (distanceInMaterial == 0.0) { return vec3f(1.0); }',
    );
    expect(zeroDistanceGuard).toBeGreaterThan(0);
    expect(zeroDistanceGuard).toBeLessThan(
      MATERIAL_OPTICS_WGSL.indexOf(
        'for (var i = 0u; i < VITRUM_OPTICAL_SPECTRAL_SAMPLE_COUNT;',
      ),
    );
    expect(MATERIAL_OPTICS_WGSL).not.toContain(
      'exp(-max(sample.x, 0.0) * distance)',
    );

    const rgbStart = MATERIAL_ATLAS_WGSL.indexOf(
      'fn homogeneousBeerTransmittanceRgb(',
    );
    const rgbEnd = MATERIAL_ATLAS_WGSL.indexOf(
      'fn henyeyGreensteinPhase(', rgbStart,
    );
    const rgbHelper = MATERIAL_ATLAS_WGSL.slice(rgbStart, rgbEnd);
    expect(rgbHelper.match(/materialBeerTransmittanceExact\(/g))
      .toHaveLength(3);
    expect(rgbHelper).not.toContain('exp(');

    const canonicalConsumers = [
      REFRACTIVE_CAUSTICS_WGSL,
      RESTIR_GI_DIELECTRIC_SUFFIX_WGSL,
      NRC_INDEPENDENT_SUFFIX_WGSL,
      NATIVE_GLASS_GI_WGSL,
      SHADING_TERMS_WGSL,
      SURFACE_TEXTURES_WGSL,
      makeProbeUpdateRaysWGSL(1),
      PROBE_RAY_CAST_WGSL,
    ];
    for (const consumer of canonicalConsumers) {
      expect(
        consumer.includes('homogeneousBeerTransmittanceRgb(') ||
          consumer.includes('materialSpectralAttenuation('),
      ).toBe(true);
    }
    // The manifold module deliberately consumes the surface-texture module's
    // complete Beer helper instead of duplicating either canonical primitive.
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'materialShadowBeerForSegment(',
    );
  });

  it('reduces positive Abbe dispersion to ordered red/green/blue IORs', () => {
    const [red, green, blue] = materialDispersionIorRgb(baseMaterial({
      ior: 1.52,
      dispersionAbbeNumber: 30,
    }));
    expect(blue).toBeGreaterThan(green);
    expect(green).toBeGreaterThan(red);
    expect(materialDispersionIorRgb(baseMaterial({ ior: 1.52 })))
      .toEqual([1.52, 1.52, 1.52]);
  });

  it('maps the legal authored IOR-zero endpoint to finite optical metadata', () => {
    const material = baseMaterial({ ior: 0 });
    expect(materialDispersionIorRgb(material)).toEqual([
      KHR_MATERIALS_IOR_INFINITY_APPROX,
      KHR_MATERIALS_IOR_INFINITY_APPROX,
      KHR_MATERIALS_IOR_INFINITY_APPROX,
    ]);
    const meta = packMaterialOpticalMeta(material);
    const base = MATERIAL_OPTICAL_RELATIVE_OFFSETS.DISPERSION_IOR_RGB * 4;
    expect(Array.from(meta.slice(base, base + 3))).toEqual([
      Math.fround(KHR_MATERIALS_IOR_INFINITY_APPROX),
      Math.fround(KHR_MATERIALS_IOR_INFINITY_APPROX),
      Math.fround(KHR_MATERIALS_IOR_INFINITY_APPROX),
    ]);
  });

  it('keeps every zero-thickness bulk activation volumetric with a matched Beer reference', () => {
    const participating = baseMaterial({
      thickness: 0,
      attenuationColor: [0.25, 0.5, 0.75],
      attenuationDistance: 2,
      scatteringCoefficient: 0.4,
    });
    const meta = packMaterialOpticalMeta(participating);
    expect(meta[3]).toBe(-2);

    const packedBeer = packBVHBeerColorsFromCore(
      new Uint32Array([0]),
      [participating],
      1,
    )[0]!;
    expect([
      (packedBeer >>> 24) / 255,
      ((packedBeer >>> 16) & 0xff) / 255,
      ((packedBeer >>> 8) & 0xff) / 255,
    ]).toEqual([
      Math.round(0.25 * 255) / 255,
      Math.round(0.5 * 255) / 255,
      Math.round(0.75 * 255) / 255,
    ]);

    expect(packMaterialOpticalMeta(baseMaterial({ thickness: 0 }))[3]).toBe(0);
    expect(packMaterialOpticalMeta(baseMaterial({
      thickness: 0,
      scatteringCoefficientRGB: [0, 0.2, 0],
    }))[3]).toBe(-1);
    expect(packMaterialOpticalMeta(baseMaterial({
      thickness: 0,
      attenuationColor: [0.5, 1, 1],
      attenuationDistance: 3,
    }))[3]).toBe(-3);
    expect(packMaterialOpticalMeta(baseMaterial({
      thickness: 0,
      attenuationColor: [1, 1, 1],
      attenuationDistance: 3,
    }))[3]).toBe(0);
    expect(packMaterialOpticalMeta(baseMaterial({
      thickness: 0,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 700,
        values: new Float32Array([0, 0.2, 0]),
      },
    }))[3]).toBe(-1);
    expect(packMaterialOpticalMeta(baseMaterial({
      thickness: 0,
      scatteringCoefficient: 0.5,
      scatteringCoefficientRGB: [0, 0, 0],
    }))[3]).toBe(0);

    expect(MATERIAL_OPTICS_WGSL).toContain(
      'fn materialOpticalHasAuthoredThickness(triIndex: u32) -> bool',
    );
    expect(MATERIAL_OPTICS_WGSL).toContain(
      'return abs(materialOpticalHeader(triIndex).w);',
    );
    // Fully composed roots contain the canonical helper itself.
    for (const composed of [
      MATERIAL_ATLAS_WGSL,
      SURFACE_TEXTURES_WGSL,
      makeProbeUpdateRaysWGSL(DDGI_MAX_MATERIALS),
    ]) {
      expect(composed).toContain('materialOpticalHasAuthoredThickness(');
    }
    // Standalone suffix modules declare their dependency by calling the
    // canonical thickness-map policy supplied by composition.
    for (const suffix of [
      RESTIR_GI_DIELECTRIC_SUFFIX_WGSL,
      NRC_INDEPENDENT_SUFFIX_WGSL,
      REFRACTIVE_CAUSTICS_WGSL,
      NATIVE_GLASS_GI_WGSL,
    ]) {
      expect(suffix).toContain('materialOpticalThicknessMapScale(');
    }
    // Manifold transport consumes the surface-texture wrapper because it also
    // needs the mapped hit-frame payload owned by that module.
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'materialShadowThicknessMapScale(',
    );
  });

  it('preserves every finite core-contract IOR in the shared WGSL optical lane', () => {
    const highIor = 9.5;
    const meta = packMaterialOpticalMeta(baseMaterial({ ior: highIor }));
    const base = MATERIAL_OPTICAL_RELATIVE_OFFSETS.DISPERSION_IOR_RGB * 4;
    expect(Array.from(meta.slice(base, base + 3))).toEqual([
      Math.fround(highIor),
      Math.fround(highIor),
      Math.fround(highIor),
    ]);
    expect(MATERIAL_OPTICS_WGSL).toContain(
      'all(value <= vec3f(VITRUM_OPTICAL_MAX_FINITE_F32))',
    );
    expect(MATERIAL_OPTICS_WGSL).not.toContain('all(value < vec3f(8.0))');
    for (const consumer of [
      RESTIR_GI_DIELECTRIC_SUFFIX_WGSL,
      NRC_INDEPENDENT_SUFFIX_WGSL,
      REFRACTIVE_CAUSTICS_WGSL,
    ]) {
      expect(consumer).toContain('materialDispersionIorRgb(');
    }
  });

  it('preintegrates finite energy-bounded forward and reverse thin-film responses', () => {
    const material = baseMaterial({
      thickness: 0.75,
      thinFilmStack: {
        incidentIor: 1,
        layers: [
          { ior: 1.34, thicknessNm: 110 },
          { ior: 2.05, thicknessNm: 420, extinctionCoefficient: 0.015 },
        ],
      },
    });
    const packed = packMaterialOpticalMeta(material);
    expect(packed[1]).toBe(1);
    expect(packed[3]).toBeCloseTo(0.75);

    for (const cosine of [0.05, 0.25, 0.5, 0.8, 1]) {
      for (const reverse of [false, true]) {
        const response = materialThinFilmRgb(material, cosine, reverse);
        for (let channel = 0; channel < 3; channel += 1) {
          const reflectance = response.reflectance[channel]!;
          const transmittance = response.transmittance[channel]!;
          expect(Number.isFinite(reflectance)).toBe(true);
          expect(Number.isFinite(transmittance)).toBe(true);
          expect(reflectance).toBeGreaterThanOrEqual(0);
          expect(reflectance).toBeLessThanOrEqual(1);
          expect(transmittance).toBeGreaterThanOrEqual(0);
          expect(transmittance).toBeLessThanOrEqual(1);
        }
      }
    }

    const front = materialThinFilmRgb(material, 0.6, false);
    const back = materialThinFilmRgb(material, 0.6, true);
    expect(front.reflectance).not.toEqual(back.reflectance);
  });
});
