import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  MATERIAL_OPTICAL_RELATIVE_OFFSETS,
  MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT,
  materialDispersionIorRgb,
  materialThinFilmRgb,
  packMaterialOpticalMeta,
} from '../bvh/materialOptics.js';

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

describe('walkaround realtime material optics preintegration', () => {
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
