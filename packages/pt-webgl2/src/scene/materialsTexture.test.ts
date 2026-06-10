// GPU-free golden for the materials packer. Mirrors pt-webgpu's
// `materialPackingCoreEquivalence` style: pack a known `@vitrum/core` MaterialSpec
// and assert the EXACT texel values at the load-bearing offsets the fork's GLSL
// `material_struct` decoder reads (verified against `MaterialsTexture.js` +
// `material_struct.glsl.js`). Any divergence here is a real render bug.

import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packMaterialsTexture, MATERIAL_PIXELS } from './materialsTexture.js';

/** Float offset of pixel `s`, channel `c` (0=r,1=g,2=b,3=a) within material `mi`. */
function texel(mi: number, s: number, c: number): number {
  return mi * MATERIAL_PIXELS * 4 + s * 4 + c;
}

describe('packMaterialsTexture — 93px RGBA32F byte layout', () => {
  it('exposes the verified MATERIAL_PIXELS constant', () => {
    // D3 (2026-06-10): fork base 85 + texels 85..92 (ao/light/bump ids + scalars
    // + envMapIntensity at 85/86, their transforms at 87..92). Single-sourced with
    // every GLSL fetch site via glsl/shader/structs/materialStride.js — see
    // materialStrideParity.test.ts for the packer↔shader guard.
    expect(MATERIAL_PIXELS).toBe(93);
  });

  it('packs a known MaterialSpec to the exact load-bearing texels', () => {
    const m: MaterialSpec = {
      baseColor: [0.25, 0.5, 0.75],
      roughness: 0.4,
      metallic: 0.6,
      emissive: [0.1, 0.2, 0.3],
      emissiveIntensity: 2.0,
      ior: 1.45,
      transmission: 0.9,
      opacity: 0.8,
      alphaMode: 'mask',
      alphaCutoff: 0.33,
    };

    const out = packMaterialsTexture([m]);
    expect(out.kind).toBe('rgba32f');
    expect(out.materialCount).toBe(1);
    // dim = ceil(sqrt(85)) = 10 → backing data is 10*10*4 = 400 floats.
    expect(out.dim).toBe(10);
    expect(out.data.length).toBe(out.dim * out.dim * 4);

    const d = out.data;

    // s0.rgb = baseColor; s0.a = map id = -1 (no atlas).
    expect(d[texel(0, 0, 0)]).toBeCloseTo(0.25, 6);
    expect(d[texel(0, 0, 1)]).toBeCloseTo(0.5, 6);
    expect(d[texel(0, 0, 2)]).toBeCloseTo(0.75, 6);
    expect(d[texel(0, 0, 3)]).toBe(-1);

    // s1 = (metalness, metalnessMap=-1, roughness, roughnessMap=-1).
    expect(d[texel(0, 1, 0)]).toBeCloseTo(0.6, 6);
    expect(d[texel(0, 1, 1)]).toBe(-1);
    expect(d[texel(0, 1, 2)]).toBeCloseTo(0.4, 6);
    expect(d[texel(0, 1, 3)]).toBe(-1);

    // s2 = (ior, transmission, transmissionMap=-1, emissiveIntensity).
    expect(d[texel(0, 2, 0)]).toBeCloseTo(1.45, 6);
    expect(d[texel(0, 2, 1)]).toBeCloseTo(0.9, 6);
    expect(d[texel(0, 2, 2)]).toBe(-1);
    expect(d[texel(0, 2, 3)]).toBeCloseTo(2.0, 6);

    // s3.rgb = emissive; s3.a = emissiveMap id = -1.
    expect(d[texel(0, 3, 0)]).toBeCloseTo(0.1, 6);
    expect(d[texel(0, 3, 1)]).toBeCloseTo(0.2, 6);
    expect(d[texel(0, 3, 2)]).toBeCloseTo(0.3, 6);
    expect(d[texel(0, 3, 3)]).toBe(-1);

    // s13 = (alphaMap=-1, opacity, alphaTest, side).
    expect(d[texel(0, 13, 0)]).toBe(-1);
    expect(d[texel(0, 13, 1)]).toBeCloseTo(0.8, 6);
    expect(d[texel(0, 13, 2)]).toBeCloseTo(0.33, 6); // alphaMode 'mask' → alphaCutoff
    // side: transmission>0 and material is thin-film by default (thickness 0,
    // attenuationDistance Infinity) ⇒ NOT the side==0 glass branch ⇒ FrontSide=1.
    expect(d[texel(0, 13, 3)]).toBe(1);
  });

  it('glass (finite attenuationDistance + transmission>0) gets side==0', () => {
    const glass: MaterialSpec = {
      baseColor: [0.95, 0.97, 1.0],
      roughness: 0.05,
      metallic: 0.0,
      transmission: 0.95,
      ior: 1.5,
      attenuationColor: [0.6, 0.8, 0.95],
      attenuationDistance: 1.2, // finite → not thin-film → glass side rule applies
    };
    const d = packMaterialsTexture([glass]).data;
    // isThinFilm == false (attenuationDistance finite) AND transmission>0 ⇒ side 0.
    expect(d[texel(0, 13, 3)]).toBe(0);
    // s12 = attenuationColor.rgb / attenuationDistance.
    expect(d[texel(0, 12, 0)]).toBeCloseTo(0.6, 6);
    expect(d[texel(0, 12, 1)]).toBeCloseTo(0.8, 6);
    expect(d[texel(0, 12, 2)]).toBeCloseTo(0.95, 6);
    expect(d[texel(0, 12, 3)]).toBeCloseTo(1.2, 6);
  });

  it('opaque defaults: ior 1.5, attenuationColor white, attenuationDistance Infinity, side front', () => {
    const opaque: MaterialSpec = { baseColor: [0.8, 0.8, 0.8], roughness: 1.0, metallic: 0.0 };
    const d = packMaterialsTexture([opaque]).data;
    expect(d[texel(0, 2, 0)]).toBeCloseTo(1.5, 6); // default ior
    expect(d[texel(0, 12, 0)]).toBe(1.0); // attenuationColor default white
    expect(d[texel(0, 12, 1)]).toBe(1.0);
    expect(d[texel(0, 12, 2)]).toBe(1.0);
    expect(d[texel(0, 12, 3)]).toBe(Infinity); // attenuationDistance default
    expect(d[texel(0, 11, 2)]).toBe(1); // isThinFilm true (thickness 0 + dist Infinity)
    expect(d[texel(0, 13, 3)]).toBe(1); // FrontSide (no transmission)
  });

  it('scatteringCoefficient sets the TRANSLUCENT flag bit (s14.a) and s15 SSS drives', () => {
    const sss: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      scatteringCoefficient: 2.5,
      scatteringAnisotropy: 0.3,
      scatteringCoefficientRGB: [0.7, 0.8, 0.9],
    };
    const d = packMaterialsTexture([sss]).data;
    const flags = d[texel(0, 14, 3)]!;
    expect((flags & (1 << 4)) !== 0).toBe(true); // TRANSLUCENT_BIT
    // s15 = sssSigmaT / sssAnisotropyG / dispersionStrength / thinFilmEnabled.
    expect(d[texel(0, 15, 0)]).toBeCloseTo(2.5, 6);
    expect(d[texel(0, 15, 1)]).toBeCloseTo(0.3, 6);
    expect(d[texel(0, 15, 3)]).toBe(0); // no thin-film stack
    // s16 = sssAlbedo.rgb / thinFilmLayerCount.
    expect(d[texel(0, 16, 0)]).toBeCloseTo(0.7, 6);
    expect(d[texel(0, 16, 1)]).toBeCloseTo(0.8, 6);
    expect(d[texel(0, 16, 2)]).toBeCloseTo(0.9, 6);
    expect(d[texel(0, 16, 3)]).toBe(0); // 0 layers
  });

  it('thin-film stack populates layer payload (s28+) + feature flags (s17.a)', () => {
    const film: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.1,
      metallic: 0,
      thinFilmStack: {
        layers: [
          { ior: 1.4, thicknessNm: 250, extinctionCoefficient: 0.01 },
          { ior: 2.0, thicknessNm: 120 },
        ],
        incidentIor: 1.0,
        angleDependent: true,
      },
    };
    const d = packMaterialsTexture([film]).data;
    // s15.a thinFilmEnabled = 1; s16.a layerCount = 2.
    expect(d[texel(0, 15, 3)]).toBe(1);
    expect(d[texel(0, 16, 3)]).toBe(2);
    // s17 = thinFilmIncidentIor / angleDependent / 0 / packedFeatureFlags.
    expect(d[texel(0, 17, 0)]).toBeCloseTo(1.0, 6);
    expect(d[texel(0, 17, 1)]).toBe(1.0); // angleDependent true
    // layer 0 payload at sample 28 (float offset 28*4).
    expect(d[texel(0, 28, 0)]).toBeCloseTo(1.4, 6);
    expect(d[texel(0, 28, 1)]).toBeCloseTo(250, 6);
    expect(d[texel(0, 28, 2)]).toBeCloseTo(0.01, 6);
    // layer 1 starts 3 floats later (no extinction → default 0).
    expect(d[texel(0, 28, 3)]).toBeCloseTo(2.0, 6); // layer1.ior spills into s28.a
    expect(d[texel(0, 29, 0)]).toBeCloseTo(120, 6);
    expect(d[texel(0, 29, 1)]).toBeCloseTo(0.0, 6);
  });

  it('spectralAttenuation fills the 32-sample grid (s20..) + sets feature bit 1', () => {
    const spec: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.1,
      metallic: 0,
      transmission: 1.0,
      ior: 1.5,
      attenuationDistance: 1.0,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 780,
        values: new Float32Array([0.0, 1.0]), // ramp 0→1 across the band
      },
    };
    const d = packMaterialsTexture([spec]).data;
    // packedFeatureFlags bit 0 (hasSpectral) set in s17.a.
    const featureFlags = d[texel(0, 17, 3)]!;
    expect((featureFlags & 1) !== 0).toBe(true);
    // sample 20 = first spectral grid sample (t=0 → value 0).
    expect(d[texel(0, 20, 0)]).toBeCloseTo(0.0, 6);
    // sample 27.a = last grid sample (t=1 → value 1).
    expect(d[texel(0, 27, 3)]).toBeCloseTo(1.0, 6);
  });

  // H49 — specularColor and specularIntensity (glTF KHR_materials_specular).
  // The packer previously hardcoded DEFAULT_SPECULAR_COLOR=[1,1,1] and 1.0 with
  // false "core has no field" comments; core/material.ts:225,232 has both fields.
  it('H49: specularColor and specularIntensity are packed from the MaterialSpec', () => {
    const m: MaterialSpec = {
      baseColor: [0.8, 0.8, 0.8],
      roughness: 0.5,
      metallic: 0.0,
      specularColor: [1, 0, 0],
      specularIntensity: 0.5,
    };
    const d = packMaterialsTexture([m]).data;
    // s10.rgb = specularColor; s10.a = specularColorMap id = -1.
    expect(d[texel(0, 10, 0)]).toBeCloseTo(1.0, 6); // red
    expect(d[texel(0, 10, 1)]).toBeCloseTo(0.0, 6);
    expect(d[texel(0, 10, 2)]).toBeCloseTo(0.0, 6);
    expect(d[texel(0, 10, 3)]).toBe(-1);
    // s11.r = specularIntensity; s11.g = specularIntensityMap=-1.
    expect(d[texel(0, 11, 0)]).toBeCloseTo(0.5, 6);
    expect(d[texel(0, 11, 1)]).toBe(-1);
  });

  it('H49 defaults: absent specularColor → [1,1,1]; absent specularIntensity → 1.0', () => {
    const m: MaterialSpec = { baseColor: [0.8, 0.8, 0.8], roughness: 0.5, metallic: 0.0 };
    const d = packMaterialsTexture([m]).data;
    expect(d[texel(0, 10, 0)]).toBe(1.0);
    expect(d[texel(0, 10, 1)]).toBe(1.0);
    expect(d[texel(0, 10, 2)]).toBe(1.0);
    expect(d[texel(0, 11, 0)]).toBe(1.0);
  });

  it('multiple materials are packed at their MATERIAL_PIXELS-strided offsets', () => {
    const a: MaterialSpec = { baseColor: [1, 0, 0], roughness: 1, metallic: 0 };
    const b: MaterialSpec = { baseColor: [0, 1, 0], roughness: 1, metallic: 0 };
    const out = packMaterialsTexture([a, b]);
    expect(out.materialCount).toBe(2);
    // dim = ceil(sqrt(170)) = 14 → 14*14*4 = 784 floats.
    expect(out.dim).toBe(14);
    // material 0 color.
    expect(out.data[texel(0, 0, 0)]).toBe(1);
    expect(out.data[texel(0, 0, 1)]).toBe(0);
    // material 1 color, at the second 340-float block.
    expect(out.data[texel(1, 0, 0)]).toBe(0);
    expect(out.data[texel(1, 0, 1)]).toBe(1);
  });
});
