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

describe('packMaterialsTexture — 95px RGBA32F byte layout', () => {
  it('exposes the verified MATERIAL_PIXELS constant', () => {
    // D3 (2026-06-10): fork base 85 + texels 85..92 (ao/light/bump ids + scalars
    // + envMapIntensity at 85/86, their transforms at 87..92) + alphaMap transform
    // at 93/94. Single-sourced with every GLSL fetch site via
    // glsl/shader/structs/materialStride.js — see materialStrideParity.test.ts for
    // the packer↔shader guard.
    expect(MATERIAL_PIXELS).toBe(95);
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
    // dim = ceil(sqrt(95)) = 10 → backing data is 10*10*4 = 400 floats.
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

  it('packs primitive-derived castShadow in s14.g, defaulting to true', () => {
    const caster: MaterialSpec = { baseColor: [0.8, 0.8, 0.8], roughness: 1.0, metallic: 0.0 };
    const nonCaster = {
      baseColor: [0.8, 0.8, 0.8],
      roughness: 1.0,
      metallic: 0.0,
      castShadow: false,
    } as MaterialSpec & { castShadow: false };
    const d = packMaterialsTexture([caster, nonCaster]).data;
    expect(d[texel(0, 14, 1)]).toBe(1);
    expect(d[texel(1, 14, 1)]).toBe(0);
  });

  it('packs the vertex-color enable flag in s14.b for materials used by colored primitives', () => {
    const plain: MaterialSpec = { baseColor: [1, 1, 1], roughness: 1.0, metallic: 0.0 };
    const colored: MaterialSpec = { baseColor: [0.8, 0.8, 0.8], roughness: 0.5, metallic: 0.0 };
    const d = packMaterialsTexture(
      [plain, colored],
      undefined,
      { vertexColorMaterialIds: new Set([1]) },
    ).data;

    expect(d[texel(0, 14, 2)]).toBe(0);
    expect(d[texel(1, 14, 2)]).toBe(1);
  });

  // Contract-honesty: emissiveIntensity default must be 1.0, not 0.0.
  // pt-webgpu (materialTextures.ts) and walkaround-hybrid both default to 1.0;
  // a host that sets emissive:[r,g,b] without emissiveIntensity expects a visible
  // emitter. Pinned here so a regression renders BLACK, not just changes a number.
  it('emissiveIntensity defaults to 1.0 when absent (matches pt-webgpu + walkaround)', () => {
    const emissive: MaterialSpec = {
      baseColor: [0, 0, 0],
      roughness: 1.0,
      metallic: 0.0,
      emissive: [0.8, 0.6, 0.4],
      // emissiveIntensity intentionally absent
    };
    const d = packMaterialsTexture([emissive]).data;
    // s2.a = emissiveIntensity (sample 2, channel 3).
    expect(d[texel(0, 2, 3)]).toBe(1.0);
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

  it('shadingModel=unlit sets the UNLIT flag bit (s14.a)', () => {
    const unlit: MaterialSpec = {
      baseColor: [0.2, 0.4, 0.8],
      roughness: 0.5,
      metallic: 0,
      shadingModel: 'unlit',
    };
    const d = packMaterialsTexture([unlit]).data;
    const flags = d[texel(0, 14, 3)]!;
    expect((flags & (1 << 5)) !== 0).toBe(true);
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

  // D3 — ao/lightMap/bumpMap transforms at texels 87/89/91 (item 26).
  //
  // The packer writes ao/lightMap/bumpMap id + scalars at texels 85/86, and their
  // UV-transform mat3s at texels 87 (aoMapTransform), 89 (lightMapTransform), 91
  // (bumpMapTransform), 2 texels per mat3 (see writeTransform + readTextureTransform
  // in material_struct.glsl.js). The GLSL decoder reads:
  //   m.aoMapTransform   = readTextureTransform(tex, i + 87u)
  //   m.lightMapTransform = readTextureTransform(tex, i + 89u)
  //   m.bumpMapTransform  = readTextureTransform(tex, i + 91u)
  //
  // All prior tests passed layerOf=undefined → aoLayer/lightMapLayer/bumpLayer are
  // always -1 → writeTransform is never called → texels 87..92 stay 0.  This test
  // exercises the non-trivial path: supply a layerOf map that assigns real atlas
  // layers and non-identity UvTransforms, then assert the exact float values that
  // readTextureTransform will read on the GPU.
  //
  // writeTransform encoding (verified against material_struct.glsl.js):
  //   texel k   row1: (sx·cos, sx·sin, offsetX, 0)
  //   texel k+1 row2: (−sy·sin, sy·cos, offsetY, 0)
  // readTextureTransform unpacks:
  //   col0 = (row1.r, row2.r, 0) = (sx·cos, −sy·sin, 0)
  //   col1 = (row1.g, row2.g, 0) = (sx·sin,  sy·cos, 0)
  //   col2 = (row1.b, row2.b, 1) = (offsetX, offsetY, 1)
  it('D3 item26: ao/lightMap/bumpMap transforms are packed at texels 87/89/91', () => {
    // Three distinct opaque handles — layerOf maps each to a real layer.
    const aoHandle    = {};
    const lightHandle = {};
    const bumpHandle  = {};

    const layerOf = new Map<unknown, number>([
      [aoHandle,    2],
      [lightHandle, 4],
      [bumpHandle,  7],
    ]);

    // Non-identity transforms for each map so the assertion can distinguish them.
    // aoMap:    scale=[2,3], offset=[0.1,0.2], rotation=0 → row1=(2,0,0.1,0), row2=(0,3,0.2,0)
    // lightMap: scale=[1,1], offset=[0,0],     rotation=Math.PI/2 →
    //           row1=(cos(π/2)≈0, sin(π/2)≈1, 0, 0), row2=(-sin(π/2)≈-1, cos(π/2)≈0, 0, 0)
    // bumpMap:  scale=[0.5,0.5], offset=[0.25,0.75], rotation=0 →
    //           row1=(0.5, 0, 0.25, 0), row2=(0, 0.5, 0.75, 0)
    const m: MaterialSpec = {
      baseColor: [0.5, 0.5, 0.5],
      roughness: 0.5,
      metallic: 0.0,
      aoMap:      { handle: aoHandle,    transform: { scale: [2, 3],     offset: [0.1, 0.2] } },
      lightMap:   { handle: lightHandle, transform: { rotation: Math.PI / 2 } },
      bumpMap:    { handle: bumpHandle,  transform: { scale: [0.5, 0.5], offset: [0.25, 0.75] } },
    };

    const out = packMaterialsTexture([m], layerOf);
    const d = out.data;

    // ── Texel 85: aoMap / lightMap / bumpMap ids + envMapIntensity ──────────
    expect(d[texel(0, 85, 0)]).toBe(2);  // aoLayer
    expect(d[texel(0, 85, 1)]).toBe(4);  // lightMapLayer
    expect(d[texel(0, 85, 2)]).toBe(7);  // bumpLayer
    expect(d[texel(0, 85, 3)]).toBe(1);  // envMapIntensity default

    // ── Texels 86: aoMapIntensity / lightMapIntensity / bumpScale / pad ─────
    expect(d[texel(0, 86, 0)]).toBe(1);  // aoMapIntensity default
    expect(d[texel(0, 86, 1)]).toBe(1);  // lightMapIntensity default
    expect(d[texel(0, 86, 2)]).toBe(1);  // bumpScale default

    // ── Texels 87/88: aoMapTransform (scale=[2,3], offset=[0.1,0.2], r=0) ──
    // row1 = (sx·cos, sx·sin, offsetX, 0) = (2·1, 2·0, 0.1, 0) = (2, 0, 0.1, 0)
    expect(d[texel(0, 87, 0)]).toBeCloseTo(2,   6);  // row1.r = sx·cos
    expect(d[texel(0, 87, 1)]).toBeCloseTo(0,   6);  // row1.g = sx·sin
    expect(d[texel(0, 87, 2)]).toBeCloseTo(0.1, 6);  // row1.b = offsetX
    expect(d[texel(0, 87, 3)]).toBe(0);               // row1.a = 0 (pad)
    // row2 = (−sy·sin, sy·cos, offsetY, 0) = (0, 3, 0.2, 0)
    expect(d[texel(0, 88, 0)]).toBeCloseTo(0,   6);  // row2.r = −sy·sin
    expect(d[texel(0, 88, 1)]).toBeCloseTo(3,   6);  // row2.g = sy·cos
    expect(d[texel(0, 88, 2)]).toBeCloseTo(0.2, 6);  // row2.b = offsetY
    expect(d[texel(0, 88, 3)]).toBe(0);               // row2.a = 0 (pad)

    // ── Texels 89/90: lightMapTransform (scale=[1,1], offset=[0,0], r=π/2) ─
    // row1 = (cos(π/2), sin(π/2), 0, 0) ≈ (0, 1, 0, 0)
    expect(d[texel(0, 89, 0)]).toBeCloseTo(0,  5);   // cos(π/2) ≈ 0
    expect(d[texel(0, 89, 1)]).toBeCloseTo(1,  6);   // sin(π/2) ≈ 1
    expect(d[texel(0, 89, 2)]).toBeCloseTo(0,  6);   // offsetX = 0
    // row2 = (−sin(π/2), cos(π/2), 0, 0) ≈ (−1, 0, 0, 0)
    expect(d[texel(0, 90, 0)]).toBeCloseTo(-1, 6);   // −sin(π/2) ≈ −1
    expect(d[texel(0, 90, 1)]).toBeCloseTo(0,  5);   // cos(π/2) ≈ 0
    expect(d[texel(0, 90, 2)]).toBeCloseTo(0,  6);   // offsetY = 0

    // ── Texels 91/92: bumpMapTransform (scale=[0.5,0.5], offset=[0.25,0.75], r=0) ─
    // row1 = (0.5, 0, 0.25, 0)
    expect(d[texel(0, 91, 0)]).toBeCloseTo(0.5,  6); // sx·cos
    expect(d[texel(0, 91, 1)]).toBeCloseTo(0,    6); // sx·sin
    expect(d[texel(0, 91, 2)]).toBeCloseTo(0.25, 6); // offsetX
    // row2 = (0, 0.5, 0.75, 0)
    expect(d[texel(0, 92, 0)]).toBeCloseTo(0,    6); // −sy·sin
    expect(d[texel(0, 92, 1)]).toBeCloseTo(0.5,  6); // sy·cos
    expect(d[texel(0, 92, 2)]).toBeCloseTo(0.75, 6); // offsetY
  });

  it('packs alphaMapTransform at texels 93/94 when alphaMap resolves to an atlas layer', () => {
    const alphaHandle = {};
    const layerOf = new Map<unknown, number>([[alphaHandle, 6]]);
    const m: MaterialSpec = {
      baseColor: [0.5, 0.5, 0.5],
      roughness: 0.5,
      metallic: 0.0,
      alphaMap: {
        handle: alphaHandle,
        transform: { scale: [2, 3], offset: [0.1, 0.2] },
      },
    };

    const d = packMaterialsTexture([m], layerOf).data;
    expect(d[texel(0, 13, 0)]).toBe(6); // alphaMap layer id
    expect(d[texel(0, 93, 0)]).toBeCloseTo(2, 6);
    expect(d[texel(0, 93, 1)]).toBeCloseTo(0, 6);
    expect(d[texel(0, 93, 2)]).toBeCloseTo(0.1, 6);
    expect(d[texel(0, 93, 3)]).toBe(0);
    expect(d[texel(0, 94, 0)]).toBeCloseTo(0, 6);
    expect(d[texel(0, 94, 1)]).toBeCloseTo(3, 6);
    expect(d[texel(0, 94, 2)]).toBeCloseTo(0.2, 6);
    expect(d[texel(0, 94, 3)]).toBe(0);
  });

  it('D3 item26: ao/lightMap/bumpMap transforms stay zero when layerOf is undefined (no atlas)', () => {
    // This is the guard for the existing behavior: without a layerOf map, all three
    // map ids are -1, writeTransform is never called, and texels 87..92 remain 0
    // (the GLSL skips reading them when the map id == -1 → identity fallback).
    const m: MaterialSpec = {
      baseColor: [0.5, 0.5, 0.5],
      roughness: 0.5,
      metallic: 0.0,
      aoMap:    { handle: {}, transform: { scale: [2, 3] } },
      lightMap: { handle: {}, transform: { scale: [4, 5] } },
      bumpMap:  { handle: {}, transform: { scale: [6, 7] } },
      alphaMap: { handle: {}, transform: { scale: [8, 9] } },
    };
    // No layerOf → all layers resolve to -1 → writeTransform not called.
    const d = packMaterialsTexture([m]).data;
    // Map ids at texel 85 must be -1 (unmapped).
    expect(d[texel(0, 85, 0)]).toBe(-1); // aoLayer
    expect(d[texel(0, 85, 1)]).toBe(-1); // lightMapLayer
    expect(d[texel(0, 85, 2)]).toBe(-1); // bumpLayer
    // Transform rows at 87/89/91 stay 0 (writeTransform was not called).
    expect(d[texel(0, 87, 0)]).toBe(0); // aoMapTransform row1.r
    expect(d[texel(0, 89, 0)]).toBe(0); // lightMapTransform row1.r
    expect(d[texel(0, 91, 0)]).toBe(0); // bumpMapTransform row1.r
    expect(d[texel(0, 93, 0)]).toBe(0); // alphaMapTransform row1.r
  });

  it('multiple materials are packed at their MATERIAL_PIXELS-strided offsets', () => {
    const a: MaterialSpec = { baseColor: [1, 0, 0], roughness: 1, metallic: 0 };
    const b: MaterialSpec = { baseColor: [0, 1, 0], roughness: 1, metallic: 0 };
    const out = packMaterialsTexture([a, b]);
    expect(out.materialCount).toBe(2);
    // dim = ceil(sqrt(190)) = 14 → 14*14*4 = 784 floats.
    expect(out.dim).toBe(14);
    // material 0 color.
    expect(out.data[texel(0, 0, 0)]).toBe(1);
    expect(out.data[texel(0, 0, 1)]).toBe(0);
    // material 1 color, at the second MATERIAL_PIXELS-strided block.
    expect(out.data[texel(1, 0, 0)]).toBe(0);
    expect(out.data[texel(1, 0, 1)]).toBe(1);
  });
});
