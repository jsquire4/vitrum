// GPU-free golden for the materials packer. Mirrors pt-webgpu's
// `materialPackingCoreEquivalence` style: pack a known `@vitrum/core` MaterialSpec
// and assert the EXACT texel values at the load-bearing offsets the fork's GLSL
// `material_struct` decoder reads (verified against `MaterialsTexture.js` +
// `material_mapped_rich.glsl.ts`). Any divergence here is a real render bug.

import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { evaluateSpectrum } from '@vitrum/shared-samplers';
import {
  MATERIAL_LAYER_NORMAL_TEXEL_OFFSET,
  MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET,
  MATERIAL_UV_SELECTOR_TEXEL_OFFSET,
  MATERIAL_WRAP_TEXEL_OFFSET,
  packMaterialsTexture,
  MATERIAL_PIXELS,
} from './materialsTexture.js';

/** Float offset of pixel `s`, channel `c` (0=r,1=g,2=b,3=a) within material `mi`. */
function texel(mi: number, s: number, c: number): number {
  return mi * MATERIAL_PIXELS * 4 + s * 4 + c;
}

describe('packMaterialsTexture — RGBA32F byte layout', () => {
  it('exposes the verified MATERIAL_PIXELS constant', () => {
    // D3 (2026-06-10): fork base 85 + texels 85..92 (ao/light/bump ids + scalars
    // + envMapIntensity at 85/86, their transforms at 87..92) + alphaMap transform
    // at 93/94 + anisotropyMap transform at 95/96 + thickness payload/transform
    // at 97..99 + per-map sampler policies at 100..120 + spectral reflectance at 121
    // + front/back layer normal payload at 122..129 + 21 scalable UV-layer
    // selectors at texels 130..135.
    // Single-sourced with every GLSL fetch site via glsl/shader/structs/materialStride.js
    // — see materialStrideParity.test.ts for the packer↔shader guard.
    expect(MATERIAL_PIXELS).toBe(136);
  });

  it('keeps direct-core maps on arbitrary UV sets and packs dense layer selectors', () => {
    const handle = {};
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle, texCoord: 2 },
      normalMap: { handle, texCoord: 3 },
    };
    const data = packMaterialsTexture(
      [material],
      new Map<unknown, number>([[handle, 7]]),
    ).data;
    expect(data[texel(0, 0, 3)]).toBe(7);
    expect(data[texel(0, 4, 0)]).toBe(7);
    // Standalone packer fallback is dense: texCoord 2 -> layer 5, 3 -> layer 6.
    expect(data[texel(0, MATERIAL_UV_SELECTOR_TEXEL_OFFSET, 0)]).toBe(5);
    expect(data[texel(0, MATERIAL_UV_SELECTOR_TEXEL_OFFSET + 1, 1)]).toBe(6);
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
      anisotropy: 0.75,
      anisotropyRotation: 0.25,
    };

    const out = packMaterialsTexture([m]);
    expect(out.kind).toBe('rgba32f');
    expect(out.materialCount).toBe(1);
    // dim = ceil(sqrt(130)) = 12 → backing data is 12*12*4 = 576 floats.
    expect(out.dim).toBe(12);
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

    // s11.a = scalar KHR_materials_anisotropy strength.
    expect(d[texel(0, 11, 3)]).toBeCloseTo(0.75, 6);
    // s17.b = scalar KHR_materials_anisotropy rotation.
    expect(d[texel(0, 17, 2)]).toBeCloseTo(0.25, 6);

    // s13 = (alphaMap=-1, opacity, alphaTest, side).
    expect(d[texel(0, 13, 0)]).toBe(-1);
    expect(d[texel(0, 13, 1)]).toBeCloseTo(0.8, 6);
    expect(d[texel(0, 13, 2)]).toBeCloseTo(0.33, 6); // alphaMode 'mask' → alphaCutoff
    // side: transmission>0 and material is thin-film by default (thickness 0,
    // attenuationDistance Infinity) ⇒ NOT the side==0 glass branch ⇒ FrontSide=1.
    expect(d[texel(0, 13, 3)]).toBe(1);
  });

  it('realizes every supported envMapIntensity at the RGBA32F boundary', () => {
    const minSubnormal = 2 ** -149;
    const maxFinite = Math.fround(3.402823466e38);
    const values = [0, 1 / 3, minSubnormal, maxFinite];
    const materials = values.map((envMapIntensity): MaterialSpec => ({
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      envMapIntensity,
    }));
    const data = packMaterialsTexture(materials).data;

    for (let i = 0; i < values.length; i += 1) {
      expect(data[texel(i, 85, 3)]).toBe(Math.fround(values[i]!));
    }
  });

  it.each([
    ['negative', -1, /must be finite and non-negative/],
    ['non-finite', Number.POSITIVE_INFINITY, /must be finite and non-negative/],
    ['overflowing', Number.MAX_VALUE, /overflows WebGL float32 storage/],
    ['positive-underflowing', Number.MIN_VALUE, /underflows WebGL float32 storage/],
  ])('rejects a %s envMapIntensity before packing', (_label, envMapIntensity, message) => {
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      envMapIntensity,
    };
    expect(() => packMaterialsTexture([material])).toThrow(message);
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

  it('packs authored double-sided opaque surfaces while preserving closed-volume exit traversal', () => {
    const frontOnly: MaterialSpec = {
      baseColor: [0.8, 0.8, 0.8],
      roughness: 1,
      metallic: 0,
      doubleSided: false,
    };
    const twoSided: MaterialSpec = { ...frontOnly, doubleSided: true };
    const closedGlass: MaterialSpec = {
      ...frontOnly,
      transmission: 1,
      attenuationDistance: 2,
    };
    const d = packMaterialsTexture([frontOnly, twoSided, closedGlass]).data;
    expect(d[texel(0, 13, 3)]).toBe(1);
    expect(d[texel(1, 13, 3)]).toBe(0);
    expect(d[texel(2, 13, 3)]).toBe(0);
    expect(Number(d[texel(0, 14, 3)]) & 0x02).toBe(0);
    expect(Number(d[texel(1, 14, 3)]) & 0x02).toBe(0x02);
    // Traversability is not emissive sidedness: closed single-sided glass keeps
    // the back interface hittable without acquiring the double-sided flag.
    expect(Number(d[texel(2, 14, 3)]) & 0x02).toBe(0);
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

  it('packs folded mesh-area emitter shadow-disable in s14.a bit 6', () => {
    const ordinary: MaterialSpec = { baseColor: [0.8, 0.8, 0.8], roughness: 1.0, metallic: 0.0 };
    const foldedShadowless = {
      baseColor: [0, 0, 0],
      roughness: 1.0,
      metallic: 0.0,
      emissive: [1, 1, 1],
      emissiveIntensity: 2,
      meshEmitterCastShadowDisabled: true,
    } as MaterialSpec & { meshEmitterCastShadowDisabled: true };
    const d = packMaterialsTexture([ordinary, foldedShadowless]).data;
    expect(d[texel(0, 14, 3)]).toBe(0);
    expect(d[texel(1, 14, 3)]).toBe(64);
  });

  it('packs forward-only mesh emission in s14.a bit 7 for NEE-excluded materials', () => {
    const exactNearest: MaterialSpec = {
      baseColor: [0, 0, 0],
      roughness: 1,
      metallic: 0,
      emissive: [1, 1, 1],
      emissiveMap: { handle: {} },
    };
    const filtered: MaterialSpec = {
      ...exactNearest,
      emissiveMap: { handle: {}, minFilter: 'linear' },
    };
    const mipmapped: MaterialSpec = {
      ...exactNearest,
      emissiveMap: { handle: {}, mipFilter: 'nearest' },
    };
    const explicitlySkipped: MaterialSpec = {
      baseColor: [0, 0, 0],
      roughness: 1,
      metallic: 0,
      emissive: [1, 1, 1],
      extensions: { skipEmitter: true },
    };
    const d = packMaterialsTexture([
      exactNearest,
      filtered,
      mipmapped,
      explicitlySkipped,
    ]).data;
    expect(d[texel(0, 14, 3)]).toBe(0);
    expect(d[texel(1, 14, 3)]).toBe(128);
    expect(d[texel(2, 14, 3)]).toBe(128);
    expect(d[texel(3, 14, 3)]).toBe(128);
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

  it('scatteringCoefficientRGB packs per-channel sigmaS override and majorant sigmaT', () => {
    const sss: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      scatteringCoefficient: 2.5,
      scatteringAnisotropy: 0.3,
      scatteringCoefficientRGB: [0.7, 0.8, 0.9],
      attenuationColor: [0.5, 0.25, 1.0],
      attenuationDistance: 2.0,
    };
    const d = packMaterialsTexture([sss]).data;
    const flags = d[texel(0, 14, 3)]!;
    expect((flags & (1 << 4)) !== 0).toBe(true); // TRANSLUCENT_BIT
    // s15.r is the scalar SSS free-flight majorant max(σ_a + σ_s).
    const sigmaTR = -Math.log(0.5) / 2.0 + 0.7;
    const sigmaTG = -Math.log(0.25) / 2.0 + 0.8;
    const sigmaTB = 0.9;
    expect(d[texel(0, 15, 0)]).toBeCloseTo(Math.max(sigmaTR, sigmaTG, sigmaTB), 6);
    expect(d[texel(0, 15, 1)]).toBeCloseTo(0.3, 6);
    expect(d[texel(0, 15, 3)]).toBe(0); // no thin-film stack
    // s16 = sssSigmaS.rgb / thinFilmLayerCount.
    expect(d[texel(0, 16, 0)]).toBeCloseTo(0.7, 6);
    expect(d[texel(0, 16, 1)]).toBeCloseTo(0.8, 6);
    expect(d[texel(0, 16, 2)]).toBeCloseTo(0.9, 6);
    expect(d[texel(0, 16, 3)]).toBe(0); // 0 layers
  });

  it('includes the packed spectral extinction peak in the free-flight majorant', () => {
    const medium: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
      scatteringCoefficientRGB: [0.1, 0.2, 0.05],
      attenuationColor: [1, 1, 1],
      attenuationDistance: Infinity,
      spectralAttenuation: {
        wavelengthStart: 380,
        wavelengthEnd: 780,
        values: new Float32Array([5, 5, 5]),
      },
    };
    const d = packMaterialsTexture([medium]).data;
    // The shader linearly interpolates the packed grid, whose peak is 5.
    // Its hero-channel scattering coefficient is bounded by max(sigmaS)=0.2.
    expect(d[texel(0, 15, 0)]).toBeCloseTo(5.2, 5);
  });

  it('activates the fog-volume lane for every positive transmitted scattering coefficient', () => {
    const participating: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      transmission: 1,
      scatteringCoefficient: 5e-7,
    };
    const active = packMaterialsTexture([participating]).data;
    expect((active[texel(0, 14, 2)]! & 4) !== 0).toBe(true);
    expect(active[texel(0, 15, 0)]).toBeGreaterThan(0);

    const nonTransmitted = packMaterialsTexture([{ ...participating, transmission: 0 }]).data;
    expect((nonTransmitted[texel(0, 14, 2)]! & 4) !== 0).toBe(false);
  });

  it('scatteringCoefficientRGB alone activates translucent SSS and packs sigmaS', () => {
    const sss: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      scatteringCoefficientRGB: [0.0, 0.3, 0.2],
    };
    const d = packMaterialsTexture([sss]).data;
    const flags = d[texel(0, 14, 3)]!;
    expect((flags & (1 << 4)) !== 0).toBe(true);
    expect(d[texel(0, 15, 0)]).toBeCloseTo(0.3, 6);
    expect(d[texel(0, 16, 0)]).toBeCloseTo(0.0, 6);
    expect(d[texel(0, 16, 1)]).toBeCloseTo(0.3, 6);
    expect(d[texel(0, 16, 2)]).toBeCloseTo(0.2, 6);
  });

  it('scalar scatteringCoefficient falls back to isotropic sigmaS packing', () => {
    const sss: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      scatteringCoefficient: 0.35,
    };
    const d = packMaterialsTexture([sss]).data;
    expect(d[texel(0, 15, 0)]).toBeCloseTo(0.35, 6);
    expect(d[texel(0, 16, 0)]).toBeCloseTo(0.35, 6);
    expect(d[texel(0, 16, 1)]).toBeCloseTo(0.35, 6);
    expect(d[texel(0, 16, 2)]).toBeCloseTo(0.35, 6);
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

  it('packs front/back layer normal maps in the appended payload', () => {
    const frontHandle = {};
    const backHandle = {};
    const layerOf = new Map<unknown, number>([
      [frontHandle, 4],
      [backHandle, 5],
    ]);
    const mat: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      frontLayer: {
        transmission: [0.9, 0.8, 0.7],
        normalMap: {
          handle: frontHandle,
          texCoord: 1,
          transform: { offset: [0.25, 0.5], scale: [0.5, 0.25], rotation: Math.PI / 2 },
          wrapS: 'clamp-to-edge',
          wrapT: 'mirrored-repeat',
          magFilter: 'linear',
          minFilter: 'nearest',
          mipFilter: 'linear',
        },
        normalScale: 0.75,
      },
      backLayer: {
        transmission: [0.7, 0.8, 0.9],
        normalMap: {
          handle: backHandle,
          texCoord: 0,
          transform: { offset: [0.1, 0.2], scale: [0.3, 0.4], rotation: 0 },
          wrapS: 'repeat',
          wrapT: 'clamp-to-edge',
          magFilter: 'nearest',
          minFilter: 'linear',
          mipFilter: 'nearest',
        },
        normalScale: 0.5,
      },
    };
    const d = packMaterialsTexture([mat], layerOf).data;
    const ln = MATERIAL_LAYER_NORMAL_TEXEL_OFFSET;
    expect(d[texel(0, ln, 0)]).toBe(4);
    expect(d[texel(0, ln, 1)]).toBeCloseTo(0.75, 6);
    expect(d[texel(0, ln, 2)]).toBe(5);
    expect(d[texel(0, ln, 3)]).toBeCloseTo(0.5, 6);
    // front transform first row: sx*cos, sx*sin, offsetX.
    expect(d[texel(0, ln + 1, 0)]).toBeCloseTo(0.0, 6);
    expect(d[texel(0, ln + 1, 1)]).toBeCloseTo(0.5, 6);
    expect(d[texel(0, ln + 1, 2)]).toBeCloseTo(0.25, 6);
    // back transform first row: sx, 0, offsetX.
    expect(d[texel(0, ln + 3, 0)]).toBeCloseTo(0.3, 6);
    expect(d[texel(0, ln + 3, 1)]).toBeCloseTo(0.0, 6);
    expect(d[texel(0, ln + 3, 2)]).toBeCloseTo(0.1, 6);
    // sampler policy: wrap repeat=0/clamp=1/mirror=2, mip none=0/nearest=1/linear=2,
    // packed filters = mag + min*2 with nearest=0, linear=1.
    expect(d[texel(0, ln + 5, 0)]).toBe(1);
    expect(d[texel(0, ln + 5, 1)]).toBe(2);
    expect(d[texel(0, ln + 5, 2)]).toBe(2);
    expect(d[texel(0, ln + 5, 3)]).toBe(1);
    expect(d[texel(0, ln + 6, 0)]).toBe(0);
    expect(d[texel(0, ln + 6, 1)]).toBe(1);
    expect(d[texel(0, ln + 6, 2)]).toBe(1);
    expect(d[texel(0, ln + 6, 3)]).toBe(2);
    expect(d[texel(0, ln + 7, 0)]).toBe(4);
    expect(d[texel(0, ln + 7, 1)]).toBe(2);
  });

  it('packs each atlas handle native extent into its sampler policy', () => {
    const handle = {};
    const lookup = {
      srgb: new Map<unknown, number>([[handle, 0]]),
      linear: new Map<unknown, number>(),
      dimensions: new Map<unknown, readonly [number, number]>([[handle, [2, 3]]]),
    };
    const mat: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle, wrapS: 'repeat', wrapT: 'clamp-to-edge' },
    };
    const d = packMaterialsTexture([mat], lookup).data;
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET, 0)]).toBe(2 * 4 + 0);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET, 1)]).toBe(3 * 4 + 1);
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
        values: new Float32Array([0.0, 0.5, 1.0]), // ramp 0→1 across the band
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

  it('packs per-material Jakob-Hanika spectral reflectance coefficients', () => {
    const red: MaterialSpec = {
      baseColor: [1, 0, 0],
      roughness: 0.5,
      metallic: 0,
    };
    const d = packMaterialsTexture([red]).data;
    const offset = MATERIAL_SPECTRAL_REFLECTANCE_TEXEL_OFFSET;
    const coeffs: [number, number, number] = [
      d[texel(0, offset, 0)]!,
      d[texel(0, offset, 1)]!,
      d[texel(0, offset, 2)]!,
    ];

    expect(d[texel(0, offset, 3)]).toBe(1);
    expect(evaluateSpectrum(coeffs, 680)).toBeGreaterThan(evaluateSpectrum(coeffs, 450));
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

  it('uses role-aware atlas layers for a handle shared by sRGB and linear maps', () => {
    const sharedHandle = {};
    const layerOfByColorSpace = {
      srgb: new Map<unknown, number>([[sharedHandle, 3]]),
      linear: new Map<unknown, number>([[sharedHandle, 9]]),
    };
    const m: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle: sharedHandle },
      roughnessMap: { handle: sharedHandle },
    };
    const d = packMaterialsTexture([m], layerOfByColorSpace).data;

    expect(d[texel(0, 0, 3)]).toBe(3); // baseColorMap: sRGB atlas layer.
    expect(d[texel(0, 1, 3)]).toBe(9); // roughnessMap: linear atlas layer.
  });

  it('packs per-map sampler policies at texels 100..120', () => {
    const baseHandle = {};
    const metalHandle = {};
    const bumpHandle = {};
    const layerOf = new Map<unknown, number>([
      [baseHandle, 0],
      [metalHandle, 1],
      [bumpHandle, 2],
    ]);
    const m: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: {
        handle: baseHandle,
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
        magFilter: 'linear',
        minFilter: 'nearest',
        mipFilter: 'linear',
      },
      metallicMap: { handle: metalHandle, wrapS: 'mirrored-repeat', wrapT: 'repeat' },
      bumpMap: { handle: bumpHandle, wrapS: 'clamp-to-edge', wrapT: 'repeat' },
    };
    const d = packMaterialsTexture([m], layerOf).data;

    // Shared map order:
    //   0 baseColorMap -> first sampler texel, 1 metallicMap -> second sampler texel,
    //   18 bumpMap -> sampler texel 18. Encodings: wrap 0 repeat/1 clamp/2 mirror,
    //   mip 0 none/1 nearest/2 linear, filter packed = mag + min*2.
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET, 0)]).toBe(1);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET, 1)]).toBe(2);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET, 2)]).toBe(2);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET, 3)]).toBe(1);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 1, 0)]).toBe(2);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 1, 1)]).toBe(0);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 1, 2)]).toBe(0);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 1, 3)]).toBe(0);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 18, 0)]).toBe(1);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 18, 1)]).toBe(0);
  });

  it('packs anisotropyMap layer, UV selector, transform, and wrap mode', () => {
    const anisoHandle = {};
    const layerOf = new Map<unknown, number>([[anisoHandle, 5]]);
    const m: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      anisotropy: 0.7,
      anisotropyRotation: 0.25,
      anisotropyMap: {
        handle: anisoHandle,
        texCoord: 1,
        wrapS: 'mirrored-repeat',
        wrapT: 'clamp-to-edge',
        transform: { scale: [2, 3], offset: [0.125, 0.25] },
      },
    };
    const d = packMaterialsTexture([m], layerOf).data;

    // s6.b is the anisotropyMap atlas layer id; s11.a/s17.b remain scalar strength/rotation.
    expect(d[texel(0, 6, 2)]).toBe(5);
    expect(d[texel(0, 11, 3)]).toBeCloseTo(0.7, 6);
    expect(d[texel(0, 17, 2)]).toBeCloseTo(0.25, 6);

    // The former UV1 bitmask lane stays reserved; selector 19 resolves UV1.
    expect(d[texel(0, 86, 3)]).toBe(0);
    expect(
      d[texel(
        0,
        MATERIAL_UV_SELECTOR_TEXEL_OFFSET + Math.floor(19 / 4),
        19 % 4,
      )],
    ).toBe(4);

    // Texels 95/96 encode the anisotropyMap UV transform.
    expect(d[texel(0, 95, 0)]).toBeCloseTo(2, 6);
    expect(d[texel(0, 95, 1)]).toBeCloseTo(0, 6);
    expect(d[texel(0, 95, 2)]).toBeCloseTo(0.125, 6);
    expect(d[texel(0, 96, 0)]).toBeCloseTo(0, 6);
    expect(d[texel(0, 96, 1)]).toBeCloseTo(3, 6);
    expect(d[texel(0, 96, 2)]).toBeCloseTo(0.25, 6);

    // Map index 19 maps to sampler-policy texel 19.
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 19, 0)]).toBe(2);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 19, 1)]).toBe(1);
  });

  it('packs thicknessMap layer, thickness scalar, UV1 bit, transform, and wrap mode', () => {
    const thicknessHandle = {};
    const layerOf = new Map<unknown, number>([[thicknessHandle, 6]]);
    const m: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      transmission: 1.0,
      attenuationDistance: 2.0,
      attenuationColor: [0.8, 0.9, 1.0],
      thickness: 0.35,
      thicknessMap: {
        handle: thicknessHandle,
        texCoord: 1,
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
        transform: { scale: [4, 5], offset: [0.375, 0.625] },
      },
    };
    const d = packMaterialsTexture([m], layerOf).data;

    // Texel 97 stores scalar thickness + thicknessMap atlas layer.
    expect(d[texel(0, 97, 0)]).toBeCloseTo(0.35, 6);
    expect(d[texel(0, 97, 1)]).toBe(6);

    // The former UV1 bitmask lane stays reserved; selector 20 resolves UV1.
    expect(d[texel(0, 86, 3)]).toBe(0);
    expect(
      d[texel(
        0,
        MATERIAL_UV_SELECTOR_TEXEL_OFFSET + Math.floor(20 / 4),
        20 % 4,
      )],
    ).toBe(4);

    // Texels 98/99 encode the thicknessMap UV transform.
    expect(d[texel(0, 98, 0)]).toBeCloseTo(4, 6);
    expect(d[texel(0, 98, 1)]).toBeCloseTo(0, 6);
    expect(d[texel(0, 98, 2)]).toBeCloseTo(0.375, 6);
    expect(d[texel(0, 99, 0)]).toBeCloseTo(0, 6);
    expect(d[texel(0, 99, 1)]).toBeCloseTo(5, 6);
    expect(d[texel(0, 99, 2)]).toBeCloseTo(0.625, 6);

    // Map index 20 maps to sampler-policy texel 20.
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 20, 0)]).toBe(1);
    expect(d[texel(0, MATERIAL_WRAP_TEXEL_OFFSET + 20, 1)]).toBe(2);
  });

  // D3 — ao/lightMap/bumpMap transforms at texels 87/89/91 (item 26).
  //
  // The packer writes ao/lightMap/bumpMap id + scalars at texels 85/86, and their
  // UV-transform mat3s at texels 87 (aoMapTransform), 89 (lightMapTransform), 91
  // (bumpMapTransform), 2 texels per mat3 (see writeTransform + readTextureTransform
  // in material_mapped_rich.glsl.ts). The GLSL decoder reads:
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
  // writeTransform encoding (verified against material_mapped_rich.glsl.ts):
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

  it('resolves layer ids from the statically selected LDR or radiance atlas', () => {
    const handle = {};
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
      baseColorMap: { handle },
      roughnessMap: { handle },
      emissiveMap: { handle },
      lightMap: { handle },
    };
    const d = packMaterialsTexture([material], {
      ldr: {
        srgb: new Map([[handle, 7]]),
        linear: new Map([[handle, 9]]),
      },
      hdr: {
        srgb: new Map([[handle, 3]]),
        linear: new Map([[handle, 5]]),
      },
    }).data;

    expect(d[texel(0, 0, 3)]).toBe(7); // baseColorMap: LDR sRGB
    expect(d[texel(0, 1, 3)]).toBe(9); // roughnessMap: LDR linear
    expect(d[texel(0, 3, 3)]).toBe(3); // emissiveMap: HDR sRGB source role
    expect(d[texel(0, 85, 1)]).toBe(5); // lightMap: HDR linear
  });

  it('multiple materials are packed at their MATERIAL_PIXELS-strided offsets', () => {
    const a: MaterialSpec = { baseColor: [1, 0, 0], roughness: 1, metallic: 0 };
    const b: MaterialSpec = { baseColor: [0, 1, 0], roughness: 1, metallic: 0 };
    const out = packMaterialsTexture([a, b]);
    expect(out.materialCount).toBe(2);
    // dim = ceil(sqrt(260)) = 17 → 17*17*4 = 1156 floats.
    expect(out.dim).toBe(17);
    // material 0 color.
    expect(out.data[texel(0, 0, 0)]).toBe(1);
    expect(out.data[texel(0, 0, 1)]).toBe(0);
    // material 1 color, at the second MATERIAL_PIXELS-strided block.
    expect(out.data[texel(1, 0, 0)]).toBe(0);
    expect(out.data[texel(1, 0, 1)]).toBe(1);
  });
});
