/**
 * materialTextures.test.ts — P2 host-side texture collection + descriptor pack.
 */
import { describe, it, expect } from 'vitest';
import {
  applyMaterialTextureUvFitScales,
  collectMaterialTextures,
  MATERIAL_TEX_FLOAT_STRIDE,
  MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET,
  MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_UV_META_VEC4S_PER_MAP,
  MATERIAL_TEX_VEC4_STRIDE,
} from '../scene/materialTextures.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import type { MaterialSpec } from '@vitrum/core';

function mat(over: Partial<MaterialSpec>): MaterialSpec {
  return { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, ...over };
}

function expectCloseArray(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(actual[i] ?? Number.NaN).toBeCloseTo(expected[i] ?? Number.NaN);
  }
}

describe('collectMaterialTextures (P2 host)', () => {
  it('dedups shared texture handles + indexes per material', () => {
    const tex = { id: 'A' };
    const { sources, descriptors } = collectMaterialTextures([
      mat({ baseColorMap: { handle: tex } }),
      mat({ baseColorMap: { handle: tex } }), // same handle → dedup to index 0
      mat({}), // no map
    ]);
    expect(sources).toEqual([tex]);
    expect(descriptors[0]).toBe(0);
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE]).toBe(0);
    expect(descriptors[2 * MATERIAL_TEX_FLOAT_STRIDE]).toBe(-1);
  });

  it('packs alpha-mode, cutoff, opacity, texCoord, and the UV transform', () => {
    const { descriptors } = collectMaterialTextures([
      mat({
        baseColorMap: { handle: {}, texCoord: 1, transform: { offset: [0.1, 0.2], scale: [2, 3], rotation: 0.5 } },
        alphaMode: 'mask',
        alphaCutoff: 0.3,
        opacity: 0.8,
      }),
    ]);
    expect(descriptors[4]).toBe(1); // alphaMode mask
    expect(descriptors[5]).toBeCloseTo(0.3);
    expect(descriptors[6]).toBeCloseTo(0.8);
    expect(descriptors[7]).toBe(1); // texCoord
    expect(descriptors[8]).toBeCloseTo(0.1);
    expect(descriptors[9]).toBeCloseTo(0.2);
    expect(descriptors[10]).toBeCloseTo(2);
    expect(descriptors[11]).toBeCloseTo(3);
    expect(descriptors[12]).toBeCloseTo(0.5);
  });

  it('collects emissiveMap into the same (sRGB) sources + packs emissiveIdx', () => {
    const tex = { id: 'base' };
    const emis = { id: 'emis' };
    const { sources, descriptors } = collectMaterialTextures([
      mat({ baseColorMap: { handle: tex }, emissiveMap: { handle: emis } }),
      mat({ emissiveMap: { handle: tex } }), // emissive reuses the baseColor handle → dedup
    ]);
    expect(sources).toEqual([tex, emis]); // baseColor + emissive share the sRGB source list
    expect(descriptors[0]).toBe(0); // mat0 baseColorIdx → tex (0)
    expect(descriptors[3]).toBe(1); // mat0 emissiveIdx → emis (1)
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + 0]).toBe(-1); // mat1 no baseColor
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + 3]).toBe(0); // mat1 emissiveIdx → tex (dedup 0)
  });

  it('collects normal + ORM into a SEPARATE linear source list (own index space)', () => {
    const baseTex = { id: 'base' };
    const normTex = { id: 'norm' };
    const mrTex = { id: 'mr' };
    const { sources, linearSources, descriptors } = collectMaterialTextures([
      mat({ baseColorMap: { handle: baseTex }, normalMap: { handle: normTex }, roughnessMap: { handle: mrTex } }),
    ]);
    expect(sources).toEqual([baseTex]);          // sRGB list: baseColor only
    expect(linearSources).toEqual([normTex, mrTex]); // linear list: normal + ORM
    expect(descriptors[0]).toBe(0);  // baseColorIdx → sRGB 0
    expect(descriptors[1]).toBe(0);  // normalIdx → linear 0 (separate space)
    expect(descriptors[2]).toBe(1);  // ormIdx → linear 1
  });

  it('packs normalScale in the normal-map descriptor lane', () => {
    const normTex = { id: 'norm' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({ normalMap: { handle: normTex }, normalScale: 0.35 }),
      mat({ normalMap: { handle: normTex } }),
    ]);
    expect(linearSources).toEqual([normTex]);
    expect(descriptors[1]).toBe(0);  // mat0 normalIdx → linear 0
    expect(descriptors[23]).toBeCloseTo(0.35);
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + 1]).toBe(0); // deduped normal handle
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + 23]).toBe(1); // glTF default scale
  });

  it('collects clearcoatNormalMap as LINEAR data with scale, wrap, and UV metadata', () => {
    const ccNormalTex = { id: 'cc-normal' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({
        clearcoatNormalMap: {
          handle: ccNormalTex,
          texCoord: 1,
          wrapS: 'clamp-to-edge',
          wrapT: 'mirrored-repeat',
          transform: { offset: [0.2, 0.3], scale: [0.4, 0.5], rotation: 0.6 },
        },
        clearcoatNormalScale: 0.45,
      }),
    ]);
    const ccBase = MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET * 4;
    const ccWrap = MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET * 4;
    const ccMeta = MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET * 4;

    expect(linearSources).toEqual([ccNormalTex]);
    expect(descriptors[ccBase]).toBe(0);
    expect(descriptors[ccBase + 1]).toBeCloseTo(0.45);
    expect(descriptors[ccBase + 2]).toBe(1);
    expect(descriptors[ccBase + 3]).toBe(1);
    expect(descriptors[ccWrap]).toBe(1);
    expect(descriptors[ccWrap + 1]).toBe(2);
    expectCloseArray(
      descriptors.slice(ccMeta, ccMeta + 8),
      [1, 0.2, 0.3, 0.6, 0.4, 0.5, 0, 0],
    );
  });

  it('collects alphaMap as LINEAR coverage data and packs its descriptor lane', () => {
    const alphaTex = { id: 'alpha' };
    const baseTex = { id: 'base' };
    const { sources, linearSources, descriptors } = collectMaterialTextures([
      mat({
        baseColorMap: { handle: baseTex },
        alphaMap: { handle: alphaTex },
        alphaMode: 'mask',
      }),
    ]);
    expect(sources).toEqual([baseTex]);
    expect(linearSources).toEqual([alphaTex]);
    expect(descriptors[0]).toBe(0);  // baseColorIdx -> sRGB 0
    expect(descriptors[24]).toBe(0); // alphaMapIdx -> linear 0
  });

  it('dedups alphaMap with other linear material maps', () => {
    const sharedLinearTex = { id: 'shared-linear' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({
        alphaMap: { handle: sharedLinearTex },
        aoMap: { handle: sharedLinearTex },
      }),
    ]);
    expect(linearSources).toEqual([sharedLinearTex]);
    expect(descriptors[13]).toBe(0); // aoMapIdx -> linear 0
    expect(descriptors[24]).toBe(0); // alphaMapIdx -> same linear layer
  });

  it('collects transmissionMap as LINEAR scalar data and packs its descriptor lane', () => {
    const transmissionTex = { id: 'transmission' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({
        transmission: 0.75,
        transmissionMap: { handle: transmissionTex },
      }),
    ]);
    expect(linearSources).toEqual([transmissionTex]);
    expect(descriptors[25]).toBe(0); // transmissionMapIdx -> linear 0
  });

  it('dedups transmissionMap with other linear material maps', () => {
    const sharedLinearTex = { id: 'shared-linear-transmission' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({
        transmissionMap: { handle: sharedLinearTex },
        normalMap: { handle: sharedLinearTex },
      }),
    ]);
    expect(linearSources).toEqual([sharedLinearTex]);
    expect(descriptors[1]).toBe(0);  // normalIdx -> linear 0
    expect(descriptors[25]).toBe(0); // transmissionMapIdx -> same linear layer
  });

  it('falls back to metallicMap for ORM when roughnessMap is absent', () => {
    const mrTex = { id: 'mr' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({ metallicMap: { handle: mrTex } }),
    ]);
    expect(linearSources).toEqual([mrTex]);
    expect(descriptors[2]).toBe(0); // ormIdx → metallicMap
  });

  it('defaults: opaque(0), cutoff 0.5, opacity 1, identity scale', () => {
    const { descriptors } = collectMaterialTextures([mat({ baseColorMap: { handle: {} } })]);
    expect(descriptors[4]).toBe(0);
    expect(descriptors[5]).toBe(0.5);
    expect(descriptors[6]).toBe(1);
    expect(descriptors[10]).toBe(1);
    expect(descriptors[11]).toBe(1);
    for (let i = 28; i < 48; i += 1) {
      expect(descriptors[i]).toBe(1);
    }
    for (let i = 156; i < 172; i += 1) {
      expect(descriptors[i]).toBe(1);
    }
  });

  it('collects extension-lobe maps into the correct color-space source arrays', () => {
    const clearcoat = { id: 'clearcoat' };
    const clearcoatRoughness = { id: 'clearcoat-roughness' };
    const sheenColor = { id: 'sheen-color' };
    const sheenRoughness = { id: 'sheen-roughness' };
    const iridescence = { id: 'iridescence' };
    const iridescenceThickness = { id: 'iridescence-thickness' };
    const specularColor = { id: 'specular-color' };
    const specularIntensity = { id: 'specular-intensity' };
    const { sources, linearSources, descriptors } = collectMaterialTextures([
      mat({
        clearcoatMap: { handle: clearcoat },
        clearcoatRoughnessMap: { handle: clearcoatRoughness },
        sheenColorMap: { handle: sheenColor },
        sheenRoughnessMap: { handle: sheenRoughness },
        iridescenceMap: { handle: iridescence },
        iridescenceThicknessMap: { handle: iridescenceThickness },
        specularColorMap: { handle: specularColor },
        specularIntensityMap: { handle: specularIntensity },
      }),
    ]);

    expect(sources).toEqual([sheenColor, specularColor]);
    expect(linearSources).toEqual([
      clearcoat,
      clearcoatRoughness,
      sheenRoughness,
      iridescence,
      iridescenceThickness,
      specularIntensity,
    ]);
    expect(Array.from(descriptors.slice(148, 156))).toEqual([0, 1, 0, 2, 3, 4, 1, 5]);
  });

  it('applies extension-lobe UV-fit scales from the right texture arrays', () => {
    const clearcoat = { id: 'clearcoat-fit' };
    const clearcoatRoughness = { id: 'clearcoat-roughness-fit' };
    const sheenColor = { id: 'sheen-color-fit' };
    const sheenRoughness = { id: 'sheen-roughness-fit' };
    const iridescence = { id: 'iridescence-fit' };
    const iridescenceThickness = { id: 'iridescence-thickness-fit' };
    const specularColor = { id: 'specular-color-fit' };
    const specularIntensity = { id: 'specular-intensity-fit' };
    const { descriptors } = collectMaterialTextures([
      mat({
        clearcoatMap: { handle: clearcoat },
        clearcoatRoughnessMap: { handle: clearcoatRoughness },
        sheenColorMap: { handle: sheenColor },
        sheenRoughnessMap: { handle: sheenRoughness },
        iridescenceMap: { handle: iridescence },
        iridescenceThicknessMap: { handle: iridescenceThickness },
        specularColorMap: { handle: specularColor },
        specularIntensityMap: { handle: specularIntensity },
      }),
    ]);

    applyMaterialTextureUvFitScales(
      descriptors,
      [[0.11, 0.12], [0.21, 0.22]],
      [
        [0.31, 0.32],
        [0.41, 0.42],
        [0.51, 0.52],
        [0.61, 0.62],
        [0.71, 0.72],
        [0.81, 0.82],
      ],
    );

    expectCloseArray(descriptors.slice(156, 172), [
      0.31, 0.32,
      0.41, 0.42,
      0.11, 0.12,
      0.51, 0.52,
      0.61, 0.62,
      0.71, 0.72,
      0.21, 0.22,
      0.81, 0.82,
    ]);
  });

  it('packs extension-lobe wrap modes and UV metadata', () => {
    const { descriptors } = collectMaterialTextures([
      mat({
        clearcoatMap: {
          handle: {},
          texCoord: 1,
          wrapS: 'clamp-to-edge',
          wrapT: 'mirrored-repeat',
          transform: { offset: [0.1, 0.2], scale: [1.1, 1.2], rotation: 0.3 },
        },
        specularIntensityMap: {
          handle: {},
          texCoord: 0,
          wrapS: 'mirrored-repeat',
          wrapT: 'clamp-to-edge',
          transform: { offset: [0.4, 0.5], scale: [1.4, 1.5], rotation: 0.6 },
        },
      }),
    ]);

    expect(Array.from(descriptors.slice(172, 176))).toEqual([1, 2, 0, 0]);
    expect(Array.from(descriptors.slice(184, 188))).toEqual([0, 0, 2, 1]);

    const uvMeta = (slot: number): number[] => {
      const start = (MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + slot * MATERIAL_TEX_UV_META_VEC4S_PER_MAP) * 4;
      return Array.from(descriptors.slice(start, start + 8));
    };

    expectCloseArray(uvMeta(0), [1, 0.1, 0.2, 0.3, 1.1, 1.2, 0, 0]);
    expectCloseArray(uvMeta(7), [0, 0.4, 0.5, 0.6, 1.4, 1.5, 0, 0]);
  });

  it('applies per-map UV-fit scales from the uploaded sRGB and linear texture arrays', () => {
    const base = { id: 'base' };
    const emissive = { id: 'emissive' };
    const normal = { id: 'normal' };
    const orm = { id: 'orm' };
    const ao = { id: 'ao' };
    const light = { id: 'light' };
    const bump = { id: 'bump' };
    const aniso = { id: 'aniso' };
    const alpha = { id: 'alpha' };
    const transmission = { id: 'transmission' };
    const { descriptors } = collectMaterialTextures([
      mat({
        baseColorMap: { handle: base },
        emissiveMap: { handle: emissive },
        normalMap: { handle: normal },
        roughnessMap: { handle: orm },
        aoMap: { handle: ao },
        lightMap: { handle: light },
        bumpMap: { handle: bump },
        anisotropyMap: { handle: aniso },
        alphaMap: { handle: alpha },
        transmissionMap: { handle: transmission },
      }),
    ]);

    applyMaterialTextureUvFitScales(
      descriptors,
      [[0.5, 1], [1, 0.25]],
      [
        [0.75, 1],
        [1, 0.5],
        [0.25, 0.5],
        [0.8, 0.6],
        [0.4, 0.3],
        [0.9, 0.7],
        [0.2, 0.1],
        [0.6, 0.4],
      ],
    );

    expectCloseArray(descriptors.slice(28, 32), [0.5, 1, 1, 0.25]);
    expectCloseArray(descriptors.slice(32, 36), [0.75, 1, 1, 0.5]);
    expectCloseArray(descriptors.slice(36, 40), [0.25, 0.5, 0.8, 0.6]);
    expectCloseArray(descriptors.slice(40, 44), [0.4, 0.3, 0.9, 0.7]);
    expectCloseArray(descriptors.slice(44, 48), [0.2, 0.1, 0.6, 0.4]);
  });

  it('packs per-map TextureRef wrap modes for the WGSL sampler', () => {
    const { descriptors } = collectMaterialTextures([
      mat({
        baseColorMap: { handle: {}, wrapS: 'clamp-to-edge', wrapT: 'mirrored-repeat' },
        emissiveMap: { handle: {}, wrapS: 'mirrored-repeat', wrapT: 'clamp-to-edge' },
        normalMap: { handle: {}, wrapS: 'clamp-to-edge', wrapT: 'clamp-to-edge' },
        metallicMap: { handle: {}, wrapS: 'mirrored-repeat', wrapT: 'repeat' },
        aoMap: { handle: {}, wrapS: 'repeat', wrapT: 'clamp-to-edge' },
        lightMap: { handle: {}, wrapS: 'mirrored-repeat', wrapT: 'mirrored-repeat' },
        bumpMap: { handle: {}, wrapS: 'clamp-to-edge', wrapT: 'repeat' },
        anisotropyMap: { handle: {}, wrapS: 'repeat', wrapT: 'mirrored-repeat' },
        alphaMap: { handle: {}, wrapS: 'clamp-to-edge', wrapT: 'mirrored-repeat' },
        transmissionMap: { handle: {}, wrapS: 'repeat', wrapT: 'clamp-to-edge' },
      }),
    ]);

    expect(Array.from(descriptors.slice(48, 52))).toEqual([1, 2, 2, 1]);
    expect(Array.from(descriptors.slice(52, 56))).toEqual([1, 1, 2, 0]);
    expect(Array.from(descriptors.slice(56, 60))).toEqual([0, 1, 2, 2]);
    expect(Array.from(descriptors.slice(60, 64))).toEqual([1, 0, 0, 2]);
    expect(Array.from(descriptors.slice(64, 68))).toEqual([1, 2, 0, 1]);
  });

  it('packs per-map texCoord and KHR_texture_transform metadata', () => {
    const { descriptors } = collectMaterialTextures([
      mat({
        baseColorMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [0.1, 0.2], scale: [1.1, 1.2], rotation: 0.3 },
        },
        emissiveMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [0.4, 0.5], scale: [1.4, 1.5], rotation: 0.6 },
        },
        normalMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [0.7, 0.8], scale: [1.7, 1.8], rotation: 0.9 },
        },
        metallicMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [1.0, 1.1], scale: [2.0, 2.1], rotation: 1.2 },
        },
        aoMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [1.3, 1.4], scale: [2.3, 2.4], rotation: 1.5 },
        },
        lightMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [1.6, 1.7], scale: [2.6, 2.7], rotation: 1.8 },
        },
        bumpMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [1.9, 2.0], scale: [2.9, 3.0], rotation: 2.1 },
        },
        anisotropyMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [2.2, 2.3], scale: [3.2, 3.3], rotation: 2.4 },
        },
        alphaMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [2.5, 2.6], scale: [3.5, 3.6], rotation: 2.7 },
        },
        transmissionMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [2.8, 2.9], scale: [3.8, 3.9], rotation: 3.0 },
        },
      }),
    ]);

    const uvMeta = (slot: number): number[] => {
      const start = (MATERIAL_TEX_UV_META_VEC4_OFFSET + slot * MATERIAL_TEX_UV_META_VEC4S_PER_MAP) * 4;
      return Array.from(descriptors.slice(start, start + 8));
    };

    expectCloseArray(uvMeta(0), [1, 0.1, 0.2, 0.3, 1.1, 1.2, 0, 0]);
    expectCloseArray(uvMeta(1), [0, 0.4, 0.5, 0.6, 1.4, 1.5, 0, 0]);
    expectCloseArray(uvMeta(2), [1, 0.7, 0.8, 0.9, 1.7, 1.8, 0, 0]);
    expectCloseArray(uvMeta(3), [0, 1.0, 1.1, 1.2, 2.0, 2.1, 0, 0]);
    expectCloseArray(uvMeta(4), [1, 1.3, 1.4, 1.5, 2.3, 2.4, 0, 0]);
    expectCloseArray(uvMeta(5), [0, 1.6, 1.7, 1.8, 2.6, 2.7, 0, 0]);
    expectCloseArray(uvMeta(6), [1, 1.9, 2.0, 2.1, 2.9, 3.0, 0, 0]);
    expectCloseArray(uvMeta(7), [0, 2.2, 2.3, 2.4, 3.2, 3.3, 0, 0]);
    expectCloseArray(uvMeta(8), [1, 2.5, 2.6, 2.7, 3.5, 3.6, 0, 0]);
    expectCloseArray(uvMeta(9), [0, 2.8, 2.9, 3.0, 3.8, 3.9, 0, 0]);
  });

  it('defaults per-map UV metadata to uv0 and identity transform', () => {
    const { descriptors } = collectMaterialTextures([mat({ normalMap: { handle: {} } })]);
    const start = (MATERIAL_TEX_UV_META_VEC4_OFFSET + 2 * MATERIAL_TEX_UV_META_VEC4S_PER_MAP) * 4;
    expect(Array.from(descriptors.slice(start, start + 8))).toEqual([0, 0, 0, 0, 1, 1, 0, 0]);
  });
});

describe('material-texture host↔WGSL contract (P2 lockstep)', () => {
  it('WGSL MATERIAL_TEX_VEC4_STRIDE matches the host descriptor stride', () => {
    expect(MATERIAL_TEX_FLOAT_STRIDE).toBe(MATERIAL_TEX_VEC4_STRIDE * 4);
    // The WGSL sampler indexes materialTexDescriptors with this exact stride;
    // drift silently misaligns every per-material texture read.
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL).toContain(
      `const MATERIAL_TEX_VEC4_STRIDE = ${MATERIAL_TEX_VEC4_STRIDE}u;`,
    );
  });

  it('group-3 WGSL declares the P2 texture bindings + the sampler fn', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain('@group(3) @binding(1) var<storage, read> meshUvs');
    expect(wgsl).toContain('@group(3) @binding(2) var<storage, read> materialTexDescriptors');
    expect(wgsl).toContain('@group(3) @binding(3) var materialTextures: texture_2d_array<f32>');
    expect(wgsl).toContain('@group(3) @binding(4) var materialTexSampler: sampler');
    expect(wgsl).toContain('@group(3) @binding(10) var<storage, read> meshTangents');
    expect(wgsl).toContain('fn sampleBaseColorTexture(');
    expect(wgsl).toContain('fn sampleAlphaTexture(');
    expect(wgsl).toContain('sampleAlphaTexture(matId, triIndex, baryVW)');
    expect(wgsl).toContain('fn sampleTransmissionTexture(');
    expect(wgsl).toContain('fn sampleClearcoatTexture(');
    expect(wgsl).toContain('fn sampleSpecularIntensityTexture(');
    expect(wgsl).toContain('fn applyClearcoatNormalMap(');
    expect(wgsl).toContain('wrappedUv * uvFitScale');
    expect(wgsl).toContain('materialTexDescriptors[base + 7u].xy');
    expect(wgsl).toContain('materialTexDescriptors[base + 8u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + 11u].zw');
    expect(wgsl).toContain('fn wrapTextureCoord(coord: f32, mode: f32) -> f32');
    expect(wgsl).toContain('materialTexDescriptors[base + 12u].xy');
    expect(wgsl).toContain('materialTexDescriptors[base + 13u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + 16u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + 37u].x');
    expect(wgsl).toContain('materialTexDescriptors[base + 38u].w');
    expect(wgsl).toContain('const MATERIAL_TEX_UV_CLEARCOAT_NORMAL = 65u;');
    expect(wgsl).toContain('let clearcoatNormalIdx = i32(materialTexDescriptors[base + 63u].x);');
    expect(wgsl).toContain('let clearcoatNormalScale = materialTexDescriptors[base + 63u].y;');
  });

  it('normal maps consume authored tangents with handedness before falling back to derived tangents', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain(
      'fn applyNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f, instanceIndex: u32)',
    );
    expect(wgsl).toContain('fn buildShadingTangentFrame(triIndex: u32, baryVW: vec2f, normal: vec3f, instanceIndex: u32)');
    expect(wgsl).toContain('let ta = meshTangents[tri.x];');
    expect(wgsl).toContain('let handednessRaw = ta.w * u + tb.w * v + tc.w * w;');
    expect(wgsl).toContain('frame.bitangent = cross(normal, tangent) * handedness;');
    expect(wgsl).toContain('if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX && params.tlasNodeCount != 0u)');
    expect(wgsl).toContain('let l2w0 = tlasInstanceLocalToWorld[m];');
    expect(wgsl).toContain('tangent = transformDirectionCols(l2w0, l2w1, l2w2, tangent);');
    expect(wgsl).toContain('let normalScale = materialTexDescriptors[base + 5u].w;');
    expect(wgsl).toContain('tn.x = tn.x * normalScale;');
    expect(wgsl).toContain('tn.y = tn.y * normalScale;');
  });

  it('group-3 WGSL samples every consumed map with its own UV metadata slot', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain('const MATERIAL_TEX_UV_BASE_COLOR = 17u;');
    expect(wgsl).toContain('const MATERIAL_TEX_UV_TRANSMISSION = 35u;');
    expect(wgsl).toContain('const MATERIAL_TEX_UV_CLEARCOAT = 47u;');
    expect(wgsl).toContain('const MATERIAL_TEX_UV_SPECULAR_INTENSITY = 61u;');
    expect(wgsl).toContain('const MATERIAL_TEX_UV_CLEARCOAT_NORMAL = 65u;');
    expect(wgsl).toContain('let uvMeta = materialTexDescriptors[base + uvMetaOffset];');
    expect(wgsl).toContain('let mipCount = f32(textureNumLevels(materialTextures));');
    expect(wgsl).toContain('let mipCount = f32(textureNumLevels(materialTexturesLinear));');
    expect(wgsl).toContain('textureSampleLevel(materialTextures, materialTexSampler, fittedUv, layerIdx, lod)');
    expect(wgsl).toContain('textureSampleLevel(materialTexturesLinear, materialTexSampler, fittedUv, layerIdx, lod)');
    expect(wgsl).toContain('sampleMaterialLayer(i32(materialTexDescriptors[base].x), base, triIndex, baryVW, MATERIAL_TEX_UV_BASE_COLOR');
    expect(wgsl).toContain('sampleMaterialLayer(i32(materialTexDescriptors[base].w), base, triIndex, baryVW, MATERIAL_TEX_UV_EMISSIVE');
    expect(wgsl).toContain('sampleMaterialLayerLinear(normalIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_NORMAL');
    expect(wgsl).toContain('sampleMaterialLayerLinear(i32(materialTexDescriptors[base].z), base, triIndex, baryVW, MATERIAL_TEX_UV_ORM');
    expect(wgsl).toContain('sampleMaterialLayerLinear(aoIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_AO');
    expect(wgsl).toContain('sampleMaterialLayerLinear(lmIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_LIGHT');
    expect(wgsl).toContain('sampleMaterialLayerLinear(anisoIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_ANISOTROPY');
    expect(wgsl).toContain('sampleMaterialLayerLinear(bumpIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_BUMP');
    expect(wgsl).toContain('sampleMaterialLayerLinear(alphaIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_ALPHA');
    expect(wgsl).toContain('sampleMaterialLayerLinear(transmissionIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_TRANSMISSION');
    expect(wgsl).toContain('sampleMaterialLayerLinear(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_CLEARCOAT');
    expect(wgsl).toContain('sampleMaterialLayer(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_SHEEN_COLOR');
    expect(wgsl).toContain('sampleMaterialLayer(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_SPECULAR_COLOR');
    expect(wgsl).toContain('sampleMaterialLayerLinear(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_SPECULAR_INTENSITY');
    expect(wgsl).toContain('clearcoatNormalIdx,');
    expect(wgsl).not.toContain('All maps of a material share its baseColor UV transform');
  });
});
