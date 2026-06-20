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
  MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET,
  MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_FILTER_POLICY_FLOATS_PER_MAP,
  MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET,
  MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET,
  MATERIAL_TEX_MIP_POLICY_MAP_COUNT,
  MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET,
  MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET,
  MATERIAL_TEX_THICKNESS_VEC4_OFFSET,
  MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET,
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

function wrapTextureCoordReference(coord: number, mode: 0 | 1 | 2): number {
  if (mode === 1) {
    return Math.min(Math.max(coord, 0), 0.999999);
  }
  if (mode === 2) {
    const period = coord - 2 * Math.floor(coord * 0.5);
    const mirrored = period <= 1 ? period : 2 - period;
    return Math.min(Math.max(mirrored, 0), 0.999999);
  }
  return coord - Math.floor(coord);
}

function transformedUvReference(
  rawUv: readonly [number, number],
  offset: readonly [number, number],
  scale: readonly [number, number],
  rotation: number,
): readonly [number, number] {
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  return [
    scale[0] * c * rawUv[0] + scale[0] * s * rawUv[1] + offset[0],
    -scale[1] * s * rawUv[0] + scale[1] * c * rawUv[1] + offset[1],
  ];
}

function bumpForwardDifferenceReference(
  height: (uv: readonly [number, number]) => number,
  rawUv: readonly [number, number],
  linearTextureDims: readonly [number, number],
  uvFitScale: readonly [number, number],
): readonly [number, number] {
  const sourceDims = [
    Math.max(linearTextureDims[0] * uvFitScale[0], 1),
    Math.max(linearTextureDims[1] * uvFitScale[1], 1),
  ] as const;
  const step = [1 / sourceDims[0], 1 / sourceDims[1]] as const;
  const hC = height(rawUv);
  const hU = height([rawUv[0] + step[0], rawUv[1]]);
  const hV = height([rawUv[0], rawUv[1] + step[1]]);
  return [(hU - hC) / step[0], (hV - hC) / step[1]];
}

function normalize3(v: readonly [number, number, number]): readonly [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 1e-12 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 0];
}

function bumpPerturbedNormalReference(
  gradient: readonly [number, number],
  bumpScale: number,
): readonly [number, number, number] {
  return normalize3([
    -bumpScale * gradient[0],
    -bumpScale * gradient[1],
    1,
  ]);
}

function parseMaterialTexUvConstants(wgsl: string): Map<string, number> {
  const constants = new Map<string, number>();
  const re = /const\s+(MATERIAL_TEX_UV_[A-Z0-9_]+)\s*=\s*(\d+)u;/g;
  let match: RegExpExecArray | null = re.exec(wgsl);
  while (match != null) {
    constants.set(match[1]!, Number(match[2]));
    match = re.exec(wgsl);
  }
  return constants;
}

function parseMaterialTexConstants(wgsl: string): Map<string, number> {
  const constants = new Map<string, number>();
  const re = /const\s+(MATERIAL_TEX(?:_[A-Z0-9]+)+)\s*=\s*(\d+)u;/g;
  let match: RegExpExecArray | null = re.exec(wgsl);
  while (match != null) {
    constants.set(match[1]!, Number(match[2]));
    match = re.exec(wgsl);
  }
  return constants;
}

function expectedMainUvMetaOffset(slot: number): number {
  return MATERIAL_TEX_UV_META_VEC4_OFFSET + slot * MATERIAL_TEX_UV_META_VEC4S_PER_MAP;
}

function expectedExtensionUvMetaOffset(slot: number): number {
  return MATERIAL_TEX_EXTENSION_UV_META_VEC4_OFFSET + slot * MATERIAL_TEX_UV_META_VEC4S_PER_MAP;
}

describe('collectMaterialTextures (P2 host)', () => {
  it('dedups shared texture handles + indexes per material', () => {
    const tex = { id: 'A' };
    const { sources, sourceInfos, descriptors } = collectMaterialTextures([
      mat({ baseColorMap: { handle: tex, texCoord: 1 } }),
      mat({ baseColorMap: { handle: tex, texCoord: 2 } }), // same handle → dedup to index 0
      mat({}), // no map
    ]);
    expect(sources).toEqual([tex]);
    expect(sourceInfos).toEqual([
      {
        layer: 0,
        uses: [
          { materialIndex: 0, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 1 },
          { materialIndex: 1, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 2 },
        ],
      },
    ]);
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

  it('collects normal + roughness/metallic maps into a SEPARATE linear source list (own index space)', () => {
    const baseTex = { id: 'base' };
    const normTex = { id: 'norm' };
    const mrTex = { id: 'mr' };
    const { sources, linearSources, descriptors } = collectMaterialTextures([
      mat({ baseColorMap: { handle: baseTex }, normalMap: { handle: normTex }, roughnessMap: { handle: mrTex } }),
    ]);
    expect(sources).toEqual([baseTex]);          // sRGB list: baseColor only
    expect(linearSources).toEqual([normTex, mrTex]); // linear list: normal + combined metallicRoughness
    expect(descriptors[0]).toBe(0);  // baseColorIdx → sRGB 0
    expect(descriptors[1]).toBe(0);  // normalIdx → linear 0 (separate space)
    expect(descriptors[2]).toBe(1);  // roughnessMapIdx → linear 1
    expect(descriptors[26]).toBe(1); // metallicMapIdx falls back to the same combined layer
  });

  it('keeps distinct roughnessMap and metallicMap handles instead of dropping metallicMap', () => {
    const roughnessTex = { id: 'roughness' };
    const metallicTex = { id: 'metallic' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({
        roughnessMap: {
          handle: roughnessTex,
          texCoord: 1,
          wrapS: 'clamp-to-edge',
          wrapT: 'repeat',
          transform: { offset: [0.1, 0.2], scale: [1.1, 1.2], rotation: 0.3 },
        },
        metallicMap: {
          handle: metallicTex,
          texCoord: 0,
          wrapS: 'mirrored-repeat',
          wrapT: 'clamp-to-edge',
          transform: { offset: [0.4, 0.5], scale: [1.4, 1.5], rotation: 0.6 },
        },
      }),
    ]);

    expect(linearSources).toEqual([roughnessTex, metallicTex]);
    expect(descriptors[2]).toBe(0);
    expect(descriptors[26]).toBe(1);
    expect(Array.from(descriptors.slice(58, 62))).toEqual([1, 0, 2, 1]);
    const uvMeta = (slot: number): number[] => {
      const start = (MATERIAL_TEX_UV_META_VEC4_OFFSET + slot * MATERIAL_TEX_UV_META_VEC4S_PER_MAP) * 4;
      return Array.from(descriptors.slice(start, start + 8));
    };
    expectCloseArray(uvMeta(3), [1, 0.1, 0.2, 0.3, 1.1, 1.2, 0, 0]);
    expectCloseArray(uvMeta(4), [0, 0.4, 0.5, 0.6, 1.4, 1.5, 0, 0]);
  });

  it('packs per-map mip and filter policies without shifting existing descriptor fields', () => {
    const baseTex = { id: 'base' };
    const roughnessTex = { id: 'roughness' };
    const sheenTex = { id: 'sheen' };
    const frontNormal = { id: 'front-normal' };
    const backNormal = { id: 'back-normal' };
    const { descriptors } = collectMaterialTextures([
      mat({
        baseColorMap: { handle: baseTex, magFilter: 'nearest', minFilter: 'nearest', mipFilter: 'none' },
        roughnessMap: { handle: roughnessTex, minFilter: 'nearest', mipFilter: 'nearest' },
        sheenColorMap: { handle: sheenTex, mipFilter: 'none' },
        frontLayer: {
          transmission: [1, 1, 1],
          normalMap: { handle: frontNormal, mipFilter: 'nearest' },
        },
        backLayer: {
          transmission: [1, 1, 1],
          normalMap: { handle: backNormal, mipFilter: 'none' },
        },
      }),
    ]);
    const mipBase = MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET * 4;
    const filterBase = MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET * 4;
    const mipPolicy = (slot: number): number => descriptors[mipBase + slot] ?? Number.NaN;
    const filterPolicy = (slot: number): readonly [number, number] => {
      const offset = filterBase + slot * MATERIAL_TEX_FILTER_POLICY_FLOATS_PER_MAP;
      return [
        descriptors[offset] ?? Number.NaN,
        descriptors[offset + 1] ?? Number.NaN,
      ];
    };

    expect(MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET).toBe(82);
    expect(MATERIAL_TEX_MIP_POLICY_MAP_COUNT).toBe(23);
    expect(MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET).toBe(88);
    expect(MATERIAL_TEX_VEC4_STRIDE).toBe(100);
    expect(mipPolicy(0)).toBe(0);  // baseColorMap: no mip lookup
    expect(mipPolicy(2)).toBe(2);  // normalMap absent/default: linear
    expect(mipPolicy(3)).toBe(1);  // roughnessMap: nearest mip selection
    expect(mipPolicy(13)).toBe(0); // sheenColorMap: no mip lookup
    expect(mipPolicy(21)).toBe(1); // frontLayer.normalMap
    expect(mipPolicy(22)).toBe(0); // backLayer.normalMap
    expect(filterPolicy(0)).toEqual([0, 0]); // baseColorMap: nearest mag/min
    expect(filterPolicy(2)).toEqual([1, 1]); // normalMap absent/default: linear
    expect(filterPolicy(3)).toEqual([1, 0]); // roughnessMap: default mag, nearest min
    expect(filterPolicy(13)).toEqual([1, 1]); // sheenColorMap: default linear
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

  it('collects front/back layer normal maps with face-specific descriptor lanes', () => {
    const topNormal = { id: 'top-normal' };
    const frontNormal = { id: 'front-normal' };
    const backNormal = { id: 'back-normal' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({
        normalMap: { handle: topNormal },
        frontLayer: {
          transmission: [1, 1, 1],
          normalMap: {
            handle: frontNormal,
            texCoord: 1,
            wrapS: 'clamp-to-edge',
            wrapT: 'mirrored-repeat',
            transform: { offset: [0.11, 0.12], scale: [0.21, 0.22], rotation: 0.31 },
          },
          normalScale: 0.75,
        },
        backLayer: {
          transmission: [1, 1, 1],
          normalMap: {
            handle: backNormal,
            texCoord: 0,
            wrapS: 'repeat',
            wrapT: 'clamp-to-edge',
            transform: { offset: [0.41, 0.42], scale: [0.51, 0.52], rotation: 0.61 },
          },
          normalScale: 0.5,
        },
      }),
    ]);
    const layerBase = MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET * 4;
    const layerFit = MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET * 4;
    const layerWrap = MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET * 4;
    const layerMeta = MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET * 4;

    expect(linearSources).toEqual([topNormal, frontNormal, backNormal]);
    expect(descriptors[layerBase]).toBe(1);
    expect(descriptors[layerBase + 1]).toBeCloseTo(0.75);
    expect(descriptors[layerBase + 2]).toBe(2);
    expect(descriptors[layerBase + 3]).toBeCloseTo(0.5);
    expectCloseArray(descriptors.slice(layerFit, layerFit + 4), [1, 1, 1, 1]);
    expectCloseArray(descriptors.slice(layerWrap, layerWrap + 4), [1, 2, 0, 1]);
    expectCloseArray(descriptors.slice(layerMeta, layerMeta + 8), [1, 0.11, 0.12, 0.31, 0.21, 0.22, 0, 0]);
    expectCloseArray(descriptors.slice(layerMeta + 8, layerMeta + 16), [0, 0.41, 0.42, 0.61, 0.51, 0.52, 0, 0]);

    applyMaterialTextureUvFitScales(descriptors, [], [[1, 1], [0.5, 0.25], [0.75, 0.5]]);
    expectCloseArray(descriptors.slice(layerFit, layerFit + 4), [0.5, 0.25, 0.75, 0.5]);
  });

  it('packs AO, light-map, and environment scalar lanes with defaults', () => {
    const { descriptors } = collectMaterialTextures([
      mat({ aoMapIntensity: 0.25, lightMapIntensity: 2.5, envMapIntensity: 0.4 }),
      mat({}),
    ]);

    expect(descriptors[16]).toBeCloseTo(0.25);
    expect(descriptors[17]).toBeCloseTo(2.5);
    expect(descriptors[19]).toBeCloseTo(0.4);
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + 16]).toBe(1);
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + 17]).toBe(1);
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + 19]).toBe(1);
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

  it('collects thicknessMap as LINEAR KHR volume data and packs layer, UV, wrap, and transform', () => {
    const thicknessTex = { id: 'thickness' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({
        thickness: 0.5,
        thicknessMap: {
          handle: thicknessTex,
          texCoord: 1,
          wrapS: 'clamp-to-edge',
          wrapT: 'mirrored-repeat',
          transform: { offset: [0.2, 0.3], scale: [0.4, 0.5], rotation: 0.6 },
        },
      }),
    ]);
    const thicknessBase = MATERIAL_TEX_THICKNESS_VEC4_OFFSET * 4;
    const thicknessWrap = MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET * 4;
    const thicknessMeta = MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET * 4;

    expect(linearSources).toEqual([thicknessTex]);
    expect(descriptors[thicknessBase]).toBe(0);
    expect(descriptors[thicknessBase + 1]).toBe(1);
    expect(descriptors[thicknessBase + 2]).toBe(1);
    expect(Array.from(descriptors.slice(thicknessWrap, thicknessWrap + 4))).toEqual([1, 2, 0, 0]);
    expectCloseArray(
      descriptors.slice(thicknessMeta, thicknessMeta + 8),
      [1, 0.2, 0.3, 0.6, 0.4, 0.5, 0, 0],
    );
  });

  it('applies thicknessMap UV-fit scale from the linear texture array', () => {
    const thicknessTex = { id: 'thickness-fit' };
    const { descriptors } = collectMaterialTextures([
      mat({ thicknessMap: { handle: thicknessTex } }),
    ]);
    applyMaterialTextureUvFitScales(descriptors, [], [[0.25, 0.5]]);
    const thicknessBase = MATERIAL_TEX_THICKNESS_VEC4_OFFSET * 4;
    expectCloseArray(descriptors.slice(thicknessBase + 1, thicknessBase + 3), [0.25, 0.5]);
  });

  it('falls back to metallicMap as a combined roughness/metallic map when roughnessMap is absent', () => {
    const mrTex = { id: 'mr' };
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({ metallicMap: { handle: mrTex } }),
    ]);
    expect(linearSources).toEqual([mrTex]);
    expect(descriptors[2]).toBe(0);  // roughnessMapIdx → metallicMap's G channel
    expect(descriptors[26]).toBe(0); // metallicMapIdx → metallicMap's B channel
  });

  it('defaults: opaque(0), cutoff 0.5, opacity 1, identity scale', () => {
    const { descriptors } = collectMaterialTextures([mat({ baseColorMap: { handle: {} } })]);
    expect(descriptors[4]).toBe(0);
    expect(descriptors[5]).toBe(0.5);
    expect(descriptors[6]).toBe(1);
    expect(descriptors[10]).toBe(1);
    expect(descriptors[11]).toBe(1);
    for (let i = 28; i < 52; i += 1) {
      expect(descriptors[i]).toBe(1);
    }
    const extUvFitStart = MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET * 4;
    for (let i = extUvFitStart; i < extUvFitStart + 16; i += 1) {
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
    const extIndexStart = MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET * 4;
    expect(Array.from(descriptors.slice(extIndexStart, extIndexStart + 8))).toEqual([0, 1, 0, 2, 3, 4, 1, 5]);
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

    const extUvFitStart = MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET * 4;
    expectCloseArray(descriptors.slice(extUvFitStart, extUvFitStart + 16), [
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

    const extWrapStart = MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET * 4;
    expect(Array.from(descriptors.slice(extWrapStart, extWrapStart + 4))).toEqual([1, 2, 0, 0]);
    expect(Array.from(descriptors.slice(extWrapStart + 12, extWrapStart + 16))).toEqual([0, 0, 2, 1]);

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
    expectCloseArray(descriptors.slice(36, 40), [1, 0.5, 0.25, 0.5]);
    expectCloseArray(descriptors.slice(40, 44), [0.8, 0.6, 0.4, 0.3]);
    expectCloseArray(descriptors.slice(44, 48), [0.9, 0.7, 0.2, 0.1]);
    expectCloseArray(descriptors.slice(48, 52), [0.6, 0.4, 1, 1]);
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

    expect(Array.from(descriptors.slice(52, 56))).toEqual([1, 2, 2, 1]);
    expect(Array.from(descriptors.slice(56, 60))).toEqual([1, 1, 2, 0]);
    expect(Array.from(descriptors.slice(60, 64))).toEqual([2, 0, 0, 1]);
    expect(Array.from(descriptors.slice(64, 68))).toEqual([2, 2, 1, 0]);
    expect(Array.from(descriptors.slice(68, 72))).toEqual([0, 2, 1, 2]);
    expect(Array.from(descriptors.slice(72, 76))).toEqual([0, 1, 0, 0]);
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
        roughnessMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [1.0, 1.1], scale: [2.0, 2.1], rotation: 1.2 },
        },
        metallicMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [1.3, 1.4], scale: [2.3, 2.4], rotation: 1.5 },
        },
        aoMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [1.6, 1.7], scale: [2.6, 2.7], rotation: 1.8 },
        },
        lightMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [1.9, 2.0], scale: [2.9, 3.0], rotation: 2.1 },
        },
        bumpMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [2.2, 2.3], scale: [3.2, 3.3], rotation: 2.4 },
        },
        anisotropyMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [2.5, 2.6], scale: [3.5, 3.6], rotation: 2.7 },
        },
        alphaMap: {
          handle: {},
          texCoord: 0,
          transform: { offset: [2.8, 2.9], scale: [3.8, 3.9], rotation: 3.0 },
        },
        transmissionMap: {
          handle: {},
          texCoord: 1,
          transform: { offset: [3.1, 3.2], scale: [4.1, 4.2], rotation: 3.3 },
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
    expectCloseArray(uvMeta(10), [1, 3.1, 3.2, 3.3, 4.1, 4.2, 0, 0]);
  });

  it('defaults per-map UV metadata to uv0 and identity transform', () => {
    const { descriptors } = collectMaterialTextures([mat({ normalMap: { handle: {} } })]);
    const start = (MATERIAL_TEX_UV_META_VEC4_OFFSET + 2 * MATERIAL_TEX_UV_META_VEC4S_PER_MAP) * 4;
    expect(Array.from(descriptors.slice(start, start + 8))).toEqual([0, 0, 0, 0, 1, 1, 0, 0]);
  });
});

describe('material-texture host↔WGSL contract (P2 lockstep)', () => {
  it('CPU oracle: wrap modes and KHR_texture_transform match the shader convention', () => {
    expect(wrapTextureCoordReference(-0.25, 0)).toBeCloseTo(0.75, 6);
    expect(wrapTextureCoordReference(1.25, 0)).toBeCloseTo(0.25, 6);
    expect(wrapTextureCoordReference(-0.25, 1)).toBe(0);
    expect(wrapTextureCoordReference(1.25, 1)).toBeCloseTo(0.999999, 6);
    expect(wrapTextureCoordReference(-0.25, 2)).toBeCloseTo(0.25, 6);
    expect(wrapTextureCoordReference(1.25, 2)).toBeCloseTo(0.75, 6);
    expect(wrapTextureCoordReference(2.25, 2)).toBeCloseTo(0.25, 6);

    const uv = transformedUvReference([0.25, 0.75], [0.1, 0.2], [2, 3], Math.PI / 2);
    expectCloseArray(uv, [1.6, -0.55]);
    expectCloseArray(
      [
        wrapTextureCoordReference(uv[0], 0) * 0.5,
        wrapTextureCoordReference(uv[1], 2) * 0.25,
      ],
      [0.3, 0.1375],
    );
  });

  it('CPU oracle: bump finite differences step in raw UV by uploaded source dimensions', () => {
    const height = ([u, v]: readonly [number, number]): number =>
      u * u + 2 * v * v + 0.25 * u * v;
    const rawUv = [0.2, 0.3] as const;

    const gradient = bumpForwardDifferenceReference(height, rawUv, [8, 10], [0.25, 0.5]);
    expectCloseArray(gradient, [0.975, 1.65]);

    const fixed512Gradient = bumpForwardDifferenceReference(height, rawUv, [512, 512], [1, 1]);
    expect(Math.abs(gradient[0] - fixed512Gradient[0])).toBeGreaterThan(0.49);
    expect(Math.abs(gradient[1] - fixed512Gradient[1])).toBeGreaterThan(0.39);

    expectCloseArray(
      bumpPerturbedNormalReference(gradient, 0.25),
      [-0.21878586532124316, -0.3704062336207192, 0.9022097549339923],
    );
  });

  it('WGSL MATERIAL_TEX_VEC4_STRIDE matches the host descriptor stride', () => {
    expect(MATERIAL_TEX_FLOAT_STRIDE).toBe(MATERIAL_TEX_VEC4_STRIDE * 4);
    // The WGSL sampler indexes materialTexDescriptors with this exact stride;
    // drift silently misaligns every per-material texture read.
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL).toContain(
      `const MATERIAL_TEX_VEC4_STRIDE = ${MATERIAL_TEX_VEC4_STRIDE}u;`,
    );
  });

  it('WGSL material descriptor block constants are derived from the host descriptor layout', () => {
    const constants = parseMaterialTexConstants(PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL);
    const expected: Readonly<Record<string, number>> = {
      MATERIAL_TEX_VEC4_STRIDE,
      MATERIAL_TEX_EXTENSION_INDEX: MATERIAL_TEX_EXTENSION_INDEX_VEC4_OFFSET,
      MATERIAL_TEX_EXTENSION_UV_FIT: MATERIAL_TEX_EXTENSION_UV_FIT_VEC4_OFFSET,
      MATERIAL_TEX_EXTENSION_WRAP: MATERIAL_TEX_EXTENSION_WRAP_VEC4_OFFSET,
      MATERIAL_TEX_CLEARCOAT_NORMAL: MATERIAL_TEX_CLEARCOAT_NORMAL_VEC4_OFFSET,
      MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP: MATERIAL_TEX_CLEARCOAT_NORMAL_WRAP_VEC4_OFFSET,
      MATERIAL_TEX_THICKNESS: MATERIAL_TEX_THICKNESS_VEC4_OFFSET,
      MATERIAL_TEX_THICKNESS_WRAP: MATERIAL_TEX_THICKNESS_WRAP_VEC4_OFFSET,
      MATERIAL_TEX_LAYER_NORMAL: MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET,
      MATERIAL_TEX_LAYER_NORMAL_UV_FIT: MATERIAL_TEX_LAYER_NORMAL_UV_FIT_VEC4_OFFSET,
      MATERIAL_TEX_LAYER_NORMAL_WRAP: MATERIAL_TEX_LAYER_NORMAL_WRAP_VEC4_OFFSET,
      MATERIAL_TEX_MIP_POLICY: MATERIAL_TEX_MIP_POLICY_VEC4_OFFSET,
      MATERIAL_TEX_FILTER_POLICY: MATERIAL_TEX_FILTER_POLICY_VEC4_OFFSET,
    };

    expect(
      Object.fromEntries(Object.keys(expected).map((name) => [name, constants.get(name)])),
    ).toEqual(expected);
  });

  it('WGSL material UV metadata constants are derived from the host descriptor layout', () => {
    const constants = parseMaterialTexUvConstants(PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL);
    const expected: Readonly<Record<string, number>> = {
      MATERIAL_TEX_UV_BASE_COLOR: expectedMainUvMetaOffset(0),
      MATERIAL_TEX_UV_EMISSIVE: expectedMainUvMetaOffset(1),
      MATERIAL_TEX_UV_NORMAL: expectedMainUvMetaOffset(2),
      MATERIAL_TEX_UV_ROUGHNESS: expectedMainUvMetaOffset(3),
      MATERIAL_TEX_UV_METALLIC: expectedMainUvMetaOffset(4),
      MATERIAL_TEX_UV_AO: expectedMainUvMetaOffset(5),
      MATERIAL_TEX_UV_LIGHT: expectedMainUvMetaOffset(6),
      MATERIAL_TEX_UV_BUMP: expectedMainUvMetaOffset(7),
      MATERIAL_TEX_UV_ANISOTROPY: expectedMainUvMetaOffset(8),
      MATERIAL_TEX_UV_ALPHA: expectedMainUvMetaOffset(9),
      MATERIAL_TEX_UV_TRANSMISSION: expectedMainUvMetaOffset(10),
      MATERIAL_TEX_UV_CLEARCOAT: expectedExtensionUvMetaOffset(0),
      MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS: expectedExtensionUvMetaOffset(1),
      MATERIAL_TEX_UV_SHEEN_COLOR: expectedExtensionUvMetaOffset(2),
      MATERIAL_TEX_UV_SHEEN_ROUGHNESS: expectedExtensionUvMetaOffset(3),
      MATERIAL_TEX_UV_IRIDESCENCE: expectedExtensionUvMetaOffset(4),
      MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS: expectedExtensionUvMetaOffset(5),
      MATERIAL_TEX_UV_SPECULAR_COLOR: expectedExtensionUvMetaOffset(6),
      MATERIAL_TEX_UV_SPECULAR_INTENSITY: expectedExtensionUvMetaOffset(7),
      MATERIAL_TEX_UV_CLEARCOAT_NORMAL: MATERIAL_TEX_CLEARCOAT_NORMAL_UV_META_VEC4_OFFSET,
      MATERIAL_TEX_UV_THICKNESS: MATERIAL_TEX_THICKNESS_UV_META_VEC4_OFFSET,
      MATERIAL_TEX_UV_FRONT_LAYER_NORMAL: MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET,
      MATERIAL_TEX_UV_BACK_LAYER_NORMAL:
        MATERIAL_TEX_LAYER_NORMAL_UV_META_VEC4_OFFSET + MATERIAL_TEX_UV_META_VEC4S_PER_MAP,
    };

    expect(Object.fromEntries(constants)).toEqual(expected);
  });

  it('group-3 WGSL declares the P2 texture bindings + the sampler fn', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain('@group(3) @binding(1) var<storage, read> meshUvs');
    expect(wgsl).toContain('@group(3) @binding(2) var<storage, read> materialTexDescriptors');
    expect(wgsl).toContain('@group(3) @binding(3) var materialTextures: texture_2d_array<f32>');
    expect(wgsl).toContain('@group(3) @binding(4) var materialTexSampler: sampler');
    expect(wgsl).toContain('@group(3) @binding(10) var<storage, read> meshTangents');
    expect(wgsl).toContain('@group(3) @binding(11) var<storage, read> meshVertexColors');
    expect(wgsl).toContain('fn sampleBaseColorTexture(');
    expect(wgsl).toContain('fn sampleVertexColor(triIndex: u32, baryVW: vec2f) -> vec4f');
    expect(wgsl).toContain('return meshVertexColors[tri.x] * u + meshVertexColors[tri.y] * v + meshVertexColors[tri.z] * w;');
    expect(wgsl).toContain('fn sampleAlphaTexture(');
    expect(wgsl).toContain('sampleVertexColor(triIndex, baryVW).a');
    expect(wgsl).toContain('sampleAlphaTexture(matId, triIndex, baryVW)');
    expect(wgsl).toContain('fn sampleTransmissionTexture(');
    expect(wgsl).toContain('fn sampleVolumeThicknessTexture(');
    expect(wgsl).toContain('fn sampleClearcoatTexture(');
    expect(wgsl).toContain('fn sampleSpecularIntensityTexture(');
    expect(wgsl).toContain('fn applyClearcoatNormalMap(');
    expect(wgsl).toContain('wrappedUv * uvFitScale');
    expect(wgsl).toContain('materialTexDescriptors[base + 7u].xy');
    expect(wgsl).toContain('materialTexDescriptors[base + 8u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + 11u].zw');
    expect(wgsl).toContain('fn wrapTextureCoord(coord: f32, mode: f32) -> f32');
    expect(wgsl).toContain('fn materialTextureMipPolicy(base: u32, slot: u32) -> f32');
    expect(wgsl).toContain('fn materialTexturePolicyLod(lod: f32, mipCount: f32, mipPolicy: f32) -> f32');
    expect(wgsl).toContain('fn materialTextureFilterPolicy(base: u32, slot: u32) -> vec2f');
    expect(wgsl).toContain('materialTexDescriptors[base + 12u].xy');
    expect(wgsl).toContain('materialTexDescriptors[base + 13u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + 16u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX].x');
    expect(wgsl).toContain('materialTexDescriptors[base + MATERIAL_TEX_EXTENSION_INDEX + 1u].w');
    expect(wgsl).toContain(
      `const MATERIAL_TEX_LAYER_NORMAL = ${MATERIAL_TEX_LAYER_NORMAL_VEC4_OFFSET}u;`,
    );
    expect(wgsl).toContain(
      'let clearcoatNormalIdx = i32(materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL].x);',
    );
    expect(wgsl).toContain(
      'let clearcoatNormalScale = materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL].y;',
    );
    expect(wgsl).toContain(
      'let thicknessIdx = i32(materialTexDescriptors[base + MATERIAL_TEX_THICKNESS].x);',
    );
    expect(wgsl).toContain('materialTexDescriptors[base + MATERIAL_TEX_THICKNESS].yz');
  });

  it('normal maps consume authored tangents with handedness before falling back to derived tangents', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain(
      'fn applyNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f, instanceIndex: u32, isFrontFace: bool)',
    );
    expect(wgsl).toContain('fn buildShadingTangentFrame(triIndex: u32, baryVW: vec2f, normal: vec3f, instanceIndex: u32)');
    expect(wgsl).toContain('let ta = meshTangents[tri.x];');
    expect(wgsl).toContain('let handednessRaw = ta.w * u + tb.w * v + tc.w * w;');
    expect(wgsl).toContain('frame.bitangent = cross(normal, tangent) * handedness;');
    expect(wgsl).toContain('if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX && params.tlasNodeCount != 0u)');
    expect(wgsl).toContain('let l2w0 = tlasInstanceLocalToWorld[m];');
    expect(wgsl).toContain('tangent = transformDirectionCols(l2w0, l2w1, l2w2, tangent);');
    expect(wgsl).toContain('var normalScale = materialTexDescriptors[base + 5u].w;');
    expect(wgsl).toContain('let layerIdx = select(i32(layerNormal.z), i32(layerNormal.x), isFrontFace);');
    expect(wgsl).toContain('normalUvMetaOffset = select(MATERIAL_TEX_UV_BACK_LAYER_NORMAL, MATERIAL_TEX_UV_FRONT_LAYER_NORMAL, isFrontFace);');
    expect(wgsl).toContain('normalUvFitScale = select(layerUvFit.zw, layerUvFit.xy, isFrontFace);');
    expect(wgsl).toContain('tn.x = tn.x * normalScale;');
    expect(wgsl).toContain('tn.y = tn.y * normalScale;');
  });

  it('group-3 WGSL samples every consumed map with its own UV metadata slot', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain('let uvMeta = materialTexDescriptors[base + uvMetaOffset];');
    expect(wgsl).toContain('let mipCount = f32(textureNumLevels(materialTextures));');
    expect(wgsl).toContain('let mipCount = f32(textureNumLevels(materialTexturesLinear));');
    expect(wgsl).toContain('let mipPolicy = materialTextureMipPolicy(base, mipPolicySlot);');
    expect(wgsl).toContain('let policyLod = materialTexturePolicyLod(lod, mipCount, mipPolicy);');
    expect(wgsl).toContain('let filterPolicy = materialTextureFilterPolicy(base, mipPolicySlot);');
    expect(wgsl).toContain('textureLoad(materialTextures, coord0, layerIdx, lod0u)');
    expect(wgsl).toContain('textureLoad(materialTexturesLinear, coord0, layerIdx, lod0u)');
    expect(wgsl).toContain('textureSampleLevel(materialTextures, materialTexSampler, fittedUv, layerIdx, policyLod)');
    expect(wgsl).toContain('textureSampleLevel(materialTexturesLinear, materialTexSampler, fittedUv, layerIdx, policyLod)');
    expect(wgsl).toContain('sampleMaterialLayer(i32(materialTexDescriptors[base].x), base, triIndex, baryVW, MATERIAL_TEX_UV_BASE_COLOR');
    expect(wgsl).toContain('sampleMaterialLayer(i32(materialTexDescriptors[base].w), base, triIndex, baryVW, MATERIAL_TEX_UV_EMISSIVE');
    expect(wgsl).toContain('sampleMaterialLayerLinear(normalIdx, base, triIndex, baryVW, normalUvMetaOffset, normalUvFitScale, normalWrapMode');
    expect(wgsl).toContain('normalMipPolicySlot = select(MATERIAL_TEX_MIP_BACK_LAYER_NORMAL, MATERIAL_TEX_MIP_FRONT_LAYER_NORMAL, isFrontFace);');
    expect(wgsl).toContain('MATERIAL_TEX_UV_ROUGHNESS');
    expect(wgsl).toContain('MATERIAL_TEX_UV_METALLIC');
    expect(wgsl).toContain('sampleMaterialLayerLinear(aoIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_AO');
    expect(wgsl).toContain('sampleMaterialLayerLinear(lmIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_LIGHT');
    expect(wgsl).toContain('sampleMaterialLayerLinear(anisoIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_ANISOTROPY');
    expect(wgsl).toContain('sampleMaterialLayerLinearRawUv(bumpIdx, base, rawUv, MATERIAL_TEX_UV_BUMP');
    expect(wgsl).toContain('sampleMaterialLayerLinear(alphaIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_ALPHA');
    expect(wgsl).toContain('sampleMaterialLayerLinear(transmissionIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_TRANSMISSION');
    expect(wgsl).toContain('sampleMaterialLayerLinear(thicknessIdx, base, triIndex, baryVW, MATERIAL_TEX_UV_THICKNESS');
    expect(wgsl).toContain('sampleMaterialLayerLinear(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_CLEARCOAT');
    expect(wgsl).toContain('sampleMaterialLayer(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_SHEEN_COLOR');
    expect(wgsl).toContain('sampleMaterialLayer(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_SPECULAR_COLOR');
    expect(wgsl).toContain('sampleMaterialLayerLinear(idx, base, triIndex, baryVW, MATERIAL_TEX_UV_SPECULAR_INTENSITY');
    expect(wgsl).toContain('clearcoatNormalIdx,');
    expect(wgsl).not.toContain('All maps of a material share its baseColor UV transform');
  });

  it('bump maps finite-difference by raw UV and the uploaded source dimensions', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain('fn sampleMaterialLayerLinearRawUv(');
    expect(wgsl).toContain('textureSampleLevel(materialTexturesLinear, materialTexSampler, fittedUv, layerIdx, 0.0)');
    expect(wgsl).toContain('let rawUv0 = uva.xy * u + uvb.xy * v + uvc.xy * w;');
    expect(wgsl).toContain('let rawUv1 = uva.zw * u + uvb.zw * v + uvc.zw * w;');
    expect(wgsl).toContain('let linearDims = vec2f(textureDimensions(materialTexturesLinear, 0));');
    expect(wgsl).toContain('let sourceDims = max(linearDims * bumpUvFitScale, vec2f(1.0));');
    expect(wgsl).toContain('let texelStep = vec2f(1.0 / sourceDims.x, 1.0 / sourceDims.y);');
    expect(wgsl).toContain('rawUv + vec2f(texelStep.x, 0.0)');
    expect(wgsl).toContain('rawUv + vec2f(0.0, texelStep.y)');
    expect(wgsl).not.toContain('1.0 / 512.0');
  });
});
