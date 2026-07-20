/**
 * materialTexDescriptorGolden.test.ts — T2-A byte-identity pin for the material
 * texture descriptor pack.
 *
 * The descriptor float array is the WIRE FORMAT the WGSL sampler reads: the map
 * order and every lane offset are load-bearing. This test packs a single material
 * exercising ALL 23 texture-map slots (distinct texCoord / wrap / filter / mip /
 * transform + per-layer UV-fit) and asserts the packed output is byte-identical to
 * a golden captured BEFORE the TEXTURE_MAP_SLOTS single-source refactor. Any drift
 * in the map order or lane layout fails here.
 */
import { describe, expect, it } from 'vitest';
import type { MaterialSpec, TextureFilterMode, TextureMipFilterMode, TextureRef, TextureWrapMode } from '@vitrum/core';
import {
  applyMaterialTextureUvFitScales,
  collectMaterialTextures,
  MATERIAL_TEX_FLOAT_STRIDE,
} from '../scene/materialTextures.js';
import { MATERIAL_TEX_DESCRIPTOR_GOLDEN } from './fixtures/materialTexDescriptorGolden.js';

/**
 * A material that touches every map slot with a distinct handle + distinct
 * texCoord / wrap / mag-min-filter / mip / transform, so the golden pins the full
 * descriptor layout (not just the default-filled lanes). Handles are indexed `t0..`
 * in exactly the slot order the packer visits them.
 */
function richAllMapsMaterial(): MaterialSpec {
  let i = 0;
  const WRAP_S: readonly TextureWrapMode[] = ['repeat', 'clamp-to-edge', 'mirrored-repeat'];
  const WRAP_T: readonly TextureWrapMode[] = ['clamp-to-edge', 'mirrored-repeat', 'repeat'];
  const MIP: readonly TextureMipFilterMode[] = ['none', 'nearest', 'linear'];
  const T = (idx: number, tc: number): TextureRef => ({
    handle: { id: `t${idx}` },
    texCoord: tc % 2,
    wrapS: WRAP_S[idx % 3]!,
    wrapT: WRAP_T[idx % 3]!,
    magFilter: (idx % 2 ? 'nearest' : 'linear') as TextureFilterMode,
    minFilter: (idx % 3 ? 'nearest' : 'linear') as TextureFilterMode,
    mipFilter: MIP[idx % 3]!,
    transform: { offset: [0.01 * idx, 0.02 * idx], scale: [1 + 0.01 * idx, 1 + 0.02 * idx], rotation: 0.03 * idx },
  });
  return {
    baseColor: [1, 1, 1],
    roughness: 0.5,
    metallic: 0.2,
    normalScale: 0.7,
    aoMapIntensity: 0.3,
    lightMapIntensity: 1.5,
    bumpScale: 0.4,
    envMapIntensity: 0.8,
    anisotropy: 0.5,
    anisotropyRotation: 0.6,
    alphaMode: 'mask',
    alphaCutoff: 0.4,
    opacity: 0.9,
    transmission: 0.5,
    thickness: 0.7,
    clearcoatNormalScale: 0.6,
    baseColorMap: T(i++, 0),
    emissiveMap: T(i++, 1),
    normalMap: T(i++, 0),
    roughnessMap: T(i++, 1),
    metallicMap: T(i++, 0),
    aoMap: T(i++, 1),
    lightMap: T(i++, 0),
    bumpMap: T(i++, 1),
    anisotropyMap: T(i++, 0),
    alphaMap: T(i++, 1),
    transmissionMap: T(i++, 0),
    clearcoatMap: T(i++, 1),
    clearcoatRoughnessMap: T(i++, 0),
    sheenColorMap: T(i++, 1),
    sheenRoughnessMap: T(i++, 0),
    iridescenceMap: T(i++, 1),
    iridescenceThicknessMap: T(i++, 0),
    specularColorMap: T(i++, 1),
    specularIntensityMap: T(i++, 0),
    clearcoatNormalMap: T(i++, 1),
    thicknessMap: T(i++, 0),
    frontLayer: { transmission: [1, 1, 1], normalMap: T(i++, 1), normalScale: 0.55 },
    backLayer: { transmission: [1, 1, 1], normalMap: T(i++, 0), normalScale: 0.44 },
  };
}

describe('T2-A material descriptor golden (byte-identity)', () => {
  it('packs the rich all-maps material byte-identically to the pre-refactor golden', () => {
    const { descriptors, sources, linearSources, emissiveSources } =
      collectMaterialTextures([richAllMapsMaterial()]);
    // Distinct per-layer UV-fit scales so the uv-fit pass is exercised on every layer.
    const srgb = sources.map((_, k) => [0.5 + 0.01 * k, 0.6 + 0.01 * k] as [number, number]);
    const lin = linearSources.map((_, k) => [0.3 + 0.01 * k, 0.4 + 0.01 * k] as [number, number]);
    // T1-6 — emissive now has its own rgba16float array + index/scale space.
    const emissive = emissiveSources.map((_, k) => [0.7 + 0.01 * k, 0.8 + 0.01 * k] as [number, number]);
    applyMaterialTextureUvFitScales(descriptors, srgb, lin, emissive);

    expect(descriptors.length).toBe(MATERIAL_TEX_FLOAT_STRIDE);
    expect(descriptors.length).toBe(MATERIAL_TEX_DESCRIPTOR_GOLDEN.length);
    const actual = Array.from(descriptors).map((x) => (Object.is(x, -0) ? 0 : x));
    expect(actual).toStrictEqual([...MATERIAL_TEX_DESCRIPTOR_GOLDEN]);
  });
});
