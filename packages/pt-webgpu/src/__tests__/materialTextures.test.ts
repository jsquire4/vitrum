/**
 * materialTextures.test.ts — P2 host-side texture collection + descriptor pack.
 */
import { describe, it, expect } from 'vitest';
import {
  applyMaterialTextureUvFitScales,
  collectMaterialTextures,
  MATERIAL_TEX_FLOAT_STRIDE,
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
    expect(wgsl).toContain('fn sampleBaseColorTexture(');
    expect(wgsl).toContain('fn sampleAlphaTexture(');
    expect(wgsl).toContain('sampleAlphaTexture(matId, triIndex, baryVW)');
    expect(wgsl).toContain('fn sampleTransmissionTexture(');
    expect(wgsl).toContain('wrappedUv * uvFitScale');
    expect(wgsl).toContain('materialTexDescriptors[base + 7u].xy');
    expect(wgsl).toContain('materialTexDescriptors[base + 8u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + 11u].zw');
    expect(wgsl).toContain('fn wrapTextureCoord(coord: f32, mode: f32) -> f32');
    expect(wgsl).toContain('materialTexDescriptors[base + 12u].xy');
    expect(wgsl).toContain('materialTexDescriptors[base + 13u].zw');
    expect(wgsl).toContain('materialTexDescriptors[base + 16u].zw');
  });

  it('normal maps transform derived tangents through the hit TLAS instance', () => {
    const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
    expect(wgsl).toContain(
      'fn applyNormalMap(matId: u32, triIndex: u32, baryVW: vec2f, geomNormal: vec3f, instanceIndex: u32)',
    );
    expect(wgsl).toContain('if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX && params.tlasNodeCount != 0u)');
    expect(wgsl).toContain('let l2w0 = tlasInstanceLocalToWorld[m];');
    expect(wgsl).toContain('tangent = transformDirectionCols(l2w0, l2w1, l2w2, tangent);');
    expect(wgsl).toContain('let normalScale = materialTexDescriptors[base + 5u].w;');
    expect(wgsl).toContain('tn.x = tn.x * normalScale;');
    expect(wgsl).toContain('tn.y = tn.y * normalScale;');
  });
});
