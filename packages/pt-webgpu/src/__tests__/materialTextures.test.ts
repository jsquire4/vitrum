/**
 * materialTextures.test.ts — P2 host-side texture collection + descriptor pack.
 */
import { describe, it, expect } from 'vitest';
import {
  collectMaterialTextures,
  MATERIAL_TEX_FLOAT_STRIDE,
  MATERIAL_TEX_VEC4_STRIDE,
} from '../scene/materialTextures.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import type { MaterialSpec } from '@vitrum/core';

function mat(over: Partial<MaterialSpec>): MaterialSpec {
  return { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, ...over };
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

  it('defaults: opaque(0), cutoff 0.5, opacity 1, identity scale', () => {
    const { descriptors } = collectMaterialTextures([mat({ baseColorMap: { handle: {} } })]);
    expect(descriptors[4]).toBe(0);
    expect(descriptors[5]).toBe(0.5);
    expect(descriptors[6]).toBe(1);
    expect(descriptors[10]).toBe(1);
    expect(descriptors[11]).toBe(1);
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
  });
});
