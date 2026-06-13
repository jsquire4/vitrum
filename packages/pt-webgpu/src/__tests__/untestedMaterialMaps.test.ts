// untestedMaterialMaps.test.ts — closes the UNTESTED-promise class for the D3
// anisotropyMap field on pt-webgpu (audit item 25, trust-remediation-plan-2026-06-10.md).
//
// Findings audit context:
//   The promise inventory found ~22 UNTESTED promises. On pt-webgpu the outstanding
//   UNTESTED map field is anisotropyMap (KHR_materials_anisotropy): packed at descriptor
//   vec4[5].z (float offset b+22) by collectMaterialTextures, consumed by the WGSL
//   materialAnisotropy / materialAnisotropyRotation functions at
//   materialTexDescriptors[base + 5u].z.
//
// Strategy:
//   (a) PACKER test — pack a MaterialSpec with anisotropyMap set; assert the EXACT
//       float offset b+22 carries the linear-source layer id. The offset is read from
//       scene/materialTextures.ts (the packer comment documents it) and cross-verified
//       against the WGSL decoder in wgsl/pathTrace/material.wgsl.ts which reads
//       `materialTexDescriptors[base + 5u].z`.
//   (b) DECODER structural test — the composed WGSL contains the anisotropyMap sampling
//       site and the two accessor functions that read it.

import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  collectMaterialTextures,
  MATERIAL_TEX_FLOAT_STRIDE,
} from '../scene/materialTextures.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL } from '../wgsl/pathTrace/material.wgsl.js';

function mat(over: Partial<MaterialSpec>): MaterialSpec {
  return { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, ...over };
}

describe('pt-webgpu anisotropyMap packer offset — UNTESTED-promise closure (item 25)', () => {

  // Float offset within the per-material descriptor block:
  //   vec4[5] starts at b+20 (5 * 4 floats = 20).
  //   .x = anisotropy      (b+20)
  //   .y = anisotropyRot   (b+21)
  //   .z = anisotropyMapIdx (b+22)  ← the field under test
  //   .w = pad             (b+23)
  const ANISO_MAP_OFFSET = 22; // within the MATERIAL_TEX_FLOAT_STRIDE block

  it('anisotropyMap: layer id packed at descriptor float offset b+22 (vec4[5].z)', () => {
    const handle = {};
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({ anisotropyMap: { handle } }),
    ]);
    // The handle is routed through the LINEAR source array (not sRGB).
    expect(linearSources).toContain(handle);
    const layerId = linearSources.indexOf(handle);
    expect(layerId).toBeGreaterThanOrEqual(0);
    // Float b+22 for material 0 (b = 0 * MATERIAL_TEX_FLOAT_STRIDE).
    expect(descriptors[ANISO_MAP_OFFSET]).toBe(layerId);
  });

  it('absent anisotropyMap packs -1 at float offset b+22', () => {
    const { descriptors } = collectMaterialTextures([mat({})]);
    expect(descriptors[ANISO_MAP_OFFSET]).toBe(-1);
  });

  it('anisotropyMap is independent of the sRGB source list (linear-only)', () => {
    const aniso = {};
    const base = {};
    const { sources, linearSources } = collectMaterialTextures([
      mat({ baseColorMap: { handle: base }, anisotropyMap: { handle: aniso } }),
    ]);
    expect(sources).toContain(base);
    expect(sources).not.toContain(aniso);    // anisotropyMap is linear, not sRGB
    expect(linearSources).toContain(aniso);
  });

  it('anisotropy scalars are packed at b+20 (strength) and b+21 (rotation)', () => {
    const { descriptors } = collectMaterialTextures([
      mat({ anisotropy: 0.6, anisotropyRotation: 0.3 }),
    ]);
    expect(descriptors[20]).toBeCloseTo(0.6, 6); // vec4[5].x
    expect(descriptors[21]).toBeCloseTo(0.3, 6); // vec4[5].y
  });

  it('anisotropy scalar defaults to 0 (isotropic path) when absent', () => {
    const { descriptors } = collectMaterialTextures([mat({})]);
    expect(descriptors[20]).toBe(0); // no anisotropy → isotropic path → byte-identical
  });

  it('anisotropyMap deduplicates shared handles across materials', () => {
    const shared = {};
    const { linearSources, descriptors } = collectMaterialTextures([
      mat({ anisotropyMap: { handle: shared } }),
      mat({ anisotropyMap: { handle: shared } }), // same handle → same layer
    ]);
    expect(linearSources.filter((s) => s === shared)).toHaveLength(1); // deduped
    const layerId = linearSources.indexOf(shared);
    // Both materials should reference the same layer.
    expect(descriptors[ANISO_MAP_OFFSET]).toBe(layerId);
    expect(descriptors[MATERIAL_TEX_FLOAT_STRIDE + ANISO_MAP_OFFSET]).toBe(layerId);
  });

  // DECODER structural tests — verify the WGSL actually consumes anisotropyMap.
  describe('WGSL decoder structural (pin-independent contains assertions)', () => {

    it('group-3 WGSL declares the linear texture array for anisotropyMap', () => {
      const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
      // Linear array binding (anisotropyMap routes to the linear/ORM array).
      expect(wgsl).toContain('@group(3) @binding(5) var materialTexturesLinear: texture_2d_array<f32>');
    });

    it('WGSL materialAnisotropy reads vec4[5].z for the anisotropyMap layer index', () => {
      const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
      // The accessor reads descriptor[base + 5u].z for the anisotropy map index.
      expect(wgsl).toContain('fn materialAnisotropy(matId: u32, triIndex: u32, baryVW: vec2f) -> f32');
      // The anisotropy map index is read from vec4[5].z (b+5u offset, .z channel).
      expect(wgsl).toContain('let anisoIdx = i32(materialTexDescriptors[base + 5u].z)');
    });

    it('WGSL materialAnisotropyRotation reads vec4[5].z for the anisotropyMap index', () => {
      const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
      expect(wgsl).toContain('fn materialAnisotropyRotation(matId: u32, triIndex: u32, baryVW: vec2f) -> f32');
      // The anisotropy rotation accessor also uses the map index from vec4[5].z.
      expect(wgsl).toContain('materialTexDescriptors[base + 5u].z');
    });

    it('WGSL anisotropy strength reads vec4[5].x and map B-channel modulates it', () => {
      const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
      // Strength at vec4[5].x.
      expect(wgsl).toContain('materialTexDescriptors[base + 5u].x');
      // Map modulates strength via the B channel (KHR_materials_anisotropy spec).
      expect(wgsl).toContain(
        'sampleMaterialLayerLinear(anisoIdx, base, triIndex, baryVW, materialTexDescriptors[base + 10u].zw, materialTexDescriptors[base + 15u].zw).b',
      );
    });

    it('WGSL anisotropy rotation reads vec4[5].y and map RG-channel offsets it', () => {
      const wgsl = PT_WEBGPU_PATH_TRACE_MATERIAL_FULL_BINDINGS_GROUP3_WGSL;
      // Rotation scalar at vec4[5].y.
      expect(wgsl).toContain('materialTexDescriptors[base + 5u].y');
      // Map RG direction encoded in [0,1]→[-1,1]: atan2(rg.y, rg.x) offset.
      expect(wgsl).toContain('atan2(rg.y, rg.x)');
    });
  });
});
