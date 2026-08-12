import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { packMaterialTextureAtlas } from '../bvh/materialTextureAtlasPack.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';

describe('unmapped base-color atlas meta', () => {
  it('stores authored linear RGB in disabled base-color meta .yzw', () => {
    const material: MaterialSpec = {
      baseColor: [0.1234567, 0.4, 0.75],
      roughness: 1,
      metallic: 0,
    };
    const atlas = packMaterialTextureAtlas([material], new Uint32Array([0]), 1);
    expect(atlas.baseColorMetaData[0]).toBe(-1);
    expect(atlas.baseColorMetaData[1]).toBeCloseTo(Math.fround(0.1234567), 5);
    expect(atlas.baseColorMetaData[2]).toBeCloseTo(0.4, 5);
    expect(atlas.baseColorMetaData[3]).toBeCloseTo(0.75, 5);
    // Other disabled map slots keep the empty yzw payload.
    expect(atlas.baseColorMetaData[8]).toBe(-1);
    expect(atlas.baseColorMetaData[9]).toBe(0);
    expect(atlas.baseColorMetaData[10]).toBe(0);
    expect(atlas.baseColorMetaData[11]).toBe(0);
  });

  it('shade derives isMetal from sampled metalness, not the packed >0 bit', () => {
    expect(SHADE_WGSL).toContain('let isMetal  = metal >= 0.5;');
    expect(SHADE_WGSL).not.toContain(
      'let isMetal  = decodeIsMetal(primaryHit.matColorPacked);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleUnmappedBaseColorRgb(');
  });
});
