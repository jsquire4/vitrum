import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  applyGltfMaterialPointerValue,
  resolveGltfMaterialAnimationPointer,
  supportedGltfMaterialAnimationPointers,
} from './materialPointerAnimation.js';

const BASE_MATERIAL: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 1,
  metallic: 0,
};

function applyPointer(pointer: string, values: readonly number[]): MaterialSpec {
  const target = resolveGltfMaterialAnimationPointer(pointer);
  expect(target).toBeDefined();
  return applyGltfMaterialPointerValue(BASE_MATERIAL, target!, new Float32Array(values));
}

describe('KHR_animation_pointer material fields', () => {
  it('supports imported scalar texture-companion fields', () => {
    expect(supportedGltfMaterialAnimationPointers()).toEqual(expect.arrayContaining([
      '/materials/{index}/normalTexture/scale',
      '/materials/{index}/occlusionTexture/strength',
      '/materials/{index}/extensions/KHR_materials_clearcoat/clearcoatNormalTexture/scale',
    ]));

    expect(applyPointer('/materials/0/normalTexture/scale', [0.5]).normalScale).toBeCloseTo(0.5);
    expect(applyPointer('/materials/0/occlusionTexture/strength', [0.75]).aoMapIntensity).toBeCloseTo(0.75);
    expect(
      applyPointer('/materials/0/extensions/KHR_materials_clearcoat/clearcoatNormalTexture/scale', [0.25])
        .clearcoatNormalScale,
    ).toBeCloseTo(0.25);
  });

  it('supports volume attenuation color and dispersion conversion', () => {
    expect(applyPointer('/materials/0/extensions/KHR_materials_volume/attenuationColor', [1.2, -0.1, 0.5])
      .attenuationColor).toEqual([1, 0, 0.5]);
    expect(applyPointer('/materials/0/extensions/KHR_materials_dispersion/dispersion', [0.4])
      .dispersionAbbeNumber).toBeCloseTo(50);
    expect(applyPointer('/materials/0/extensions/KHR_materials_dispersion/dispersion', [0])
      .dispersionAbbeNumber).toBeUndefined();
  });
});
