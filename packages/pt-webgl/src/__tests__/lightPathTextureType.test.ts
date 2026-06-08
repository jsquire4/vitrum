import { describe, expect, it } from 'vitest';
import { FloatType, HalfFloatType } from 'three';
import { bdptLightPathTextureType } from '../legacy/three/bdpt/lightPathTextureType.js';

describe('bdptLightPathTextureType', () => {
  it('uses half float on ANGLE', () => {
    expect(
      bdptLightPathTextureType('ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 Direct3D11 vs_5_0 ps_5_0)'),
    ).toBe(HalfFloatType);
  });

  it('uses full float on native GL', () => {
    expect(bdptLightPathTextureType('Mesa Intel(R) UHD Graphics')).toBe(FloatType);
  });
});
