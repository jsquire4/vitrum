import { describe, expect, it } from 'vitest';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL } from '@vitrum/shared-samplers';

describe('hero-wavelength FrameParams plumbing', () => {
  it('reserves spectral slots after tlasNodeCount', () => {
    expect(FrameParamsSlot.spectralEnabled).toBe(20);
    expect(FrameParamsSlot.heroLambdaNm).toBe(22);
    expect(FrameParamsSlot.cmfIntegralY).toBe(25);
    expect(FrameParamsSlot.cameraPos).toBe(32);
  });

  it('exports CMF integrals matching shared-samplers', () => {
    expect(Y_CMF_INTEGRAL).toBeGreaterThan(100);
    expect(X_CMF_INTEGRAL).toBeCloseTo(Y_CMF_INTEGRAL, 0);
    expect(Z_CMF_INTEGRAL).toBeGreaterThan(0);
  });
});
