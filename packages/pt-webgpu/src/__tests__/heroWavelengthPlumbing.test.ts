import { describe, expect, it } from 'vitest';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL } from '@vitrum/shared-samplers';

describe('hero-wavelength FrameParams plumbing', () => {
  it('reserves spectral slots after tlasNodeCount', () => {
    // 2026-06-06: heroStrategy (always-0, never read by WGSL) was dropped from
    // FrameParams; scalar slots after spectralEnabled shifted down by one and
    // _padAuto0 keeps cameraPos at its vec4-aligned slot 32.
    expect(FrameParamsSlot.spectralEnabled).toBe(20);
    expect(FrameParamsSlot.heroLambdaNm).toBe(21);
    expect(FrameParamsSlot.cmfIntegralY).toBe(24);
    expect(FrameParamsSlot.cameraPos).toBe(32);
  });

  it('exports CMF integrals matching shared-samplers', () => {
    expect(Y_CMF_INTEGRAL).toBeGreaterThan(100);
    expect(X_CMF_INTEGRAL).toBeCloseTo(Y_CMF_INTEGRAL, 0);
    expect(Z_CMF_INTEGRAL).toBeGreaterThan(0);
  });
});
