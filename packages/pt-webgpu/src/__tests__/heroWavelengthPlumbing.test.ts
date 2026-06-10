import { describe, expect, it } from 'vitest';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';
import { X_CMF_INTEGRAL, Y_CMF_INTEGRAL, Z_CMF_INTEGRAL } from '@vitrum/shared-samplers';

describe('hero-wavelength FrameParams plumbing', () => {
  it('reserves spectral slots after tlasNodeCount', () => {
    // 2026-06-06: heroStrategy (always-0, never read by WGSL) was dropped from
    // FrameParams; scalar slots after spectralEnabled shifted down by one and
    // the f32 pad slot keeps cameraPos at its vec4-aligned slot 32.
    // 2026-06-09: that pad slot (31) became environmentHdriIntensity: f32 (H14-E),
    // still occupying a single f32 so the cameraPos alignment is unchanged.
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
