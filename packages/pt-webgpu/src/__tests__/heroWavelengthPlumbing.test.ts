import { describe, expect, it } from 'vitest';
import { FrameParamsSlot } from '../scene/frameParamsLayout.js';

describe('hero-wavelength FrameParams plumbing', () => {
  it('keeps only invocation-seed spectral fields after tlasNodeCount', () => {
    // 2026-06-06: heroStrategy (always-0, never read by WGSL) was dropped from
    // FrameParams. 2026-07-27: three equally unread CMF-integral payloads were
    // removed; cameraPos.xyz plus environmentHdriIntensity now fill one aligned
    // 16-byte block without a synthetic cameraPos.w semantic.
    expect(FrameParamsSlot.spectralEnabled).toBe(20);
    expect(FrameParamsSlot.heroLambdaNm).toBe(21);
    expect(FrameParamsSlot.heroPdf).toBe(22);
    expect(FrameParamsSlot.cameraPos).toBe(28);
    expect(FrameParamsSlot.environmentHdriIntensity).toBe(31);
    expect('cmfIntegralX' in FrameParamsSlot).toBe(false);
    expect('cmfIntegralY' in FrameParamsSlot).toBe(false);
    expect('cmfIntegralZ' in FrameParamsSlot).toBe(false);
  });
});
