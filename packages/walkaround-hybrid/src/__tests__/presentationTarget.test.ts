import { describe, expect, it } from 'vitest';
import { linearToSrgb } from '@vitrum/shared-samplers';
import {
  HYBRID_PRESENTATION_FORMATS,
  clampHybridPresentationRgb,
  hybridCompositeFragmentConstants,
  hybridPresentationMaxRgb,
  resolveHybridPresentationTarget,
} from '../presentationTarget.js';
import { COMPOSITE_FRAG_WGSL } from '../shaders/composite.wgsl.js';

describe('walkaround presentation-target transfer boundary', () => {
  it('preserves the historical software OETF for ordinary bgra8unorm', () => {
    expect(
      resolveHybridPresentationTarget('bgra8unorm', 'srgb'),
    ).toEqual({
      format: 'bgra8unorm',
      outputColorSpace: 'srgb',
      attachmentSrgb: false,
      applySoftwareSrgbOetf: true,
    });
    expect(hybridCompositeFragmentConstants('bgra8unorm')).toEqual({
      VT_ATTACHMENT_SRGB: 0,
      VT_TARGET_MAX_R: 1,
      VT_TARGET_MAX_G: 1,
      VT_TARGET_MAX_B: 1,
    });
  });

  it('delegates exactly one OETF to an sRGB attachment', () => {
    const target = resolveHybridPresentationTarget(
      'bgra8unorm-srgb',
      'srgb',
    );
    expect(target.applySoftwareSrgbOetf).toBe(false);
    expect(target.attachmentSrgb).toBe(true);
    expect(hybridCompositeFragmentConstants(target.format)).toEqual({
      VT_ATTACHMENT_SRGB: 1,
      VT_TARGET_MAX_R: 1,
      VT_TARGET_MAX_G: 1,
      VT_TARGET_MAX_B: 1,
    });

    // Semantic counterexample for the former unconditional shader OETF:
    // 0.5 linear must be encoded once (~0.735), not twice (~0.873).
    const encodedOnce = linearToSrgb(0.5);
    const encodedTwice = linearToSrgb(encodedOnce);
    expect(encodedOnce).toBeCloseTo(0.7353569, 6);
    expect(encodedTwice).toBeGreaterThan(0.87);
    expect(encodedTwice - encodedOnce).toBeGreaterThan(0.13);
  });

  it('rejects linear output on an attachment whose sRGB conversion is mandatory', () => {
    for (const format of ['rgba8unorm-srgb', 'bgra8unorm-srgb'] as const) {
      expect(() =>
        resolveHybridPresentationTarget(format, 'linear'),
      ).toThrow(/outputColorSpace 'linear' is incompatible/);
    }
  });

  it('accepts only RGB float-compatible color-renderable targets', () => {
    for (const format of HYBRID_PRESENTATION_FORMATS) {
      expect(
        resolveHybridPresentationTarget(format, 'srgb').format,
      ).toBe(format);
    }
    for (const format of [
      'depth24plus',
      'rgba8uint',
      'r8unorm',
      'bc1-rgba-unorm',
      'not-a-format',
      null,
    ]) {
      expect(() =>
        resolveHybridPresentationTarget(format, 'srgb'),
      ).toThrow(/swapChainFormat is unsupported/);
    }
  });

  it('preserves wide-target HDR and saturates only at each concrete target', () => {
    const hdr = [131_008, 131_008, 131_008] as const;
    expect(clampHybridPresentationRgb(hdr, 'rgba32float')).toEqual(hdr);
    expect(clampHybridPresentationRgb(hdr, 'rgba16float')).toEqual([
      65_504,
      65_504,
      65_504,
    ]);
    expect(clampHybridPresentationRgb(hdr, 'rg11b10ufloat')).toEqual([
      65_024,
      65_024,
      64_512,
    ]);
    expect(clampHybridPresentationRgb(hdr, 'rgba8unorm')).toEqual([1, 1, 1]);
    expect(hybridPresentationMaxRgb('rgba32float')[0]).toBeGreaterThan(1e38);
  });

  it('pins the shader override and its software-OETF gate', () => {
    expect(COMPOSITE_FRAG_WGSL).toContain(
      'override VT_ATTACHMENT_SRGB: u32 = 0u;',
    );
    expect(COMPOSITE_FRAG_WGSL).toContain(
      '&& VT_ATTACHMENT_SRGB == 0u',
    );
    expect(COMPOSITE_FRAG_WGSL).toContain(
      'override VT_TARGET_MAX_R: f32 = 1.0;',
    );
    expect(COMPOSITE_FRAG_WGSL).toContain(
      'clamp(presented, vec3f(0.0), targetMaximum)',
    );
  });
});
