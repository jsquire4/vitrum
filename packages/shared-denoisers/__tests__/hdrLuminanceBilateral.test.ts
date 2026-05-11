import { describe, it, expect } from 'vitest';
import {
  HDR_LUMINANCE_BILATERAL_ENTRY,
  HDR_LUMINANCE_BILATERAL_WGSL,
} from '../src/wgsl/hdrLuminanceBilateral.wgsl.js';

describe('HDR_LUMINANCE_BILATERAL_WGSL', () => {
  it('declares compute entry', () => {
    expect(HDR_LUMINANCE_BILATERAL_WGSL).toContain(`fn ${HDR_LUMINANCE_BILATERAL_ENTRY}`);
  });

  it('uses rgba32float storage output', () => {
    expect(HDR_LUMINANCE_BILATERAL_WGSL).toContain('texture_storage_2d<rgba32float');
  });

  it('declares luminance edge-stop', () => {
    expect(HDR_LUMINANCE_BILATERAL_WGSL).toContain('sigmaLuminance');
  });
});
