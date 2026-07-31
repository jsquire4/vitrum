import { applyTonemap } from '@vitrum/shared-samplers';
import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_PRESENT_WGSL } from '../wgsl/present.wgsl.js';

describe('pt-webgpu presentation range boundary', () => {
  it('keeps none+linear exposure raw until the concrete output store', () => {
    expect(applyTonemap([65_504, 65_504, 65_504], 'none', 2)).toEqual([
      131_008,
      131_008,
      131_008,
    ]);
    expect(PT_WEBGPU_PRESENT_WGSL).toContain(
      'const PT_PRESENT_RGBA16F_MAX: f32 = 65504.0;',
    );
    expect(PT_WEBGPU_PRESENT_WGSL).toContain(
      'vec3f(PT_PRESENT_RGBA16F_MAX)',
    );
    expect(PT_WEBGPU_PRESENT_WGSL).toContain(
      'textureStore(presentTex, px, outColor);',
    );
  });

  it('clamps after the optional OETF so rgba16float publication stays finite', () => {
    const oetfIndex = PT_WEBGPU_PRESENT_WGSL.indexOf(
      'outColor = vec4f(vt_linearToSrgb(tonemapped), 1.0);',
    );
    const clampIndex = PT_WEBGPU_PRESENT_WGSL.indexOf(
      'clamp(',
      oetfIndex,
    );
    const storeIndex = PT_WEBGPU_PRESENT_WGSL.indexOf(
      'textureStore(presentTex, px, outColor);',
    );
    expect(oetfIndex).toBeGreaterThan(-1);
    expect(clampIndex).toBeGreaterThan(oetfIndex);
    expect(storeIndex).toBeGreaterThan(clampIndex);
  });
});
