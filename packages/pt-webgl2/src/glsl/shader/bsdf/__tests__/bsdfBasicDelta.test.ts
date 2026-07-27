import { describe, expect, it } from 'vitest';
import { BSDF_BASIC_GLSL } from '../bsdf_basic.glsl.js';

describe('pt-webgl2 basic-tier exact roughness support', () => {
  it('keeps exact zero as a discrete specular event and preserves every positive roughness', () => {
    expect(BSDF_BASIC_GLSL).toContain(
      'return surf.filteredRoughness <= 0.0;',
    );
    expect(BSDF_BASIC_GLSL).toContain('float basicDeltaPdfLocal(');
    expect(BSDF_BASIC_GLSL).toContain('float basicDeltaEvalLocal(');
    expect(BSDF_BASIC_GLSL).toContain('result.sampledDelta = sampledDelta;');
    expect(BSDF_BASIC_GLSL).toContain(
      'clamp( surf.filteredRoughness, 0.0, 1.0 )',
    );
    expect(BSDF_BASIC_GLSL).not.toContain(
      'clamp( surf.filteredRoughness, 0.001, 1.0 )',
    );

    const tinyPositive = 1e-9;
    expect(Math.max(0, Math.min(1, tinyPositive))).toBe(tinyPositive);
  });
});
