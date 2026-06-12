import { describe, expect, it } from 'vitest';

import { MATERIAL_DECODE_WGSL } from '../shaders/materialDecode.wgsl.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';

describe('walkaround unlit material shader contract', () => {
  it('decodes bit 1 of bvh_material as the unlit material flag', () => {
    expect(MATERIAL_DECODE_WGSL).toContain('fn decodeIsUnlitMaterial(packed: u32) -> bool');
    expect(MATERIAL_DECODE_WGSL).toContain('return (packed & 0x2u) != 0u;');
  });

  it('emits unlit base color directly and bypasses GI lighting terms', () => {
    expect(SHADE_WGSL).toContain('let materialWord = textureLoad(bvh_material, vec2i(rmCoord), 0).r;');
    expect(SHADE_WGSL).toContain('if (decodeIsUnlitMaterial(materialWord))');
    expect(SHADE_WGSL).toContain('textureStore(hdrColorOut,    pix, vec4f(albedo,      1.0));');
    expect(SHADE_WGSL).toContain('textureStore(hdrIndirectOut, pix, vec4f(vec3f(0.0), 1.0));');
    expect(SHADE_WGSL).toContain('textureStore(hdrTotalOut,    pix, vec4f(albedo,      1.0));');
  });
});
