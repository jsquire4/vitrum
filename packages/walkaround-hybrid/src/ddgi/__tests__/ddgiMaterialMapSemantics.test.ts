import { describe, expect, it } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';

describe('DDGI material-map semantics', () => {
  it('applies roughness/metallic maps as scalar factors, not replacements', () => {
    const src = makeProbeUpdateRaysWGSL(64);

    expect(src).toContain('fallback * ddgiMaterialMapChannel(texel.value, channel)');
    expect(src).toContain('out.roughness = ddgiSampleMaterialScalarMap(');
    expect(src).toContain('out.metalness = ddgiSampleMaterialScalarMap(');
  });

  it('keeps atlas validity separate from legal signed texel values and decodes signed normals once', () => {
    const src = makeProbeUpdateRaysWGSL(64);

    expect(src).toContain('struct DdgiMaterialAtlasSample {');
    expect(src).toContain('value: vec4f,');
    expect(src).toContain('encoding: u32,');
    expect(src).toContain('valid: u32,');
    expect(src).toContain('out.valid = 0u;');
    expect(src).not.toContain('return vec4f(-1.0);');
    expect(src).toContain(
      'texelColor.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM',
    );
    expect(src).toContain(
      'texelColor.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM',
    );
    expect(src).toContain(
      'clamp(texelColor.value.rgb, vec3f(-1.0), vec3f(1.0))',
    );
  });

  it('decodes anisotropy direction according to the atlas encoding', () => {
    const src = makeProbeUpdateRaysWGSL(64);

    expect(src).toContain(
      'anisoMap.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM',
    );
    expect(src).toContain(
      'anisoMap.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM',
    );
    expect(src).toContain(
      'clamp(anisoMap.value.rg, vec2f(0.0), vec2f(1.0)) * 2.0 - vec2f(1.0)',
    );
    expect(src).toContain(
      'clamp(anisoMap.value.rg, vec2f(-1.0), vec2f(1.0))',
    );
  });
});
