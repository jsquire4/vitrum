import { describe, expect, it } from 'vitest';

import { TRANSPARENT_OIT_MODULE, TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';

describe('transparent OIT material parity', () => {
  it('shades camera-visible blend layers with the extension-aware material BRDF for direct sun', () => {
    expect(TRANSPARENT_OIT_WGSL).toContain('fn oitLayerRadiance(hit: IntersectionResult, rayDir: vec3f, materialWord: u32) -> vec3f');
    expect(TRANSPARENT_OIT_WGSL).toContain('let payload = sampleRestirDIMaterialPayloadForHit(hit, normals.smoothNormal, normal, scalarBase, materialWord);');
    expect(TRANSPARENT_OIT_WGSL).toContain('let sunBrdf = evalGGXWithSpecularClearcoatSheen(');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.specular.rgb,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.anisotropy.x,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.iridescence,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.clearcoatNormal,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.sheen.rgb,');
    expect(TRANSPARENT_OIT_WGSL).toContain('let sunDirect = vec3f(ubo.sunIntensity) * sunBrdf;');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('sunDiffuse');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('vec3f(ubo.sunIntensity) * albedo * INV_PI');
  });

  it('keeps camera-visible emissive/light-map terms while declaring BRDF dependencies', () => {
    expect(TRANSPARENT_OIT_WGSL).toContain('sampleEmissiveMap(');
    expect(TRANSPARENT_OIT_WGSL).toContain('sampleLightMap(');
    expect(TRANSPARENT_OIT_MODULE.requires).toContain('materialAtlas');
    expect(TRANSPARENT_OIT_MODULE.requires).toContain('ggxBrdf');
  });
});
