import { describe, expect, it } from 'vitest';

import { TRANSPARENT_OIT_MODULE, TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';

describe('transparent OIT material parity', () => {
  it('shades camera-visible blend layers with extension-aware material BRDFs for sky and direct sun', () => {
    expect(TRANSPARENT_OIT_WGSL).toContain('fn oitLayerRadiance(hit: IntersectionResult, hitPos: vec3f, rayDir: vec3f, materialWord: u32) -> vec3f');
    expect(TRANSPARENT_OIT_WGSL).toContain('let payload = sampleRestirDIMaterialPayloadForHit(hit, normals.smoothNormal, normal, scalarBase, materialWord);');
    expect(TRANSPARENT_OIT_WGSL).toContain('fn oitLayerSkyRadiance(payload: RestirDIMaterialPayload, normal: vec3f, wo: vec3f) -> vec3f');
    expect(TRANSPARENT_OIT_WGSL).toContain('fn oitLayerEnvSampleRadiance(');
    expect(TRANSPARENT_OIT_WGSL).toContain('if (!envHasMap()) {');
    expect(TRANSPARENT_OIT_WGSL).toContain('return envRadiance(normal) * max(payload.envMapIntensity, 0.0) * payload.albedo * INV_PI;');
    expect(TRANSPARENT_OIT_WGSL).toContain('let d1 = oitEnvSampleDir(normal,  0.70,  0.00);');
    expect(TRANSPARENT_OIT_WGSL).toContain('return avg * (2.0 * PI / 5.0);');
    expect(TRANSPARENT_OIT_WGSL).toContain('let skyAmbient = oitLayerSkyRadiance(payload, normal, wo);');
    expect(TRANSPARENT_OIT_WGSL).toContain('let sunBrdf = evalGGXWithSpecularClearcoatSheen(');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.specular.rgb,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.anisotropy.x,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.iridescence,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.clearcoatNormal,');
    expect(TRANSPARENT_OIT_WGSL).toContain('payload.sheen.rgb,');
    expect(TRANSPARENT_OIT_WGSL).toContain('let toSun = safe_normalize(ubo.sunDirection);');
    expect(TRANSPARENT_OIT_WGSL).toContain('traceSceneAnyCastMask(');
    expect(TRANSPARENT_OIT_WGSL).toContain('hitPos + hit.normal * 1e-3,');
    expect(TRANSPARENT_OIT_WGSL).toContain('sunVisibility = select(1.0, 0.0, sunOccluded);');
    expect(TRANSPARENT_OIT_WGSL).toContain('let sunDirect = vec3f(ubo.sunIntensity) * sunBrdf * sunVisibility;');
    expect(TRANSPARENT_OIT_WGSL).toContain('let hitPos = walkRay.origin + walkRay.direction * hit.dist;');
    expect(TRANSPARENT_OIT_WGSL).toContain('oitLayerRadiance(hit, hitPos, primaryRay.direction, word);');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('sunDiffuse');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('vec3f(ubo.sunIntensity) * albedo * INV_PI');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('let skyAmbient = envRadiance(normal) * payload.albedo * INV_PI;');
  });

  it('keeps camera-visible emissive/light-map terms while declaring BRDF dependencies', () => {
    expect(TRANSPARENT_OIT_WGSL).toContain('sampleEmissiveMap(');
    expect(TRANSPARENT_OIT_WGSL).toContain('sampleLightMap(');
    expect(TRANSPARENT_OIT_MODULE.requires).toContain('materialAtlas');
    expect(TRANSPARENT_OIT_MODULE.requires).toContain('ggxBrdf');
  });
});
