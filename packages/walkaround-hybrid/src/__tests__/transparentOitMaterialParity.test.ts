import { describe, expect, it } from 'vitest';

import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';
import { TRANSPARENT_OIT_MODULE, TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';

function functionBody(source: string, name: string): string {
  const signature = source.indexOf(`fn ${name}(`);
  expect(signature, `${name} is declared`).toBeGreaterThanOrEqual(0);
  const open = source.indexOf('{', signature);
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(open + 1, i);
    }
  }
  throw new Error(`${name} has an unterminated body`);
}

describe('transparent OIT estimator and ownership', () => {
  it('owns deterministic camera blend coverage while opaque primary shading skips it', () => {
    const main = functionBody(TRANSPARENT_OIT_WGSL, 'transparentOitMain');
    expect(SHADE_WGSL).toContain('traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(');
    expect(main).toContain('let coverage = clamp(alpha.coverage, 0.0, 1.0);');
    expect(main).toContain('accum = accum + layerRadiance * coverage * transmittance;');
    expect(main).toContain('transmittance = transmittance * (1.0 - coverage);');
    expect(main).toContain('vec4f(accum + background * transmittance, 1.0)');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialAlphaBlendCoverageHash(');
  });

  it('uses a scene-derived saturated layer bound instead of a fixed black tail', () => {
    const main = functionBody(TRANSPARENT_OIT_WGSL, 'transparentOitMain');
    expect(main).toContain('let triangleCount = bvhIndexCount();');
    expect(main).toContain('triangleCount > 0xffffffffu / instanceMultiplier');
    expect(main).toContain('for (var layer = 0u; layer < layerBudget; layer = layer + 1u)');
    expect(main).not.toMatch(/layer\s*<\s*32u/);
    expect(main).not.toMatch(/layer\s*==\s*31u/);
    expect(main).not.toContain('transmittance <= 0.001');

    const layerCount = 64;
    const coverage = 0.1;
    const layerRadiance = 2;
    const background = 0.25;
    let transmittance = 1;
    let accumulated = 0;
    for (let i = 0; i < layerCount; i += 1) {
      accumulated += layerRadiance * coverage * transmittance;
      transmittance *= 1 - coverage;
    }
    const analyticT = (1 - coverage) ** layerCount;
    const analytic = layerRadiance * (1 - analyticT) + background * analyticT;
    expect(accumulated + background * transmittance).toBeCloseTo(analytic, 14);
    expect(transmittance).toBeGreaterThan(0);
  });

  it('pairs its temporal environment proposal with the exact cosine PDF', () => {
    const sky = functionBody(TRANSPARENT_OIT_WGSL, 'oitLayerSkyRadiance');
    expect(sky).toContain('oitSamplingHashToF32(seed ^ 0x9e3779b9u)');
    expect(sky).toContain('let wi = oitCosineHemisphereDir(normal, xi);');
    expect(sky).toContain('let pdf = max(dot(normal, wi), 0.0) * INV_PI;');
    expect(sky).toContain('oitBoundedCosineImportanceDivide(');
    expect(sky).toContain(
      'oitLayerEnvSampleRadiance(payload, normal, wo, wi, transmission),',
    );
  });

  it('uses the configured DDGI/RC mixture for every layer and ULP ray offsets', () => {
    const layer = functionBody(TRANSPARENT_OIT_WGSL, 'oitLayerRadiance');
    const offset = functionBody(TRANSPARENT_OIT_WGSL, 'oitOffsetRayOrigin');
    expect(layer).toContain(
      'mix(ddgiIndirect, rcIndirect, clamp(rcParams.rcWeight, 0.0, 1.0))',
    );
    expect(layer).not.toMatch(/(?:ddgi|rc).*HasEnergy/i);
    expect(offset).toContain('bitcast<f32>(bitcast<i32>(p.x)');
    expect(offset).toContain('abs(p.x) < (1.0 / 32.0)');
    expect(TRANSPARENT_OIT_WGSL).not.toMatch(/hitPos\s*[+-]\s*[^;]*(?:1e-3|0\.001)/);
  });

  it('retains mapped emission and receiver-local baked irradiance in primary shading', () => {
    expect(SHADE_WGSL).toContain('sampleEmissiveMap(');
    expect(SHADE_WGSL).toContain('let lightMapIrradiance = sampleLightMap(');
    expect(SHADE_WGSL).toContain(
      'let Lo_lightMap = albedo * INV_PI * lightMapIrradiance *',
    );
    expect(SHADE_WGSL).toContain(
      '(1.0 - clamp(matColor.a, 0.0, 1.0));',
    );
    expect(TRANSPARENT_OIT_MODULE.requires).toContain('materialAtlas');
    expect(TRANSPARENT_OIT_MODULE.requires).toContain('surfaceTextures');
  });

  it('layers absolute thin-film reflection separately from base/source transmission', () => {
    const brdf = functionBody(TRANSPARENT_OIT_WGSL, 'oitLayerSurfaceBrdf');
    const layer = functionBody(TRANSPARENT_OIT_WGSL, 'oitLayerRadiance');
    expect(brdf).toContain(
      'evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(',
    );
    expect(brdf).toContain('evalGGXReflectionWithTransmissionMix(');
    expect(brdf).toContain('applyMaterialLayerTransmissionToBrdf(');
    expect(brdf).toContain('payload.reflectionLayerTransmission,');
    expect(layer).toContain(
      '(emissive + (indirect + baked) * diffuseWeight) *',
    );
    expect(layer).toContain(
      'let transmission = sampleTransmissionMapForHit(hit, scalarMaterial.a);',
    );
    expect(layer).not.toMatch(
      /\(skyAmbient[^;]+\)\s*\*\s*payload\.layerTransmission/s,
    );
  });
});
