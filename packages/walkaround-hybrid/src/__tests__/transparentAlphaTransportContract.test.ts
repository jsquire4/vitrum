import { describe, expect, it } from 'vitest';

import { RESTIR_CAST_PRIMARY_WGSL } from '../shaders/restirCastPrimary.wgsl.js';
import { RIS_WGSL } from '../shaders/ris.wgsl.js';
import { RIS_GI_WGSL } from '../shaders/risGi.wgsl.js';
import { buildRisGiNrcModule } from '../shaders/risGiNrc.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../shaders/nrcIndependentSuffix.wgsl.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shaders/shadingTerms.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../shaders/surfaceTextures.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';
import { RESTIR_GI_MATERIAL_WGSL } from '../shaders/restirGiMaterial.wgsl.js';
import { makeProbeUpdateRaysWGSL } from '../ddgi/wgsl/probeUpdateRays.wgsl.js';

const nrcGiModule = buildRisGiNrcModule({
  levels: 1,
  featuresPerEntry: 4,
  oneBlobBins: 4,
  width: 16,
  outWidth: 3,
  hidden: 1,
});

describe('transparent alpha transport contract', () => {
  it('keeps camera primaries deterministic while secondary transport samples blend coverage', () => {
    for (const [name, shader] of [
      ['restir-cast-primary', RESTIR_CAST_PRIMARY_WGSL],
      ['ris-di', RIS_WGSL],
      ['ris-gi', RIS_GI_WGSL],
      ['ris-gi-nrc', nrcGiModule.source],
      ['shade-primary', SHADE_WGSL],
    ] as const) {
      expect(shader, name).toContain('traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(');
    }
    for (const [name, shader] of [
      ['ris-gi', RIS_GI_WGSL],
      ['ris-gi-nrc', nrcGiModule.source],
    ] as const) {
      expect(shader, name).toMatch(
        /traceSceneFirstHitAlphaMaskTextured(?:WithMetadata)?\(/,
      );
      expect(shader, name).toContain('ubo.frameSeed');
    }
    expect(MATERIAL_ATLAS_WGSL).toContain('fn materialAlphaBlendCoverageHash(');
    expect(MATERIAL_ATLAS_WGSL).toContain('sampleSeed: u32,');
    expect(MATERIAL_ATLAS_WGSL).toContain('let originBits = bitcast<vec3u>(ray.origin);');
    expect(MATERIAL_ATLAS_WGSL).toContain('let directionBits = bitcast<vec3u>(ray.direction);');
    expect(MATERIAL_ATLAS_WGSL).toContain('(layer * 0x9e3779b9u)');
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'materialAlphaBlendCoverageHash(hit, ray, layer, sampleSeed) >= representedCoverage',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'fn materialRepresentedAlphaBlendCoverage(coverage: f32)',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'return materialRepresentedAlphaBlendCoverage(alpha.coverage);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'Conservative overflow: after the bounded transparent-layer budget',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('traceSceneFirstHitAlphaMaskTexturedOpaqueOnly');
  });

  it('keeps shade-side direct-light visibility on alpha-aware transmittance predicates', () => {
    expect(SHADING_TERMS_WGSL).toContain(
      'traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(',
    );
    expect(SHADE_WGSL).toContain(
      'let directShadowContainingMedia = materialShadowClassifyContainingMedia(',
    );
    expect(SHADING_TERMS_WGSL).toContain('var shadowT = vec3f(1.0);');
    expect(SHADING_TERMS_WGSL).toContain('var sunShadowT = vec3f(1.0);');
    expect(SHADING_TERMS_WGSL).toContain(
      'r.logW, envColor, layeredBrdfE, shadowTint',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'geometryLogW, Le, layeredBrdf, shadowTint,',
    );
    expect(SHADING_TERMS_WGSL).toContain('restirShadeDirectionalVolumeLog(');
    expect(SHADING_TERMS_WGSL).not.toContain('shadowColorCorrection');
    expect(SHADING_TERMS_WGSL).not.toContain('sunShadowT = traceSceneAlphaTransmittanceTextured(');
  });

  it('preserves explicit transmission ownership through the complete ordered walk', () => {
    const statefulWalker = SURFACE_TEXTURES_WGSL.slice(
      SURFACE_TEXTURES_WGSL.indexOf(
        'fn traceSceneAlphaTintTransmittanceTexturedWithState(',
      ),
      SURFACE_TEXTURES_WGSL.indexOf(
        'fn traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(',
      ),
    );
    expect(statefulWalker).toContain(
      'let surfaceBudget = materialShadowWorldSurfaceBudget(',
    );
    expect(statefulWalker).toContain(
      'for (var i = 0u; i < surfaceBudget; i = i + 1u)',
    );
    expect(statefulWalker).toContain(
      'tau = tau * vec3f(1.0 - coverage);',
    );
    expect(statefulWalker).toContain(
      'mediumState.materialId[mediumState.depth - 1u] == boundaryId',
    );
    expect(statefulWalker).toContain(
      'mediumState.instance[mediumState.depth - 1u] == representedId',
    );
    expect(statefulWalker).toContain('return vec3f(0.0);');
    expect(statefulWalker).not.toContain('traceSceneAny');
    const ownershipWrapper = SURFACE_TEXTURES_WGSL.slice(
      SURFACE_TEXTURES_WGSL.indexOf(
        'fn traceSceneAlphaTintTransmittanceTexturedWithOwnership(',
      ),
      SURFACE_TEXTURES_WGSL.indexOf(
        'fn traceSceneAlphaTintTransmittanceTextured(',
      ),
    );
    expect(ownershipWrapper).toContain(
      'let containingMedia = materialShadowClassifyContainingMedia(',
    );
    expect(ownershipWrapper).toContain(
      'return traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'return traceSceneAlphaTintTransmittanceTextured(',
    );
    expect(SURFACE_TEXTURES_WGSL).not.toContain(
      'fn _bvhTintedTriAccumulate(',
    );
    expect(SURFACE_TEXTURES_WGSL).not.toContain(
      'fn _bvhTraceTintedBlasLeaves(',
    );
  });

  it('orders and composites camera blend layers exactly once without a fixed tail cutoff', () => {
    expect(TRANSPARENT_OIT_WGSL).toContain('fn transparentOitMain(@builtin(global_invocation_id) gid: vec3u)');
    expect(TRANSPARENT_OIT_WGSL).toContain('let background = textureLoad(oit_background, vec2i(pix), 0).rgb;');
    expect(TRANSPARENT_OIT_WGSL).toContain('let triangleCount = bvhIndexCount();');
    expect(TRANSPARENT_OIT_WGSL).toContain('let layerBudget = max(layerCapacity, 1u);');
    expect(TRANSPARENT_OIT_WGSL).toContain('for (var layer = 0u; layer < layerBudget; layer = layer + 1u)');
    expect(TRANSPARENT_OIT_WGSL).toContain('accum = accum + layerRadiance * coverage * transmittance;');
    expect(TRANSPARENT_OIT_WGSL).toContain('transmittance = transmittance * (1.0 - coverage);');
    expect(TRANSPARENT_OIT_WGSL).toContain('if (coverage >= 1.0)');
    expect(TRANSPARENT_OIT_WGSL).not.toMatch(/layer\s*<\s*32u/);
    expect(TRANSPARENT_OIT_WGSL).not.toMatch(/layer\s*==\s*31u/);
    expect(TRANSPARENT_OIT_WGSL).not.toContain('transmittance <= 0.001');
    expect(TRANSPARENT_OIT_WGSL).toContain('accum + background * transmittance');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('viewFacing');
    expect(TRANSPARENT_OIT_WGSL).not.toMatch(/var<storage,\s*(read|read_write)>[^;]*reservoir/i);
    expect(TRANSPARENT_OIT_WGSL).not.toMatch(/\b(load|store|update|resolve)\w*Reservoir\b/);
    expect(TRANSPARENT_OIT_WGSL).not.toContain('selectedEmitter');
    expect(TRANSPARENT_OIT_WGSL).not.toContain('risFinal');
  });

  it('treats an unlit blend layer as authored outgoing radiance before coverage compositing', () => {
    expect(TRANSPARENT_OIT_WGSL).toContain(
      'if (decodeIsUnlitMaterial(materialWord))',
    );
    expect(TRANSPARENT_OIT_WGSL).toContain(
      'return payload.albedo * payload.layerTransmission;',
    );
    expect(TRANSPARENT_OIT_WGSL).toContain(
      'accum = accum + layerRadiance * coverage * transmittance;',
    );
  });

  it('treats lightMap as receiver-local diffuse irradiance exactly once', () => {
    expect(SHADE_WGSL).toContain('let lightMapIrradiance = sampleLightMap(');
    expect(SHADE_WGSL).toContain(
      'let Lo_lightMap = albedo * INV_PI * lightMapIrradiance *',
    );
    expect(SHADE_WGSL).toContain(
      '(1.0 - clamp(matColor.a, 0.0, 1.0));',
    );
    expect(TRANSPARENT_OIT_WGSL).toContain('let bakedIrradiance = sampleLightMap(');
    expect(TRANSPARENT_OIT_WGSL).toContain('let baked = payload.albedo * INV_PI * bakedIrradiance;');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('let bakedLo = albedo * INV_PI * sampleLightMap(hit);');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'let bakedDiffuse = payload.albedo * INV_PI * sampleLightMap(hit);',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'let diffuseLo = (bakedDiffuse +',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'incomingIrradiance * payload.albedo * INV_PI) *',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      '(1.0 - clamp(transmission, 0.0, 1.0));',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).not.toContain('incomingIrradiance * brdf');
    const ddgi = makeProbeUpdateRaysWGSL(8);
    expect(ddgi).toContain('let bakedOutgoing = probeMat.albedo * (1.0 / PI) *');
    expect(ddgi).toContain('ddgiSampleLightMapIrradiance(hit);');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain(
      'let bakedDiffuse = payload.albedo * INV_PI * sampleLightMap(hit) *',
    );
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain(
      'radiance = radiance + throughput * nrcTeacherLocalSourceForHit(',
    );

    const whiteLambertian = [2, 4, 6].map((irradiance) => irradiance / Math.PI);
    expect(whiteLambertian[0]).toBeCloseTo(2 / Math.PI, 12);
    expect(whiteLambertian[1]).toBeCloseTo(4 / Math.PI, 12);
    expect(whiteLambertian[2]).toBeCloseTo(6 / Math.PI, 12);
  });
});
