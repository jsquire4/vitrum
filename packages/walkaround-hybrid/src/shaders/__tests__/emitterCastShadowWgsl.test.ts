import { describe, expect, it } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../../ddgi/wgsl/probeUpdateRays.wgsl.js';
import { EMITTER_LE_AT_XI_WGSL } from '../emitterLeAtXi.wgsl.js';
import { EMITTER_SAMPLING_WGSL } from '../emitterSampling.wgsl.js';
import { MATERIAL_ATLAS_MODULE } from '../materialAtlas.wgsl.js';
import { SCENE_STORAGE_ARENA_WGSL } from '../sceneStorageArena.wgsl.js';
import { SHADE_MODULE } from '../shade.wgsl.js';
import { RIS_WGSL } from '../ris.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { RESTIR_GI_MATERIAL_WGSL } from '../restirGiMaterial.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { TRANSPARENT_OIT_MODULE } from '../transparentOit.wgsl.js';
import { TEMPORAL_GI_MODULE } from '../temporalGi.wgsl.js';
import { SPATIAL_GI_MODULE } from '../spatialGi.wgsl.js';
import { composeWgsl } from '../../pipeline/wgslComposer.js';
import { WGSL_MODULES } from '../../pipeline/wgslModules.js';

describe('emitter castShadow:false shader gates', () => {
  it('uses the sampled CDF segment as the flat emitter PMF', () => {
    expect(EMITTER_SAMPLING_WGSL).toContain('fn emitterCdfPmf(');
    expect(EMITTER_SAMPLING_WGSL).toContain('here - prev');
    expect(RIS_WGSL).toContain('emitterSelPmf = emitterCdfPmf(emCount, lid);');
    expect(RIS_WGSL).not.toContain('luminance(emitters[lid].Le) * emitters[lid].area) / totalPower');
  });

  it('threads the shared EmitterTri flag lane through ReSTIR-DI visibility', () => {
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('sourceTriIndex: f32');
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('sourceSubdivLevel: f32');
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('sourceSubdivOrdinal: f32');
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('emitterFlags: f32');
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('fn emitterTriCastShadowDisabled(');
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('fn emitterTriIsTwoSided(');
    expect(SCENE_STORAGE_ARENA_WGSL).toContain('fn emitterTriCosineTowardReceiver(');
    expect(RIS_WGSL).toContain('if (!emitterTriCastShadowDisabled(e))');
    expect(SHADING_TERMS_WGSL).toContain('if (!emitterTriCastShadowDisabled(e))');
    expect(RIS_WGSL).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(RIS_WGSL).toContain('traceSceneAlphaTintTransmittanceTexturedWithOwnership(');
    expect(RIS_WGSL).toContain('restirDirectVisibilityScalar(shadowTint)');
    expect(RIS_WGSL).not.toContain('traceSceneAlphaTransmittanceTextured(');
    expect(SHADING_TERMS_WGSL).toContain('traceSceneAlphaTintTransmittanceTexturedWithOwnership(');
  });

  it('maps micro-emitter samples back to parent-triangle UV barycentrics', () => {
    expect(EMITTER_LE_AT_XI_WGSL).toContain('fn emitterParentBarycentricFromXi(e: EmitterTri, xi: vec2f) -> vec3f');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('e.sourceSubdivLevel');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('e.sourceSubdivOrdinal');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('return localBary.x * a + localBary.y * b + localBary.z * c;');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('return localBary.x * b + localBary.y * d + localBary.z * c;');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('var bary = emitterParentBarycentricFromXi(e, xi);');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('bary = vec3f(bary.z, bary.y, bary.x);');

    const ddgi = makeProbeUpdateRaysWGSL(4);
    expect(ddgi).toContain('fn ddgiEmitterParentBarycentricFromLocal(localBary: vec3f, levelF: f32, ordinalF: f32) -> vec3f');
    expect(ddgi).toContain('let encodedSourceTri = i32(round(ddgiEmitterTris[base + 0u].w));');
    expect(ddgi).toContain('ddgiEmitterTris[base + 1u].w');
    expect(ddgi).toContain('ddgiEmitterTris[base + 2u].w');
    expect(ddgi).toContain('bary = vec3f(bary.z, bary.y, bary.x);');
    expect(ddgi).toContain('let Le = ddgiSampleEmitterLeAtBary(base, localBary, scalarLe);');
  });

  it('threads analytic point/spot and DDGI area-emitter flags into shadow-ray gates', () => {
    expect(SHADING_TERMS_WGSL).toContain('let castShadowDisabled = light3.y > 0.5;');
    expect(SHADING_TERMS_WGSL).toContain('if (!castShadowDisabled)');
    expect(SHADING_TERMS_WGSL).toContain('SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED');

    const ddgi = makeProbeUpdateRaysWGSL(4);
    expect(ddgi).toContain('LIGHT_CAST_SHADOW_DISABLED');
    expect(ddgi).toContain('fn ddgiLightKind(light: DDGILight) -> u32');
    expect(ddgi).toContain('if (!ddgiLightCastShadowDisabled(light))');
    expect(ddgi).toContain('castShadowDisabled: bool');
    expect(ddgi).toContain('Le.rgb + emitterFlags');
    expect(ddgi).toContain('let castShadowDisabled = (emitterFlags & 1u) != 0u;');
    expect(ddgi).toContain('let twoSidedEmitter = (emitterFlags & 2u) != 0u;');
    expect(ddgi).toContain('abs(signedCosLight)');
    expect(ddgi).toContain('if (!castShadowDisabled)');
  });

  it('keeps back-interface traversal separate from one-sided forward emission', () => {
    expect(MATERIAL_ATLAS_MODULE.source).toContain(
      'fn materialEmissionSideAdmittedForHit(hit: IntersectionResult) -> bool',
    );
    expect(SHADE_MODULE.source).toContain(
      'materialEmissionSideAdmittedForHit(primaryHit)',
    );
    expect(TRANSPARENT_OIT_MODULE.source).toContain(
      'materialEmissionSideAdmittedForHit(hit)',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'if (!materialEmissionSideAdmittedForHit(hit))',
    );

    const ddgi = makeProbeUpdateRaysWGSL(4);
    expect(ddgi).toContain(
      '(mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u',
    );
    expect(ddgi).toContain('ddgiSampleEmissiveMap(hit, scalarSurfaceEmission)');
  });

  it('threads primitive castShadow:false into DDGI and ReSTIR-GI shadow visibility', () => {
    const ddgi = makeProbeUpdateRaysWGSL(4);
    expect(ddgi).toContain('fn ddgiWorldSurfaceBudget() -> u32');
    expect(ddgi).toContain('MATERIAL_FLAG_CAST_SHADOW_DISABLED');
    expect(ddgi).toContain('fn ddgiTraceShadowVisibility(');
    expect(ddgi).toContain('shadowVisibility = ddgiTraceShadowVisibility(shadowOrig, lightDir, dist - normalBias_p)');
    expect(ddgi).toContain('shadowT = ddgiTraceShadowVisibility(hitPos + n * normalBias, wi, dist - normalBias)');
    expect(ddgi).toContain('visibility = visibility * alphaT');
    expect(ddgi).toContain('let boundaryTransmission = clamp(');
    expect(ddgi).toContain('ddgiSampleTransmissionMapForHit(hit, mat.transmission),');
    expect(ddgi).toContain('mediumMaterial[mediumDepth] = matId;');
    expect(ddgi).toContain('mediumInstance[mediumDepth] = hit.instanceIndex;');
    expect(ddgi).not.toContain('fn bvhTraceAnyCastShadow(');

    for (const src of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(src).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
      expect(src).toContain('traceSceneAlphaTintTransmittanceTextured(');
      expect(src).toContain('clamp(luminance(shadowTint');
      expect(src).not.toContain('traceSceneAlphaTransmittanceTextured(');
      expect(src).toContain('BVH_MATERIAL_TEX_WIDTH');
    }

    for (const src of [
      composeWgsl(TEMPORAL_GI_MODULE, WGSL_MODULES),
      composeWgsl(SPATIAL_GI_MODULE, WGSL_MODULES),
    ]) {
      expect(src).toContain('traceSceneAlphaTintTransmittanceTextured(');
      expect(src).not.toContain('traceSceneAnyAlphaMaskTextured(');
      expect(src).toContain('BVH_MATERIAL_TEX_WIDTH');
    }
  });
});
