import { describe, expect, it } from 'vitest';

import { EMITTER_LE_AT_XI_WGSL } from '../shaders/emitterLeAtXi.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';
import { RESTIR_CAST_PRIMARY_WGSL } from '../shaders/restirCastPrimary.wgsl.js';
import { RESTIR_PHAT_WGSL } from '../shaders/restirPHat.wgsl.js';
import { buildReservoirGiWgsl } from '../shaders/reservoirGi.wgsl.js';
import { RIS_MODULE, RIS_WGSL } from '../shaders/ris.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shaders/shadingTerms.wgsl.js';

describe('ReSTIR-DI material parity', () => {
  it('widens PrimarySurface with the material payload needed by extension-aware pHat', () => {
    const reservoirGi = buildReservoirGiWgsl();

    for (const field of [
      'clearcoatNormal: vec3f,',
      'specular: vec4f,',
      'anisotropy: vec2f,',
      'anisotropyTangent: vec3f,',
      'anisotropyBitangent: vec3f,',
      'iridescence: vec4f,',
      'clearcoat: vec2f,',
      'sheen: vec4f,',
      'sheenRoughness: f32,',
      'layerTransmission: vec3f,',
      'volumeScattering: vec4f,',
    ]) {
      expect(reservoirGi).toContain(field);
    }
  });

  it('loads ReSTIR-DI receiver material payloads from the texture atlas', () => {
    expect(MATERIAL_ATLAS_WGSL).toContain('struct RestirDIMaterialPayload');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleRestirDIMaterialPayloadForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain('let layerControls = sampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);');
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.albedo = sampleBaseColorMap(hit.indices.w, hit.uv, uv1, scalarBaseColor * vertexColor.rgb);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.rough = faceLayerRoughness(');
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.metal = sampleMaterialScalarMap(hit.indices.w, MATERIAL_MAP_SLOT_METALLIC, 2u, hit.uv, uv1, decodeRoughMetal(materialWord).y);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.clearcoatNormal = applyClearcoatNormalMapForHit(hit, smoothNormal, shadingNormal);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.specular = sampleSpecularControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.anisotropy = sampleAnisotropyControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.anisotropyTangent = anisotropyFrame.tangent;');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.anisotropyBitangent = anisotropyFrame.bitangent;');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.iridescence = sampleIridescenceControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.clearcoat = sampleClearcoatControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.sheen = sampleSheenControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.sheenRoughness = sampleSheenRoughness(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.layerTransmission = faceLayerTransmission(layerControls);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.volumeScattering = sampleVolumeScatteringControls(hit.indices.w);');
  });

  it('threads atlas material payloads through RIS and temporal/spatial primary casts', () => {
    for (const shader of [RESTIR_CAST_PRIMARY_WGSL, RIS_WGSL]) {
      expect(shader).toContain('let normalMapped = applyNormalMapForHit(hit, smoothNormal);');
      expect(shader).toContain('applyBumpMapForHit(hit, normalMapped)');
      expect(shader).toContain('let payload = sampleRestirDIMaterialPayloadForHit(hit, smoothNormal,');
      expect(shader).toContain('clearcoatNormal = payload.clearcoatNormal');
      expect(shader).toContain('specular = payload.specular');
      expect(shader).toContain('anisotropy = payload.anisotropy');
      expect(shader).toContain('anisotropyTangent = payload.anisotropyTangent');
      expect(shader).toContain('anisotropyBitangent = payload.anisotropyBitangent');
      expect(shader).toContain('iridescence = payload.iridescence');
      expect(shader).toContain('clearcoat = payload.clearcoat');
      expect(shader).toContain('sheen = payload.sheen');
      expect(shader).toContain('sheenRoughness = payload.sheenRoughness');
      expect(shader).toContain('layerTransmission = payload.layerTransmission');
      expect(shader).toContain('volumeScattering = payload.volumeScattering');
    }
  });

  it('uses the extension-aware BRDF for canonical pHat and RIS candidate scoring', () => {
    expect(RESTIR_PHAT_WGSL).toContain('fn restir_di_eval_surface_brdf(surf: PrimarySurface, wi: vec3f) -> vec3f');
    expect(RESTIR_PHAT_WGSL).toContain('let brdf = surf.layerTransmission * evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(');
    expect(RESTIR_PHAT_WGSL).toContain('return applyVolumeScatteringApproximation(brdf, surf.albedo, surf.volumeScattering, surf.normal, surf.wo);');
    expect(RESTIR_PHAT_WGSL).toContain('surf.specular.rgb,');
    expect(RESTIR_PHAT_WGSL).toContain('surf.anisotropyTangent,');
    expect(RESTIR_PHAT_WGSL).toContain('surf.anisotropyBitangent,');
    expect(RESTIR_PHAT_WGSL).toContain('surf.clearcoatNormal,');
    expect(RESTIR_PHAT_WGSL).toContain('let brdf  = restir_di_eval_surface_brdf(surf, wi);');
    expect(RESTIR_PHAT_WGSL).toContain('let brdf = restir_di_eval_surface_brdf(surf, wi);');

    expect(RIS_WGSL).toContain('let brdf = restir_di_eval_surface_brdf(surf, wi);');
    expect(RIS_WGSL).toContain('let brdfB = restir_di_eval_surface_brdf(surf, wiB);');
    expect(RIS_WGSL).toContain('let brdfE = restir_di_eval_surface_brdf(surf, envS.dir);');
    expect(RIS_WGSL).not.toContain('let brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi);');
    expect(RIS_WGSL).not.toContain('let brdfB = evalGGX(albedo, roughness, metalness, normal, wo, wiB);');
    expect(RIS_WGSL).not.toContain('let brdfE = evalGGX(albedo, roughness, metalness, normal, wo, envS.dir);');
  });

  it('samples UV-varying emissive-map radiance for ReSTIR-DI emitter evaluation', () => {
    expect(EMITTER_LE_AT_XI_WGSL).toContain('fn sampleEmitterLeAtXi(e: EmitterTri, xi: vec2f) -> vec3f');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('-(tri + 2) = mirrored TLAS instance');
    expect(EMITTER_LE_AT_XI_WGSL).not.toContain('if (ubo.bvhMode != 0u)');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('if (encodedSourceTri == -1)');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('let mirroredSourceTri = encodedSourceTri < -1;');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('bary = vec3f(bary.z, bary.y, bary.x);');
    expect(EMITTER_LE_AT_XI_WGSL).toContain('return sampleEmissiveMap(triIndex, uv0, uv1, e.Le);');
    expect(RESTIR_PHAT_WGSL).toContain('let Le = sampleEmitterLeAtXi(e, xi);');
    expect(RIS_WGSL).toContain('let Le = sampleEmitterLeAtXi(e, xiTri);');
    expect(RIS_WGSL).toContain('bestLe = sampleEmitterLeAtXi(eb, bestXi);');
    expect(SHADING_TERMS_WGSL).toContain('let Le = sampleEmitterLeAtXi(e, r.xi);');
  });

  it('uses RGB transparent-shadow visibility for ReSTIR-DI finalization and shade consumption', () => {
    expect(RIS_MODULE.requires).toContain('surfaceTextures');
    expect(RIS_WGSL).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(RIS_WGSL).toContain('fn restirDirectVisibilityScalar(tint: vec3f) -> f32');
    expect(RIS_WGSL).toContain('let shadowTint = traceSceneAlphaTintTransmittanceTextured(');
    expect(RIS_WGSL).toContain('let shadowT = restirDirectVisibilityScalar(shadowTint);');
    expect(RIS_WGSL).not.toContain('traceSceneAlphaTransmittanceTextured(');

    expect(SHADING_TERMS_WGSL).toContain('let shadowTint = traceSceneAlphaTintTransmittanceTextured(');
    expect(SHADING_TERMS_WGSL).toContain('let shadowScalar = clamp(luminance(shadowTint), 0.0, 1.0);');
    expect(SHADING_TERMS_WGSL).toContain('let shadowColorCorrection = shadowTint / vec3f(max(shadowScalar, 1e-4));');
    expect(SHADING_TERMS_WGSL).toContain('return Le * brdf * G * r.W * shadowColorCorrection;');
  });
});
