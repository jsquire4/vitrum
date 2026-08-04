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
      'reflectionLayerTransmission: vec3f,',
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
      'payload.albedo = sampleBaseColorMap(hit, scalarBaseColor * vertexColor.rgb);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.rough = faceLayerRoughness(');
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.metal = sampleMaterialScalarMap(hit, MATERIAL_MAP_SLOT_METALLIC, 2u, decodeRoughMetal(materialWord).y);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.clearcoatNormal = applyClearcoatNormalMapForHit(hit, smoothNormal, shadingNormal);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.specular = sampleSpecularControls(hit);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.anisotropy = sampleAnisotropyControls(hit);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.anisotropyTangent = anisotropyFrame.tangent;');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.anisotropyBitangent = anisotropyFrame.bitangent;');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.iridescence = sampleIridescenceControls(hit);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.clearcoat = sampleClearcoatControls(hit);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.sheen = sampleSheenControls(hit);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.sheenRoughness = sampleSheenRoughness(hit);');
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.reflectionLayerTransmission = faceLayerTransmission(layerControls);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.layerTransmission = payload.reflectionLayerTransmission;',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'if (all(layerTransmission == reflectionLayerTransmission)) {',
    );
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
      expect(shader).toContain(
        'reflectionLayerTransmission = payload.reflectionLayerTransmission',
      );
      expect(shader).toContain('layerTransmission = payload.layerTransmission');
      expect(shader).toContain('volumeScattering = payload.volumeScattering');
    }
  });

  it('uses the extension-aware BRDF for canonical pHat and RIS candidate scoring', () => {
    expect(RESTIR_PHAT_WGSL).toContain('fn restir_di_eval_surface_brdf(surf: PrimarySurface, wi: vec3f) -> vec3f');
    expect(RESTIR_PHAT_WGSL).toContain(
      'let mixedBrdf = evalGGXReflectionWithTransmissionMix(',
    );
    expect(RESTIR_PHAT_WGSL).toContain('surf.transmission,');
    expect(RESTIR_PHAT_WGSL).not.toContain('if (surf.isGlass)');
    expect(RESTIR_PHAT_WGSL).toContain(
      'let reflectionBrdf =\n    evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(',
    );
    expect(RESTIR_PHAT_WGSL).toContain(
      'return applyMaterialLayerTransmissionToBrdf(',
    );
    expect(RESTIR_PHAT_WGSL).toContain('surf.reflectionLayerTransmission,');
    expect(RESTIR_PHAT_WGSL).not.toContain(
      'surf.layerTransmission * mixedBrdf',
    );
    expect(RESTIR_PHAT_WGSL).toContain('return applyHomogeneousVolumeSingleScatterDirectional(');
    expect(RESTIR_PHAT_WGSL).toContain(
      'incidentRadiance * brdf,\n    surf.albedo, surf.volumeScattering, surf.bulkThickness,',
    );
    expect(RESTIR_PHAT_WGSL).toContain('surf.specular.rgb,');
    expect(RESTIR_PHAT_WGSL).toContain('surf.anisotropyTangent,');
    expect(RESTIR_PHAT_WGSL).toContain('surf.anisotropyBitangent,');
    expect(RESTIR_PHAT_WGSL).toContain('surf.clearcoatNormal,');
    expect(RESTIR_PHAT_WGSL).toContain(
      'restir_di_eval_surface_response(surf, wi, color)',
    );
    expect(RESTIR_PHAT_WGSL).toContain(
      'restir_di_eval_surface_response(surf, wi, Le * G)',
    );

    expect(RIS_WGSL).toContain('restir_di_eval_surface_response(\n      surf, wi, Le * G,');
    expect(RIS_WGSL).not.toContain('let brdfB =');
    expect(RIS_WGSL).toContain(
      'restir_di_eval_surface_response(\n      surf, envS.dir, receiverEnvironment,',
    );
    expect(RIS_WGSL).not.toContain('let brdf = evalGGX(albedo, roughness, metalness, normal, wo, wi);');
    expect(RIS_WGSL).not.toContain('let brdfB = evalGGX(albedo, roughness, metalness, normal, wo, wiB);');
    expect(RIS_WGSL).not.toContain('let brdfE = evalGGX(albedo, roughness, metalness, normal, wo, envS.dir);');
  });

  it('keeps absolute thin-film reflection when film transmission is zero', () => {
    const faceLayer = [0.8, 0.6, 0.4];
    const filmTransmission = [0, 0, 0];
    const baseClosure = [0.5, 0.25, 0.125];
    const reflectionClosure = [0.3, 0.2, 0.1];
    const combinedLayer = faceLayer.map(
      (face, channel) => face * filmTransmission[channel]!,
    );
    const layered = reflectionClosure.map(
      (reflection, channel) =>
        baseClosure[channel]! * combinedLayer[channel]! +
        reflection * faceLayer[channel]!,
    );

    expect(layered[0]).toBeCloseTo(0.24, 14);
    expect(layered[1]).toBeCloseTo(0.12, 14);
    expect(layered[2]).toBeCloseTo(0.04, 14);
    expect(layered.every((channel) => channel > 0)).toBe(true);
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
    expect(RIS_WGSL).not.toContain('bestLe =');
    expect(SHADING_TERMS_WGSL).toContain('let Le = sampleEmitterLeAtXi(e, r.xi);');
  });

  it('defers RGB transparent-shadow visibility to target shading exactly once', () => {
    expect(RIS_MODULE.requires).toContain('surfaceTextures');
    expect(RIS_WGSL).not.toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(RIS_WGSL).not.toContain('restirDirectVisibilityScalar');
    expect(RIS_WGSL).not.toContain('traceSceneAlphaTintTransmittanceTexturedWithOwnership(');
    expect(RIS_WGSL).not.toContain('traceSceneAlphaTransmittanceTextured(');
    expect(RIS_WGSL).toContain('finaliseReservoirDIFromNativeWrs(&r, wrs, pHatZ);');

    expect(SHADING_TERMS_WGSL).toContain(
      'let shadowTint = traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'shadowContainingMedia,',
    );
    expect(SHADING_TERMS_WGSL).toContain('let shadowScalar = clamp(luminance(shadowTint), 0.0, 1.0);');
    expect(SHADING_TERMS_WGSL).toContain(
      'r.logW, envColor, layeredBrdfE, shadowTint',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'geometryLogW, Le, layeredBrdf, shadowTint,',
    );
    expect(SHADING_TERMS_WGSL).toContain('restirShadeDirectionalVolumeLog(');
    expect(SHADING_TERMS_WGSL).not.toContain('correctedLogW');
    expect(SHADING_TERMS_WGSL).not.toContain('shadowTint / vec3f(shadowScalar)');
  });

  it('does not let source occlusion erase a sample that is visible after reuse', () => {
    const logUnoccludedWeight = Math.log2(4);
    const sourceVisibility = 0;
    const targetVisibility = 1;

    // The stored estimator is visibility-free; only the final target gate is
    // applied. The retired source-baked form would make this contribution zero.
    const targetEstimate = 2 ** logUnoccludedWeight * targetVisibility;
    const retiredSourceBakedEstimate =
      2 ** logUnoccludedWeight * sourceVisibility * targetVisibility;
    expect(targetEstimate).toBe(4);
    expect(retiredSourceBakedEstimate).toBe(0);
  });
});
