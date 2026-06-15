import { describe, expect, it } from 'vitest';

import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';
import { RESTIR_CAST_PRIMARY_WGSL } from '../shaders/restirCastPrimary.wgsl.js';
import { RESTIR_PHAT_WGSL } from '../shaders/restirPHat.wgsl.js';
import { buildReservoirGiWgsl } from '../shaders/reservoirGi.wgsl.js';
import { RIS_WGSL } from '../shaders/ris.wgsl.js';

describe('ReSTIR-DI material parity', () => {
  it('widens PrimarySurface with the material payload needed by extension-aware pHat', () => {
    const reservoirGi = buildReservoirGiWgsl();

    for (const field of [
      'clearcoatNormal: vec3f,',
      'specular: vec4f,',
      'anisotropy: vec2f,',
      'iridescence: vec4f,',
      'clearcoat: vec2f,',
      'sheen: vec4f,',
      'sheenRoughness: f32,',
    ]) {
      expect(reservoirGi).toContain(field);
    }
  });

  it('loads ReSTIR-DI receiver material payloads from the texture atlas', () => {
    expect(MATERIAL_ATLAS_WGSL).toContain('struct RestirDIMaterialPayload');
    expect(MATERIAL_ATLAS_WGSL).toContain('fn sampleRestirDIMaterialPayloadForHit(');
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.albedo = sampleBaseColorMap(hit.indices.w, hit.uv, uv1, scalarBaseColor * vertexColor.rgb);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.rough = sampleMaterialScalarMap(hit.indices.w, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, hit.uv, uv1, decodeRoughMetal(materialWord).x);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.metal = sampleMaterialScalarMap(hit.indices.w, MATERIAL_MAP_SLOT_METALLIC, 2u, hit.uv, uv1, decodeRoughMetal(materialWord).y);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain(
      'payload.clearcoatNormal = applyClearcoatNormalMapForHit(hit, smoothNormal, shadingNormal);',
    );
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.specular = sampleSpecularControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.anisotropy = sampleAnisotropyControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.iridescence = sampleIridescenceControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.clearcoat = sampleClearcoatControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.sheen = sampleSheenControls(hit.indices.w, hit.uv, uv1);');
    expect(MATERIAL_ATLAS_WGSL).toContain('payload.sheenRoughness = sampleSheenRoughness(hit.indices.w, hit.uv, uv1);');
  });

  it('threads atlas material payloads through RIS and temporal/spatial primary casts', () => {
    for (const shader of [RESTIR_CAST_PRIMARY_WGSL, RIS_WGSL]) {
      expect(shader).toContain('let normalMapped = applyNormalMapForHit(hit, smoothNormal);');
      expect(shader).toContain('applyBumpMapForHit(hit, normalMapped)');
      expect(shader).toContain('let payload = sampleRestirDIMaterialPayloadForHit(hit, smoothNormal,');
      expect(shader).toContain('clearcoatNormal = payload.clearcoatNormal');
      expect(shader).toContain('specular = payload.specular');
      expect(shader).toContain('anisotropy = payload.anisotropy');
      expect(shader).toContain('iridescence = payload.iridescence');
      expect(shader).toContain('clearcoat = payload.clearcoat');
      expect(shader).toContain('sheen = payload.sheen');
      expect(shader).toContain('sheenRoughness = payload.sheenRoughness');
    }
  });

  it('uses the extension-aware BRDF for canonical pHat and RIS candidate scoring', () => {
    expect(RESTIR_PHAT_WGSL).toContain('fn restir_di_eval_surface_brdf(surf: PrimarySurface, wi: vec3f) -> vec3f');
    expect(RESTIR_PHAT_WGSL).toContain('return evalGGXWithSpecularClearcoatSheen(');
    expect(RESTIR_PHAT_WGSL).toContain('surf.specular.rgb,');
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
});
