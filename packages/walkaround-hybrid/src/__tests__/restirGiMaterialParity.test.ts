import { describe, expect, it } from 'vitest';

import { WGSL_MODULES } from '../pipeline/wgslModules.js';
import { composeWgsl } from '../pipeline/wgslComposer.js';
import { RIS_GI_MODULE, RIS_GI_WGSL } from '../shaders/risGi.wgsl.js';
import { buildRisGiNrcModule } from '../shaders/risGiNrc.wgsl.js';
import { RESTIR_GI_MATERIAL_WGSL } from '../shaders/restirGiMaterial.wgsl.js';
import { TEMPORAL_GI_GRIS_MODULE, TEMPORAL_GI_MODULE, TEMPORAL_GI_WGSL } from '../shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_GRIS_MODULE, SPATIAL_GI_MODULE, SPATIAL_GI_WGSL } from '../shaders/spatialGi.wgsl.js';

const nrcModule = buildRisGiNrcModule({
  levels: 1,
  featuresPerEntry: 4,
  oneBlobBins: 4,
  width: 16,
  outWidth: 3,
  hidden: 1,
});

describe('ReSTIR-GI material parity', () => {
  it('defines a GI suffix material helper over the same atlas payload used by DI', () => {
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('struct RestirGIHitMaterial');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_smooth_normal_for_hit(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_shading_normal_for_hit(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn sampleRestirGIHitMaterialForHit(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('sampleRestirDIMaterialPayloadForHit(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('applyNormalMapForHit(hit, smoothNormal)');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('applyBumpMapForHit(hit, normalMapped)');
  });

  it('routes rich GI suffix materials through extension-aware BRDF shading', () => {
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('restir_gi_has_rich_suffix_payload(payload)');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.metal > 1e-4');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.rough < 0.84');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.clearcoat.x > 1e-4');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.sheen.a > 1e-4');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.iridescence.x > 1e-4');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.anisotropyTangent,');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.anisotropyBitangent,');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('out.Lo = surfaceEmission + incomingIrradiance * brdf;');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('out.Lo = surfaceEmission + diffuseLo;');
  });

  it('adds readable emissive-map surface emission to GI suffix hits', () => {
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('@group(1) @binding(12) var restir_gi_bvh_emissive');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_surface_emission_for_hit(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('textureLoad(restir_gi_bvh_emissive, vec2i(coord), 0).rgb');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('hit.indices.w % BVH_MATERIAL_TEX_WIDTH');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('sampleEmissiveMap(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('materialAtlasUv1ForHit(hit)');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('let surfaceEmission = restir_gi_surface_emission_for_hit(hit);');
  });

  it('defines receiver-lobe p-hat helpers for rich-material GI reuse', () => {
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_receiver_phat_from_payload(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_receiver_phat_from_surface_or_geometry(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('contribution = contribution + Lo * specBrdf;');
  });

  it('wires default risGi bounce hits through mapped material payloads', () => {
    expect(RIS_GI_MODULE.requires).toContain('restirGiMaterial');
    expect(RIS_GI_WGSL).toContain('let normalMapped = applyNormalMapForHit(hit, smoothNormal);');
    expect(RIS_GI_WGSL).toContain('let normal = applyBumpMapForHit(hit, normalMapped);');
    expect(RIS_GI_WGSL).toContain('let smoothNs = restir_gi_smooth_normal_for_hit(bounceHit, bounceHit.normal);');
    expect(RIS_GI_WGSL).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    expect(RIS_GI_WGSL).toContain('Lo = xsPayload.Lo;');
    expect(RIS_GI_WGSL).toContain('let receiverPayload = sampleRestirDIMaterialPayloadForHit(');
    expect(RIS_GI_WGSL).toContain('restir_gi_receiver_phat_from_payload(');
    expect(RIS_GI_WGSL).not.toContain('Lo = irrAtXs * xsMat.rgb * INV_PI;');
  });

  it('wires NRC risGi bounce hits and training records through the same payload', () => {
    const body = nrcModule.source;
    expect(nrcModule.requires).toContain('restirGiMaterial');
    expect(body).toContain('let normalMapped = applyNormalMapForHit(hit, smoothNormal);');
    expect(body).toContain('let normal = applyBumpMapForHit(hit, normalMapped);');
    expect(body).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    expect(body).toContain('let ddgiLo = xsPayload.Lo;');
    expect(body).toContain('let xsAlbedo = xsPayload.albedo;');
    expect(body).toContain('let xsRough = xsPayload.rough;');
    expect(body).toContain('let receiverPayload = sampleRestirDIMaterialPayloadForHit(');
    expect(body).toContain('restir_gi_receiver_phat_from_payload(');
    expect(body).not.toContain('let ddgiLo = irrAtXs * xsMat.rgb * INV_PI;');
  });

  it('wires GI temporal/spatial reuse through receiver-material p-hat', () => {
    for (const module of [TEMPORAL_GI_MODULE, TEMPORAL_GI_GRIS_MODULE, SPATIAL_GI_MODULE, SPATIAL_GI_GRIS_MODULE]) {
      expect(module.requires).toContain('restirCastPrimary');
      expect(module.requires).toContain('restirGiMaterial');
    }
    for (const shader of [TEMPORAL_GI_WGSL, SPATIAL_GI_WGSL]) {
      expect(shader).toContain('@group(1) @binding(0) var<storage, read> bvh');
      expect(
        shader.includes('let centerSurf = castPrimary(') ||
        shader.includes('let curSurf = castPrimary('),
      ).toBe(true);
      expect(shader).toContain('restir_gi_receiver_phat_from_surface_or_geometry(');
      expect(shader).toContain('finaliseGIReservoirWFromPHat(');
    }
  });

  it('composes both GI producers with the new material helper', () => {
    const defaultGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    const nrcGi = composeWgsl(nrcModule, WGSL_MODULES);

    for (const shader of [defaultGi, nrcGi]) {
      expect(shader).toContain('fn sampleRestirGIHitMaterialForHit(');
      expect(shader).toContain('fn evalGGXWithSpecularClearcoatSheen(');
      expect(shader).toContain('fn evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(');
      expect(shader).toContain('struct RestirDIMaterialPayload');
      expect(shader).toContain('fn applyNormalMapForHit(');
      expect(shader).toContain('fn applyBumpMapForHit(');
      expect(shader).toContain('var restir_gi_bvh_emissive: texture_2d<f32>');
      expect(shader).toContain('fn sampleEmissiveMap(');
    }
  });
});
