import { describe, expect, it } from 'vitest';

import { WGSL_MODULES } from '../pipeline/wgslModules.js';
import { composeWgsl } from '../pipeline/wgslComposer.js';
import { RIS_GI_MODULE, RIS_GI_WGSL } from '../shaders/risGi.wgsl.js';
import { buildRisGiNrcModule } from '../shaders/risGiNrc.wgsl.js';
import { RESTIR_GI_MATERIAL_WGSL } from '../shaders/restirGiMaterial.wgsl.js';
import { TEMPORAL_GI_MODULE, TEMPORAL_GI_WGSL } from '../shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_MODULE, SPATIAL_GI_WGSL } from '../shaders/spatialGi.wgsl.js';

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

  it('keeps the irradiance-fed suffix measure-correct while preserving layers and volume', () => {
    expect(RESTIR_GI_MATERIAL_WGSL).not.toContain('restir_gi_proxy_incoming_dir');
    expect(RESTIR_GI_MATERIAL_WGSL).not.toContain('restir_gi_has_rich_suffix_payload');
    expect(RESTIR_GI_MATERIAL_WGSL).not.toContain('incomingIrradiance * brdf');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'let bakedDiffuse = payload.albedo * INV_PI * sampleLightMap(hit);',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'let diffuseLo = (bakedDiffuse +\n    incomingIrradiance * payload.albedo * INV_PI) *',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      '(1.0 - clamp(transmission, 0.0, 1.0));',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('out.Lo = applyHomogeneousVolumeSingleScatter(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('(surfaceEmission + diffuseLo) * payload.layerTransmission');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.volumeScattering,');
  });

  it('adds readable emissive-map surface emission to GI suffix hits', () => {
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('@group(1) @binding(12) var restir_gi_bvh_emissive');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_surface_emission_for_hit(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('textureLoad(restir_gi_bvh_emissive, vec2i(coord), 0).rgb');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('hit.indices.w % BVH_MATERIAL_TEX_WIDTH');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('sampleEmissiveMap(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('materialAtlasUv1ForHit(hit)');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'let surfaceEmission = restir_gi_surface_emission_for_hit(hit);',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'let bakedDiffuse = payload.albedo * INV_PI * sampleLightMap(hit);',
    );
  });

  it('defines receiver-lobe p-hat helpers for rich-material GI reuse', () => {
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_receiver_phat_from_payload(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('fn restir_gi_receiver_phat_from_surface_or_geometry(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'contribution = contribution + Lo * specBrdf *\n      payload.reflectionLayerTransmission;',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'cosTheta * INV_PI *\n    payload.layerTransmission;',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'payload.reflectionLayerTransmission = surf.reflectionLayerTransmission;',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.volumeScattering = surf.volumeScattering;');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain(
      'return applyHomogeneousVolumeSingleScatterDirectional(',
    );
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('payload.bulkThickness,');
    expect(RESTIR_GI_MATERIAL_WGSL).not.toContain(
      'contribution * payload.layerTransmission',
    );
  });

  it('wires default risGi bounce hits through mapped material payloads', () => {
    expect(RIS_GI_MODULE.requires).toContain('restirGiMaterial');
    expect(RIS_GI_MODULE.requires).toContain('surfaceTextures');
    expect(RIS_GI_WGSL).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(RIS_GI_WGSL).toContain('let normalMapped = applyNormalMapForHit(hit, smoothNormal);');
    expect(RIS_GI_WGSL).toContain('let normal = applyBumpMapForHit(hit, normalMapped);');
    expect(RIS_GI_WGSL).toContain('let smoothNs = restir_gi_smooth_normal_for_hit(bounceHit, bounceHit.normal);');
    expect(RIS_GI_WGSL).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    expect(RIS_GI_WGSL).toContain('Lo = xsPayload.Lo;');
    expect(RIS_GI_WGSL).toContain('let receiverPayload = sampleRestirDIMaterialPayloadForHit(');
    expect(RIS_GI_WGSL).toContain('let receiverPHat = restir_gi_receiver_phat_from_payload(');
    expect(RIS_GI_WGSL).toContain(
      'let logPHat = reservoirGiLogPositiveProduct(receiverPHat, candidateVisibility);',
    );
    expect(RIS_GI_WGSL).toContain('traceSceneFirstHitAlphaMaskTexturedWithMetadata(');
    expect(RIS_GI_WGSL).toContain('bounceTrace.requiresNativeEstimator');
    expect(RIS_GI_WGSL).toContain('let candidateVisibility: f32 = 1.0;');
    expect(RIS_GI_WGSL).not.toContain('let shadowTint = traceSceneAlphaTintTransmittanceTextured(');
    expect(RIS_GI_WGSL).not.toContain('traceSceneAlphaTransmittanceTextured(');
    expect(RIS_GI_WGSL).not.toContain('Lo = irrAtXs * xsMat.rgb * INV_PI;');
  });

  it('wires NRC risGi bounce hits and training records through the same payload', () => {
    const body = nrcModule.source;
    expect(nrcModule.requires).toContain('restirGiMaterial');
    expect(nrcModule.requires).toContain('surfaceTextures');
    expect(body).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(body).toContain('let normalMapped = applyNormalMapForHit(hit, smoothNormal);');
    expect(body).toContain('let normal = applyBumpMapForHit(hit, normalMapped);');
    expect(body).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    // The canonical producer always uses the receiver-independent proxy; NRC
    // may substitute its learned independent suffix after that baseline.
    expect(body).toContain('let ddgiLo = ddgiProxyLo;');
    expect(body).toContain('let xsAlbedo = xsPayload.albedo;');
    expect(body).toContain('let xsRough = xsPayload.rough;');
    expect(body).toContain('let receiverPayload = sampleRestirDIMaterialPayloadForHit(');
    expect(body).toContain('let receiverPHat = restir_gi_receiver_phat_from_payload(');
    expect(body).toContain(
      'let logPHat = reservoirGiLogPositiveProduct(receiverPHat, candidateVisibility);',
    );
    expect(body).toContain('let candidateVisibility: f32 = 1.0;');
    expect(body).not.toContain('let shadowTint = traceSceneAlphaTintTransmittanceTextured(');
    expect(body).not.toContain('traceSceneAlphaTransmittanceTextured(');
    expect(body).not.toContain('let ddgiLo = irrAtXs * xsMat.rgb * INV_PI;');
  });

  it('keeps camera-prefix glass out of reusable GI while retaining the ordinary PPG mixture', () => {
    for (const body of [RIS_GI_WGSL, nrcModule.source]) {
      expect(body).toContain('let ppgGuidedOn = ubo.ppgEnabled == 1u;');
      expect(body).toContain('let alpha = represented_bernoulli_probability_f32(');
      expect(body).toContain('select(0.0, ubo.ppgMixAlpha, ppgGuidedOn),');
      expect(body).toContain(
        'logPSrc = reservoirGiLogProposalMixture(alpha, pGuide, pCos);',
      );
      expect(body).not.toContain('ppgGuidedOn_g');
      expect(body).not.toContain('walkHitPos');
    }
  });

  it('wires GI temporal/spatial reuse through canonical material targets', () => {
    for (const module of [TEMPORAL_GI_MODULE, SPATIAL_GI_MODULE]) {
      expect(module.requires).toContain('restirCastPrimary');
      expect(module.requires).toContain('restirGiMaterial');
    }
    for (const [module, shader] of [
      [TEMPORAL_GI_MODULE, TEMPORAL_GI_WGSL],
      [SPATIAL_GI_MODULE, SPATIAL_GI_WGSL],
    ] as const) {
      const composed = composeWgsl(module, WGSL_MODULES);
      expect(composed).toContain('@group(1) @binding(0) var<storage, read> bvhSceneGeometryArena');
      expect(composed).toContain('fn bvhLoadNode(');
      expect(shader).toContain('grisLogMaterialPHatAt(');
      expect(shader).not.toContain('grisMaterialTargetAt(');
      expect(shader).toContain('grisFinaliseRepresentedReservoir(');
      expect(shader).toContain('candidate.H,');
      expect(shader).toContain('GI_SAMPLE_FLAG_RECAST_TINT');
      expect(composed).toContain('restir_gi_receiver_phat_from_surface_or_geometry(');
    }
  });

  it('composes both GI producers with the new material helper', () => {
    const defaultGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    const nrcGi = composeWgsl(nrcModule, WGSL_MODULES);

    for (const shader of [defaultGi, nrcGi]) {
      expect(shader).toContain('fn sampleRestirGIHitMaterialForHit(');
      expect(shader).toContain('fn evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(');
      expect(shader).toContain('struct RestirDIMaterialPayload');
      expect(shader).toContain('fn applyNormalMapForHit(');
      expect(shader).toContain('fn applyBumpMapForHit(');
      expect(shader).toContain('var restir_gi_bvh_emissive: texture_2d<f32>');
      expect(shader).toContain('fn sampleEmissiveMap(');
      expect(shader).toContain('fn traceSceneAlphaTintTransmittanceTextured(');
      expect(shader).toContain('textureLoad(bvh_beer');
    }
  });
});
