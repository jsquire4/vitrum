import { describe, expect, it } from 'vitest';

import { WGSL_MODULES } from '../pipeline/wgslModules.js';
import { composeWgsl } from '../pipeline/wgslComposer.js';
import { RIS_GI_MODULE, RIS_GI_WGSL } from '../shaders/risGi.wgsl.js';
import { buildRisGiNrcModule } from '../shaders/risGiNrc.wgsl.js';
import { RESTIR_GI_MATERIAL_WGSL } from '../shaders/restirGiMaterial.wgsl.js';

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
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('evalGGXWithSpecularClearcoatSheen(');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('out.Lo = incomingIrradiance * brdf;');
    expect(RESTIR_GI_MATERIAL_WGSL).toContain('out.Lo = diffuseLo;');
  });

  it('wires default risGi bounce hits through mapped material payloads', () => {
    expect(RIS_GI_MODULE.requires).toContain('restirGiMaterial');
    expect(RIS_GI_WGSL).toContain('let normalMapped = applyNormalMapForHit(hit, smoothNormal);');
    expect(RIS_GI_WGSL).toContain('let normal = applyBumpMapForHit(hit, normalMapped);');
    expect(RIS_GI_WGSL).toContain('let smoothNs = restir_gi_smooth_normal_for_hit(bounceHit, bounceHit.normal);');
    expect(RIS_GI_WGSL).toContain('let xsPayload = sampleRestirGIHitMaterialForHit(');
    expect(RIS_GI_WGSL).toContain('Lo = xsPayload.Lo;');
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
    expect(body).not.toContain('let ddgiLo = irrAtXs * xsMat.rgb * INV_PI;');
  });

  it('composes both GI producers with the new material helper', () => {
    const defaultGi = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    const nrcGi = composeWgsl(nrcModule, WGSL_MODULES);

    for (const shader of [defaultGi, nrcGi]) {
      expect(shader).toContain('fn sampleRestirGIHitMaterialForHit(');
      expect(shader).toContain('fn evalGGXWithSpecularClearcoatSheen(');
      expect(shader).toContain('struct RestirDIMaterialPayload');
      expect(shader).toContain('fn applyNormalMapForHit(');
      expect(shader).toContain('fn applyBumpMapForHit(');
    }
  });
});
