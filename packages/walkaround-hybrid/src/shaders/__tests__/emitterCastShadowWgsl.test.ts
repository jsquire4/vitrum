import { describe, expect, it } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../../ddgi/wgsl/probeUpdateRays.wgsl.js';
import { EMITTER_SAMPLING_WGSL } from '../emitterSampling.wgsl.js';
import { RESERVOIR_DI_WGSL } from '../reservoirDi.wgsl.js';
import { RIS_WGSL } from '../ris.wgsl.js';
import { RIS_GI_WGSL } from '../risGi.wgsl.js';
import { RIS_GI_NRC_BODY } from '../risGiNrc.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { TEMPORAL_GI_GRIS_WGSL } from '../temporalGi.wgsl.js';
import { SPATIAL_GI_GRIS_WGSL } from '../spatialGi.wgsl.js';

describe('emitter castShadow:false shader gates', () => {
  it('uses the sampled CDF segment as the flat emitter PMF', () => {
    expect(EMITTER_SAMPLING_WGSL).toContain('fn emitterCdfPmf(');
    expect(EMITTER_SAMPLING_WGSL).toContain('here - prev');
    expect(RIS_WGSL).toContain('emitterSelPmf = emitterCdfPmf(&emitterCdf, emCount, lid);');
    expect(RIS_WGSL).not.toContain('luminance(emitters[lid].Le) * emitters[lid].area) / totalPower');
  });

  it('threads the shared EmitterTri castShadowDisabled lane through ReSTIR-DI visibility', () => {
    expect(RESERVOIR_DI_WGSL).toContain('sourceTriIndex: f32');
    expect(RESERVOIR_DI_WGSL).toContain('castShadowDisabled: f32');
    expect(RIS_WGSL).toContain('if (e.castShadowDisabled < 0.5)');
    expect(SHADING_TERMS_WGSL).toContain('if (e.castShadowDisabled < 0.5)');
    expect(RIS_WGSL).toContain('traceSceneAlphaTransmittanceTextured(');
    expect(SHADING_TERMS_WGSL).toContain('traceSceneAlphaTransmittanceTextured(');
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
    expect(ddgi).toContain('Le.rgb + castShadowDisabled');
    expect(ddgi).toContain('let castShadowDisabled = ddgiEmitterTris[base + 4u].w > 0.5;');
    expect(ddgi).toContain('if (!castShadowDisabled)');
  });

  it('threads primitive castShadow:false into DDGI and ReSTIR-GI shadow visibility', () => {
    const ddgi = makeProbeUpdateRaysWGSL(4);
    expect(ddgi).toContain('fn bvhCastShadowDisabledForTri(triIdx: u32) -> bool');
    expect(ddgi).toContain('MATERIAL_FLAG_CAST_SHADOW_DISABLED');
    expect(ddgi).toContain('fn bvhTraceAnyCastShadow(');
    expect(ddgi).toContain('fn ddgiTraceShadowTransmittance(');
    expect(ddgi).toContain('let shadowT = ddgiTraceShadowTransmittance(shadowOrig, lightDir, dist - normalBias_p, false)');
    expect(ddgi).toContain('tau = tau * ddgiAlphaShadowTransmittanceForHit(hit)');

    for (const src of [RIS_GI_WGSL, RIS_GI_NRC_BODY]) {
      expect(src).toContain('traceSceneAlphaTransmittanceTextured(');
      expect(src).toContain('BVH_MATERIAL_TEX_WIDTH');
    }

    for (const src of [TEMPORAL_GI_GRIS_WGSL, SPATIAL_GI_GRIS_WGSL]) {
      expect(src).toContain('traceSceneAnyAlphaMaskTextured(');
      expect(src).toContain('BVH_MATERIAL_TEX_WIDTH');
    }
  });
});
