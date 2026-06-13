import { describe, expect, it } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../../ddgi/wgsl/probeUpdateRays.wgsl.js';
import { RESERVOIR_DI_WGSL } from '../reservoirDi.wgsl.js';
import { RIS_WGSL } from '../ris.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';

describe('emitter castShadow:false shader gates', () => {
  it('threads the shared EmitterTri castShadowDisabled lane through ReSTIR-DI visibility', () => {
    expect(RESERVOIR_DI_WGSL).toContain('castShadowDisabled: f32');
    expect(RIS_WGSL).toContain('if (e.castShadowDisabled < 0.5)');
    expect(SHADING_TERMS_WGSL).toContain('if (e.castShadowDisabled < 0.5)');
  });

  it('threads analytic point/spot and DDGI area-emitter flags into shadow-ray gates', () => {
    expect(SHADING_TERMS_WGSL).toContain('let castShadowDisabled = light3.y > 0.5;');
    expect(SHADING_TERMS_WGSL).toContain('if (!castShadowDisabled)');

    const ddgi = makeProbeUpdateRaysWGSL(4);
    expect(ddgi).toContain('Le.rgb + castShadowDisabled');
    expect(ddgi).toContain('let castShadowDisabled = ddgiEmitterTris[base + 4u].w > 0.5;');
    expect(ddgi).toContain('if (!castShadowDisabled)');
  });
});
