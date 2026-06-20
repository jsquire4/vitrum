import { describe, expect, it } from 'vitest';
import { makeProbeUpdateRaysWGSL } from '../wgsl/probeUpdateRays.wgsl.js';

describe('DDGI material-map semantics', () => {
  it('applies roughness/metallic maps as scalar factors, not replacements', () => {
    const src = makeProbeUpdateRaysWGSL(64);

    expect(src).toContain('fallback * ddgiMaterialMapChannel(texel, channel)');
    expect(src).toContain('out.roughness = ddgiSampleMaterialScalarMap(');
    expect(src).toContain('out.metalness = ddgiSampleMaterialScalarMap(');
  });
});
