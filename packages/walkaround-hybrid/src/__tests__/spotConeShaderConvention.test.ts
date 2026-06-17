import { describe, expect, it } from 'vitest';

import { SHADING_TERMS_WGSL } from '../shaders/shadingTerms.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';

function functionBody(source: string, name: string): string {
  const marker = `fn ${name}(`;
  const start = source.indexOf(marker);
  expect(start, `${name} should be present`).toBeGreaterThanOrEqual(0);
  const brace = source.indexOf('{', start);
  expect(brace, `${name} should have a body`).toBeGreaterThanOrEqual(0);

  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(brace + 1, i);
    }
  }
  throw new Error(`Could not find end of ${name}`);
}

describe('walkaround spot cone shader convention', () => {
  it('uses the forward spot axis consistently for opaque analytic NEE', () => {
    const helper = functionBody(SHADING_TERMS_WGSL, 'analyticSpotConeFalloff');
    const analytic = functionBody(SHADING_TERMS_WGSL, 'lo_analyticNEE');

    expect(helper).toContain('let cosTheta = dot(-axis, wi);');
    expect(helper).toContain('abs(cosInner - cosOuter) < 1e-5');
    expect(analytic).toContain('let cone = analyticSpotConeFalloff(lightDir, wi, cosInner, cosOuter);');
    expect(analytic).toContain('let cutoffDistance = light3.z;');
    expect(analytic).toContain('let decay = light3.w;');
    expect(analytic).toContain('let attenuation = analyticPointSpotAttenuation(dist, cutoffDistance, decay, ubo.emitterDist2Floor);');
    expect(analytic).not.toContain('smoothstep(cosOuter, cosInner, cosTheta)');
  });

  it('uses the same forward spot axis convention for transparent OIT analytic NEE', () => {
    const helper = functionBody(TRANSPARENT_OIT_WGSL, 'oitSpotConeFalloff');
    const analytic = functionBody(TRANSPARENT_OIT_WGSL, 'oitLayerAnalyticNEE');

    expect(helper).toContain('let cosTheta = dot(-axis, wi);');
    expect(helper).toContain('abs(cosInner - cosOuter) < 1e-5');
    expect(analytic).toContain('let cone = oitSpotConeFalloff(lightDir, wi, cosInner, cosOuter);');
    expect(analytic).toContain('let cutoffDistance = light3.z;');
    expect(analytic).toContain('let decay = light3.w;');
    expect(analytic).toContain('let attenuation = oitPointSpotAttenuation(dist, cutoffDistance, decay, ubo.emitterDist2Floor);');
    expect(analytic).not.toContain('smoothstep(cosOuter, cosInner, cosTheta)');
  });
});
