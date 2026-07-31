import { describe, expect, it } from 'vitest';

import { SHADING_TERMS_WGSL } from '../shaders/shadingTerms.wgsl.js';
import { REFRACTIVE_CAUSTICS_WGSL } from '../shaders/refractiveCaustics.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../shaders/transparentOit.wgsl.js';
import { WALKAROUND_UBO_WGSL } from '../shaders/walkaroundUbo.wgsl.js';
import { makeProbeUpdateRaysWGSL } from '../ddgi/wgsl/probeUpdateRays.wgsl.js';

describe('walkaround directional angularDiameter shader plumbing', () => {
  it('threads authored sun cone radius through the shared UBO and visible direct-sun paths', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('sunAngular:');
    expect(WALKAROUND_UBO_WGSL).toContain('x = direct sun cone radius in radians');
    expect(SHADING_TERMS_WGSL).toContain('let sunAngularRadius = max(ubo.sunAngular.x, 0.0);');
    expect(SHADING_TERMS_WGSL).toContain(
      'let xi = pixelHash2(gid, ubo.frameSeed ^ 0x53474341u);',
    );
    expect(SHADING_TERMS_WGSL).toContain('let r2  = sunAngularRadius * sqrt(xi.x);');
    expect(TRANSPARENT_OIT_WGSL).toContain(
      'worldHash2(hitPos, hit.indices.w ^ ubo.frameSeed ^ 0x4f495431u)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('let sunRadius = max(ubo.sunAngular.x, 0.001);');
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain('let omegaSun = 6.2831853 * (1.0 - cos(sunRadius));');
    expect(SHADING_TERMS_WGSL).not.toContain('SUN_ANGULAR_RADIUS');
    expect(REFRACTIVE_CAUSTICS_WGSL).not.toContain('SUN_ANGULAR_RADIUS');
  });

  it('threads authored sun cone radius through DDGI probe direct lighting', () => {
    const ddgiProbeUpdate = makeProbeUpdateRaysWGSL(4);
    expect(ddgiProbeUpdate).toContain('fn ddgiSoftSunDirection');
    expect(ddgiProbeUpdate).toContain('light.innerCone');
    expect(ddgiProbeUpdate).toMatch(
      /ddgiSoftSunDirection\(\s*-ddgiNormalizeOr\(light\.direction, vec3f\(0\.0, -1\.0, 0\.0\)\),\s*light\.innerCone,\s*hitPos,\s*\)/,
    );
  });
});
