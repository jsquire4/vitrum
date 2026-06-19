import { describe, expect, it } from 'vitest';

import { SHADING_TERMS_WGSL } from '../shaders/shadingTerms.wgsl.js';
import { STAINED_GLASS_SHADE_WGSL } from '../shaders/stainedGlassShade.wgsl.js';
import { WALKAROUND_UBO_WGSL } from '../shaders/walkaroundUbo.wgsl.js';
import { makeProbeUpdateRaysWGSL } from '../ddgi/wgsl/probeUpdateRays.wgsl.js';

describe('walkaround directional angularDiameter shader plumbing', () => {
  it('threads authored sun cone radius through the shared UBO and visible direct-sun paths', () => {
    expect(WALKAROUND_UBO_WGSL).toContain('sunAngular:');
    expect(WALKAROUND_UBO_WGSL).toContain('x = direct sun cone radius in radians');
    expect(SHADING_TERMS_WGSL).toContain('let sunAngularRadius = max(ubo.sunAngular.x, 0.0);');
    expect(SHADING_TERMS_WGSL).toContain('let xi = pixelHash2(gid, 0x53474341u);');
    expect(SHADING_TERMS_WGSL).toContain('let r2  = sunAngularRadius * sqrt(xi.x);');
    expect(STAINED_GLASS_SHADE_WGSL).toContain('let sunAngularRadius = max(ubo.sunAngular.x, 0.0);');
    expect(STAINED_GLASS_SHADE_WGSL).toContain('let xi = pixelHash2(gid, 0x53474341u);');
    expect(STAINED_GLASS_SHADE_WGSL).toContain('let r2 = sunAngularRadius * sqrt(xi.x);');
    expect(SHADING_TERMS_WGSL).not.toContain('SUN_ANGULAR_RADIUS');
    expect(STAINED_GLASS_SHADE_WGSL).not.toContain('SUN_ANGULAR_RADIUS');
  });

  it('threads authored sun cone radius through DDGI probe direct lighting', () => {
    const ddgiProbeUpdate = makeProbeUpdateRaysWGSL(4);
    expect(ddgiProbeUpdate).toContain('fn ddgiSoftSunDirection');
    expect(ddgiProbeUpdate).toContain('light.innerCone');
    expect(ddgiProbeUpdate).toContain('ddgiSoftSunDirection(normalize(-light.direction), light.innerCone, hitPos)');
  });
});
