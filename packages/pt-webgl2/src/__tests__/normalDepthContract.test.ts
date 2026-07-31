import { describe, expect, it } from 'vitest';

import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';

describe('normalDepth output contract', () => {
  it('writes packed world normals and the no-hit sentinel', () => {
    const glsl = composeTraceGlsl(DEFAULT_TRACE_FEATURES);

    expect(glsl).toContain('surf.normal * 0.5 + 0.5');
    expect(glsl).toContain('vec3( 0.5, 1.0, 0.5 )');
    expect(glsl).toContain('gNormalDepth = vec4( 0.5, 1.0, 0.5, 0.0 );');
    expect(glsl).toContain('float gbufLinearDepth = 0.0');
    expect(glsl).toContain('vec3 primaryRayOrigin = ray.origin;');
    expect(glsl).toContain('if ( ! gbufWritten )');
    expect(glsl).not.toContain('if ( state.firstRay && ! gbufWritten )');
    expect(glsl).toContain(
      'gbufLinearDepth = vitrumSaturatedLengthVec3(',
    );
    expect(glsl).toContain('geometricHitPoint - primaryRayOrigin');
    expect(glsl).not.toContain(
      'gbufLinearDepth = distance( ray.origin + ray.direction * surfaceHit.dist, primaryRayOrigin );',
    );
    expect(glsl).toContain('gNormalDepth = vec4( gbufNormalEnc, gbufLinearDepth );');
  });
});
