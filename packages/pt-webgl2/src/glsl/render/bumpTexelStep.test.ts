import { describe, expect, it } from 'vitest';
import { composeTraceGlsl } from '../composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES } from '../../featureTypes.js';
import * as SurfaceRecordSource from './get_surface_record_function.glsl.js';

const surfaceRecordSource = (
  SurfaceRecordSource as unknown as Record<string, string>
).get_surface_record_function!;

describe('mapped bump derivatives', () => {
  it('derives independent UV steps from the map source dimensions', () => {
    expect(surfaceRecordSource).toContain(
      'textures, bumpMapPolicy, 0, bumpSize',
    );
    expect(surfaceRecordSource).toContain(
      'uvPrime.xy + vec2( bumpTexel.x, 0.0 )',
    );
    expect(surfaceRecordSource).toContain(
      'uvPrime.xy + vec2( 0.0, bumpTexel.y )',
    );
    expect(surfaceRecordSource).toContain('float gradientU = dU * float( sourceSize.x );');
    expect(surfaceRecordSource).toContain('float gradientV = dV * float( sourceSize.y );');
    expect(surfaceRecordSource).toContain('centerValid && uValid && vValid');
    expect(surfaceRecordSource).not.toMatch(/\b512(?:\.0)?\b/);
  });

  it('uses the source extent in the mapped-PBR compiler tier too', () => {
    const source = composeTraceGlsl({
      ...DEFAULT_TRACE_FEATURES,
      mappedPbrMaterials: true,
      mappedRichMaterials: false,
    });

    expect(source).toContain(
      'textures, material.bumpMapWrap, 0, bumpSize',
    );
    expect(source).toContain(
      'uvPrime.xy + vec2( bumpTexel.x, 0.0 )',
    );
    expect(source).toContain(
      'uvPrime.xy + vec2( 0.0, bumpTexel.y )',
    );
    expect(source).toContain('centerValid && uValid && vValid');
    expect(source).not.toMatch(/\b512(?:\.0)?\b/);
  });
});
