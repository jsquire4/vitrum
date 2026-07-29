import { describe, expect, it } from 'vitest';
import * as LightSamplingModule from './shader/sampling/light_sampling_functions.glsl.js';
import * as BdptLightSubpathModule from './render/bdpt_light_subpath.glsl.js';

function glslChunk(module: Record<string, unknown>, name: string): string {
  const chunk = module[name];
  if (typeof chunk !== 'string') {
    throw new TypeError(`Expected GLSL module "${name}" to export a string.`);
  }
  return chunk;
}

function dot(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function oneSidedAreaPdf(
  area: number,
  distance: number,
  normal: readonly [number, number, number],
  receiverToLight: readonly [number, number, number],
): number {
  const cosLight = dot(normal, [
    -receiverToLight[0],
    -receiverToLight[1],
    -receiverToLight[2],
  ]);
  return cosLight > 0 ? (distance * distance) / (area * cosLight) : 0;
}

describe('analytic area-light one-sided NEE / forward-hit parity', () => {
  const sampling = glslChunk(
    LightSamplingModule,
    'light_sampling_functions',
  );
  const bdpt = glslChunk(
    BdptLightSubpathModule,
    'bdpt_light_subpath',
  );

  it('uses the same emitting hemisphere and solid-angle density in NEE and forward-hit MIS', () => {
    expect(sampling).toContain('if ( dot( normal, rayDirection ) < 0.0 )');
    expect(sampling).toContain('float cosTheta = dot( - rayDirection, normal );');
    expect(sampling).toContain('float cosLight = dot( lightNormal, - direction );');
    expect(sampling).toContain('cosLight > 0.0');
    expect(sampling).not.toContain('abs( light.area');

    const normal = [0, 0, 1] as const;
    const frontDirection = [0, 0, -1] as const;
    const backDirection = [0, 0, 1] as const;
    const area = 24;
    const distance = 3;

    const forwardHitAccepts = (
      direction: readonly [number, number, number],
    ): boolean => dot(normal, direction) < 0;

    expect(forwardHitAccepts(frontDirection)).toBe(true);
    expect(oneSidedAreaPdf(area, distance, normal, frontDirection)).toBeCloseTo(9 / 24, 12);
    expect(forwardHitAccepts(backDirection)).toBe(false);
    expect(oneSidedAreaPdf(area, distance, normal, backDirection)).toBe(0);
  });

  it('marks analytic BDPT roots one-sided while retaining two-sided mesh-area roots', () => {
    expect(bdpt).toContain('vec4( light.castShadowDisabled, 0.0, 0.0, 0.0 )');
    expect(bdpt).toContain('vec4( tri.castShadowDisabled, 1.0, 0.0, 0.0 )');
  });
});
