import { describe, expect, it } from 'vitest';
import * as LightSamplingModule from './shader/sampling/light_sampling_functions.glsl.js';

const light_sampling_functions = (
  LightSamplingModule as unknown as Record<string, string>
)['light_sampling_functions']!;

describe('KHR_lights_punctual range attenuation', () => {
  it('uses the unsquared quartic range window required by KHR_lights_punctual', () => {
    const compact = light_sampling_functions.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'distanceFalloff *= saturate( 1.0 - pow4( lightDistance / cutoffDistance ) )',
    );
    expect(compact).not.toContain(
      'distanceFalloff *= pow2( saturate( 1.0 - pow4(',
    );

    const normalizedDistance = 0.5;
    const khrWindow = 1 - normalizedDistance ** 4;
    const oldSquaredWindow = khrWindow ** 2;
    expect(khrWindow).toBe(0.9375);
    expect(oldSquaredWindow).toBe(0.87890625);
  });

  it('defines the intentional zero-penumbra spot cone without equal-edge smoothstep', () => {
    const compact = light_sampling_functions.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'if ( penumbraCosine == coneCosine ) { return angleCosine >= coneCosine ? 1.0 : 0.0; }',
    );
    expect(compact).toContain('penumbraCosine < coneCosine');
  });

  it('fails receiver-dependent point and spot radiance closed before publication', () => {
    const compact = light_sampling_functions.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'vec3 realizedRadiance = sourceRadiance * distanceAttenuation * angularAttenuation',
    );
    expect(compact.match(/finitePunctualRadiance\(/g)).toHaveLength(3);
    expect(compact).not.toContain(
      'light.color * light.intensity * distanceAttenuation * spotAttenuation',
    );
    expect(compact).not.toContain(
      'light.color * light.intensity * distanceFalloff',
    );
  });

  it('initializes every mesh LightRecord field before its first early return', () => {
    const functionStart = light_sampling_functions.indexOf(
      'LightRecord sampleMeshAreaLight(',
    );
    const firstEarlyReturn = light_sampling_functions.indexOf(
      'if ( meshLightCount == 0u',
      functionStart,
    );
    expect(functionStart).toBeGreaterThanOrEqual(0);
    expect(firstEarlyReturn).toBeGreaterThan(functionStart);
    const initialization = light_sampling_functions.slice(
      functionStart,
      firstEarlyReturn,
    );
    for (const field of [
      'point',
      'normal',
      'dist',
      'direction',
      'pdf',
      'emission',
      'type',
      'discretePdf',
      'castShadowDisabled',
      'delta',
      'hasTargetFace',
      'targetFaceIndex',
    ]) {
      expect(initialization, field).toContain(`rec.${field} =`);
    }
    expect(light_sampling_functions).toContain(
      'rec.targetFaceIndex = tri.sourceFaceIndex;',
    );
  });

  it('uses an ignored finite point for the max-f32 directional distance sentinel', () => {
    expect(light_sampling_functions).toContain('rec.dist = INFINITY;');
    expect(light_sampling_functions).toContain('rec.point = vec3( 0.0 );');
    expect(light_sampling_functions).not.toContain(
      'rec.point = - rec.direction * rec.dist;',
    );
  });
});
