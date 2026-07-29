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
});
