import { describe, expect, it } from 'vitest';
import type {
  DirectionalEmitter,
  PointEmitter,
  RectAreaEmitter,
} from '@vitrum/core';
import { LIGHT_PIXELS, packLightsTexture } from './lightsTexture.js';

// GPU-FREE: assert the packed RGBA32F byte layout matches the §5 spec / the kept
// GLSL decoder (`lights_struct.glsl.js readLightInfo`, 6 texels/light). We read
// s0.a (type), s1.a (intensity) and s2.a (power) for a directional, a point and
// a rect-area emitter and pin the per-type power/area math.

const luminance = (r: number, g: number, b: number) =>
  0.2126 * r + 0.7152 * g + 0.0722 * b;

/** texel `t` (0..5), channel `c` (0..3) of light `i` in the packed grid.
 *  `LightsTextureData.data` is typed as the `Float32Array | Uint32Array`
 *  contract union; the lights packer always emits `rgba32f`. */
function texel(
  data: Float32Array | Uint32Array,
  i: number,
  t: number,
  c: number,
): number {
  return data[(i * LIGHT_PIXELS + t) * 4 + c]!;
}

describe('packLightsTexture', () => {
  const directional: DirectionalEmitter = {
    id: 'dir',
    kind: 'directional',
    color: [1, 1, 1],
    intensity: 3,
    direction: [0, 1, 0], // already unit
  };
  const point: PointEmitter = {
    id: 'pt',
    kind: 'point',
    color: [0.5, 0.25, 0.75],
    intensity: 2,
    position: [1, 2, 3],
    decay: 2,
    distance: 10,
  };
  // Unit-area rect: uAxis=(2,0,0), vAxis=(0,3,0) → width 2, height 3, area 6.
  const rect: RectAreaEmitter = {
    id: 'rect',
    kind: 'rect-area',
    color: [0.2, 0.4, 0.6],
    intensity: 5,
    position: [-1, 0, 1],
    uAxis: [2, 0, 0],
    vAxis: [0, 3, 0],
  };

  const data = packLightsTexture([directional, point, rect]);

  it('emits a square RGBA32F grid sized for 6 texels/light', () => {
    expect(data.kind).toBe('rgba32f');
    expect(data.lightCount).toBe(3);
    // 3 lights × 6 texels = 18 texels → dim = ceil(sqrt(18)) = 5.
    expect(data.dim).toBe(5);
    expect(data.data.length).toBe(5 * 5 * 4);
  });

  it('packs the directional emitter (type 3, intensity, power)', () => {
    // s0.a = type DIR = 3
    expect(texel(data.data, 0, 0, 3)).toBe(3);
    // s1.a = intensity
    expect(texel(data.data, 0, 1, 3)).toBe(3);
    // s2.xyz = direction toward light
    expect(texel(data.data, 0, 2, 0)).toBeCloseTo(0, 6);
    expect(texel(data.data, 0, 2, 1)).toBeCloseTo(1, 6);
    expect(texel(data.data, 0, 2, 2)).toBeCloseTo(0, 6);
    // s2.a = power = luminance * intensity
    expect(texel(data.data, 0, 2, 3)).toBeCloseTo(
      luminance(1, 1, 1) * 3,
      6,
    );
  });

  it('packs the point emitter (type 4, intensity, power, decay/distance)', () => {
    // s0.a = type POINT = 4
    expect(texel(data.data, 1, 0, 3)).toBe(4);
    // s1.a = intensity
    expect(texel(data.data, 1, 1, 3)).toBe(2);
    // s2.xyz = world position (repeated), s2.a = power
    expect(texel(data.data, 1, 2, 0)).toBe(1);
    expect(texel(data.data, 1, 2, 1)).toBe(2);
    expect(texel(data.data, 1, 2, 2)).toBe(3);
    expect(texel(data.data, 1, 2, 3)).toBeCloseTo(
      luminance(0.5, 0.25, 0.75) * 2,
      6,
    );
    // s4.g = decay, s4.b = distance
    expect(texel(data.data, 1, 4, 1)).toBe(2);
    expect(texel(data.data, 1, 4, 2)).toBe(10);
  });

  it('packs the rect-area emitter (type 0, intensity, power, area)', () => {
    // s0.a = type RECT_AREA = 0
    expect(texel(data.data, 2, 0, 3)).toBe(0);
    // s1.a = intensity
    expect(texel(data.data, 2, 1, 3)).toBe(5);
    // s2.xyz = u-vector
    expect(texel(data.data, 2, 2, 0)).toBe(2);
    expect(texel(data.data, 2, 2, 1)).toBe(0);
    expect(texel(data.data, 2, 2, 2)).toBe(0);
    // s2.a = power = luminance * intensity * (width * height) = lum*5*(2*3)
    expect(texel(data.data, 2, 2, 3)).toBeCloseTo(
      luminance(0.2, 0.4, 0.6) * 5 * 6,
      6,
    );
    // s3.xyz = v-vector, s3.a = area = |u × v| = 6
    expect(texel(data.data, 2, 3, 1)).toBe(3);
    expect(texel(data.data, 2, 3, 3)).toBeCloseTo(6, 6);
  });

  it('handles an empty emitter list (1×1 grid, no crash)', () => {
    const empty = packLightsTexture([]);
    expect(empty.lightCount).toBe(0);
    expect(empty.dim).toBe(1);
    expect(empty.data.length).toBe(4);
  });
});
