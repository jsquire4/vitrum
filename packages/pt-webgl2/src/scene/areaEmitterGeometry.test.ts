import { describe, expect, it } from 'vitest';
import type { DiscAreaEmitter, RectAreaEmitter } from '@vitrum/core';
import {
  classifyAreaVectorF32,
  normalizeDirectionF32,
} from './areaEmitterGeometry.js';
import { LIGHT_PIXELS, packLightsTexture } from './lightsTexture.js';
import * as UtilFunctionsNS from '../glsl/shader/common/util_functions.glsl.js';
import * as ShapeIntersectionFunctionsNS from '../glsl/shader/common/shape_intersection_functions.glsl.js';
import * as LightSamplingFunctionsNS from '../glsl/shader/sampling/light_sampling_functions.glsl.js';
import * as BdptLightSubpathNS from '../glsl/render/bdpt_light_subpath.glsl.js';

const util_functions =
  (UtilFunctionsNS as unknown as Record<string, string>)['util_functions'] ?? '';
const shape_intersection_functions =
  (ShapeIntersectionFunctionsNS as unknown as Record<string, string>)[
    'shape_intersection_functions'
  ] ?? '';
const light_sampling_functions =
  (LightSamplingFunctionsNS as unknown as Record<string, string>)[
    'light_sampling_functions'
  ] ?? '';
const bdpt_light_subpath =
  (BdptLightSubpathNS as unknown as Record<string, string>)['bdpt_light_subpath'] ?? '';

function texel(
  data: Float32Array | Uint32Array,
  light: number,
  texelIndex: number,
  channel: number,
): number {
  return data[(light * LIGHT_PIXELS + texelIndex) * 4 + channel]!;
}

function rect(uAxis: [number, number, number], vAxis: [number, number, number]): RectAreaEmitter {
  return {
    kind: 'rect-area',
    id: 'scale-rect',
    position: [0, 0, 0],
    uAxis,
    vAxis,
    color: [1, 1, 1],
    intensity: 1,
  };
}

describe('pt-webgl2 area-emitter Float32 scale contract', () => {
  it('packs tiny and huge-near-parallel finite areas without squared-cross loss', () => {
    const tiny = packLightsTexture([rect([1e-18, 0, 0], [0, 1e-18, 0])]);
    expect(texel(tiny.data, 0, 3, 3) / 4e-36).toBeCloseTo(1, 5);

    const base = Math.fround(5e19);
    const next = Math.fround(base * (1 + 1e-6));
    const huge = packLightsTexture([rect([base, base, 0], [base, next, 0])]);
    const area = texel(huge.data, 0, 3, 3);
    expect(area).toBeGreaterThan(1e33);
    expect(area).toBeLessThan(1e36);
  });

  it('fails closed for exact degeneracy and unrepresentable area reciprocals', () => {
    expect(classifyAreaVectorF32([1, 2, 3], [2, 4, 6], 1)).toEqual({
      valid: false,
      reason: 'degenerate',
    });
    expect(classifyAreaVectorF32([1e-20, 0, 0], [0, 1e-20, 0], 1)).toEqual({
      valid: false,
      reason: 'unrepresentable-area',
    });
    expect(() => packLightsTexture([rect([1e20, 0, 0], [0, 1e20, 0])]))
      .toThrow(/unrepresentable-area/);
  });

  it('builds a correct disc basis from scale-free tiny normals', () => {
    const disc: DiscAreaEmitter = {
      kind: 'disc-area',
      id: 'tiny-normal-disc',
      position: [0, 0, 0],
      normal: [1e-300, 0, 0],
      radius: 2,
      color: [1, 1, 1],
      intensity: 1,
    };
    const packed = packLightsTexture([disc]);
    const u: [number, number, number] = [
      texel(packed.data, 0, 2, 0),
      texel(packed.data, 0, 2, 1),
      texel(packed.data, 0, 2, 2),
    ];
    const v: [number, number, number] = [
      texel(packed.data, 0, 3, 0),
      texel(packed.data, 0, 3, 1),
      texel(packed.data, 0, 3, 2),
    ];
    const measured = classifyAreaVectorF32(u, v, Math.PI / 4);
    expect(measured.valid).toBe(true);
    if (measured.valid) {
      expect(measured.normal).toEqual([1, 0, 0]);
      expect(measured.area).toBeCloseTo(Math.PI * 4, 5);
    }
    expect(normalizeDirectionF32([1e300, 0, 0])).toEqual([1, 0, 0]);
  });

  it('wires equilibrated area and affine-coordinate helpers through production GLSL', () => {
    expect(util_functions).toContain('vitrumMeasureAreaVector');
    expect(util_functions).toContain('vitrumAreaVectorCoordinates');
    expect(shape_intersection_functions).toContain('vitrumAreaVectorCoordinates');
    expect(light_sampling_functions).toContain('vitrumMeasureAreaVector');
    expect(bdpt_light_subpath).toContain('vitrumMeasureAreaVector');
  });
});
