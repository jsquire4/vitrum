import { describe, expect, it } from 'vitest';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';
import { composeTraceGlsl } from './composeTraceGlsl.js';
import * as CameraModule from './render/camera_util_functions.glsl.js';
import * as EquirectModule from './shader/sampling/equirect_sampling_functions.glsl.js';

function glslChunk(module: Record<string, unknown>, name: string): string {
  const value = module[name];
  if (typeof value !== 'string') {
    throw new TypeError(`Expected GLSL module "${name}" to export a string.`);
  }
  return value;
}

const cameraUtil = glslChunk(CameraModule, 'camera_util_functions');
const equirectFunctions = glslChunk(EquirectModule, 'equirect_functions');

type Vec3 = readonly [number, number, number];

function f32(value: number): number {
  return Math.fround(value);
}

function f32Add(a: number, b: number): number {
  return f32(f32(a) + f32(b));
}

function f32Subtract(a: number, b: number): number {
  return f32(f32(a) - f32(b));
}

function f32Multiply(a: number, b: number): number {
  return f32(f32(a) * f32(b));
}

function stableNormalize(value: Vec3, fallback: Vec3): Vec3 {
  const scale = Math.max(Math.abs(value[0]), Math.abs(value[1]), Math.abs(value[2]));
  if (!(scale > 0) || !Number.isFinite(scale)) return fallback;
  const scaled: Vec3 = [
    value[0] / scale,
    value[1] / scale,
    value[2] / scale,
  ];
  const length = Math.hypot(...scaled);
  return [
    scaled[0] / length,
    scaled[1] / length,
    scaled[2] / length,
  ];
}

function naiveF32Normalize(value: Vec3): Vec3 {
  const squaredLength = f32Add(
    f32Add(
      f32Multiply(value[0], value[0]),
      f32Multiply(value[1], value[1]),
    ),
    f32Multiply(value[2], value[2]),
  );
  const length = f32(Math.sqrt(squaredLength));
  return [
    f32(value[0] / length),
    f32(value[1] / length),
    f32(value[2] / length),
  ];
}

function oldWorldSpaceFocusDirection(
  origin: Vec3,
  baseDirection: Vec3,
  focusDistance: number,
  apertureOffset: Vec3,
): Vec3 {
  return [0, 1, 2].map((axis) => {
    const focalPoint = f32Add(
      origin[axis]!,
      f32Multiply(baseDirection[axis]!, focusDistance),
    );
    const shiftedOrigin = f32Add(origin[axis]!, apertureOffset[axis]!);
    return f32Subtract(focalPoint, shiftedOrigin);
  }) as unknown as Vec3;
}

function unscaledRelativeFocusDirection(
  baseDirection: Vec3,
  focusDistance: number,
  apertureOffset: Vec3,
): Vec3 {
  return [0, 1, 2].map((axis) =>
    f32Subtract(
      f32Multiply(baseDirection[axis]!, focusDistance),
      apertureOffset[axis]!,
    ),
  ) as unknown as Vec3;
}

function scaledRelativeFocusDirection(
  baseDirection: Vec3,
  focusDistance: number,
  apertureOffset: Vec3,
): Vec3 {
  const scale = Math.max(
    focusDistance,
    Math.abs(apertureOffset[0]),
    Math.abs(apertureOffset[1]),
    Math.abs(apertureOffset[2]),
  );
  return [0, 1, 2].map((axis) =>
    f32Subtract(
      f32Multiply(baseDirection[axis]!, f32(focusDistance / scale)),
      f32(apertureOffset[axis]! / scale),
    ),
  ) as unknown as Vec3;
}

describe('pt-webgl2 camera and background scale safety', () => {
  it('constructs DOF direction in relative space and keeps the pinhole fallback', () => {
    const origin: Vec3 = [3e38, 0, 0];
    const baseDirection: Vec3 = [1, 0, 0];
    const apertureOffset: Vec3 = [0, 0, 0];

    expect(
      oldWorldSpaceFocusDirection(
        origin,
        baseDirection,
        1e30,
        apertureOffset,
      ),
    ).toEqual([0, 0, 0]);
    expect(
      scaledRelativeFocusDirection(baseDirection, 1e30, apertureOffset),
    ).toEqual([1, 0, 0]);

    const compact = cameraUtil.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'float relativeFocusScale = max( physicalCamera.focusDistance, apertureOffsetScale );',
    );
    expect(compact).toContain(
      'vec3 relativeFocusDirection = baseDirection * ( physicalCamera.focusDistance / relativeFocusScale ) - apertureWorldOffset / relativeFocusScale;',
    );
    expect(compact).toContain(
      'candidateOriginFinite && vitrumFiniteNonZeroVec3( relativeFocusDirection )',
    );
    expect(compact).toContain(
      'ray.direction = vitrumNormalizeVec3( relativeFocusDirection, baseDirection );',
    );
    expect(compact).not.toContain('vec3 focalPoint');
  });

  it('bypasses aperture sampling at zero bokeh and uses stable camera normalization', () => {
    const compact = cameraUtil.replace(/\s+/g, ' ');
    const apertureGuard = compact.indexOf(
      'if ( physicalCamera.bokehSize > 0.0 )',
    );
    const apertureRng = compact.indexOf('vec3 shapeUVW= rand3( 1 );');
    expect(apertureGuard).toBeGreaterThan(-1);
    expect(apertureRng).toBeGreaterThan(apertureGuard);
    expect(compact).toContain(
      'vec3 baseDirection = vitrumNormalizeVec3( ray.direction, cameraForward );',
    );
    expect(cameraUtil).not.toMatch(/\bnormalize\s*\(/);
  });

  it('does not evaluate a reciprocal for sub-unit anamorphic ratios', () => {
    const compact = cameraUtil.replace(/\s+/g, ' ');
    const lowRatioBranch = compact.indexOf(
      'if ( anamorphicRatio < 1.0 )',
    );
    const directLowRatioScale = compact.indexOf(
      'anamorphicScale = vec2( anamorphicRatio, 1.0 );',
    );
    const reciprocalScale = compact.indexOf(
      'anamorphicScale = vec2( 1.0, 1.0 / anamorphicRatio );',
    );
    expect(lowRatioBranch).toBeGreaterThan(-1);
    expect(directLowRatioScale).toBeGreaterThan(lowRatioBranch);
    expect(reciprocalScale).toBeGreaterThan(directLowRatioScale);
    expect(compact).not.toContain(
      'saturate( vec2( anamorphicRatio, 1.0 / anamorphicRatio ) )',
    );
  });

  it('keeps extreme relative DOF subtraction finite by scaling both terms', () => {
    const overflowed = unscaledRelativeFocusDirection(
      [1, 0, 0],
      Math.fround(3.4e38),
      [Math.fround(-3.4e38), 0, 0],
    );
    expect(overflowed[0]).toBe(Number.POSITIVE_INFINITY);
    const scaled = scaledRelativeFocusDirection(
      [1, 0, 0],
      Math.fround(3.4e38),
      [Math.fround(-3.4e38), 0, 0],
    );
    expect(scaled).toEqual([2, 0, 0]);
    expect(stableNormalize(scaled, [1, 0, 0])).toEqual([1, 0, 0]);
  });

  it('normalizes an extreme but finite background perturbation without squared-length overflow', () => {
    const perturbation: Vec3 = [
      Math.fround(Math.fround(3.402823466e38 * 0.5) * Math.SQRT1_2),
      Math.fround(Math.fround(3.402823466e38 * 0.5) * Math.SQRT1_2),
      0,
    ];
    expect(perturbation.every(Number.isFinite)).toBe(true);
    expect(naiveF32Normalize(perturbation)).toEqual([0, 0, 0]);

    const normalized = stableNormalize(perturbation, [1, 0, 0]);
    expect(normalized[0]).toBeCloseTo(Math.SQRT1_2, 7);
    expect(normalized[1]).toBeCloseTo(Math.SQRT1_2, 7);

    const source = composeTraceGlsl(DEFAULT_TRACE_FEATURES).replace(/\s+/g, ' ');
    expect(source).toContain(
      'vec3 rotatedDirection = vitrumNormalizeVec3( envRotation3x3 * backgroundDirection, backgroundDirection );',
    );
    expect(source).toContain(
      'vec3 combinedDirection = rotatedDirection / combinationScale + perturbation / combinationScale;',
    );
    expect(source).toContain(
      'sampleDir = vitrumNormalizeVec3( combinedDirection, rotatedDirection );',
    );
  });

  it('uses the same stable fallback at the equirectangular lookup boundary', () => {
    const compact = equirectFunctions.replace(/\s+/g, ' ');
    expect(compact).toContain(
      'vec3 n = vitrumNormalizeVec3( direction, vec3( 0.0, 1.0, 0.0 ) );',
    );
    expect(compact).toContain(
      'float u = horizontalScale > 0.0 ? fract( atan( n.z, n.x ) / ( 2.0 * PI ) + 0.5 ) : 0.5;',
    );
    expect(compact).not.toContain('vec3 n = normalize( direction );');

    const ordinary = stableNormalize([3, 4, 0], [0, 1, 0]);
    expect(ordinary).toEqual([0.6, 0.8, 0]);
    expect(stableNormalize([0, 0, 0], [0, 1, 0])).toEqual([0, 1, 0]);
  });
});
