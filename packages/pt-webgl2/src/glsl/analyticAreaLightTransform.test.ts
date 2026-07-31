import { describe, expect, it } from 'vitest';
import * as ShapeIntersectionModule from './shader/common/shape_intersection_functions.glsl.js';
import * as LightSamplingModule from './shader/sampling/light_sampling_functions.glsl.js';

type Vec3 = readonly [number, number, number];

function glslChunk(module: Record<string, unknown>, name: string): string {
  const chunk = module[name];
  if (typeof chunk !== 'string') {
    throw new TypeError(`Expected GLSL module "${name}" to export a string.`);
  }
  return chunk;
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function gramCoordinates(relative: Vec3, u: Vec3, v: Vec3): readonly [number, number] {
  const uLengthSquared = dot(u, u);
  const vLengthSquared = dot(v, v);
  const axisDot = dot(u, v);
  const relativeU = dot(relative, u);
  const relativeV = dot(relative, v);
  const gramDeterminant = uLengthSquared * vLengthSquared - axisDot * axisDot;
  return [
    (relativeU * vLengthSquared - relativeV * axisDot) / gramDeterminant,
    (relativeV * uLengthSquared - relativeU * axisDot) / gramDeterminant,
  ];
}

describe('analytic area-light affine containment', () => {
  const shapeIntersectionFunctions = glslChunk(
    ShapeIntersectionModule,
    'shape_intersection_functions',
  );
  const lightSamplingFunctions = glslChunk(
    LightSamplingModule,
    'light_sampling_functions',
  );

  it('pins rectangle and circle intersection to the scale-safe affine solve', () => {
    expect(shapeIntersectionFunctions).toContain(
      'VitrumAreaVectorMeasure areaMeasure =',
    );
    expect(shapeIntersectionFunctions).toContain(
      'vitrumMeasureAreaVector( u, v, 1.0 )',
    );
    expect(shapeIntersectionFunctions).toContain(
      'vec3 areaCoordinates = vitrumAreaVectorCoordinates(',
    );
    expect(shapeIntersectionFunctions).toContain('u, v, relative, areaMeasure');
    expect(shapeIntersectionFunctions).not.toContain('float axisDot = dot( u, v );');
    expect(shapeIntersectionFunctions).toContain(
      'all( lessThanEqual( abs( coordinates ), vec2( 0.5 ) ) )',
    );
    expect(shapeIntersectionFunctions).toContain(
      'dot( coordinates, coordinates ) <= 0.25',
    );
    expect(lightSamplingFunctions).not.toContain('u *= 1.0 / dot( u, u );');
    expect(lightSamplingFunctions).not.toContain('v *= 1.0 / dot( v, v );');
  });

  it('recovers sheared rect and disc coordinates that independent projection rejects', () => {
    const u = [2, 0, 0] as const;
    const v = [1, 3, 0] as const;

    // q=(0.5,0.5) is a valid rectangle corner. Independent axis projection
    // produces (0.75,0.6) and incorrectly rejects it.
    const rectPoint = [1.5, 1.5, 0] as const;
    const rectCoordinates = gramCoordinates(rectPoint, u, v);
    expect(rectCoordinates[0]).toBeCloseTo(0.5, 12);
    expect(rectCoordinates[1]).toBeCloseTo(0.5, 12);
    expect(dot(rectPoint, u) / dot(u, u)).toBeGreaterThan(0.5);

    // q=(0.3,0.4) lies on the affine disc boundary. Independent projection
    // produces (0.5,0.46), whose squared radius is > 0.25.
    const discPoint = [1, 1.2, 0] as const;
    const discCoordinates = gramCoordinates(discPoint, u, v);
    expect(discCoordinates[0]).toBeCloseTo(0.3, 12);
    expect(discCoordinates[1]).toBeCloseTo(0.4, 12);
    const projectedU = dot(discPoint, u) / dot(u, u);
    const projectedV = dot(discPoint, v) / dot(v, v);
    expect(projectedU * projectedU + projectedV * projectedV).toBeGreaterThan(0.25);
  });
});
