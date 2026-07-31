import { describe, expect, it } from 'vitest';

import {
  classifyAreaVectorF32,
  type AreaEmitterVec3,
} from '../scene/areaEmitterGeometry.js';
import { PT_WEBGPU_PATH_TRACE_CONNECT_WGSL } from '../wgsl/pathTrace/connect.wgsl.js';

type Vec3 = readonly [number, number, number];

function f32(value: number): number {
  return Math.fround(value);
}

function f32Mul(a: number, b: number): number {
  return f32(f32(a) * f32(b));
}

function f32Sub(a: number, b: number): number {
  return f32(f32(a) - f32(b));
}

function scaledVec(
  value: AreaEmitterVec3,
  divisor: number,
): [number, number, number] {
  return [
    f32(value[0] / divisor),
    f32(value[1] / divisor),
    f32(value[2] / divisor),
  ];
}

function meshWeightsScaleSafe(
  ab: AreaEmitterVec3,
  ac: AreaEmitterVec3,
  ap: AreaEmitterVec3,
): Vec3 {
  const measure = classifyAreaVectorF32(ab, ac, 0.5);
  if (!measure.valid) return [1, 0, 0];

  const u = scaledVec(ab, measure.edgeScale);
  const v = scaledVec(ac, measure.edgeScale);
  const rel = scaledVec(ap, measure.edgeScale);
  const absNormal = measure.normal.map(Math.abs);

  let det: number;
  let wbNumerator: number;
  let wcNumerator: number;
  if (absNormal[0]! >= absNormal[1]! && absNormal[0]! >= absNormal[2]!) {
    det = f32Sub(f32Mul(u[1], v[2]), f32Mul(u[2], v[1]));
    wbNumerator = f32Sub(f32Mul(rel[1], v[2]), f32Mul(rel[2], v[1]));
    wcNumerator = f32Sub(f32Mul(u[1], rel[2]), f32Mul(u[2], rel[1]));
  } else if (absNormal[1]! >= absNormal[2]!) {
    det = f32Sub(f32Mul(u[2], v[0]), f32Mul(u[0], v[2]));
    wbNumerator = f32Sub(f32Mul(rel[2], v[0]), f32Mul(rel[0], v[2]));
    wcNumerator = f32Sub(f32Mul(u[2], rel[0]), f32Mul(u[0], rel[2]));
  } else {
    det = f32Sub(f32Mul(u[0], v[1]), f32Mul(u[1], v[0]));
    wbNumerator = f32Sub(f32Mul(rel[0], v[1]), f32Mul(rel[1], v[0]));
    wcNumerator = f32Sub(f32Mul(u[0], rel[1]), f32Mul(u[1], rel[0]));
  }
  if (det === 0) return [1, 0, 0];

  const wb = f32(wbNumerator / det);
  const wc = f32(wcNumerator / det);
  return [f32(1 - wb - wc), wb, wc];
}

describe('textured mesh-area light barycentric scale contract', () => {
  it.each([
    1e-12,
    1,
    1e10,
  ])('recovers the same weights at representable edge scale %s', (scale) => {
    const ab: Vec3 = [scale, 0, 0];
    const ac: Vec3 = [0, scale, 0];
    const ap: Vec3 = [0.3 * scale, 0.5 * scale, 0];
    const weights = meshWeightsScaleSafe(ab, ac, ap);
    expect(weights[0]).toBeCloseTo(0.2, 6);
    expect(weights[1]).toBeCloseTo(0.3, 6);
    expect(weights[2]).toBeCloseTo(0.5, 6);
    expect(weights[0] + weights[1] + weights[2]).toBeCloseTo(1, 7);
  });

  it('uses the shared equilibrated measure and dominant-projection solver', () => {
    const start = PT_WEBGPU_PATH_TRACE_CONNECT_WGSL.indexOf(
      'fn meshAreaLightWeightsAtPoint(',
    );
    const end = PT_WEBGPU_PATH_TRACE_CONNECT_WGSL.indexOf(
      '\nfn sampleMeshAreaLightRadiance(',
      start,
    );
    const source = PT_WEBGPU_PATH_TRACE_CONNECT_WGSL.slice(start, end);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    expect(source).toContain('measureAreaVector(ab, ac, 0.5)');
    expect(source).toContain('solveAreaVectorCoordinates(');
    expect(source).not.toContain('let d00 = dot(ab, ab);');
    expect(source).not.toContain('let denom = d00 * d11 - d01 * d01;');
  });
});
