import { describe, expect, it } from 'vitest';
import {
  applyBeerLambertColor,
  BEER_LAMBERT_WGSL,
  BVH_INTERSECT_CORE_WGSL,
  MOLLER_TRUMBORE_WGSL,
} from '../index.js';

type Vec3 = readonly [number, number, number];
interface TriangleHit {
  readonly t: number;
  readonly bary: Vec3;
  readonly det: number;
  readonly normal: Vec3;
}

function intersect(
  origin: Vec3,
  dir: Vec3,
  a: Vec3,
  b: Vec3,
  c: Vec3,
  tMin: number,
): TriangleHit | null {
  const sub = (x: Vec3, y: Vec3): Vec3 => [
    x[0] - y[0],
    x[1] - y[1],
    x[2] - y[2],
  ];
  const cross = (x: Vec3, y: Vec3): Vec3 => [
    x[1] * y[2] - x[2] * y[1],
    x[2] * y[0] - x[0] * y[2],
    x[0] * y[1] - x[1] * y[0],
  ];
  const dot = (x: Vec3, y: Vec3): number =>
    x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  const vecLength = (x: Vec3): number => Math.sqrt(dot(x, x));
  const scaleVec = (x: Vec3, scale: number): Vec3 => [
    x[0] * scale,
    x[1] * scale,
    x[2] * scale,
  ];
  const e1Raw = sub(b, a);
  const e2Raw = sub(c, a);
  const edgeScale = Math.max(...e1Raw.map(Math.abs), ...e2Raw.map(Math.abs));
  const directionScale = Math.max(...dir.map(Math.abs));
  if (!(edgeScale > 0) || !(directionScale > 0)) return null;
  const e1 = scaleVec(e1Raw, 1 / edgeScale);
  const e2 = scaleVec(e2Raw, 1 / edgeScale);
  const direction = scaleVec(dir, 1 / directionScale);
  const n = cross(e1, e2);
  const nLen = vecLength(n);
  const dLen = vecLength(direction);
  if (!(nLen > 0) || !(dLen > 0)) return null;
  const det = -dot(direction, n);
  if (Math.abs(det) / (nLen * dLen) <= 1e-7) return null;
  const invDet = 1 / det;
  const ao = scaleVec(sub(origin, a), 1 / edgeScale);
  const dao = cross(ao, direction);
  const u = dot(e2, dao) * invDet;
  const v = -dot(e1, dao) * invDet;
  const w = 1 - u - v;
  const t = dot(ao, n) * invDet * edgeScale / directionScale;
  if (
    u < -1e-6 ||
    v < -1e-6 ||
    w < -1e-6 ||
    t < Math.max(tMin, 0)
  ) {
    return null;
  }
  return {
    t,
    bary: [w, u, v],
    det,
    normal: scaleVec(n, 1 / nLen),
  };
}

describe('shared transport invariants', () => {
  it('keeps determinant, barycentric, and ray-distance tolerances independent', () => {
    expect(MOLLER_TRUMBORE_WGSL).toContain(
      'const MOLLER_TRUMBORE_ANGULAR_EPSILON: f32 = 1e-7;',
    );
    expect(MOLLER_TRUMBORE_WGSL).toContain(
      'const MOLLER_TRUMBORE_BARYCENTRIC_EPSILON: f32 = 1e-6;',
    );
    expect(MOLLER_TRUMBORE_WGSL).toContain('t < max(tMin, 0.0)');
    expect(MOLLER_TRUMBORE_WGSL).not.toContain('abs(det) < triEps');
    expect(MOLLER_TRUMBORE_WGSL).toContain('let edgeScale = max(');
    expect(MOLLER_TRUMBORE_WGSL).not.toContain('normalLen2 <= 1e-30');

    // A perpendicular 10 nm-edge triangle has an unscaled f32 cross product
    // below the former area floor. Its conditioning, not its units, decides.
    expect(intersect(
      [2.5e-9, 2.5e-9, 1],
      [0, 0, -1],
      [0, 0, 0],
      [1e-8, 0, 0],
      [0, 1e-8, 0],
      1e-5,
    )?.t).toBeCloseTo(1, 12);
  });

  it.each([
    ['tiny', 1e-24, 0],
    ['huge', 1e20, Number.POSITIVE_INFINITY],
  ] as const)(
    'preserves %s-triangle barycentrics, determinant side, and normal when a raw f32 cross is unusable',
    (_label, scale, rawF32Cross) => {
      expect(Math.fround(Math.fround(scale) * Math.fround(scale))).toBe(rawF32Cross);

      const hit = intersect(
        [scale * 0.25, scale * 0.5, 1],
        [0, 0, -1],
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        0,
      );

      expect(hit).not.toBeNull();
      expect(hit?.t).toBeCloseTo(1, 12);
      expect(hit?.bary[0]).toBeCloseTo(0.25, 12);
      expect(hit?.bary[1]).toBeCloseTo(0.25, 12);
      expect(hit?.bary[2]).toBeCloseTo(0.5, 12);
      expect(hit?.det).toBeGreaterThan(0);
      expect(hit?.normal).toEqual([0, 0, 1]);
    },
  );

  it('keeps the canonical payload as the only shared binary reconstruction source', () => {
    expect(MOLLER_TRUMBORE_WGSL).toContain('normal: vec3f');
    expect(MOLLER_TRUMBORE_WGSL).toContain(
      'result.normal = normalDirection / normalLength;',
    );
    expect(BVH_INTERSECT_CORE_WGSL).toContain(
      'result.normal    = result.side * core.normal;',
    );

    const adapterStart = BVH_INTERSECT_CORE_WGSL.indexOf('fn intersectTriangle(');
    const traversalStart = BVH_INTERSECT_CORE_WGSL.indexOf(
      'fn bvhIntersectFirstHit(',
      adapterStart,
    );
    const adapter = BVH_INTERSECT_CORE_WGSL.slice(adapterStart, traversalStart);
    expect(adapter).not.toContain('cross(e1');
    expect(adapter).not.toContain('normalize(');
  });

  it('uses the no-absorption identity for omitted Beer operands', () => {
    expect(applyBeerLambertColor([0.25, 0.5, 1], undefined, 2)).toEqual([1, 1, 1]);
    expect(applyBeerLambertColor([0.25, 0.5, 1], 2, undefined)).toEqual([1, 1, 1]);
    expect(applyBeerLambertColor([0.25, 0.5, 1], 0, 2)).toEqual([1, 1, 1]);
    expect(applyBeerLambertColor([0.25, 0.5, 1], 2, 2)).toEqual([0.25, 0.5, 1]);
    expect(BEER_LAMBERT_WGSL).toContain(
      'let exponent = distance / attenuationDistance;',
    );
    expect(BEER_LAMBERT_WGSL).toContain(
      'return select(positive, vec3f(0.0), color <= vec3f(0.0));',
    );
  });
});
