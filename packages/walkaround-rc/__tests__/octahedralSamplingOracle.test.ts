import { describe, expect, it } from 'vitest';
import { RC_OCTAHEDRAL_SOLID_ANGLE_WGSL } from '../src/wgsl/octahedralSampling.wgsl.js';

type Vec3 = readonly [number, number, number];

const FOUR_PI = 4 * Math.PI;

function rawOctVector(u: number, v: number): Vec3 {
  let x = u;
  let y = v;
  const z = 1 - Math.abs(u) - Math.abs(v);
  if (z < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * (oldX >= 0 ? 1 : -1);
    y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1);
  }
  return [x, y, z];
}

function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / length, v[1] / length, v[2] / length];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function octSolidAngleDensity(u: number, v: number): number {
  const p = rawOctVector(u, v);
  const lengthSquared = dot(p, p);
  return 1 / (lengthSquared * Math.sqrt(lengthSquared));
}

function sphericalTriangleSolidAngle(a: Vec3, b: Vec3, c: Vec3): number {
  const numerator = Math.abs(dot(a, cross(b, c)));
  const denominator = 1 + dot(a, b) + dot(b, c) + dot(c, a);
  return 2 * Math.atan2(numerator, denominator);
}

function octCellSolidAngle(column: number, row: number, gridSize: number): number {
  const width = 2 / gridSize;
  const u0 = -1 + column * width;
  const v0 = -1 + row * width;
  const u1 = u0 + width;
  const v1 = v0 + width;
  const p00 = normalize(rawOctVector(u0, v0));
  const p10 = normalize(rawOctVector(u1, v0));
  const p01 = normalize(rawOctVector(u0, v1));
  const p11 = normalize(rawOctVector(u1, v1));
  return sphericalTriangleSolidAngle(p00, p10, p01)
    + sphericalTriangleSolidAngle(p10, p11, p01);
}

function integrateJacobian(gridSize: number, strataPerCell = 16): number {
  const resolution = gridSize * strataPerCell;
  const step = 2 / resolution;
  let total = 0;
  for (let row = 0; row < resolution; row++) {
    const v = -1 + (row + 0.5) * step;
    for (let column = 0; column < resolution; column++) {
      const u = -1 + (column + 0.5) * step;
      total += octSolidAngleDensity(u, v) * step * step;
    }
  }
  return total;
}

function sumExactCells(gridSize: number): number {
  let total = 0;
  for (let row = 0; row < gridSize; row++) {
    for (let column = 0; column < gridSize; column++) {
      total += octCellSolidAngle(column, row, gridSize);
    }
  }
  return total;
}

describe('RC octahedral solid-angle CPU oracles', () => {
  it('pins the WGSL helpers exercised by the numerical oracles', () => {
    expect(RC_OCTAHEDRAL_SOLID_ANGLE_WGSL).toContain('fn rcOctahedralSolidAngleDensity');
    expect(RC_OCTAHEDRAL_SOLID_ANGLE_WGSL).toContain('fn rcOctCellSolidAngle');
    expect(RC_OCTAHEDRAL_SOLID_ANGLE_WGSL).toContain('fn rcStratifiedSampleSolidAngle');
  });

  for (const gridSize of [4, 8, 16]) {
    it(`integrates the octahedral Jacobian to 4pi for N=${gridSize}`, () => {
      const total = integrateJacobian(gridSize);
      expect(Math.abs(total - FOUR_PI) / FOUR_PI).toBeLessThan(1e-6);
    });

    it(`tiles the sphere with exact cell solid angles for N=${gridSize}`, () => {
      const total = sumExactCells(gridSize);
      expect(Math.abs(total - FOUR_PI) / FOUR_PI).toBeLessThan(1e-10);
    });
  }
});
