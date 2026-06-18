import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_KERNEL_WGSL } from '../wgsl/pathTrace/kernel.wgsl.js';
import { PT_WEBGPU_ADJOINT_PASS_WGSL } from '../wgsl/pathTrace/adjointPass.wgsl.js';

type V3 = readonly [number, number, number];

const TAU = 2 * Math.PI;

const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const len = (v: V3): number => Math.hypot(v[0], v[1], v[2]);
const normalize = (v: V3): V3 => {
  const l = len(v);
  return [v[0] / l, v[1] / l, v[2] / l];
};
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (v: V3, s: number): V3 => [v[0] * s, v[1] * s, v[2] * s];

function decodeAngularDiameter(raw: number): { angularDiameter: number; shadowDisabled: boolean } {
  const shadowDisabled = raw < 0;
  return {
    angularDiameter: shadowDisabled ? -1 - raw : raw,
    shadowDisabled,
  };
}

function sampleCone(axisIn: V3, angularDiameter: number, xi1: number, xi2: number): V3 {
  const axis = normalize(axisIn);
  const cosHalfAngle = Math.cos(angularDiameter * 0.5);
  const cosTheta = cosHalfAngle + (1 - cosHalfAngle) * xi1;
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const phi = TAU * xi2;
  const tangentX: V3 = Math.abs(axis[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const basisY = normalize(cross(axis, tangentX));
  const basisX = cross(basisY, axis);
  return normalize(add(
    add(scale(basisX, sinTheta * Math.cos(phi)), scale(basisY, sinTheta * Math.sin(phi))),
    scale(axis, cosTheta),
  ));
}

describe('pt-webgpu directional soft-sun cone sampler — independent oracle', () => {
  it('decodes castShadow sign-encoded angularDiameter without changing the cone width', () => {
    expect(decodeAngularDiameter(0.014)).toEqual({ angularDiameter: 0.014, shadowDisabled: false });
    expect(decodeAngularDiameter(-1.014)).toEqual({ angularDiameter: 0.014000000000000012, shadowDisabled: true });
  });

  it('maps xi1 endpoints to cone rim and centre, with unit-length directions', () => {
    const axis = normalize([0.25, -0.35, 0.9]);
    const angularDiameter = 0.12;
    const cosHalfAngle = Math.cos(angularDiameter * 0.5);
    const rim = sampleCone(axis, angularDiameter, 0, 0.125);
    const centre = sampleCone(axis, angularDiameter, 1, 0.875);

    expect(len(rim)).toBeCloseTo(1, 12);
    expect(len(centre)).toBeCloseTo(1, 12);
    expect(dot(axis, rim)).toBeCloseTo(cosHalfAngle, 12);
    expect(dot(axis, centre)).toBeCloseTo(1, 12);
  });

  it('is uniform in solid angle: cos(theta) is uniform over [cosHalfAngle, 1]', () => {
    const axis = normalize([0.93, 0.21, 0.3]); // exercises the alternate tangent branch.
    const angularDiameter = 0.24;
    const cosHalfAngle = Math.cos(angularDiameter * 0.5);
    const n = 256;
    let sumCos = 0;
    let lowerHalf = 0;

    for (let i = 0; i < n; i++) {
      const xi1 = (i + 0.5) / n;
      const xi2 = (i * 0.6180339887498948) % 1;
      const dir = sampleCone(axis, angularDiameter, xi1, xi2);
      const cosTheta = dot(axis, dir);
      expect(cosTheta).toBeGreaterThanOrEqual(cosHalfAngle - 1e-12);
      expect(cosTheta).toBeLessThanOrEqual(1 + 1e-12);
      sumCos += cosTheta;
      if (cosTheta < (cosHalfAngle + 1) * 0.5) lowerHalf++;
    }

    expect(sumCos / n).toBeCloseTo((cosHalfAngle + 1) * 0.5, 12);
    expect(lowerHalf).toBe(n / 2);
  });

  it('forward full-tier and adjoint replay shaders are linked to the same solid-angle cone mapping', () => {
    for (const wgsl of [PT_WEBGPU_PATH_TRACE_KERNEL_WGSL, PT_WEBGPU_ADJOINT_PASS_WGSL]) {
      expect(wgsl).toContain('let cosHalfAngle = cos(');
      expect(wgsl).toContain('let cosTheta = mix(cosHalfAngle, 1.0, xi1);');
      expect(wgsl).toContain('let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));');
      expect(wgsl).toContain('let tangentX = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(');
    }
    expect(PT_WEBGPU_PATH_TRACE_KERNEL_WGSL).toContain('let angDiam = select(angDiamRaw, -1.0 - angDiamRaw, dirShadowDisabled);');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let angularDiameter = select(angularDiameterRaw, -1.0 - angularDiameterRaw, directionalShadowDisabled);');
  });
});
